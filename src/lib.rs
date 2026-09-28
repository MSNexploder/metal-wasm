//! Execute a supported WebAssembly function once per Metal thread.

use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use metal::objc::{Message, runtime::Sel};
use metal::{
    CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};
use wasmparser::{ExternalKind, FunctionBody, Operator, Parser, Payload, ValType, Validator};

/// A compiled `(param i32) (result i32)` WebAssembly export.
pub struct Kernel {
    device: Device,
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
}

/// An error while validating, compiling, or running a WebAssembly export.
#[derive(Debug)]
pub enum Error {
    InvalidWasm(wasmparser::BinaryReaderError),
    Unsupported(String),
    Metal(String),
}

impl Kernel {
    /// Compile a WebAssembly export for the system's default Metal device.
    pub fn new(wasm: &[u8], export: &str) -> Result<Self, Error> {
        let source = translate(wasm, export)?;
        let device = Device::system_default()
            .ok_or_else(|| Error::Metal("no Metal device is available".into()))?;
        let library = device
            .new_library_with_source(&source, &CompileOptions::new())
            .map_err(Error::Metal)?;
        let function = library
            .get_function("run_wasm", None)
            .map_err(Error::Metal)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| Error::Metal(error.to_string()))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    /// Run one independent invocation for each input. Values are raw i32 bit patterns.
    pub fn run(&self, inputs: &[u32]) -> Result<Vec<u32>, Error> {
        self.run_timed(inputs).map(|(output, _)| output)
    }

    /// Run the batch and return Metal's GPU execution time when available.
    /// This excludes host-side buffer copies and command submission.
    pub fn run_timed(&self, inputs: &[u32]) -> Result<(Vec<u32>, Option<Duration>), Error> {
        if inputs.is_empty() {
            return Ok((Vec::new(), None));
        }
        let count = u32::try_from(inputs.len())
            .map_err(|_| Error::Unsupported("more than u32::MAX invocations".into()))?;
        let bytes = inputs
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| Error::Unsupported("input buffer size overflow".into()))?;
        let bytes = u64::try_from(bytes)
            .map_err(|_| Error::Unsupported("input buffer size overflow".into()))?;
        if bytes > self.device.max_buffer_length() {
            return Err(Error::Unsupported(
                "inputs exceed the Metal buffer limit".into(),
            ));
        }

        let input = self.device.new_buffer_with_data(
            inputs.as_ptr().cast(),
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let output = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, size_of::<u32>() as u64, (&count as *const u32).cast());
        let width = self
            .pipeline
            .thread_execution_width()
            .min(self.pipeline.max_total_threads_per_threadgroup());
        let groups = (count as u64).div_ceil(width);
        encoder.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(width, 1, 1));
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != MTLCommandBufferStatus::Completed {
            return Err(Error::Metal(format!(
                "Metal command failed: {:?}",
                command.status()
            )));
        }

        // Metal-rs does not wrap these MTLCommandBuffer properties.
        let gpu_start: Option<f64> =
            unsafe { command.send_message(Sel::register("GPUStartTime"), ()) }.ok();
        let gpu_end: Option<f64> =
            unsafe { command.send_message(Sel::register("GPUEndTime"), ()) }.ok();
        let gpu_time = gpu_start.zip(gpu_end).and_then(|(start, end)| {
            (start > 0.0 && end.is_finite() && end >= start)
                .then(|| Duration::from_secs_f64(end - start))
        });

        // The shared output buffer is readable after the command completes.
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), inputs.len()) };
        Ok((values.to_vec(), gpu_time))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWasm(error) => write!(formatter, "invalid WebAssembly: {error}"),
            Self::Unsupported(reason) => write!(formatter, "unsupported WebAssembly: {reason}"),
            Self::Metal(reason) => write!(formatter, "Metal error: {reason}"),
        }
    }
}

impl StdError for Error {}

impl From<wasmparser::BinaryReaderError> for Error {
    fn from(error: wasmparser::BinaryReaderError) -> Self {
        Self::InvalidWasm(error)
    }
}

fn translate(wasm: &[u8], export_name: &str) -> Result<String, Error> {
    Validator::new().validate_all(wasm)?;

    let mut types = Vec::new();
    let mut function_types = Vec::new();
    let mut bodies = Vec::new();
    let mut export_index = None;
    for payload in Parser::new(0).parse_all(wasm) {
        match payload? {
            Payload::Version { .. } | Payload::CustomSection(_) | Payload::End(_) => {}
            Payload::TypeSection(section) => {
                types = section
                    .into_iter_err_on_gc_types()
                    .collect::<Result<_, _>>()?;
            }
            Payload::FunctionSection(section) => {
                function_types = section.into_iter().collect::<Result<_, _>>()?;
            }
            Payload::ExportSection(section) => {
                for item in section {
                    let item = item?;
                    if item.name == export_name && item.kind == ExternalKind::Func {
                        export_index = Some(item.index);
                    }
                }
            }
            Payload::CodeSectionEntry(body) => bodies.push(body),
            Payload::CodeSectionStart { .. } => {}
            other => {
                return Err(Error::Unsupported(format!(
                    "module section {:?}",
                    other.as_section().map(|(id, _)| id)
                )));
            }
        }
    }

    let index = export_index
        .ok_or_else(|| Error::Unsupported(format!("export {export_name:?} is not a function")))?
        as usize;
    let type_index = *function_types
        .get(index)
        .ok_or_else(|| Error::Unsupported("exported function index is out of range".into()))?
        as usize;
    let ty = types
        .get(type_index)
        .ok_or_else(|| Error::Unsupported("exported function type is out of range".into()))?;
    if ty.params() != [ValType::I32] || ty.results() != [ValType::I32] {
        return Err(Error::Unsupported(
            "export must have type (param i32) (result i32)".into(),
        ));
    }
    let body = bodies
        .get(index)
        .ok_or_else(|| Error::Unsupported("exported function body is missing".into()))?;
    translate_body(body)
}

fn translate_body(body: &FunctionBody<'_>) -> Result<String, Error> {
    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str("kernel void run_wasm(device const uint* input [[buffer(0)]], device uint* output [[buffer(1)]], constant uint& count [[buffer(2)]], uint id [[thread_position_in_grid]]) {\n");
    source.push_str("if (id >= count) return;\n");
    source.push_str("uint l0 = input[id];\n");
    let mut local_count = 1usize;
    for local in body.get_locals_reader()? {
        let (count, ty) = local?;
        if ty != ValType::I32 {
            return Err(Error::Unsupported("only i32 locals are supported".into()));
        }
        let end = local_count
            .checked_add(count as usize)
            .ok_or_else(|| Error::Unsupported("too many locals".into()))?;
        for index in local_count..end {
            source.push_str(&format!("uint l{index} = 0u;\n"));
        }
        local_count = end;
    }

    let mut stack = Vec::<String>::new();
    let mut next_value = 0usize;
    let mut ended = false;
    for operator in body.get_operators_reader()? {
        let operator = operator?;
        match operator {
            Operator::End => {
                ended = true;
                break;
            }
            Operator::I32Const { value } => stack.push(format!("{}u", value as u32)),
            Operator::LocalGet { local_index } => {
                check_local(local_index, local_count)?;
                emit(
                    &mut source,
                    &mut stack,
                    &mut next_value,
                    format!("l{local_index}"),
                );
            }
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index } => {
                check_local(local_index, local_count)?;
                let value = pop(&mut stack)?;
                source.push_str(&format!("l{local_index} = {value};\n"));
                if matches!(operator, Operator::LocalTee { .. }) {
                    stack.push(value);
                }
            }
            Operator::Drop => {
                pop(&mut stack)?;
            }
            Operator::I32Eqz => {
                let value = pop(&mut stack)?;
                emit(
                    &mut source,
                    &mut stack,
                    &mut next_value,
                    format!("uint({value} == 0u)"),
                );
            }
            Operator::I32Add
            | Operator::I32Sub
            | Operator::I32Mul
            | Operator::I32And
            | Operator::I32Or
            | Operator::I32Xor
            | Operator::I32Shl
            | Operator::I32ShrU
            | Operator::I32Rotl
            | Operator::I32Rotr
            | Operator::I32Eq
            | Operator::I32Ne => {
                let rhs = pop(&mut stack)?;
                let lhs = pop(&mut stack)?;
                let expression = match operator {
                    Operator::I32Add => format!("{lhs} + {rhs}"),
                    Operator::I32Sub => format!("{lhs} - {rhs}"),
                    Operator::I32Mul => format!("{lhs} * {rhs}"),
                    Operator::I32And => format!("{lhs} & {rhs}"),
                    Operator::I32Or => format!("{lhs} | {rhs}"),
                    Operator::I32Xor => format!("{lhs} ^ {rhs}"),
                    Operator::I32Shl => format!("{lhs} << ({rhs} & 31u)"),
                    Operator::I32ShrU => format!("{lhs} >> ({rhs} & 31u)"),
                    Operator::I32Rotl => format!(
                        "({lhs} << ({rhs} & 31u)) | ({lhs} >> ((32u - ({rhs} & 31u)) & 31u))"
                    ),
                    Operator::I32Rotr => format!(
                        "({lhs} >> ({rhs} & 31u)) | ({lhs} << ((32u - ({rhs} & 31u)) & 31u))"
                    ),
                    Operator::I32Eq => format!("uint({lhs} == {rhs})"),
                    Operator::I32Ne => format!("uint({lhs} != {rhs})"),
                    _ => unreachable!(),
                };
                emit(&mut source, &mut stack, &mut next_value, expression);
            }
            _ => return Err(Error::Unsupported(format!("instruction {operator:?}"))),
        }
    }
    if !ended || stack.len() != 1 {
        return Err(Error::Unsupported(
            "function must leave one i32 result".into(),
        ));
    }
    source.push_str(&format!("output[id] = {};\n}}\n", stack.pop().unwrap()));
    Ok(source)
}

fn check_local(index: u32, count: usize) -> Result<(), Error> {
    if index as usize >= count {
        return Err(Error::Unsupported(format!(
            "local index {index} is out of range"
        )));
    }
    Ok(())
}

fn pop(stack: &mut Vec<String>) -> Result<String, Error> {
    stack
        .pop()
        .ok_or_else(|| Error::Unsupported("operand stack underflow".into()))
}

fn emit(source: &mut String, stack: &mut Vec<String>, next_value: &mut usize, expression: String) {
    let name = format!("v{}", *next_value);
    *next_value += 1;
    source.push_str(&format!("uint {name} = {expression};\n"));
    stack.push(name);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_many_independent_instances() {
        let wasm = wat::parse_str(
            "(module (func (export \"calculate\") (param i32) (result i32) (local i32) local.get 0 i32.const 7 i32.mul local.set 1 local.get 1 i32.const 3 i32.add))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "calculate").unwrap();
        let inputs: Vec<_> = (0..1_000_000u32).collect();
        let result = kernel.run(&inputs).unwrap();
        assert_eq!(result.len(), inputs.len());
        for (index, value) in result.iter().enumerate() {
            assert_eq!(*value, (index as u32).wrapping_mul(7).wrapping_add(3));
        }
        assert!(kernel.run(&[]).unwrap().is_empty());
    }

    #[test]
    fn preserves_i32_wrapping_and_shift_masking() {
        let wasm = wat::parse_str(
            "(module (func (export \"shift\") (param i32) (result i32) local.get 0 i32.const 33 i32.shl i32.const -1 i32.add))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "shift").unwrap();
        assert_eq!(
            kernel.run(&[0, 1, u32::MAX]).unwrap(),
            [u32::MAX, 1, u32::MAX - 2]
        );
    }

    #[test]
    fn rejects_unsupported_control_flow() {
        let wasm = wat::parse_str(
            "(module (func (export \"branch\") (param i32) (result i32) local.get 0 if (result i32) i32.const 1 else i32.const 2 end))",
        )
        .unwrap();
        let error = Kernel::new(&wasm, "branch").err().unwrap();
        assert!(matches!(error, Error::Unsupported(_)));
    }

    #[test]
    fn local_get_keeps_the_value_before_a_later_set() {
        let wasm = wat::parse_str(
            "(module (func (export \"update\") (param i32) (result i32) local.get 0 i32.const 2 local.set 0 local.get 0 i32.add))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "update").unwrap();
        assert_eq!(kernel.run(&[3, 9]).unwrap(), [5, 11]);
    }

    #[test]
    fn rotates_mask_the_shift_amount() {
        let wasm = wat::parse_str(
            "(module (func (export \"left\") (param i32) (result i32) local.get 0 i32.const 33 i32.rotl) (func (export \"right\") (param i32) (result i32) local.get 0 i32.const 33 i32.rotr))",
        )
        .unwrap();
        let inputs = [0, 1, 0x8000_0001, u32::MAX];
        for (export, rotate) in [
            ("left", u32::rotate_left as fn(u32, u32) -> u32),
            ("right", u32::rotate_right as fn(u32, u32) -> u32),
        ] {
            let kernel = Kernel::new(&wasm, export).unwrap();
            let expected: Vec<_> = inputs.iter().map(|&value| rotate(value, 33)).collect();
            assert_eq!(kernel.run(&inputs).unwrap(), expected);
        }
    }
}
