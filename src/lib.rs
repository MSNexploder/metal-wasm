//! Execute a supported WebAssembly function once per Metal thread.

use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use metal::objc::{Message, runtime::Sel};
use metal::{
    CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus, MTLResourceOptions,
    MTLSize,
};
use wasmparser::{
    BlockType, ConstExpr, DataKind, ExternalKind, FuncType, FunctionBody, Operator, Parser,
    Payload, ValType, Validator,
};

const MAX_MEMORY_BYTES_PER_INSTANCE: usize = 16 * 1024 * 1024;
const MAX_GROWTH_PAGES: usize = 4;

/// One raw WebAssembly numeric value. Floating-point variants hold their bit patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    I32(u32),
    I64(u64),
    F32(u32),
    F64(u64),
}

impl Value {
    fn ty(self) -> ValType {
        match self {
            Self::I32(_) => ValType::I32,
            Self::I64(_) => ValType::I64,
            Self::F32(_) => ValType::F32,
            Self::F64(_) => ValType::F64,
        }
    }

    fn bits(self) -> u64 {
        match self {
            Self::I32(bits) | Self::F32(bits) => bits as u64,
            Self::I64(bits) | Self::F64(bits) => bits,
        }
    }
}

/// A compiled WebAssembly export, run once per Metal thread.
pub struct Kernel {
    device: Device,
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
    can_trap: bool,
    params: Vec<ValType>,
    results: Vec<ValType>,
    memory_template: Option<Vec<u8>>,
    memory_capacity: Option<usize>,
    globals_template: Vec<u32>,
}

struct Translation {
    source: String,
    can_trap: bool,
    params: Vec<ValType>,
    results: Vec<ValType>,
    memory_template: Option<Vec<u8>>,
    memory_capacity: Option<usize>,
    globals_template: Vec<u32>,
}

/// An error while validating, compiling, or running a WebAssembly export.
#[derive(Debug)]
pub enum Error {
    InvalidWasm(wasmparser::BinaryReaderError),
    Unsupported(String),
    Metal(String),
    Trap {
        input_index: usize,
        reason: &'static str,
    },
}

impl Kernel {
    /// Compile a WebAssembly export for the system's default Metal device.
    pub fn new(wasm: &[u8], export: &str) -> Result<Self, Error> {
        let translation = translate(wasm, export)?;
        let device = Device::system_default()
            .ok_or_else(|| Error::Metal("no Metal device is available".into()))?;
        let library = device
            .new_library_with_source(&translation.source, &CompileOptions::new())
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
            can_trap: translation.can_trap,
            params: translation.params,
            results: translation.results,
            memory_template: translation.memory_template,
            memory_capacity: translation.memory_capacity,
            globals_template: translation.globals_template,
        })
    }

    /// Run one independent invocation for each input. Values are raw i32 bit patterns.
    pub fn run(&self, inputs: &[u32]) -> Result<Vec<u32>, Error> {
        self.run_timed(inputs).map(|(output, _)| output)
    }

    /// Run the batch and return Metal's GPU execution time when available.
    /// This excludes host-side buffer copies and command submission.
    pub fn run_timed(&self, inputs: &[u32]) -> Result<(Vec<u32>, Option<Duration>), Error> {
        if self.params != [ValType::I32] || self.results != [ValType::I32] {
            return Err(Error::Unsupported(
                "run requires type (param i32) (result i32); use run_values".into(),
            ));
        }
        if inputs.is_empty() {
            return Ok((Vec::new(), None));
        }
        let count = u32::try_from(inputs.len())
            .map_err(|_| Error::Unsupported("more than u32::MAX invocations".into()))?;
        let bytes = inputs
            .len()
            .checked_mul(size_of::<u32>())
            .ok_or_else(|| Error::Unsupported("input buffer size overflow".into()))?;
        let input = unsafe { std::slice::from_raw_parts(inputs.as_ptr().cast::<u8>(), bytes) };
        let (output, gpu_time) = self.dispatch(input, bytes, count)?;
        let values =
            unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), inputs.len()) };
        Ok((values.to_vec(), gpu_time))
    }

    /// Run one independent invocation per row. Each row must match the export's parameters.
    pub fn run_values(&self, inputs: &[Vec<Value>]) -> Result<Vec<Vec<Value>>, Error> {
        let count = u32::try_from(inputs.len())
            .map_err(|_| Error::Unsupported("more than u32::MAX invocations".into()))?;
        let mut encoded = Vec::new();
        for row in inputs {
            if row.len() != self.params.len()
                || row
                    .iter()
                    .zip(&self.params)
                    .any(|(value, ty)| value.ty() != *ty)
            {
                return Err(Error::Unsupported(
                    "input values do not match the export signature".into(),
                ));
            }
            encoded.extend(row.iter().map(|value| value.bits()));
        }
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        if self.params == [ValType::I32] && self.results == [ValType::I32] {
            let flat: Vec<_> = inputs
                .iter()
                .map(|row| match row[0] {
                    Value::I32(bits) => bits,
                    _ => unreachable!(),
                })
                .collect();
            return Ok(self
                .run(&flat)?
                .into_iter()
                .map(|bits| vec![Value::I32(bits)])
                .collect());
        }
        let output_len = inputs
            .len()
            .checked_mul(self.results.len())
            .and_then(|count| count.checked_mul(size_of::<u64>()))
            .ok_or_else(|| Error::Unsupported("output buffer size overflow".into()))?;
        let input_bytes = encoded
            .len()
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| Error::Unsupported("input buffer size overflow".into()))?;
        let input =
            unsafe { std::slice::from_raw_parts(encoded.as_ptr().cast::<u8>(), input_bytes) };
        let (output, _) = self.dispatch(input, output_len, count)?;
        let values = unsafe {
            std::slice::from_raw_parts(
                output.contents().cast::<u64>(),
                inputs.len() * self.results.len(),
            )
        };
        Ok((0..inputs.len())
            .map(|index| {
                self.results
                    .iter()
                    .enumerate()
                    .map(|(result, ty)| match ty {
                        ValType::I32 => {
                            Value::I32(values[index * self.results.len() + result] as u32)
                        }
                        ValType::I64 => Value::I64(values[index * self.results.len() + result]),
                        ValType::F32 => {
                            Value::F32(values[index * self.results.len() + result] as u32)
                        }
                        ValType::F64 => Value::F64(values[index * self.results.len() + result]),
                        _ => unreachable!(),
                    })
                    .collect()
            })
            .collect())
    }

    fn dispatch(
        &self,
        input_data: &[u8],
        output_bytes: usize,
        count: u32,
    ) -> Result<(metal::Buffer, Option<Duration>), Error> {
        let input_bytes = u64::try_from(input_data.len().max(4))
            .map_err(|_| Error::Unsupported("input buffer size overflow".into()))?;
        let output_bytes = u64::try_from(output_bytes.max(4))
            .map_err(|_| Error::Unsupported("output buffer size overflow".into()))?;
        let trap_bytes = u64::from(count) * size_of::<u32>() as u64;
        let memory_stride = self.memory_capacity.map(|bytes| bytes.max(1));
        let memory_bytes = memory_stride
            .and_then(|stride| stride.checked_mul(count as usize))
            .unwrap_or(0);
        let globals_bytes = self
            .globals_template
            .len()
            .checked_mul(count as usize)
            .and_then(|values| values.checked_mul(size_of::<u32>()))
            .ok_or_else(|| Error::Unsupported("globals buffer size overflow".into()))?;
        if self.memory_template.is_some() && memory_bytes == 0 {
            return Err(Error::Unsupported("memory buffer size overflow".into()));
        }
        let largest = input_bytes
            .max(output_bytes)
            .max(trap_bytes)
            .max(memory_bytes as u64)
            .max(globals_bytes as u64);
        if largest > self.device.max_buffer_length() {
            return Err(Error::Unsupported(
                "batch exceeds the Metal buffer limit".into(),
            ));
        }
        let memory_data = if let Some(template) = &self.memory_template {
            let stride = memory_stride.unwrap();
            let mut data = Vec::new();
            data.try_reserve_exact(memory_bytes)
                .map_err(|_| Error::Unsupported("cannot allocate per-lane memory".into()))?;
            data.resize(memory_bytes, 0u8);
            for row in data.chunks_mut(stride) {
                row[..template.len()].copy_from_slice(template);
            }
            Some(data)
        } else {
            None
        };
        let globals_data = if self.globals_template.is_empty() {
            None
        } else {
            let mut data = Vec::new();
            data.try_reserve_exact(globals_bytes / size_of::<u32>())
                .map_err(|_| Error::Unsupported("cannot allocate per-lane globals".into()))?;
            for _ in 0..count {
                data.extend_from_slice(&self.globals_template);
            }
            Some(data)
        };
        let empty = [0u8; 4];
        let input_ptr = if input_data.is_empty() {
            empty.as_ptr()
        } else {
            input_data.as_ptr()
        };
        let input = self.device.new_buffer_with_data(
            input_ptr.cast(),
            input_bytes,
            MTLResourceOptions::StorageModeShared,
        );
        let output = self
            .device
            .new_buffer(output_bytes, MTLResourceOptions::StorageModeShared);
        let traps = self.can_trap.then(|| {
            self.device
                .new_buffer(trap_bytes, MTLResourceOptions::StorageModeShared)
        });
        let memory = memory_data.as_ref().map(|data| {
            self.device.new_buffer_with_data(
                data.as_ptr().cast(),
                data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        });
        let pages = self.memory_template.as_ref().map(|template| {
            let initial = template.len() as u32 / 65536;
            let data = vec![initial; count as usize];
            self.device.new_buffer_with_data(
                data.as_ptr().cast(),
                trap_bytes,
                MTLResourceOptions::StorageModeShared,
            )
        });
        let globals = globals_data.as_ref().map(|data| {
            self.device.new_buffer_with_data(
                data.as_ptr().cast(),
                globals_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            )
        });
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, size_of::<u32>() as u64, (&count as *const u32).cast());
        if let Some(traps) = &traps {
            encoder.set_buffer(3, Some(traps), 0);
        }
        if let Some(memory) = &memory {
            encoder.set_buffer(4, Some(memory), 0);
        }
        if let Some(pages) = &pages {
            encoder.set_buffer(6, Some(pages), 0);
        }
        if let Some(globals) = &globals {
            encoder.set_buffer(5, Some(globals), 0);
        }
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
        if let Some(traps) = &traps {
            let codes = unsafe {
                std::slice::from_raw_parts(traps.contents().cast::<u32>(), count as usize)
            };
            if let Some((input_index, code)) =
                codes.iter().enumerate().find(|(_, code)| **code != 0)
            {
                let reason = match code {
                    1 => "integer divide by zero",
                    2 => "integer overflow",
                    3 => "unreachable executed",
                    4 => "out of bounds memory access",
                    _ => "unknown trap",
                };
                return Err(Error::Trap {
                    input_index,
                    reason,
                });
            }
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

        Ok((output, gpu_time))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWasm(error) => write!(formatter, "invalid WebAssembly: {error}"),
            Self::Unsupported(reason) => write!(formatter, "unsupported WebAssembly: {reason}"),
            Self::Metal(reason) => write!(formatter, "Metal error: {reason}"),
            Self::Trap {
                input_index,
                reason,
            } => {
                write!(
                    formatter,
                    "WebAssembly trap at input {input_index}: {reason}"
                )
            }
        }
    }
}

impl StdError for Error {}

impl From<wasmparser::BinaryReaderError> for Error {
    fn from(error: wasmparser::BinaryReaderError) -> Self {
        Self::InvalidWasm(error)
    }
}

fn translate(wasm: &[u8], export_name: &str) -> Result<Translation, Error> {
    Validator::new().validate_all(wasm)?;

    let mut types = Vec::new();
    let mut function_types = Vec::new();
    let mut bodies = Vec::new();
    let mut export_index = None;
    let mut memory_template: Option<Vec<u8>> = None;
    let mut memory_max_pages = None;
    let mut globals_template = Vec::new();
    let mut globals_mutable = Vec::new();
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
            Payload::MemorySection(section) => {
                for memory in section {
                    let memory = memory?;
                    if memory_template.is_some()
                        || memory.memory64
                        || memory.shared
                        || memory.page_size_log2() != 16
                    {
                        return Err(Error::Unsupported(
                            "only one unshared 32-bit memory is supported".into(),
                        ));
                    }
                    let bytes = memory
                        .initial
                        .checked_mul(65536)
                        .and_then(|bytes| usize::try_from(bytes).ok())
                        .ok_or_else(|| Error::Unsupported("memory size overflow".into()))?;
                    if bytes > MAX_MEMORY_BYTES_PER_INSTANCE {
                        return Err(Error::Unsupported(
                            "initial memory exceeds 16 MiB per instance".into(),
                        ));
                    }
                    memory_template = Some(vec![0; bytes]);
                    memory_max_pages = memory.maximum;
                }
            }
            Payload::GlobalSection(section) => {
                for global in section {
                    let global = global?;
                    if global.ty.content_type != ValType::I32 || global.ty.shared {
                        return Err(Error::Unsupported(
                            "only unshared i32 globals are supported".into(),
                        ));
                    }
                    globals_template.push(eval_i32_const(&global.init_expr, &globals_template)?);
                    globals_mutable.push(global.ty.mutable);
                }
            }
            Payload::DataSection(section) => {
                for data in section {
                    let data = data?;
                    let DataKind::Active {
                        memory_index: 0,
                        offset_expr,
                    } = data.kind
                    else {
                        return Err(Error::Unsupported(
                            "only active data for memory 0 is supported".into(),
                        ));
                    };
                    let offset = eval_i32_const(&offset_expr, &globals_template)? as usize;
                    let memory = memory_template
                        .as_mut()
                        .ok_or_else(|| Error::Unsupported("data without memory".into()))?;
                    let end = offset
                        .checked_add(data.data.len())
                        .ok_or_else(|| Error::Unsupported("data segment offset overflow".into()))?;
                    if end > memory.len() {
                        return Err(Error::Unsupported(
                            "data segment exceeds initial memory".into(),
                        ));
                    }
                    memory[offset..end].copy_from_slice(data.data);
                }
            }
            Payload::DataCountSection { .. } => {}
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
    function_signature(
        index,
        ty,
        memory_template.is_some(),
        !globals_template.is_empty(),
    )?;
    let fast_i32 = ty.params() == [ValType::I32] && ty.results() == [ValType::I32];
    let mut visit_state = vec![0u8; bodies.len()];
    let mut order = Vec::new();
    let mut can_trap = memory_template.is_some();
    let mut uses_growth = false;
    visit_function(
        index,
        &bodies,
        &mut visit_state,
        &mut order,
        &mut can_trap,
        &mut uses_growth,
    )?;
    let memory_capacity = memory_template.as_ref().map(|template| {
        let initial_pages = template.len() / 65536;
        let wanted = initial_pages + if uses_growth { MAX_GROWTH_PAGES } else { 0 };
        let declared = memory_max_pages.unwrap_or(u64::MAX);
        wanted
            .min(declared as usize)
            .min(MAX_MEMORY_BYTES_PER_INSTANCE / 65536)
            * 65536
    });

    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    for &function_index in &order {
        let ty = &types[function_types[function_index] as usize];
        source.push_str(&function_signature(
            function_index,
            ty,
            memory_template.is_some(),
            !globals_template.is_empty(),
        )?);
        source.push_str(";\n");
    }
    for &function_index in &order {
        let ty = &types[function_types[function_index] as usize];
        source.push_str(&translate_body(
            &bodies[function_index],
            function_index,
            ty,
            &types,
            &function_types,
            memory_template.as_ref().map(Vec::len),
            memory_capacity,
            &globals_mutable,
        )?);
    }
    let input_ty = if fast_i32 { "uint" } else { "ulong" };
    source.push_str(&format!("kernel void run_wasm(device const {input_ty}* input [[buffer(0)]], device {input_ty}* output [[buffer(1)]], constant uint& count [[buffer(2)]]"));
    if can_trap {
        source.push_str(", device uint* traps [[buffer(3)]]");
    }
    if memory_template.is_some() {
        source.push_str(", device uchar* memory [[buffer(4)]]");
        source.push_str(", device uint* pages [[buffer(6)]]");
    }
    if !globals_template.is_empty() {
        source.push_str(", device uint* globals [[buffer(5)]]");
    }
    source.push_str(", uint id [[thread_position_in_grid]]) {\n");
    source.push_str("if (id >= count) return;\nuint trap = 0u;\n");
    let mut args = Vec::new();
    for param in 0..ty.params().len() {
        let offset = if fast_i32 {
            "id".into()
        } else {
            format!("id * {}u + {param}u", ty.params().len())
        };
        args.push(format!("uint(input[{offset}])"));
    }
    for result in 0..ty.results().len() {
        source.push_str(&format!("uint result{result} = 0u;\n"));
        args.push(format!("result{result}"));
    }
    if memory_template.is_some() {
        args.push(format!(
            "memory + ulong(id) * {}ul",
            memory_capacity.unwrap().max(1)
        ));
        args.push("pages + id".into());
    }
    if !globals_template.is_empty() {
        args.push(format!(
            "globals + ulong(id) * {}ul",
            globals_template.len()
        ));
    }
    args.push("trap".into());
    source.push_str(&format!("f{index}({});\n", args.join(", ")));
    if can_trap {
        source.push_str("traps[id] = trap;\n");
    }
    for result in 0..ty.results().len() {
        let offset = if fast_i32 {
            "id".into()
        } else {
            format!("id * {}u + {result}u", ty.results().len())
        };
        source.push_str(&format!("output[{offset}] = {input_ty}(result{result});\n"));
    }
    source.push_str("}\n");
    Ok(Translation {
        source,
        can_trap,
        params: ty.params().to_vec(),
        results: ty.results().to_vec(),
        memory_template,
        memory_capacity,
        globals_template,
    })
}

fn eval_i32_const(expr: &ConstExpr<'_>, globals: &[u32]) -> Result<u32, Error> {
    let mut operators = expr.get_operators_reader().into_iter();
    let value = match operators.next().transpose()? {
        Some(Operator::I32Const { value }) => value as u32,
        Some(Operator::GlobalGet { global_index }) => *globals
            .get(global_index as usize)
            .ok_or_else(|| Error::Unsupported("constant global index is out of range".into()))?,
        _ => return Err(Error::Unsupported("unsupported i32 initializer".into())),
    };
    if !matches!(operators.next().transpose()?, Some(Operator::End)) || operators.next().is_some() {
        return Err(Error::Unsupported("unsupported i32 initializer".into()));
    }
    Ok(value)
}

fn visit_function(
    index: usize,
    bodies: &[FunctionBody<'_>],
    state: &mut [u8],
    order: &mut Vec<usize>,
    can_trap: &mut bool,
    uses_growth: &mut bool,
) -> Result<(), Error> {
    match state.get(index).copied() {
        Some(2) => return Ok(()),
        Some(1) => {
            return Err(Error::Unsupported(
                "recursive calls are not supported".into(),
            ));
        }
        Some(0) => {}
        _ => return Err(Error::Unsupported("function index is out of range".into())),
    }
    state[index] = 1;
    for operator in bodies[index].get_operators_reader()? {
        match operator? {
            Operator::Call { function_index } => {
                visit_function(
                    function_index as usize,
                    bodies,
                    state,
                    order,
                    can_trap,
                    uses_growth,
                )?;
            }
            Operator::MemoryGrow { .. } => *uses_growth = true,
            Operator::I32DivS
            | Operator::I32DivU
            | Operator::I32RemS
            | Operator::I32RemU
            | Operator::Unreachable => *can_trap = true,
            _ => {}
        }
    }
    state[index] = 2;
    order.push(index);
    Ok(())
}

fn function_signature(
    index: usize,
    ty: &FuncType,
    has_memory: bool,
    has_globals: bool,
) -> Result<String, Error> {
    if ty
        .params()
        .iter()
        .chain(ty.results())
        .any(|ty| *ty != ValType::I32)
    {
        return Err(Error::Unsupported(
            "only i32 function signatures are supported".into(),
        ));
    }
    let mut args: Vec<_> = ty
        .params()
        .iter()
        .enumerate()
        .map(|(index, _)| format!("uint a{index}"))
        .collect();
    args.extend(
        ty.results()
            .iter()
            .enumerate()
            .map(|(index, _)| format!("thread uint& r{index}")),
    );
    if has_memory {
        args.push("device uchar* memory".into());
        args.push("device uint* pages".into());
    }
    if has_globals {
        args.push("device uint* globals".into());
    }
    args.push("thread uint& trap".into());
    Ok(format!("void f{index}({})", args.join(", ")))
}

fn translate_body(
    body: &FunctionBody<'_>,
    function_index: usize,
    ty: &FuncType,
    types: &[FuncType],
    function_types: &[u32],
    memory_bytes: Option<usize>,
    memory_capacity: Option<usize>,
    globals_mutable: &[bool],
) -> Result<String, Error> {
    let mut source = function_signature(
        function_index,
        ty,
        memory_bytes.is_some(),
        !globals_mutable.is_empty(),
    )?;
    source.push_str(" {\n");
    for index in 0..ty.params().len() {
        source.push_str(&format!("uint l{index} = a{index};\n"));
    }
    let mut local_count = ty.params().len();
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

    let mut operators = body.get_operators_reader()?.into_iter();
    let (instructions, end) = read_instructions(&mut operators)?;
    if end != End::Block || operators.next().is_some() {
        return Err(Error::Unsupported("invalid function body".into()));
    }
    let declarations_at = source.len();
    source.push_str("uint flow = 0u;\n");
    let mut compiler = Compiler {
        source,
        declarations: String::new(),
        local_count,
        next_value: 0,
        next_frame: 1,
        types,
        function_types,
        memory_bytes,
        memory_capacity,
        globals_mutable,
        frames: vec![Frame {
            id: 1,
            params: Vec::new(),
            results: (0..ty.results().len())
                .map(|index| format!("r{index}"))
                .collect(),
            loop_label: false,
        }],
    };
    let mut stack = Vec::new();
    let reachable = compiler.emit_instructions(&instructions, &mut stack)?;
    let results = compiler.frames[0].results.clone();
    compiler.finish_path(reachable, &mut stack, 0, &results)?;
    compiler.source.push_str("}\n");
    compiler
        .source
        .insert_str(declarations_at, &compiler.declarations);
    Ok(compiler.source)
}

#[derive(PartialEq, Eq)]
enum End {
    Block,
    Else,
}

enum Instruction<'a> {
    Basic(Operator<'a>),
    Block(BlockType, Vec<Instruction<'a>>),
    Loop(BlockType, Vec<Instruction<'a>>),
    If(BlockType, Vec<Instruction<'a>>, Vec<Instruction<'a>>),
}

fn read_instructions<'a>(
    operators: &mut impl Iterator<Item = Result<Operator<'a>, wasmparser::BinaryReaderError>>,
) -> Result<(Vec<Instruction<'a>>, End), Error> {
    let mut instructions = Vec::new();
    while let Some(operator) = operators.next() {
        match operator? {
            Operator::End => return Ok((instructions, End::Block)),
            Operator::Else => return Ok((instructions, End::Else)),
            Operator::Block { blockty } => {
                let (body, end) = read_instructions(operators)?;
                if end != End::Block {
                    return Err(Error::Unsupported("else outside if".into()));
                }
                instructions.push(Instruction::Block(blockty, body));
            }
            Operator::Loop { blockty } => {
                let (body, end) = read_instructions(operators)?;
                if end != End::Block {
                    return Err(Error::Unsupported("else outside if".into()));
                }
                instructions.push(Instruction::Loop(blockty, body));
            }
            Operator::If { blockty } => {
                let (then_body, end) = read_instructions(operators)?;
                let else_body = if end == End::Else {
                    let (else_body, end) = read_instructions(operators)?;
                    if end != End::Block {
                        return Err(Error::Unsupported("invalid if".into()));
                    }
                    else_body
                } else {
                    Vec::new()
                };
                instructions.push(Instruction::If(blockty, then_body, else_body));
            }
            operator => instructions.push(Instruction::Basic(operator)),
        }
    }
    Err(Error::Unsupported("missing end instruction".into()))
}

#[derive(Clone)]
struct Frame {
    id: usize,
    params: Vec<String>,
    results: Vec<String>,
    loop_label: bool,
}

struct Compiler<'a> {
    source: String,
    declarations: String,
    local_count: usize,
    next_value: usize,
    next_frame: usize,
    types: &'a [FuncType],
    function_types: &'a [u32],
    memory_bytes: Option<usize>,
    memory_capacity: Option<usize>,
    globals_mutable: &'a [bool],
    frames: Vec<Frame>,
}

impl Compiler<'_> {
    fn fresh_value(&mut self) -> String {
        let name = format!("v{}", self.next_value);
        self.next_value += 1;
        self.declarations.push_str(&format!("uint {name} = 0u;\n"));
        name
    }

    fn emit_value(&mut self, stack: &mut Vec<String>, expression: String) {
        let name = self.fresh_value();
        self.source.push_str(&format!("{name} = {expression};\n"));
        stack.push(name);
    }

    fn memory_address(
        &mut self,
        address: &str,
        memarg: &wasmparser::MemArg,
        width: usize,
    ) -> Result<String, Error> {
        self.memory_bytes
            .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
        if memarg.memory != 0 {
            return Err(Error::Unsupported("only memory 0 is supported".into()));
        }
        let index = self.fresh_value();
        self.source.push_str(&format!(
            "if (ulong({address}) + {}ul + {width}ul > ulong(pages[0]) * 65536ul) {{ trap = 4u; return; }}\n",
            memarg.offset
        ));
        self.source.push_str(&format!(
            "{index} = uint(ulong({address}) + {}ul);\n",
            memarg.offset
        ));
        Ok(index)
    }

    fn emit_i32_load(
        &mut self,
        stack: &mut Vec<String>,
        memarg: &wasmparser::MemArg,
        width: usize,
        signed: bool,
    ) -> Result<(), Error> {
        let address = pop(stack)?;
        let index = self.memory_address(&address, memarg, width)?;
        let value = self.fresh_value();
        for byte in 0..width {
            self.source.push_str(&format!(
                "{value} |= uint(memory[ulong({index}) + {byte}ul]) << {}u;\n",
                byte * 8
            ));
        }
        if signed && width < 4 {
            let shift = (4 - width) * 8;
            self.source.push_str(&format!(
                "{value} = as_type<uint>(as_type<int>({value} << {shift}u) >> {shift}u);\n"
            ));
        }
        stack.push(value);
        Ok(())
    }

    fn emit_i32_store(
        &mut self,
        stack: &mut Vec<String>,
        memarg: &wasmparser::MemArg,
        width: usize,
    ) -> Result<(), Error> {
        let value = pop(stack)?;
        let address = pop(stack)?;
        let index = self.memory_address(&address, memarg, width)?;
        for byte in 0..width {
            self.source.push_str(&format!(
                "memory[ulong({index}) + {byte}ul] = uchar({value} >> {}u);\n",
                byte * 8
            ));
        }
        Ok(())
    }

    fn new_frame(&mut self, blockty: BlockType, loop_label: bool) -> Result<Frame, Error> {
        let (params, results) = match blockty {
            BlockType::Empty => (0, 0),
            BlockType::Type(ValType::I32) => (0, 1),
            BlockType::FuncType(index) => {
                let ty = self
                    .types
                    .get(index as usize)
                    .ok_or_else(|| Error::Unsupported("block type index is out of range".into()))?;
                if ty
                    .params()
                    .iter()
                    .chain(ty.results())
                    .any(|ty| *ty != ValType::I32)
                {
                    return Err(Error::Unsupported(
                        "only i32 block signatures are supported".into(),
                    ));
                }
                (ty.params().len(), ty.results().len())
            }
            _ => {
                return Err(Error::Unsupported(
                    "only i32 block signatures are supported".into(),
                ));
            }
        };
        self.next_frame += 1;
        Ok(Frame {
            id: self.next_frame,
            params: (0..params).map(|_| self.fresh_value()).collect(),
            results: (0..results).map(|_| self.fresh_value()).collect(),
            loop_label,
        })
    }

    fn assign_slots(&mut self, slots: &[String], values: &[String]) {
        if values.iter().any(|value| slots.contains(value)) {
            // Loop branches can swap parameter slots; read all values before writing any.
            let snapshots: Vec<_> = values
                .iter()
                .map(|value| {
                    let name = self.fresh_value();
                    self.source.push_str(&format!("{name} = {value};\n"));
                    name
                })
                .collect();
            for (slot, value) in slots.iter().zip(snapshots) {
                self.source.push_str(&format!("{slot} = {value};\n"));
            }
        } else {
            for (slot, value) in slots.iter().zip(values) {
                self.source.push_str(&format!("{slot} = {value};\n"));
            }
        }
    }

    fn enter_frame(&mut self, frame: &Frame, stack: &mut Vec<String>) -> Result<usize, Error> {
        if stack.len() < frame.params.len() {
            return Err(Error::Unsupported("block parameter is missing".into()));
        }
        let base = stack.len() - frame.params.len();
        self.assign_slots(&frame.params, &stack[base..]);
        stack.truncate(base);
        Ok(base)
    }

    fn finish_path(
        &mut self,
        reachable: bool,
        stack: &mut Vec<String>,
        base: usize,
        results: &[String],
    ) -> Result<(), Error> {
        if !reachable {
            return Ok(());
        }
        if stack.len() != base + results.len() {
            return Err(Error::Unsupported(
                "block has the wrong result stack".into(),
            ));
        }
        if !results.is_empty() {
            self.source.push_str("if (flow == 0u) {\n");
            self.assign_slots(results, &stack[base..]);
            self.source.push_str("}\n");
        }
        Ok(())
    }

    fn emit_instructions(
        &mut self,
        instructions: &[Instruction<'_>],
        stack: &mut Vec<String>,
    ) -> Result<bool, Error> {
        let mut guards = 0;
        let mut reachable = true;
        for instruction in instructions {
            let guard = match instruction {
                Instruction::Basic(operator) => match operator {
                    Operator::Br { relative_depth } => {
                        self.emit_branch(*relative_depth, stack, None)?;
                        reachable = false;
                        break;
                    }
                    Operator::BrIf { relative_depth } => {
                        let condition = pop(stack)?;
                        self.emit_branch(*relative_depth, stack, Some(condition))?;
                        true
                    }
                    Operator::BrTable { targets } => {
                        let index = pop(stack)?;
                        for (case, depth) in targets.targets().enumerate() {
                            self.source.push_str(&format!(
                                "{}if ({index} == {case}u) {{\n",
                                if case == 0 { "" } else { "else " }
                            ));
                            self.emit_branch(depth?, stack, None)?;
                            self.source.push_str("} ");
                        }
                        if !targets.is_empty() {
                            self.source.push_str("else ");
                        }
                        self.source.push_str("{\n");
                        self.emit_branch(targets.default(), stack, None)?;
                        self.source.push_str("}\n");
                        reachable = false;
                        break;
                    }
                    Operator::Return => {
                        self.emit_branch((self.frames.len() - 1) as u32, stack, None)?;
                        reachable = false;
                        break;
                    }
                    Operator::Unreachable => {
                        self.source.push_str("trap = 3u; return;\n");
                        reachable = false;
                        break;
                    }
                    _ => {
                        self.emit_basic(operator, stack)?;
                        false
                    }
                },
                Instruction::Block(blockty, body) => {
                    self.emit_block(*blockty, body, stack, false)?;
                    true
                }
                Instruction::Loop(blockty, body) => {
                    self.emit_block(*blockty, body, stack, true)?;
                    true
                }
                Instruction::If(blockty, then_body, else_body) => {
                    self.emit_if(*blockty, then_body, else_body, stack)?;
                    true
                }
            };
            if guard {
                self.source.push_str("if (flow == 0u) {\n");
                guards += 1;
            }
        }
        for _ in 0..guards {
            self.source.push_str("}\n");
        }
        Ok(reachable)
    }

    fn emit_branch(
        &mut self,
        depth: u32,
        stack: &[String],
        condition: Option<String>,
    ) -> Result<(), Error> {
        let frame = self
            .frames
            .iter()
            .rev()
            .nth(depth as usize)
            .ok_or_else(|| Error::Unsupported("branch depth out of range".into()))?
            .clone();
        if let Some(ref condition) = condition {
            self.source
                .push_str(&format!("if ({condition} != 0u) {{\n"));
        }
        let slots = if frame.loop_label {
            &frame.params
        } else {
            &frame.results
        };
        if stack.len() < slots.len() {
            return Err(Error::Unsupported("branch value is missing".into()));
        }
        self.assign_slots(slots, &stack[stack.len() - slots.len()..]);
        self.source.push_str(&format!("flow = {}u;\n", frame.id));
        if condition.is_some() {
            self.source.push_str("}\n");
        }
        Ok(())
    }

    fn emit_block(
        &mut self,
        blockty: BlockType,
        body: &[Instruction<'_>],
        stack: &mut Vec<String>,
        loop_label: bool,
    ) -> Result<(), Error> {
        let frame = self.new_frame(blockty, loop_label)?;
        let base = self.enter_frame(&frame, stack)?;
        let mut inner_stack = stack.clone();
        inner_stack.extend(frame.params.iter().cloned());
        self.frames.push(frame.clone());
        if loop_label {
            self.source.push_str("while (true) {\n");
        } else {
            self.source.push_str("{\n");
        }
        let reachable = self.emit_instructions(body, &mut inner_stack)?;
        self.finish_path(reachable, &mut inner_stack, base, &frame.results)?;
        if loop_label {
            self.source.push_str(&format!(
                "if (flow == {}u) {{ flow = 0u; continue; }}\nbreak;\n",
                frame.id
            ));
        }
        self.source.push_str("}\n");
        if !loop_label {
            self.source
                .push_str(&format!("if (flow == {}u) flow = 0u;\n", frame.id));
        }
        self.frames.pop();
        stack.extend(frame.results);
        Ok(())
    }

    fn emit_if(
        &mut self,
        blockty: BlockType,
        then_body: &[Instruction<'_>],
        else_body: &[Instruction<'_>],
        stack: &mut Vec<String>,
    ) -> Result<(), Error> {
        let condition = pop(stack)?;
        let frame = self.new_frame(blockty, false)?;
        let base = self.enter_frame(&frame, stack)?;
        self.frames.push(frame.clone());
        self.source
            .push_str(&format!("if ({condition} != 0u) {{\n"));
        let mut then_stack = stack.clone();
        then_stack.extend(frame.params.iter().cloned());
        let then_reachable = self.emit_instructions(then_body, &mut then_stack)?;
        self.finish_path(then_reachable, &mut then_stack, base, &frame.results)?;
        self.source.push_str("} else {\n");
        let mut else_stack = stack.clone();
        else_stack.extend(frame.params.iter().cloned());
        let else_reachable = self.emit_instructions(else_body, &mut else_stack)?;
        self.finish_path(else_reachable, &mut else_stack, base, &frame.results)?;
        self.source.push_str("}\n");
        self.source
            .push_str(&format!("if (flow == {}u) flow = 0u;\n", frame.id));
        self.frames.pop();
        stack.extend(frame.results);
        Ok(())
    }

    fn emit_basic(
        &mut self,
        operator: &Operator<'_>,
        stack: &mut Vec<String>,
    ) -> Result<(), Error> {
        match operator {
            Operator::Nop => {}
            Operator::I32Const { value } => stack.push(format!("{}u", *value as u32)),
            Operator::GlobalGet { global_index } => {
                let index = *global_index as usize;
                if index >= self.globals_mutable.len() {
                    return Err(Error::Unsupported("global index is out of range".into()));
                }
                self.emit_value(stack, format!("globals[{index}u]"));
            }
            Operator::GlobalSet { global_index } => {
                let index = *global_index as usize;
                if !self.globals_mutable.get(index).copied().unwrap_or(false) {
                    return Err(Error::Unsupported("global is immutable or missing".into()));
                }
                let value = pop(stack)?;
                self.source
                    .push_str(&format!("globals[{index}u] = {value};\n"));
            }
            Operator::I32Load { memarg } => self.emit_i32_load(stack, memarg, 4, false)?,
            Operator::I32Load8U { memarg } => self.emit_i32_load(stack, memarg, 1, false)?,
            Operator::I32Load8S { memarg } => self.emit_i32_load(stack, memarg, 1, true)?,
            Operator::I32Load16U { memarg } => self.emit_i32_load(stack, memarg, 2, false)?,
            Operator::I32Load16S { memarg } => self.emit_i32_load(stack, memarg, 2, true)?,
            Operator::I32Store { memarg } => self.emit_i32_store(stack, memarg, 4)?,
            Operator::I32Store8 { memarg } => self.emit_i32_store(stack, memarg, 1)?,
            Operator::I32Store16 { memarg } => self.emit_i32_store(stack, memarg, 2)?,
            Operator::MemorySize { mem: 0 } => {
                self.memory_bytes
                    .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
                self.emit_value(stack, "pages[0]".into());
            }
            Operator::MemoryGrow { mem: 0 } => {
                let capacity = self
                    .memory_capacity
                    .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
                let delta = pop(stack)?;
                let prior = self.fresh_value();
                self.source.push_str(&format!(
                    "{prior} = pages[0];\nif ({delta} <= {}u - pages[0]) pages[0] += {delta}; else {prior} = 4294967295u;\n",
                    capacity / 65536
                ));
                stack.push(prior);
            }
            Operator::Call { function_index } => {
                let index = *function_index as usize;
                let type_index = *self
                    .function_types
                    .get(index)
                    .ok_or_else(|| Error::Unsupported("function index is out of range".into()))?;
                let ty = self
                    .types
                    .get(type_index as usize)
                    .ok_or_else(|| Error::Unsupported("function type is out of range".into()))?;
                if ty
                    .params()
                    .iter()
                    .chain(ty.results())
                    .any(|ty| *ty != ValType::I32)
                {
                    return Err(Error::Unsupported(
                        "only i32 function signatures are supported".into(),
                    ));
                }
                if stack.len() < ty.params().len() {
                    return Err(Error::Unsupported("call argument is missing".into()));
                }
                let args = stack.split_off(stack.len() - ty.params().len());
                let results: Vec<_> = (0..ty.results().len())
                    .map(|_| self.fresh_value())
                    .collect();
                let mut call_args = args;
                call_args.extend(results.iter().cloned());
                if self.memory_bytes.is_some() {
                    call_args.push("memory".into());
                    call_args.push("pages".into());
                }
                if !self.globals_mutable.is_empty() {
                    call_args.push("globals".into());
                }
                call_args.push("trap".into());
                self.source.push_str(&format!(
                    "f{index}({});\nif (trap != 0u) return;\n",
                    call_args.join(", ")
                ));
                stack.extend(results);
            }
            Operator::LocalGet { local_index } => {
                check_local(*local_index, self.local_count)?;
                self.emit_value(stack, format!("l{local_index}"));
            }
            Operator::LocalSet { local_index } | Operator::LocalTee { local_index } => {
                check_local(*local_index, self.local_count)?;
                let value = pop(stack)?;
                self.source
                    .push_str(&format!("l{local_index} = {value};\n"));
                if matches!(operator, Operator::LocalTee { .. }) {
                    stack.push(value);
                }
            }
            Operator::Drop => {
                pop(stack)?;
            }
            Operator::I32Eqz => {
                let value = pop(stack)?;
                self.emit_value(stack, format!("uint({value} == 0u)"));
            }
            Operator::I32Clz
            | Operator::I32Ctz
            | Operator::I32Popcnt
            | Operator::I32Extend8S
            | Operator::I32Extend16S => {
                let value = pop(stack)?;
                let expression = match operator {
                    Operator::I32Clz => format!("({value} == 0u ? 32u : clz({value}))"),
                    Operator::I32Ctz => format!("({value} == 0u ? 32u : ctz({value}))"),
                    Operator::I32Popcnt => format!("popcount({value})"),
                    Operator::I32Extend8S => {
                        format!("as_type<uint>(as_type<int>({value} << 24u) >> 24u)")
                    }
                    Operator::I32Extend16S => {
                        format!("as_type<uint>(as_type<int>({value} << 16u) >> 16u)")
                    }
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            Operator::Select | Operator::TypedSelect { ty: ValType::I32 } => {
                let condition = pop(stack)?;
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_value(stack, format!("({condition} != 0u ? {lhs} : {rhs})"));
            }
            Operator::I32Add
            | Operator::I32Sub
            | Operator::I32Mul
            | Operator::I32DivS
            | Operator::I32DivU
            | Operator::I32RemS
            | Operator::I32RemU
            | Operator::I32And
            | Operator::I32Or
            | Operator::I32Xor
            | Operator::I32Shl
            | Operator::I32ShrU
            | Operator::I32ShrS
            | Operator::I32Rotl
            | Operator::I32Rotr
            | Operator::I32Eq
            | Operator::I32Ne
            | Operator::I32LtS
            | Operator::I32LtU
            | Operator::I32GtS
            | Operator::I32GtU
            | Operator::I32LeS
            | Operator::I32LeU
            | Operator::I32GeS
            | Operator::I32GeU => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                if matches!(
                    operator,
                    Operator::I32DivS | Operator::I32DivU | Operator::I32RemS | Operator::I32RemU
                ) {
                    self.source
                        .push_str(&format!("if ({rhs} == 0u) {{ trap = 1u; return; }}\n"));
                }
                if matches!(operator, Operator::I32DivS) {
                    self.source.push_str(&format!(
                        "if ({lhs} == 2147483648u && {rhs} == 4294967295u) {{ trap = 2u; return; }}\n"
                    ));
                }
                let expression = match operator {
                    Operator::I32Add => format!("{lhs} + {rhs}"),
                    Operator::I32Sub => format!("{lhs} - {rhs}"),
                    Operator::I32Mul => format!("{lhs} * {rhs}"),
                    Operator::I32DivS => {
                        format!("as_type<uint>(as_type<int>({lhs}) / as_type<int>({rhs}))")
                    }
                    Operator::I32DivU => format!("{lhs} / {rhs}"),
                    Operator::I32RemS => format!(
                        "({lhs} == 2147483648u && {rhs} == 4294967295u ? 0u : as_type<uint>(as_type<int>({lhs}) % as_type<int>({rhs})))"
                    ),
                    Operator::I32RemU => format!("{lhs} % {rhs}"),
                    Operator::I32And => format!("{lhs} & {rhs}"),
                    Operator::I32Or => format!("{lhs} | {rhs}"),
                    Operator::I32Xor => format!("{lhs} ^ {rhs}"),
                    Operator::I32Shl => format!("{lhs} << ({rhs} & 31u)"),
                    Operator::I32ShrU => format!("{lhs} >> ({rhs} & 31u)"),
                    Operator::I32ShrS => {
                        format!("as_type<uint>(as_type<int>({lhs}) >> ({rhs} & 31u))")
                    }
                    Operator::I32Rotl => format!(
                        "({lhs} << ({rhs} & 31u)) | ({lhs} >> ((32u - ({rhs} & 31u)) & 31u))"
                    ),
                    Operator::I32Rotr => format!(
                        "({lhs} >> ({rhs} & 31u)) | ({lhs} << ((32u - ({rhs} & 31u)) & 31u))"
                    ),
                    Operator::I32Eq => format!("uint({lhs} == {rhs})"),
                    Operator::I32Ne => format!("uint({lhs} != {rhs})"),
                    Operator::I32LtS => format!("uint(as_type<int>({lhs}) < as_type<int>({rhs}))"),
                    Operator::I32LtU => format!("uint({lhs} < {rhs})"),
                    Operator::I32GtS => format!("uint(as_type<int>({lhs}) > as_type<int>({rhs}))"),
                    Operator::I32GtU => format!("uint({lhs} > {rhs})"),
                    Operator::I32LeS => format!("uint(as_type<int>({lhs}) <= as_type<int>({rhs}))"),
                    Operator::I32LeU => format!("uint({lhs} <= {rhs})"),
                    Operator::I32GeS => format!("uint(as_type<int>({lhs}) >= as_type<int>({rhs}))"),
                    Operator::I32GeU => format!("uint({lhs} >= {rhs})"),
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            _ => return Err(Error::Unsupported(format!("instruction {operator:?}"))),
        }
        Ok(())
    }
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
    fn chooses_each_if_result() {
        let wasm = wat::parse_str(
            "(module (func (export \"branch\") (param i32) (result i32) local.get 0 if (result i32) i32.const 1 else i32.const 2 end))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "branch").unwrap();
        assert_eq!(kernel.run(&[0, 1, u32::MAX]).unwrap(), [2, 1, 1]);
    }

    #[test]
    fn repeats_a_loop_with_br_if() {
        let wasm = wat::parse_str(
            "(module (func (export \"repeat\") (param i32) (result i32) (local i32) i32.const 0 local.set 1 loop local.get 0 i32.const 3 i32.mul local.set 0 local.get 1 i32.const 1 i32.add local.tee 1 i32.const 4 i32.ne br_if 0 end local.get 0))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "repeat").unwrap();
        assert_eq!(kernel.run(&[0, 1, 3]).unwrap(), [0, 81, 243]);
    }

    #[test]
    fn branches_out_of_nested_blocks_with_a_result() {
        let wasm = wat::parse_str(
            "(module (func (export \"nested\") (param i32) (result i32) block (result i32) loop local.get 0 i32.const 1 i32.and if i32.const 42 br 2 end i32.const 7 br 1 end i32.const 99 end))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "nested").unwrap();
        assert_eq!(kernel.run(&[0, 1, 2, 3]).unwrap(), [7, 42, 7, 42]);
    }

    #[test]
    fn conditional_branch_keeps_its_value_on_fallthrough() {
        let wasm = wat::parse_str(
            "(module (func (export \"select\") (param i32) (result i32) block (result i32) i32.const 7 local.get 0 br_if 0 i32.const 2 i32.add end))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "select").unwrap();
        assert_eq!(kernel.run(&[0, 1]).unwrap(), [9, 7]);
    }

    #[test]
    fn returns_from_inside_a_conditional() {
        let wasm = wat::parse_str(
            "(module (func (export \"early\") (param i32) (result i32) local.get 0 if i32.const 99 return end local.get 0))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "early").unwrap();
        assert_eq!(kernel.run(&[0, 1, 2]).unwrap(), [0, 99, 99]);
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

    fn matches_wasmtime(wat: &str, inputs: &[u32]) {
        let wasm = wat::parse_str(wat).unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
        let function = instance
            .get_typed_func::<i32, i32>(&mut store, "f")
            .unwrap();
        let expected: Vec<_> = inputs
            .iter()
            .map(|&input| function.call(&mut store, input as i32).unwrap() as u32)
            .collect();
        assert_eq!(kernel.run(inputs).unwrap(), expected, "{wat}");
    }

    #[test]
    fn scalar_i32_operations_match_wasmtime() {
        let mut inputs = vec![
            0,
            1,
            2,
            7,
            31,
            32,
            33,
            127,
            128,
            255,
            256,
            i32::MAX as u32,
            i32::MIN as u32,
            u32::MAX,
        ];
        let mut state = 0x1234_5678u32;
        for _ in 0..128 {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            inputs.push(state);
        }
        for op in [
            "i32.clz",
            "i32.ctz",
            "i32.popcnt",
            "i32.extend8_s",
            "i32.extend16_s",
        ] {
            matches_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param i32) (result i32) local.get 0 {op}))"
                ),
                &inputs,
            );
        }
        for (op, rhs) in [
            ("i32.div_s", 3),
            ("i32.div_s", -1),
            ("i32.div_u", 3),
            ("i32.rem_s", 3),
            ("i32.rem_s", -1),
            ("i32.rem_u", 3),
            ("i32.shr_s", 33),
            ("i32.lt_s", -1),
            ("i32.lt_u", -1),
            ("i32.gt_s", -1),
            ("i32.gt_u", -1),
            ("i32.le_s", -1),
            ("i32.le_u", -1),
            ("i32.ge_s", -1),
            ("i32.ge_u", -1),
        ] {
            let safe_inputs: Vec<_> = inputs
                .iter()
                .copied()
                .filter(|&input| !(op == "i32.div_s" && rhs == -1 && input == i32::MIN as u32))
                .collect();
            matches_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param i32) (result i32) local.get 0 i32.const {rhs} {op}))"
                ),
                &safe_inputs,
            );
        }
        for select in ["select", "select (result i32)"] {
            matches_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param i32) (result i32) i32.const 7 i32.const 9 local.get 0 {select} nop))"
                ),
                &inputs,
            );
        }
    }

    #[test]
    fn branch_table_targets_nested_labels() {
        matches_wasmtime(
            "(module (func (export \"f\") (param i32) (result i32) block (result i32) block (result i32) i32.const 42 local.get 0 br_table 0 1 2 end i32.const 3 i32.add end))",
            &[0, 1, 2, 3, u32::MAX],
        );
        matches_wasmtime(
            "(module (func (export \"f\") (param i32) (result i32) i32.const 42 local.get 0 br_table 0))",
            &[0, 1, u32::MAX],
        );
    }

    #[test]
    fn typed_blocks_pass_multiple_values_and_loop_arguments() {
        matches_wasmtime(
            "(module (type $pair (func (param i32 i32) (result i32 i32))) (func (export \"f\") (param i32) (result i32) local.get 0 i32.const 7 block (type $pair) i32.const 1 i32.add end i32.sub))",
            &[0, 1, 7, u32::MAX],
        );
        matches_wasmtime(
            "(module (type $pair (func (param i32 i32) (result i32 i32))) (func (export \"f\") (param i32) (result i32) (local i32 i32) local.get 0 i32.const 7 loop (type $pair) local.set 2 local.set 1 local.get 0 if i32.const 0 local.set 0 local.get 2 local.get 1 br 1 end local.get 1 local.get 2 end i32.sub))",
            &[0, 1, 7, u32::MAX],
        );
        matches_wasmtime(
            "(module (type $value (func (param i32) (result i32))) (func (export \"f\") (param i32) (result i32) local.get 0 local.get 0 if (type $value) i32.const 1 i32.add else i32.const 2 i32.add end))",
            &[0, 1, 7, u32::MAX],
        );
        matches_wasmtime(
            "(module (type $pair (func (param i32 i32) (result i32 i32))) (func (export \"f\") (param i32) (result i32) local.get 0 i32.const 7 block (type $pair) local.get 0 if i32.const 1 i32.const 2 br 1 end end i32.sub))",
            &[0, 1, 7, u32::MAX],
        );
    }

    #[test]
    fn reports_only_executed_traps_with_input_index() {
        for (body, inputs, index, reason) in [
            (
                "local.get 0 i32.const 0 i32.div_u",
                vec![1, 2, 3],
                0,
                "integer divide by zero",
            ),
            (
                "local.get 0 i32.const -1 i32.div_s",
                vec![1, i32::MIN as u32],
                1,
                "integer overflow",
            ),
            (
                "local.get 0 if unreachable end i32.const 7",
                vec![0, 0, 1],
                2,
                "unreachable executed",
            ),
        ] {
            let wasm = wat::parse_str(format!(
                "(module (func (export \"f\") (param i32) (result i32) {body}))"
            ))
            .unwrap();
            let kernel = Kernel::new(&wasm, "f").unwrap();
            let engine = wasmtime::Engine::default();
            let module = wasmtime::Module::new(&engine, &wasm).unwrap();
            let mut store = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let function = instance
                .get_typed_func::<i32, i32>(&mut store, "f")
                .unwrap();
            assert!(function.call(&mut store, inputs[index] as i32).is_err());
            assert!(matches!(
                kernel.run(&inputs),
                Err(Error::Trap { input_index, reason: actual })
                    if input_index == index && actual == reason
            ));
        }
        matches_wasmtime(
            "(module (func (export \"f\") (param i32) (result i32) local.get 0 if (result i32) i32.const 8 else i32.const 0 i32.const 1 i32.div_u end))",
            &[0, 1, 2],
        );
    }

    #[test]
    fn direct_calls_pass_multiple_values_and_propagate_traps() {
        matches_wasmtime(
            "(module (func $pair (param i32 i32) (result i32 i32) local.get 0 local.get 1 i32.add local.get 0 local.get 1 i32.sub) (func $mix (param i32) (result i32) local.get 0 i32.const 7 call $pair i32.xor) (func (export \"f\") (param i32) (result i32) local.get 0 call $mix))",
            &[0, 1, 7, u32::MAX, i32::MIN as u32],
        );
        let wasm = wat::parse_str(
            "(module (func $divide (param i32) (result i32) local.get 0 i32.const 1 i32.sub local.get 0 i32.div_u) (func (export \"f\") (param i32) (result i32) local.get 0 call $divide))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[2, 3]).unwrap(), [0, 0]);
        assert!(matches!(
            kernel.run(&[2, 0, 3]),
            Err(Error::Trap {
                input_index: 1,
                reason: "integer divide by zero"
            })
        ));
    }

    #[test]
    fn exported_i32_signatures_accept_multiple_and_empty_values() {
        let wasm = wat::parse_str(
            "(module (func (export \"f\") (param i32 i32) (result i32 i32) local.get 0 local.get 1 i32.add local.get 0 local.get 1 i32.sub))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(
            kernel
                .run_values(&[
                    vec![Value::I32(7), Value::I32(3)],
                    vec![Value::I32(0), Value::I32(u32::MAX)],
                ])
                .unwrap(),
            [
                vec![Value::I32(10), Value::I32(4)],
                vec![Value::I32(u32::MAX), Value::I32(1)],
            ]
        );
        assert!(kernel.run(&[7]).is_err());
        assert!(kernel.run_values(&[vec![Value::I32(7)]]).is_err());
        let no_params =
            wat::parse_str("(module (func (export \"f\") (result i32) i32.const 7))").unwrap();
        let kernel = Kernel::new(&no_params, "f").unwrap();
        assert_eq!(
            kernel.run_values(&[vec![], vec![]]).unwrap(),
            [vec![Value::I32(7)], vec![Value::I32(7)]]
        );
        let no_results =
            wat::parse_str("(module (func (export \"f\") (param i32) local.get 0 drop))").unwrap();
        let kernel = Kernel::new(&no_results, "f").unwrap();
        assert_eq!(
            kernel
                .run_values(&[vec![Value::I32(1)], vec![Value::I32(2)]])
                .unwrap(),
            [Vec::<Value>::new(), Vec::new()]
        );
    }

    #[test]
    fn memory_data_and_globals_are_isolated_per_lane() {
        let wat = r#"(module
            (memory 1)
            (data (i32.const 4) "\80\ff\00\01")
            (global $g (mut i32) (i32.const 5))
            (func $helper (param i32)
                i32.const 8
                global.get $g
                local.get 0
                i32.add
                i32.store
                global.get $g
                i32.const 1
                i32.add
                global.set $g)
            (func (export "f") (param i32) (result i32)
                local.get 0
                call $helper
                i32.const 8
                i32.load
                i32.const 4
                i32.load8_s
                i32.add
                global.get $g
                i32.add))"#;
        let wasm = wat::parse_str(wat).unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let inputs = [0, 1, 7, u32::MAX];
        let expected: Vec<_> = inputs
            .iter()
            .map(|&input| {
                let mut store = wasmtime::Store::new(&engine, ());
                let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
                let function = instance
                    .get_typed_func::<i32, i32>(&mut store, "f")
                    .unwrap();
                function.call(&mut store, input as i32).unwrap() as u32
            })
            .collect();
        assert_eq!(kernel.run(&inputs).unwrap(), expected);
        assert_eq!(kernel.run(&inputs).unwrap(), expected);
    }

    #[test]
    fn memory_loads_and_stores_check_bounds() {
        let wasm = wat::parse_str(
            "(module (memory 1) (func (export \"f\") (param i32) (result i32) local.get 0 i32.const 4660 i32.store16 local.get 0 i32.load16_u))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 65534]).unwrap(), [4660, 4660]);
        assert!(matches!(
            kernel.run(&[0, 65535]),
            Err(Error::Trap {
                input_index: 1,
                reason: "out of bounds memory access"
            })
        ));
    }

    #[test]
    fn memory_growth_updates_only_the_current_lane() {
        let wasm = wat::parse_str(
            "(module (memory 1 2) (func (export \"f\") (param i32) (result i32) memory.size local.get 0 memory.grow i32.add memory.size i32.add))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 1, 2]).unwrap(), [3, 4, 1]);
        let wasm = wat::parse_str(
            "(module (memory 1 2) (func (export \"f\") (param i32) (result i32) local.get 0 memory.grow drop i32.const 65536 i32.const 42 i32.store8 i32.const 65536 i32.load8_u))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[1, 1]).unwrap(), [42, 42]);
        assert!(matches!(
            kernel.run(&[1, 0]),
            Err(Error::Trap {
                input_index: 1,
                reason: "out of bounds memory access"
            })
        ));
    }
}
