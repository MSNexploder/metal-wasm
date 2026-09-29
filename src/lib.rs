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
    BlockType, ConstExpr, DataKind, ElementItems, ElementKind, ExternalKind, FuncType,
    FunctionBody, Operator, Parser, Payload, RefType, TableInit, ValType, Validator,
};

const MAX_MEMORY_BYTES_PER_INSTANCE: usize = 16 * 1024 * 1024;
const MAX_GROWTH_PAGES: usize = 4;
const MAX_TABLE_ELEMENTS: usize = 65536;

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
    globals_template: Vec<u64>,
    table: Option<metal::Buffer>,
}

struct Translation {
    source: String,
    can_trap: bool,
    params: Vec<ValType>,
    results: Vec<ValType>,
    memory_template: Option<Vec<u8>>,
    memory_capacity: Option<usize>,
    globals_template: Vec<u64>,
    table_template: Option<Vec<u32>>,
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
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(&translation.source, &options)
            .map_err(Error::Metal)?;
        let function = library
            .get_function("run_wasm", None)
            .map_err(Error::Metal)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|error| Error::Metal(error.to_string()))?;
        let queue = device.new_command_queue();
        let table = translation.table_template.as_ref().map(|data| {
            let empty = [0u32];
            device.new_buffer_with_data(
                if data.is_empty() {
                    empty.as_ptr().cast()
                } else {
                    data.as_ptr().cast()
                },
                (data.len() * size_of::<u32>()).max(4) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        });
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
            table,
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
            .and_then(|values| values.checked_mul(size_of::<u64>()))
            .ok_or_else(|| Error::Unsupported("globals buffer size overflow".into()))?;
        if self.memory_template.is_some() && memory_bytes == 0 {
            return Err(Error::Unsupported("memory buffer size overflow".into()));
        }
        let largest = input_bytes
            .max(output_bytes)
            .max(trap_bytes)
            .max(memory_bytes as u64)
            .max(globals_bytes as u64)
            .max(self.table.as_ref().map_or(0, |table| table.length()));
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
            data.try_reserve_exact(globals_bytes / size_of::<u64>())
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
        if let Some(table) = &self.table {
            encoder.set_buffer(7, Some(table), 0);
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
                    5 => "invalid conversion to integer",
                    6 => "undefined or uninitialized table element",
                    7 => "indirect call type mismatch",
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
    let mut table_template: Option<Vec<u32>> = None;
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
            Payload::TableSection(section) => {
                for table in section {
                    let table = table?;
                    if table_template.is_some()
                        || table.ty.element_type != RefType::FUNCREF
                        || table.ty.table64
                        || table.ty.shared
                    {
                        return Err(Error::Unsupported(
                            "only one unshared 32-bit funcref table is supported".into(),
                        ));
                    }
                    let size = usize::try_from(table.ty.initial)
                        .map_err(|_| Error::Unsupported("table size overflow".into()))?;
                    if size > MAX_TABLE_ELEMENTS {
                        return Err(Error::Unsupported(
                            "initial table exceeds 65536 elements".into(),
                        ));
                    }
                    let value = match table.init {
                        TableInit::RefNull => 0,
                        TableInit::Expr(expr) => eval_ref_const(&expr)?,
                    };
                    table_template = Some(vec![value; size]);
                }
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
                    if !is_numeric(global.ty.content_type) || global.ty.shared {
                        return Err(Error::Unsupported(
                            "only unshared numeric globals are supported".into(),
                        ));
                    }
                    globals_template.push(eval_const(&global.init_expr, &globals_template)?);
                    globals_mutable.push(global.ty.mutable);
                }
            }
            Payload::ElementSection(section) => {
                for element in section {
                    let element = element?;
                    let ElementKind::Active {
                        table_index: None | Some(0),
                        offset_expr,
                    } = element.kind
                    else {
                        return Err(Error::Unsupported(
                            "only active elements for table 0 are supported".into(),
                        ));
                    };
                    let offset = eval_const(&offset_expr, &globals_template)? as u32 as usize;
                    let values: Vec<u32> = match element.items {
                        ElementItems::Functions(indices) => indices
                            .into_iter()
                            .map(|index| {
                                index?.checked_add(1).ok_or_else(|| {
                                    Error::Unsupported("function index overflow".into())
                                })
                            })
                            .collect::<Result<_, Error>>()?,
                        ElementItems::Expressions(RefType::FUNCREF, expressions) => expressions
                            .into_iter()
                            .map(|expr| eval_ref_const(&expr?))
                            .collect::<Result<_, Error>>()?,
                        _ => {
                            return Err(Error::Unsupported(
                                "only funcref elements are supported".into(),
                            ));
                        }
                    };
                    let table = table_template
                        .as_mut()
                        .ok_or_else(|| Error::Unsupported("elements without a table".into()))?;
                    let end = offset.checked_add(values.len()).ok_or_else(|| {
                        Error::Unsupported("element segment offset overflow".into())
                    })?;
                    if end > table.len() {
                        return Err(Error::Unsupported(
                            "element segment exceeds initial table".into(),
                        ));
                    }
                    table[offset..end].copy_from_slice(&values);
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
                    let offset = eval_const(&offset_expr, &globals_template)? as u32 as usize;
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
        table_template.is_some(),
    )?;
    let fast_i32 = ty.params() == [ValType::I32] && ty.results() == [ValType::I32];
    let mut visit_state = vec![0u8; bodies.len()];
    let mut order = Vec::new();
    let mut can_trap = memory_template.is_some();
    let mut uses_growth = false;
    let mut table_targets: Vec<usize> = table_template
        .as_ref()
        .into_iter()
        .flat_map(|table| table.iter().copied())
        .filter(|&value| value != 0)
        .map(|value| (value - 1) as usize)
        .collect();
    table_targets.sort_unstable();
    table_targets.dedup();
    visit_function(
        index,
        &bodies,
        &types,
        &function_types,
        &table_targets,
        table_template.as_deref().unwrap_or(&[]),
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
    source.push_str(include_str!("softfloat64.metal"));
    source.push_str(include_str!("wasm_float.metal"));
    for &function_index in &order {
        let ty = &types[function_types[function_index] as usize];
        source.push_str(&function_signature(
            function_index,
            ty,
            memory_template.is_some(),
            !globals_template.is_empty(),
            table_template.is_some(),
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
            table_template.as_deref(),
            &table_targets,
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
        source.push_str(", device ulong* globals [[buffer(5)]]");
    }
    if table_template.is_some() {
        source.push_str(", device const uint* table [[buffer(7)]]");
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
        args.push(format!("ulong(input[{offset}])"));
    }
    for result in 0..ty.results().len() {
        source.push_str(&format!("ulong result{result} = 0ul;\n"));
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
    if table_template.is_some() {
        args.push("table".into());
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
        table_template,
    })
}

fn is_numeric(ty: ValType) -> bool {
    matches!(
        ty,
        ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64
    )
}

fn eval_const(expr: &ConstExpr<'_>, globals: &[u64]) -> Result<u64, Error> {
    let mut operators = expr.get_operators_reader().into_iter();
    let value = match operators.next().transpose()? {
        Some(Operator::I32Const { value }) => value as u32 as u64,
        Some(Operator::I64Const { value }) => value as u64,
        Some(Operator::F32Const { value }) => value.bits() as u64,
        Some(Operator::F64Const { value }) => value.bits(),
        Some(Operator::GlobalGet { global_index }) => *globals
            .get(global_index as usize)
            .ok_or_else(|| Error::Unsupported("constant global index is out of range".into()))?,
        _ => return Err(Error::Unsupported("unsupported numeric initializer".into())),
    };
    if !matches!(operators.next().transpose()?, Some(Operator::End)) || operators.next().is_some() {
        return Err(Error::Unsupported("unsupported numeric initializer".into()));
    }
    Ok(value)
}

fn eval_ref_const(expr: &ConstExpr<'_>) -> Result<u32, Error> {
    let mut operators = expr.get_operators_reader().into_iter();
    let value = match operators.next().transpose()? {
        Some(Operator::RefNull { .. }) => 0,
        Some(Operator::RefFunc { function_index }) => function_index
            .checked_add(1)
            .ok_or_else(|| Error::Unsupported("function index overflow".into()))?,
        _ => return Err(Error::Unsupported("unsupported funcref initializer".into())),
    };
    if !matches!(operators.next().transpose()?, Some(Operator::End)) || operators.next().is_some() {
        return Err(Error::Unsupported("unsupported funcref initializer".into()));
    }
    Ok(value)
}

fn visit_function(
    index: usize,
    bodies: &[FunctionBody<'_>],
    types: &[FuncType],
    function_types: &[u32],
    table_targets: &[usize],
    table_values: &[u32],
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
    let mut previous_constant = None;
    for operator in bodies[index].get_operators_reader()? {
        let operator = operator?;
        let current_constant = match &operator {
            Operator::I32Const { value } => Some(*value as u32),
            _ => None,
        };
        match operator {
            Operator::Call { function_index } => {
                visit_function(
                    function_index as usize,
                    bodies,
                    types,
                    function_types,
                    table_targets,
                    table_values,
                    state,
                    order,
                    can_trap,
                    uses_growth,
                )?;
            }
            Operator::CallIndirect { type_index, .. } => {
                *can_trap = true;
                let expected = types.get(type_index as usize).ok_or_else(|| {
                    Error::Unsupported("indirect type index is out of range".into())
                })?;
                for &target in table_targets {
                    if let Some(slot) = previous_constant {
                        if table_values.get(slot as usize).copied() != Some(target as u32 + 1) {
                            continue;
                        }
                    }
                    let target_type = function_types
                        .get(target)
                        .and_then(|index| types.get(*index as usize))
                        .ok_or_else(|| {
                            Error::Unsupported("table function type is out of range".into())
                        })?;
                    if target_type == expected {
                        visit_function(
                            target,
                            bodies,
                            types,
                            function_types,
                            table_targets,
                            table_values,
                            state,
                            order,
                            can_trap,
                            uses_growth,
                        )?;
                    }
                }
            }
            Operator::MemoryGrow { .. } => *uses_growth = true,
            Operator::I32DivS
            | Operator::I32DivU
            | Operator::I32RemS
            | Operator::I32RemU
            | Operator::I64DivS
            | Operator::I64DivU
            | Operator::I64RemS
            | Operator::I64RemU
            | Operator::I32TruncF32S
            | Operator::I32TruncF32U
            | Operator::I32TruncF64S
            | Operator::I32TruncF64U
            | Operator::I64TruncF32S
            | Operator::I64TruncF32U
            | Operator::I64TruncF64S
            | Operator::I64TruncF64U
            | Operator::Unreachable => *can_trap = true,
            _ => {}
        }
        previous_constant = current_constant;
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
    has_table: bool,
) -> Result<String, Error> {
    if ty
        .params()
        .iter()
        .chain(ty.results())
        .any(|ty| !is_numeric(*ty))
    {
        return Err(Error::Unsupported(
            "only numeric function signatures are supported".into(),
        ));
    }
    let mut args: Vec<_> = ty
        .params()
        .iter()
        .enumerate()
        .map(|(index, _)| format!("ulong a{index}"))
        .collect();
    args.extend(
        ty.results()
            .iter()
            .enumerate()
            .map(|(index, _)| format!("thread ulong& r{index}")),
    );
    if has_memory {
        args.push("device uchar* memory".into());
        args.push("device uint* pages".into());
    }
    if has_globals {
        args.push("device ulong* globals".into());
    }
    if has_table {
        args.push("device const uint* table".into());
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
    table_values: Option<&[u32]>,
    table_targets: &[usize],
) -> Result<String, Error> {
    let mut source = function_signature(
        function_index,
        ty,
        memory_bytes.is_some(),
        !globals_mutable.is_empty(),
        table_values.is_some(),
    )?;
    source.push_str(" {\n");
    for index in 0..ty.params().len() {
        source.push_str(&format!("ulong l{index} = a{index};\n"));
    }
    let mut local_count = ty.params().len();
    for local in body.get_locals_reader()? {
        let (count, ty) = local?;
        if !is_numeric(ty) {
            return Err(Error::Unsupported(
                "only numeric locals are supported".into(),
            ));
        }
        let end = local_count
            .checked_add(count as usize)
            .ok_or_else(|| Error::Unsupported("too many locals".into()))?;
        for index in local_count..end {
            source.push_str(&format!("ulong l{index} = 0ul;\n"));
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
        table_len: table_values.map(<[u32]>::len),
        table_values,
        table_targets,
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
    table_len: Option<usize>,
    table_values: Option<&'a [u32]>,
    table_targets: &'a [usize],
    frames: Vec<Frame>,
}

impl Compiler<'_> {
    fn fresh_value(&mut self) -> String {
        let name = format!("v{}", self.next_value);
        self.next_value += 1;
        self.declarations
            .push_str(&format!("ulong {name} = 0ul;\n"));
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
            "{index} = ulong({address}) + {}ul;\n",
            memarg.offset
        ));
        Ok(index)
    }

    fn emit_load(
        &mut self,
        stack: &mut Vec<String>,
        memarg: &wasmparser::MemArg,
        width: usize,
        signed: bool,
        result_bits: usize,
    ) -> Result<(), Error> {
        let address = pop(stack)?;
        let index = self.memory_address(&address, memarg, width)?;
        let value = self.fresh_value();
        for byte in 0..width {
            self.source.push_str(&format!(
                "{value} |= ulong(memory[ulong({index}) + {byte}ul]) << {}u;\n",
                byte * 8
            ));
        }
        if signed && width * 8 < result_bits {
            let shift = result_bits - width * 8;
            let expression = if result_bits == 32 {
                format!("as_type<uint>(as_type<int>(uint({value}) << {shift}u) >> {shift}u)")
            } else {
                format!("as_type<ulong>(as_type<long>({value} << {shift}u) >> {shift}u)")
            };
            self.source.push_str(&format!("{value} = {expression};\n"));
        }
        stack.push(value);
        Ok(())
    }

    fn emit_store(
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
            BlockType::Type(ty) if is_numeric(ty) => (0, 1),
            BlockType::FuncType(index) => {
                let ty = self
                    .types
                    .get(index as usize)
                    .ok_or_else(|| Error::Unsupported("block type index is out of range".into()))?;
                if ty
                    .params()
                    .iter()
                    .chain(ty.results())
                    .any(|ty| !is_numeric(*ty))
                {
                    return Err(Error::Unsupported(
                        "only numeric block signatures are supported".into(),
                    ));
                }
                (ty.params().len(), ty.results().len())
            }
            _ => {
                return Err(Error::Unsupported(
                    "only numeric block signatures are supported".into(),
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
            Operator::I32Const { value } => stack.push(format!("{}ul", *value as u32)),
            Operator::I64Const { value } => stack.push(format!("{}ul", *value as u64)),
            Operator::F32Const { value } => stack.push(format!("{}ul", value.bits())),
            Operator::F64Const { value } => stack.push(format!("{}ul", value.bits())),
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
            Operator::I32Load { memarg } | Operator::F32Load { memarg } => {
                self.emit_load(stack, memarg, 4, false, 32)?
            }
            Operator::I64Load { memarg } | Operator::F64Load { memarg } => {
                self.emit_load(stack, memarg, 8, false, 64)?
            }
            Operator::I32Load8U { memarg } => self.emit_load(stack, memarg, 1, false, 32)?,
            Operator::I32Load8S { memarg } => self.emit_load(stack, memarg, 1, true, 32)?,
            Operator::I32Load16U { memarg } => self.emit_load(stack, memarg, 2, false, 32)?,
            Operator::I32Load16S { memarg } => self.emit_load(stack, memarg, 2, true, 32)?,
            Operator::I64Load8U { memarg } => self.emit_load(stack, memarg, 1, false, 64)?,
            Operator::I64Load8S { memarg } => self.emit_load(stack, memarg, 1, true, 64)?,
            Operator::I64Load16U { memarg } => self.emit_load(stack, memarg, 2, false, 64)?,
            Operator::I64Load16S { memarg } => self.emit_load(stack, memarg, 2, true, 64)?,
            Operator::I64Load32U { memarg } => self.emit_load(stack, memarg, 4, false, 64)?,
            Operator::I64Load32S { memarg } => self.emit_load(stack, memarg, 4, true, 64)?,
            Operator::I32Store { memarg }
            | Operator::F32Store { memarg }
            | Operator::I64Store32 { memarg } => self.emit_store(stack, memarg, 4)?,
            Operator::I64Store { memarg } | Operator::F64Store { memarg } => {
                self.emit_store(stack, memarg, 8)?
            }
            Operator::I32Store8 { memarg } | Operator::I64Store8 { memarg } => {
                self.emit_store(stack, memarg, 1)?
            }
            Operator::I32Store16 { memarg } | Operator::I64Store16 { memarg } => {
                self.emit_store(stack, memarg, 2)?
            }
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
                    "{prior} = pages[0];\nif ({delta} <= ulong({}u - pages[0])) pages[0] += uint({delta}); else {prior} = 4294967295ul;\n",
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
                    .any(|ty| !is_numeric(*ty))
                {
                    return Err(Error::Unsupported(
                        "only numeric function signatures are supported".into(),
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
                if self.table_len.is_some() {
                    call_args.push("table".into());
                }
                call_args.push("trap".into());
                self.source.push_str(&format!(
                    "f{index}({});\nif (trap != 0u) return;\n",
                    call_args.join(", ")
                ));
                stack.extend(results);
            }
            Operator::CallIndirect {
                type_index,
                table_index: 0,
            } => {
                let table_len = self
                    .table_len
                    .ok_or_else(|| Error::Unsupported("table is missing".into()))?;
                let ty = self.types.get(*type_index as usize).ok_or_else(|| {
                    Error::Unsupported("indirect type index is out of range".into())
                })?;
                if ty
                    .params()
                    .iter()
                    .chain(ty.results())
                    .any(|ty| !is_numeric(*ty))
                {
                    return Err(Error::Unsupported(
                        "only numeric indirect signatures are supported".into(),
                    ));
                }
                let element = pop(stack)?;
                if stack.len() < ty.params().len() {
                    return Err(Error::Unsupported(
                        "indirect call argument is missing".into(),
                    ));
                }
                let args = stack.split_off(stack.len() - ty.params().len());
                let results: Vec<_> = (0..ty.results().len())
                    .map(|_| self.fresh_value())
                    .collect();
                self.source.push_str(&format!(
                    "if ({element} >= {table_len}ul) {{ trap = 6u; return; }}\n"
                ));
                let target = self.fresh_value();
                self.source
                    .push_str(&format!("{target} = table[uint({element})];\n"));
                self.source.push_str(&format!(
                    "if ({target} == 0ul) {{ trap = 6u; return; }}\nswitch ({target}) {{\n"
                ));
                let constant_slot = element
                    .strip_suffix("ul")
                    .and_then(|digits| digits.parse::<usize>().ok());
                let constant_target = constant_slot.and_then(|slot| {
                    self.table_values
                        .and_then(|values| values.get(slot))
                        .copied()
                });
                for &index in self.table_targets {
                    if constant_slot.is_some() && constant_target != Some(index as u32 + 1) {
                        continue;
                    }
                    let target_ty = self
                        .function_types
                        .get(index)
                        .and_then(|type_index| self.types.get(*type_index as usize))
                        .ok_or_else(|| {
                            Error::Unsupported("table function type is out of range".into())
                        })?;
                    if target_ty != ty {
                        continue;
                    }
                    let mut call_args = args.clone();
                    call_args.extend(results.iter().cloned());
                    if self.memory_bytes.is_some() {
                        call_args.push("memory".into());
                        call_args.push("pages".into());
                    }
                    if !self.globals_mutable.is_empty() {
                        call_args.push("globals".into());
                    }
                    call_args.push("table".into());
                    call_args.push("trap".into());
                    self.source.push_str(&format!(
                        "case {}u: f{index}({}); break;\n",
                        index + 1,
                        call_args.join(", ")
                    ));
                }
                self.source
                    .push_str("default: trap = 7u; return;\n}\nif (trap != 0u) return;\n");
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
            Operator::I64Eqz => {
                let value = pop(stack)?;
                self.emit_value(stack, format!("uint({value} == 0ul)"));
            }
            Operator::I32Clz
            | Operator::I32Ctz
            | Operator::I32Popcnt
            | Operator::I32Extend8S
            | Operator::I32Extend16S => {
                let value = pop(stack)?;
                let expression = match operator {
                    Operator::I32Clz => format!("({value} == 0u ? 32u : clz(uint({value})))"),
                    Operator::I32Ctz => format!("({value} == 0u ? 32u : ctz(uint({value})))"),
                    Operator::I32Popcnt => format!("popcount(uint({value}))"),
                    Operator::I32Extend8S => {
                        format!("as_type<uint>(as_type<int>(uint({value}) << 24u) >> 24u)")
                    }
                    Operator::I32Extend16S => {
                        format!("as_type<uint>(as_type<int>(uint({value}) << 16u) >> 16u)")
                    }
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            Operator::I64Clz
            | Operator::I64Ctz
            | Operator::I64Popcnt
            | Operator::I64Extend8S
            | Operator::I64Extend16S
            | Operator::I64Extend32S => {
                let value = pop(stack)?;
                let expression = match operator {
                    Operator::I64Clz => format!("({value} == 0ul ? 64ul : ulong(clz({value})))"),
                    Operator::I64Ctz => format!("({value} == 0ul ? 64ul : ulong(ctz({value})))"),
                    Operator::I64Popcnt => format!("ulong(popcount({value}))"),
                    Operator::I64Extend8S => {
                        format!("as_type<ulong>(as_type<long>({value} << 56u) >> 56u)")
                    }
                    Operator::I64Extend16S => {
                        format!("as_type<ulong>(as_type<long>({value} << 48u) >> 48u)")
                    }
                    Operator::I64Extend32S => {
                        format!("as_type<ulong>(as_type<long>({value} << 32u) >> 32u)")
                    }
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            Operator::Select
            | Operator::TypedSelect {
                ty: ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64,
            } => {
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
                let rhs = format!("uint({})", pop(stack)?);
                let lhs = format!("uint({})", pop(stack)?);
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
            Operator::I64Add
            | Operator::I64Sub
            | Operator::I64Mul
            | Operator::I64DivS
            | Operator::I64DivU
            | Operator::I64RemS
            | Operator::I64RemU
            | Operator::I64And
            | Operator::I64Or
            | Operator::I64Xor
            | Operator::I64Shl
            | Operator::I64ShrU
            | Operator::I64ShrS
            | Operator::I64Rotl
            | Operator::I64Rotr
            | Operator::I64Eq
            | Operator::I64Ne
            | Operator::I64LtS
            | Operator::I64LtU
            | Operator::I64GtS
            | Operator::I64GtU
            | Operator::I64LeS
            | Operator::I64LeU
            | Operator::I64GeS
            | Operator::I64GeU => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                if matches!(
                    operator,
                    Operator::I64DivS | Operator::I64DivU | Operator::I64RemS | Operator::I64RemU
                ) {
                    self.source
                        .push_str(&format!("if ({rhs} == 0ul) {{ trap = 1u; return; }}\n"));
                }
                if matches!(operator, Operator::I64DivS) {
                    self.source.push_str(&format!("if ({lhs} == 9223372036854775808ul && {rhs} == 18446744073709551615ul) {{ trap = 2u; return; }}\n"));
                }
                let expression = match operator {
                    Operator::I64Add => format!("{lhs} + {rhs}"),
                    Operator::I64Sub => format!("{lhs} - {rhs}"),
                    Operator::I64Mul => format!("{lhs} * {rhs}"),
                    Operator::I64DivS => {
                        format!("as_type<ulong>(as_type<long>({lhs}) / as_type<long>({rhs}))")
                    }
                    Operator::I64DivU => format!("{lhs} / {rhs}"),
                    Operator::I64RemS => format!(
                        "({lhs} == 9223372036854775808ul && {rhs} == 18446744073709551615ul ? 0ul : as_type<ulong>(as_type<long>({lhs}) % as_type<long>({rhs})))"
                    ),
                    Operator::I64RemU => format!("{lhs} % {rhs}"),
                    Operator::I64And => format!("{lhs} & {rhs}"),
                    Operator::I64Or => format!("{lhs} | {rhs}"),
                    Operator::I64Xor => format!("{lhs} ^ {rhs}"),
                    Operator::I64Shl => format!("{lhs} << ({rhs} & 63ul)"),
                    Operator::I64ShrU => format!("{lhs} >> ({rhs} & 63ul)"),
                    Operator::I64ShrS => {
                        format!("as_type<ulong>(as_type<long>({lhs}) >> ({rhs} & 63ul))")
                    }
                    Operator::I64Rotl => format!(
                        "({lhs} << ({rhs} & 63ul)) | ({lhs} >> ((64ul - ({rhs} & 63ul)) & 63ul))"
                    ),
                    Operator::I64Rotr => format!(
                        "({lhs} >> ({rhs} & 63ul)) | ({lhs} << ((64ul - ({rhs} & 63ul)) & 63ul))"
                    ),
                    Operator::I64Eq => format!("uint({lhs} == {rhs})"),
                    Operator::I64Ne => format!("uint({lhs} != {rhs})"),
                    Operator::I64LtS => {
                        format!("uint(as_type<long>({lhs}) < as_type<long>({rhs}))")
                    }
                    Operator::I64LtU => format!("uint({lhs} < {rhs})"),
                    Operator::I64GtS => {
                        format!("uint(as_type<long>({lhs}) > as_type<long>({rhs}))")
                    }
                    Operator::I64GtU => format!("uint({lhs} > {rhs})"),
                    Operator::I64LeS => {
                        format!("uint(as_type<long>({lhs}) <= as_type<long>({rhs}))")
                    }
                    Operator::I64LeU => format!("uint({lhs} <= {rhs})"),
                    Operator::I64GeS => {
                        format!("uint(as_type<long>({lhs}) >= as_type<long>({rhs}))")
                    }
                    Operator::I64GeU => format!("uint({lhs} >= {rhs})"),
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            Operator::F32Abs
            | Operator::F32Neg
            | Operator::F32Ceil
            | Operator::F32Floor
            | Operator::F32Trunc
            | Operator::F32Nearest
            | Operator::F32Sqrt
            | Operator::F64Abs
            | Operator::F64Neg
            | Operator::F64Ceil
            | Operator::F64Floor
            | Operator::F64Trunc
            | Operator::F64Nearest
            | Operator::F64Sqrt => {
                let value = pop(stack)?;
                let expression = match operator {
                    Operator::F32Abs => format!("uint({value}) & 0x7fffffffu"),
                    Operator::F32Neg => format!("uint({value}) ^ 0x80000000u"),
                    Operator::F32Ceil => format!(
                        "__softfloat64_cvt_f64_to_f32(wasm_f64_ceil(__softfloat64_cvt_f32_to_f64(uint({value}))), 0u)"
                    ),
                    Operator::F32Floor => format!(
                        "__softfloat64_cvt_f64_to_f32(wasm_f64_floor(__softfloat64_cvt_f32_to_f64(uint({value}))), 0u)"
                    ),
                    Operator::F32Trunc => format!(
                        "__softfloat64_cvt_f64_to_f32(wasm_f64_trunc(__softfloat64_cvt_f32_to_f64(uint({value}))), 0u)"
                    ),
                    Operator::F32Nearest => format!(
                        "__softfloat64_cvt_f64_to_f32(wasm_f64_nearest(__softfloat64_cvt_f32_to_f64(uint({value}))), 0u)"
                    ),
                    Operator::F32Sqrt => format!("wasm_f32_sqrt(uint({value}))"),
                    Operator::F64Abs => format!("{value} & 0x7ffffffffffffffful"),
                    Operator::F64Neg => format!("{value} ^ 0x8000000000000000ul"),
                    Operator::F64Ceil => format!("wasm_f64_ceil({value})"),
                    Operator::F64Floor => format!("wasm_f64_floor({value})"),
                    Operator::F64Trunc => format!("wasm_f64_trunc({value})"),
                    Operator::F64Nearest => format!("wasm_f64_nearest({value})"),
                    Operator::F64Sqrt => format!("__softfloat64_fsqrt({value}, 0u)"),
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            Operator::F32Add
            | Operator::F32Sub
            | Operator::F32Mul
            | Operator::F32Div
            | Operator::F32Min
            | Operator::F32Max
            | Operator::F32Copysign
            | Operator::F32Eq
            | Operator::F32Ne
            | Operator::F32Lt
            | Operator::F32Gt
            | Operator::F32Le
            | Operator::F32Ge
            | Operator::F64Add
            | Operator::F64Sub
            | Operator::F64Mul
            | Operator::F64Div
            | Operator::F64Min
            | Operator::F64Max
            | Operator::F64Copysign
            | Operator::F64Eq
            | Operator::F64Ne
            | Operator::F64Lt
            | Operator::F64Gt
            | Operator::F64Le
            | Operator::F64Ge => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                let a = format!("__softfloat64_cvt_f32_to_f64(uint({lhs}))");
                let b = format!("__softfloat64_cvt_f32_to_f64(uint({rhs}))");
                let expression = match operator {
                    Operator::F32Add => format!("wasm_f32_add(uint({lhs}), uint({rhs}))"),
                    Operator::F32Sub => format!("wasm_f32_sub(uint({lhs}), uint({rhs}))"),
                    Operator::F32Mul => format!("wasm_f32_mul(uint({lhs}), uint({rhs}))"),
                    Operator::F32Div => format!("wasm_f32_div(uint({lhs}), uint({rhs}))"),
                    Operator::F32Min => format!("wasm_f32_min(uint({lhs}), uint({rhs}))"),
                    Operator::F32Max => format!("wasm_f32_max(uint({lhs}), uint({rhs}))"),
                    Operator::F32Copysign => {
                        format!("(uint({lhs}) & 0x7fffffffu) | (uint({rhs}) & 0x80000000u)")
                    }
                    Operator::F32Eq => format!("uint(__softfloat64_feq({a}, {b}))"),
                    Operator::F32Ne => format!("uint(!__softfloat64_feq({a}, {b}))"),
                    Operator::F32Lt => format!("uint(__softfloat64_flt({a}, {b}))"),
                    Operator::F32Gt => format!("uint(__softfloat64_fgt({a}, {b}))"),
                    Operator::F32Le => format!("uint(__softfloat64_fle({a}, {b}))"),
                    Operator::F32Ge => format!("uint(__softfloat64_fge({a}, {b}))"),
                    Operator::F64Add => format!("__softfloat64_fadd({lhs}, {rhs}, 0u)"),
                    Operator::F64Sub => format!("__softfloat64_fsub({lhs}, {rhs}, 0u)"),
                    Operator::F64Mul => format!("__softfloat64_fmul({lhs}, {rhs}, 0u)"),
                    Operator::F64Div => format!("__softfloat64_fdiv({lhs}, {rhs}, 0u)"),
                    Operator::F64Min => format!("wasm_f64_min({lhs}, {rhs})"),
                    Operator::F64Max => format!("wasm_f64_max({lhs}, {rhs})"),
                    Operator::F64Copysign => {
                        format!("({lhs} & 0x7ffffffffffffffful) | ({rhs} & 0x8000000000000000ul)")
                    }
                    Operator::F64Eq => format!("uint(__softfloat64_feq({lhs}, {rhs}))"),
                    Operator::F64Ne => format!("uint(!__softfloat64_feq({lhs}, {rhs}))"),
                    Operator::F64Lt => format!("uint(__softfloat64_flt({lhs}, {rhs}))"),
                    Operator::F64Gt => format!("uint(__softfloat64_fgt({lhs}, {rhs}))"),
                    Operator::F64Le => format!("uint(__softfloat64_fle({lhs}, {rhs}))"),
                    Operator::F64Ge => format!("uint(__softfloat64_fge({lhs}, {rhs}))"),
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            Operator::I32WrapI64
            | Operator::I64ExtendI32S
            | Operator::I64ExtendI32U
            | Operator::I32ReinterpretF32
            | Operator::F32ReinterpretI32
            | Operator::I64ReinterpretF64
            | Operator::F64ReinterpretI64
            | Operator::F32ConvertI32S
            | Operator::F32ConvertI32U
            | Operator::F32ConvertI64S
            | Operator::F32ConvertI64U
            | Operator::F64ConvertI32S
            | Operator::F64ConvertI32U
            | Operator::F64ConvertI64S
            | Operator::F64ConvertI64U
            | Operator::F32DemoteF64
            | Operator::F64PromoteF32 => {
                let value = pop(stack)?;
                let expression = match operator {
                    Operator::I32WrapI64 => format!("ulong(uint({value}))"),
                    Operator::I64ExtendI32S => {
                        format!("as_type<ulong>(long(as_type<int>(uint({value}))))")
                    }
                    Operator::I64ExtendI32U => format!("ulong(uint({value}))"),
                    Operator::I32ReinterpretF32 | Operator::F32ReinterpretI32 => {
                        format!("uint({value})")
                    }
                    Operator::I64ReinterpretF64 | Operator::F64ReinterpretI64 => value,
                    Operator::F32ConvertI32S => {
                        format!("as_type<uint>(float(as_type<int>(uint({value}))))")
                    }
                    Operator::F32ConvertI32U => format!("as_type<uint>(float(uint({value})))"),
                    Operator::F32ConvertI64S => {
                        format!("as_type<uint>(float(as_type<long>({value})))")
                    }
                    Operator::F32ConvertI64U => format!("as_type<uint>(float({value}))"),
                    Operator::F64ConvertI32S => format!(
                        "__softfloat64_cvt_i64_to_f64(long(as_type<int>(uint({value}))), 0u)"
                    ),
                    Operator::F64ConvertI32U => {
                        format!("__softfloat64_cvt_u64_to_f64(ulong(uint({value})), 0u)")
                    }
                    Operator::F64ConvertI64S => {
                        format!("__softfloat64_cvt_i64_to_f64(as_type<long>({value}), 0u)")
                    }
                    Operator::F64ConvertI64U => {
                        format!("__softfloat64_cvt_u64_to_f64({value}, 0u)")
                    }
                    Operator::F32DemoteF64 => format!("__softfloat64_cvt_f64_to_f32({value}, 0u)"),
                    Operator::F64PromoteF32 => {
                        format!("__softfloat64_cvt_f32_to_f64(uint({value}))")
                    }
                    _ => unreachable!(),
                };
                self.emit_value(stack, expression);
            }
            Operator::I32TruncF32S
            | Operator::I32TruncF32U
            | Operator::I32TruncF64S
            | Operator::I32TruncF64U
            | Operator::I64TruncF32S
            | Operator::I64TruncF32U
            | Operator::I64TruncF64S
            | Operator::I64TruncF64U
            | Operator::I32TruncSatF32S
            | Operator::I32TruncSatF32U
            | Operator::I32TruncSatF64S
            | Operator::I32TruncSatF64U
            | Operator::I64TruncSatF32S
            | Operator::I64TruncSatF32U
            | Operator::I64TruncSatF64S
            | Operator::I64TruncSatF64U => {
                let value = pop(stack)?;
                let f32_source = matches!(
                    operator,
                    Operator::I32TruncF32S
                        | Operator::I32TruncF32U
                        | Operator::I64TruncF32S
                        | Operator::I64TruncF32U
                        | Operator::I32TruncSatF32S
                        | Operator::I32TruncSatF32U
                        | Operator::I64TruncSatF32S
                        | Operator::I64TruncSatF32U
                );
                let width = if matches!(
                    operator,
                    Operator::I32TruncF32S
                        | Operator::I32TruncF32U
                        | Operator::I32TruncF64S
                        | Operator::I32TruncF64U
                        | Operator::I32TruncSatF32S
                        | Operator::I32TruncSatF32U
                        | Operator::I32TruncSatF64S
                        | Operator::I32TruncSatF64U
                ) {
                    32
                } else {
                    64
                };
                let signed = matches!(
                    operator,
                    Operator::I32TruncF32S
                        | Operator::I32TruncF64S
                        | Operator::I64TruncF32S
                        | Operator::I64TruncF64S
                        | Operator::I32TruncSatF32S
                        | Operator::I32TruncSatF64S
                        | Operator::I64TruncSatF32S
                        | Operator::I64TruncSatF64S
                );
                let saturating = matches!(
                    operator,
                    Operator::I32TruncSatF32S
                        | Operator::I32TruncSatF32U
                        | Operator::I32TruncSatF64S
                        | Operator::I32TruncSatF64U
                        | Operator::I64TruncSatF32S
                        | Operator::I64TruncSatF32U
                        | Operator::I64TruncSatF64S
                        | Operator::I64TruncSatF64U
                );
                let float_bits = if f32_source {
                    format!("__softfloat64_cvt_f32_to_f64(uint({value}))")
                } else {
                    value
                };
                self.emit_value(
                    stack,
                    format!(
                        "wasm_f64_to_int({float_bits}, {width}u, {signed}, {saturating}, trap)"
                    ),
                );
                if !saturating {
                    self.source.push_str("if (trap != 0u) return;\n");
                }
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

    #[test]
    fn i64_arithmetic_calls_and_globals_match_wasmtime() {
        let wasm = wat::parse_str(
            r#"(module
            (global $g (mut i64) (i64.const -9))
            (func $rotate (param i64 i64) (result i64)
                local.get 0 local.get 1 i64.rotl)
            (func (export "f") (param i64 i64) (result i64 i64)
                local.get 0 local.get 1 call $rotate
                global.get $g local.get 0 i64.add
                global.set $g
                global.get $g))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let inputs = [
            vec![Value::I64(0), Value::I64(0)],
            vec![Value::I64(u64::MAX), Value::I64(65)],
            vec![Value::I64(0x8000_0000_0000_0001), Value::I64(13)],
        ];
        let actual = kernel.run_values(&inputs).unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        for (row, result) in inputs.iter().zip(actual) {
            let mut store = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let function = instance
                .get_typed_func::<(i64, i64), (i64, i64)>(&mut store, "f")
                .unwrap();
            let (a, b) = function
                .call(&mut store, (row[0].bits() as i64, row[1].bits() as i64))
                .unwrap();
            assert_eq!(result, vec![Value::I64(a as u64), Value::I64(b as u64)]);
        }
    }

    #[test]
    fn i64_memory_and_division_traps() {
        let wasm = wat::parse_str("(module (memory 1) (func (export \"f\") (param i64) (result i64) i32.const 0 local.get 0 i64.store i32.const 0 i64.load32_s i64.const 3 i64.div_s))").unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let rows = [vec![Value::I64(u64::MAX)], vec![Value::I64(0x8000_0000)]];
        assert_eq!(
            kernel.run_values(&rows).unwrap(),
            [
                vec![Value::I64(0)],
                vec![Value::I64((-715827882i64) as u64)]
            ]
        );
        let wasm = wat::parse_str("(module (func (export \"f\") (param i64) (result i64) local.get 0 i64.const 0 i64.div_u))").unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert!(matches!(
            kernel.run_values(&[vec![Value::I64(1)]]),
            Err(Error::Trap {
                reason: "integer divide by zero",
                ..
            })
        ));
    }

    #[test]
    fn f64_software_add_runs_on_metal() {
        let wasm = wat::parse_str("(module (func (export \"f\") (param f64 f64) (result f64) local.get 0 local.get 1 f64.add))").unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let rows = [vec![
            Value::F64(1.25f64.to_bits()),
            Value::F64(2.5f64.to_bits()),
        ]];
        assert_eq!(
            kernel.run_values(&rows).unwrap(),
            [vec![Value::F64(3.75f64.to_bits())]]
        );
    }

    fn compare_values_to_wasmtime(wat: &str, rows: &[Vec<Value>]) {
        let wasm = wat::parse_str(wat).unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let actual = kernel.run_values(rows).unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        for (row, actual_row) in rows.iter().zip(actual) {
            let mut store = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let function = instance.get_func(&mut store, "f").unwrap();
            let params: Vec<_> = row
                .iter()
                .map(|value| match value {
                    Value::I32(bits) => wasmtime::Val::I32(*bits as i32),
                    Value::I64(bits) => wasmtime::Val::I64(*bits as i64),
                    Value::F32(bits) => wasmtime::Val::F32(*bits),
                    Value::F64(bits) => wasmtime::Val::F64(*bits),
                })
                .collect();
            let mut results: Vec<_> = kernel
                .results
                .iter()
                .map(|ty| match ty {
                    ValType::I32 => wasmtime::Val::I32(0),
                    ValType::I64 => wasmtime::Val::I64(0),
                    ValType::F32 => wasmtime::Val::F32(0),
                    ValType::F64 => wasmtime::Val::F64(0),
                    _ => unreachable!(),
                })
                .collect();
            function.call(&mut store, &params, &mut results).unwrap();
            for (got, expected) in actual_row.iter().zip(results) {
                let equal = match (*got, expected) {
                    (Value::I32(a), wasmtime::Val::I32(b)) => a == b as u32,
                    (Value::I64(a), wasmtime::Val::I64(b)) => a == b as u64,
                    (Value::F32(a), wasmtime::Val::F32(b)) => {
                        a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
                    }
                    (Value::F64(a), wasmtime::Val::F64(b)) => {
                        a == b || (f64::from_bits(a).is_nan() && f64::from_bits(b).is_nan())
                    }
                    _ => false,
                };
                assert!(
                    equal,
                    "{wat}: input {row:?}, got {got:?}, expected {expected:?}"
                );
            }
        }
    }

    #[test]
    fn f64_arithmetic_matches_wasmtime_at_boundaries() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            f64::from_bits(0x000f_ffff_ffff_ffff),
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let rows: Vec<_> = values
            .iter()
            .flat_map(|a| {
                values
                    .iter()
                    .map(move |b| vec![Value::F64(a.to_bits()), Value::F64(b.to_bits())])
            })
            .collect();
        for op in [
            "add", "sub", "mul", "div", "min", "max", "copysign", "eq", "ne", "lt", "gt", "le",
            "ge",
        ] {
            let result = if ["eq", "ne", "lt", "gt", "le", "ge"].contains(&op) {
                "i32"
            } else {
                "f64"
            };
            compare_values_to_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param f64 f64) (result {result}) local.get 0 local.get 1 f64.{op}))"
                ),
                &rows,
            );
        }
        let rows: Vec<_> = values
            .iter()
            .map(|a| vec![Value::F64(a.to_bits())])
            .collect();
        for op in ["abs", "neg", "ceil", "floor", "trunc", "nearest", "sqrt"] {
            compare_values_to_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param f64) (result f64) local.get 0 f64.{op}))"
                ),
                &rows,
            );
        }
    }

    #[test]
    fn f32_arithmetic_matches_wasmtime_at_boundaries() {
        let values = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x007f_ffff),
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let rows: Vec<_> = values
            .iter()
            .flat_map(|a| {
                values
                    .iter()
                    .map(move |b| vec![Value::F32(a.to_bits()), Value::F32(b.to_bits())])
            })
            .collect();
        for op in [
            "add", "sub", "mul", "div", "min", "max", "copysign", "eq", "ne", "lt", "gt", "le",
            "ge",
        ] {
            let result = if ["eq", "ne", "lt", "gt", "le", "ge"].contains(&op) {
                "i32"
            } else {
                "f32"
            };
            compare_values_to_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param f32 f32) (result {result}) local.get 0 local.get 1 f32.{op}))"
                ),
                &rows,
            );
        }
        let rows: Vec<_> = values
            .iter()
            .map(|a| vec![Value::F32(a.to_bits())])
            .collect();
        for op in ["abs", "neg", "ceil", "floor", "trunc", "nearest", "sqrt"] {
            compare_values_to_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param f32) (result f32) local.get 0 f32.{op}))"
                ),
                &rows,
            );
        }
    }

    #[test]
    fn numeric_conversions_match_wasmtime() {
        for (input, rows) in [
            (
                "i32",
                vec![
                    0,
                    1,
                    i32::MAX as u64,
                    i32::MIN as u32 as u64,
                    u32::MAX as u64,
                ],
            ),
            (
                "i64",
                vec![
                    0,
                    1,
                    i64::MAX as u64,
                    i64::MIN as u64,
                    u64::MAX,
                    (1u64 << 53) + 1,
                ],
            ),
        ] {
            for output in ["f32", "f64"] {
                for sign in ["s", "u"] {
                    let values: Vec<_> = rows
                        .iter()
                        .map(|bits| {
                            vec![if input == "i32" {
                                Value::I32(*bits as u32)
                            } else {
                                Value::I64(*bits)
                            }]
                        })
                        .collect();
                    compare_values_to_wasmtime(
                        &format!(
                            "(module (func (export \"f\") (param {input}) (result {output}) local.get 0 {output}.convert_{input}_{sign}))"
                        ),
                        &values,
                    );
                }
            }
        }
        let f64_values = [
            0.0,
            -0.0,
            1.5,
            -1.5,
            2147483647.0,
            2147483648.0,
            4294967295.0,
            4294967296.0,
            9223372036854775808.0,
            18446744073709551616.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let f32_values = [
            0.0f32,
            -0.0,
            1.5,
            -1.5,
            2147483648.0,
            4294967296.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        for input in ["f32", "f64"] {
            let rows: Vec<_> = if input == "f32" {
                f32_values
                    .iter()
                    .map(|value| vec![Value::F32(value.to_bits())])
                    .collect()
            } else {
                f64_values
                    .iter()
                    .map(|value| vec![Value::F64(value.to_bits())])
                    .collect()
            };
            for output in ["i32", "i64"] {
                for sign in ["s", "u"] {
                    let wat = format!(
                        "(module (func (export \"f\") (param {input}) (result {output}) local.get 0 {output}.trunc_sat_{input}_{sign}))"
                    );
                    compare_values_to_wasmtime(&wat, &rows);
                    let wat = format!(
                        "(module (func (export \"f\") (param {input}) (result {output}) local.get 0 {output}.trunc_{input}_{sign}))"
                    );
                    let wasm = wat::parse_str(&wat).unwrap();
                    let kernel = Kernel::new(&wasm, "f").unwrap();
                    assert!(
                        kernel
                            .run_values(&[vec![if input == "f32" {
                                Value::F32(f32::NAN.to_bits())
                            } else {
                                Value::F64(f64::NAN.to_bits())
                            }]])
                            .is_err()
                    );
                }
            }
        }
        let rows: Vec<_> = f32_values
            .iter()
            .map(|value| vec![Value::F32(value.to_bits())])
            .collect();
        compare_values_to_wasmtime(
            "(module (func (export \"f\") (param f32) (result f64) local.get 0 f64.promote_f32))",
            &rows,
        );
        let rows: Vec<_> = f64_values
            .iter()
            .map(|value| vec![Value::F64(value.to_bits())])
            .collect();
        compare_values_to_wasmtime(
            "(module (func (export \"f\") (param f64) (result f32) local.get 0 f32.demote_f64))",
            &rows,
        );
    }

    #[test]
    fn f32_subnormal_midpoint_rounds_up() {
        let just_above_midpoint = 3.0f64 * 2.0f64.powi(-151);
        compare_values_to_wasmtime(
            "(module (func (export \"f\") (param f64) (result f32) local.get 0 f32.demote_f64))",
            &[vec![Value::F64(just_above_midpoint.to_bits())]],
        );
        compare_values_to_wasmtime(
            "(module (func (export \"f\") (param f32 f32) (result f32) local.get 0 local.get 1 f32.div))",
            &[vec![Value::F32(1), Value::F32(1.5f32.to_bits())]],
        );
    }

    #[test]
    fn mixed_numeric_calls_memory_and_globals_match_wasmtime() {
        let wat = r#"(module
            (memory 1)
            (global $g (mut f64) (f64.const 0.75))
            (func $step (param f64 i64) (result f64 i64) (local f32)
                local.get 0 f32.demote_f64 local.set 2
                i32.const 0 local.get 1 i64.store
                global.get $g local.get 0 f64.add global.set $g
                global.get $g local.get 2 f64.promote_f32 f64.add
                i32.const 0 i64.load)
            (func (export "f") (param f64 i64) (result f64 i64)
                local.get 0 local.get 1 call $step))"#;
        compare_values_to_wasmtime(
            wat,
            &[
                vec![Value::F64(1.25f64.to_bits()), Value::I64(7)],
                vec![Value::F64((-2.5f64).to_bits()), Value::I64(u64::MAX)],
            ],
        );
    }

    #[test]
    fn valid_float_to_int_truncation_matches_wasmtime() {
        for (input, values) in [
            (
                "f32",
                vec![
                    Value::F32(0.0f32.to_bits()),
                    Value::F32(1.75f32.to_bits()),
                    Value::F32((-1.75f32).to_bits()),
                ],
            ),
            (
                "f64",
                vec![
                    Value::F64(0.0f64.to_bits()),
                    Value::F64(1.75f64.to_bits()),
                    Value::F64((-1.75f64).to_bits()),
                ],
            ),
        ] {
            for output in ["i32", "i64"] {
                for sign in ["s", "u"] {
                    let rows: Vec<_> = values
                        .iter()
                        .copied()
                        .filter(|value| {
                            sign == "s"
                                || (value.bits()
                                    & if input == "f32" {
                                        1u64 << 31
                                    } else {
                                        1u64 << 63
                                    })
                                    == 0
                        })
                        .map(|value| vec![value])
                        .collect();
                    compare_values_to_wasmtime(
                        &format!(
                            "(module (func (export \"f\") (param {input}) (result {output}) local.get 0 {output}.trunc_{input}_{sign}))"
                        ),
                        &rows,
                    );
                }
            }
        }
    }

    #[test]
    fn indirect_calls_use_initialized_table_slots() {
        let wat = r#"(module
            (type $t (func (param i32) (result i32)))
            (table 3 funcref)
            (func $add (type $t) local.get 0 i32.const 3 i32.add)
            (func $mul (type $t) local.get 0 i32.const 7 i32.mul)
            (elem (i32.const 0) $add $mul)
            (func (export "f") (param i32 i32) (result i32)
                local.get 0 local.get 1 call_indirect (type $t)))"#;
        compare_values_to_wasmtime(
            wat,
            &[
                vec![Value::I32(11), Value::I32(0)],
                vec![Value::I32(11), Value::I32(1)],
                vec![Value::I32(u32::MAX), Value::I32(0)],
            ],
        );
        let wasm = wat::parse_str(wat).unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        for index in [2, 3] {
            assert!(matches!(
                kernel.run_values(&[vec![Value::I32(11), Value::I32(index)]]),
                Err(Error::Trap {
                    reason: "undefined or uninitialized table element",
                    ..
                })
            ));
        }
    }

    #[test]
    fn indirect_calls_check_type_and_reject_recursion() {
        let wat = r#"(module
            (type $t (func (param i32) (result i32)))
            (table 1 funcref)
            (func $other (param i64) (result i64) local.get 0)
            (elem (i32.const 0) $other)
            (func (export "f") (param i32) (result i32)
                local.get 0 i32.const 0 call_indirect (type $t)))"#;
        let kernel = Kernel::new(&wat::parse_str(wat).unwrap(), "f").unwrap();
        assert!(matches!(
            kernel.run(&[1]),
            Err(Error::Trap {
                reason: "indirect call type mismatch",
                ..
            })
        ));

        let recursive = r#"(module
            (type $t (func (param i32) (result i32)))
            (table 1 funcref)
            (func $f (export "f") (type $t)
                local.get 0 i32.const 0 call_indirect (type $t))
            (elem (i32.const 0) $f))"#;
        assert!(matches!(
            Kernel::new(&wat::parse_str(recursive).unwrap(), "f"),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn expression_elements_call_mixed_numeric_function() {
        let wat = r#"(module
            (type $t (func (param f64 i64) (result f64)))
            (table 2 funcref)
            (func $mul (type $t)
                local.get 0 local.get 1 f64.convert_i64_s f64.mul)
            (elem (i32.const 0) funcref (ref.func $mul) (ref.null func))
            (func (export "f") (param f64 i64 i32) (result f64)
                local.get 0 local.get 1 local.get 2 call_indirect (type $t)))"#;
        compare_values_to_wasmtime(
            wat,
            &[
                vec![Value::F64(1.25f64.to_bits()), Value::I64(3), Value::I32(0)],
                vec![
                    Value::F64((-0.5f64).to_bits()),
                    Value::I64(u64::MAX),
                    Value::I32(0),
                ],
            ],
        );
        let kernel = Kernel::new(&wat::parse_str(wat).unwrap(), "f").unwrap();
        assert!(matches!(
            kernel.run_values(&[vec![
                Value::F64(1.0f64.to_bits()),
                Value::I64(3),
                Value::I32(1)
            ]]),
            Err(Error::Trap {
                reason: "undefined or uninitialized table element",
                ..
            })
        ));
    }

    #[test]
    fn constant_indirect_slot_excludes_unused_recursive_target() {
        let wat = r#"(module
            (type $t (func (param i32) (result i32)))
            (table 2 funcref)
            (func $f (export "f") (type $t)
                local.get 0 i32.const 1 call_indirect (type $t))
            (func $g (type $t) local.get 0 i32.const 4 i32.add)
            (elem (i32.const 0) $f $g))"#;
        compare_values_to_wasmtime(wat, &[vec![Value::I32(0)], vec![Value::I32(7)]]);
    }
}
