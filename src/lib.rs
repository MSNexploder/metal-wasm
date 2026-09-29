//! Execute a supported WebAssembly function once per Metal thread.

use std::collections::HashMap;
use std::error::Error as StdError;
use std::fmt;
use std::time::Duration;

use metal::objc::{Message, runtime::Sel};
use metal::{
    CompileOptions, ComputePipelineDescriptor, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use wasmparser::{
    BlockType, ConstExpr, DataKind, ElementItems, ElementKind, ExternalKind, FuncType,
    FunctionBody, Operator, Parser, Payload, RefType, TableInit, ValType, Validator,
};

const MAX_MEMORY_BYTES_PER_INSTANCE: usize = 16 * 1024 * 1024;
const MAX_GROWTH_PAGES: usize = 4;
const MAX_TABLE_ELEMENTS: usize = 65536;
const MAX_TABLE_GROWTH: usize = 256;
const MAX_CALL_DEPTH: usize = 32;

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
    instances: Vec<InstanceBuffers>,
}

struct Translation {
    source: String,
    can_trap: bool,
    params: Vec<ValType>,
    results: Vec<ValType>,
    instances: Vec<InstanceTemplate>,
    slots: Vec<BufferSlots>,
    recursive: bool,
}

struct InstanceTemplate {
    memory_template: Option<Vec<u8>>,
    memory_capacity: Option<usize>,
    globals_template: Vec<u64>,
    table_template: Option<Vec<u32>>,
    table_info: TableInfo,
    passive_data: Vec<u8>,
    data_lengths: Vec<u32>,
}

#[derive(Default)]
struct BufferSlots {
    memory: Option<u64>,
    pages: Option<u64>,
    globals: Option<u64>,
    table: Option<u64>,
    passive_data: Option<u64>,
}

struct InstanceBuffers {
    template: InstanceTemplate,
    slots: BufferSlots,
    table: Option<metal::Buffer>,
    passive_data: Option<metal::Buffer>,
}

#[derive(Default)]
struct TableInfo {
    mutable: bool,
    capacity: usize,
    elements: Vec<Vec<u32>>,
    lengths: Vec<u32>,
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
        Self::new_with_imports(wasm, export, &[])
    }

    /// Compile an export with Wasm modules registered as function-import providers.
    /// A provider namespace owns one independent instance per lane, shared by all
    /// imports from that namespace. Provider dependencies initialize before importers.
    /// Host callbacks, imported resources, and cyclic provider dependencies are rejected.
    pub fn new_with_imports(
        wasm: &[u8],
        export: &str,
        providers: &[(&str, &[u8])],
    ) -> Result<Self, Error> {
        let translation = translate(wasm, export, providers)?;
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
        let pipeline = if translation.recursive {
            let descriptor = ComputePipelineDescriptor::new();
            descriptor.set_compute_function(Some(&function));
            descriptor.set_max_call_stack_depth(MAX_CALL_DEPTH as u64 + 2);
            device.new_compute_pipeline_state(&descriptor)
        } else {
            device.new_compute_pipeline_state_with_function(&function)
        }
        .map_err(Error::Metal)?;
        let queue = device.new_command_queue();
        let mut instances = Vec::new();
        for (template, slots) in translation.instances.into_iter().zip(translation.slots) {
            let table = template
                .table_template
                .as_ref()
                .filter(|_| !template.table_info.mutable)
                .map(|data| make_buffer(&device, data))
                .transpose()?;
            let passive_data = (!template.data_lengths.is_empty())
                .then(|| make_buffer(&device, &template.passive_data))
                .transpose()?;
            instances.push(InstanceBuffers {
                template,
                slots,
                table,
                passive_data,
            });
        }
        Ok(Self {
            device,
            queue,
            pipeline,
            can_trap: translation.can_trap,
            params: translation.params,
            results: translation.results,
            instances,
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
        if input_bytes.max(output_bytes).max(trap_bytes) > self.device.max_buffer_length() {
            return Err(Error::Unsupported(
                "batch exceeds the Metal buffer limit".into(),
            ));
        }
        let mut instance_buffers = Vec::new();
        for instance in &self.instances {
            instance_buffers.extend(instance.for_batch(&self.device, count)?);
        }
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
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&input), 0);
        encoder.set_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, size_of::<u32>() as u64, (&count as *const u32).cast());
        if let Some(traps) = &traps {
            encoder.set_buffer(3, Some(traps), 0);
        }
        for (slot, buffer) in &instance_buffers {
            encoder.set_buffer(*slot, Some(buffer), 0);
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
                    8 => "call depth limit exceeded",
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

fn make_buffer<T>(device: &Device, data: &[T]) -> Result<metal::Buffer, Error> {
    let bytes = size_of_val(data).max(4) as u64;
    if bytes > device.max_buffer_length() {
        return Err(Error::Unsupported(
            "batch exceeds the Metal buffer limit".into(),
        ));
    }
    let empty = [0u8; 4];
    Ok(device.new_buffer_with_data(
        if data.is_empty() {
            empty.as_ptr().cast()
        } else {
            data.as_ptr().cast()
        },
        bytes,
        MTLResourceOptions::StorageModeShared,
    ))
}

fn lane_buffer<T: Copy + Default>(
    device: &Device,
    template: &[T],
    capacity: usize,
    count: u32,
) -> Result<metal::Buffer, Error> {
    let stride = capacity.max(1);
    let len = stride
        .checked_mul(count as usize)
        .ok_or_else(|| Error::Unsupported("instance buffer size overflow".into()))?;
    let bytes = len
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::Unsupported("instance buffer size overflow".into()))?;
    if bytes as u64 > device.max_buffer_length() {
        return Err(Error::Unsupported(
            "batch exceeds the Metal buffer limit".into(),
        ));
    }
    let mut data = Vec::new();
    data.try_reserve_exact(len)
        .map_err(|_| Error::Unsupported("cannot allocate per-lane state".into()))?;
    data.resize(len, T::default());
    for lane in data.chunks_mut(stride) {
        lane[..template.len()].copy_from_slice(template);
    }
    make_buffer(device, &data)
}

impl InstanceBuffers {
    fn for_batch(&self, device: &Device, count: u32) -> Result<Vec<(u64, metal::Buffer)>, Error> {
        let mut buffers = Vec::new();
        let template = &self.template;
        if let Some(slot) = self.slots.memory {
            let initial = template.memory_template.as_ref().unwrap();
            buffers.push((
                slot,
                lane_buffer(device, initial, template.memory_capacity.unwrap(), count)?,
            ));
            buffers.push((
                self.slots.pages.unwrap(),
                lane_buffer(device, &[initial.len() as u32 / 65536], 1, count)?,
            ));
        }
        if let Some(slot) = self.slots.globals {
            buffers.push((
                slot,
                lane_buffer(
                    device,
                    &template.globals_template,
                    template.globals_template.len(),
                    count,
                )?,
            ));
        }
        if let Some(slot) = self.slots.table {
            let table = if template.table_info.mutable {
                lane_buffer(
                    device,
                    template.table_template.as_ref().unwrap(),
                    template.table_info.capacity,
                    count,
                )?
            } else {
                self.table.as_ref().unwrap().clone()
            };
            buffers.push((slot, table));
        }
        if let Some(slot) = self.slots.passive_data {
            buffers.push((slot, self.passive_data.as_ref().unwrap().clone()));
        }
        Ok(buffers)
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

struct ParsedModule<'a> {
    types: Vec<FuncType>,
    function_types: Vec<u32>,
    bodies: Vec<FunctionBody<'a>>,
    exports: HashMap<String, (ExternalKind, usize)>,
    imports: Vec<(String, String)>,
    resolved_imports: Vec<(usize, usize)>,
    resolved_functions: Vec<usize>,
    start_index: Option<usize>,
    globals_mutable: Vec<bool>,
    reference_globals: Vec<usize>,
    memory_max_pages: Option<u64>,
    data_offsets: Vec<usize>,
    table_targets: Vec<usize>,
    instance: InstanceTemplate,
}

impl ParsedModule<'_> {
    fn function_type(&self, index: usize) -> Result<&FuncType, Error> {
        self.function_types
            .get(index)
            .and_then(|index| self.types.get(*index as usize))
            .ok_or_else(|| Error::Unsupported("function type is out of range".into()))
    }
}

fn parse_module(wasm: &[u8]) -> Result<ParsedModule<'_>, Error> {
    Validator::new().validate_all(wasm)?;

    let mut types = Vec::new();
    let mut function_types = Vec::new();
    let mut bodies = Vec::new();
    let mut exports = HashMap::new();
    let mut imports = Vec::new();
    let mut start_index = None;
    let mut memory_template: Option<Vec<u8>> = None;
    let mut memory_max_pages = None;
    let mut globals_template = Vec::new();
    let mut globals_mutable = Vec::new();
    let mut reference_globals = Vec::new();
    let mut table_template: Option<Vec<u32>> = None;
    let mut table_info = TableInfo::default();
    let mut table_max = None;
    let mut passive_data = Vec::new();
    let mut data_offsets = Vec::new();
    let mut data_lengths = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        match payload? {
            Payload::Version { .. } | Payload::CustomSection(_) | Payload::End(_) => {}
            Payload::TypeSection(section) => {
                types = section
                    .into_iter_err_on_gc_types()
                    .collect::<Result<_, _>>()?;
            }
            Payload::ImportSection(section) => {
                for import in section.into_imports() {
                    let import = import?;
                    let wasmparser::TypeRef::Func(ty) = import.ty else {
                        return Err(Error::Unsupported(format!(
                            "import {}.{} is not a function",
                            import.module, import.name
                        )));
                    };
                    function_types.push(ty);
                    imports.push((import.module.to_owned(), import.name.to_owned()));
                }
            }
            Payload::FunctionSection(section) => {
                function_types.extend(section.into_iter().collect::<Result<Vec<_>, _>>()?);
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
                    table_max = table.ty.maximum;
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
                    if !is_supported_value(global.ty.content_type) || global.ty.shared {
                        return Err(Error::Unsupported(
                            "only unshared numeric and funcref globals are supported".into(),
                        ));
                    }
                    globals_template.push(eval_const(&global.init_expr, &globals_template)?);
                    globals_mutable.push(global.ty.mutable);
                    if global.ty.content_type == ValType::Ref(RefType::FUNCREF) {
                        reference_globals.push(globals_template.len() - 1);
                    }
                }
            }
            Payload::ElementSection(section) => {
                for element in section {
                    let element = element?;
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
                    let length = match element.kind {
                        ElementKind::Active {
                            table_index: None | Some(0),
                            offset_expr,
                        } => {
                            let offset =
                                eval_const(&offset_expr, &globals_template)? as u32 as usize;
                            let table = table_template.as_mut().ok_or_else(|| {
                                Error::Unsupported("elements without a table".into())
                            })?;
                            let end = offset.checked_add(values.len()).ok_or_else(|| {
                                Error::Unsupported("element segment offset overflow".into())
                            })?;
                            if end > table.len() {
                                return Err(Error::Unsupported(
                                    "element segment exceeds initial table".into(),
                                ));
                            }
                            table[offset..end].copy_from_slice(&values);
                            0
                        }
                        ElementKind::Passive => values.len() as u32,
                        ElementKind::Declared => 0,
                        _ => return Err(Error::Unsupported("only table 0 is supported".into())),
                    };
                    table_info.lengths.push(length);
                    table_info.elements.push(values);
                }
            }
            Payload::DataSection(section) => {
                for data in section {
                    let data = data?;
                    match data.kind {
                        DataKind::Active {
                            memory_index: 0,
                            offset_expr,
                        } => {
                            let offset =
                                eval_const(&offset_expr, &globals_template)? as u32 as usize;
                            let memory = memory_template
                                .as_mut()
                                .ok_or_else(|| Error::Unsupported("data without memory".into()))?;
                            let end = offset.checked_add(data.data.len()).ok_or_else(|| {
                                Error::Unsupported("data segment offset overflow".into())
                            })?;
                            if end > memory.len() {
                                return Err(Error::Unsupported(
                                    "data segment exceeds initial memory".into(),
                                ));
                            }
                            memory[offset..end].copy_from_slice(data.data);
                            data_offsets.push(0);
                            data_lengths.push(0);
                        }
                        DataKind::Passive => {
                            let length = u32::try_from(data.data.len()).map_err(|_| {
                                Error::Unsupported("passive data segment is too large".into())
                            })?;
                            data_offsets.push(passive_data.len());
                            data_lengths.push(length);
                            passive_data.extend_from_slice(data.data);
                        }
                        _ => {
                            return Err(Error::Unsupported(
                                "only data for memory 0 is supported".into(),
                            ));
                        }
                    }
                }
            }
            Payload::DataCountSection { .. } => {}
            Payload::StartSection { func, .. } => start_index = Some(func as usize),
            Payload::ExportSection(section) => {
                for item in section {
                    let item = item?;
                    exports.insert(item.name.to_owned(), (item.kind, item.index as usize));
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

    let mut table_grows = false;
    for body in &bodies {
        for op in body.get_operators_reader()? {
            if matches!(op?, Operator::TableGrow { .. }) {
                table_grows = true;
            }
        }
    }
    table_info.capacity = table_template.as_ref().map_or(0, |t| {
        (t.len() + if table_grows { MAX_TABLE_GROWTH } else { 0 })
            .min(table_max.unwrap_or(u64::MAX).min(MAX_TABLE_ELEMENTS as u64) as usize)
    });
    Ok(ParsedModule {
        types,
        function_types,
        bodies,
        exports,
        imports,
        start_index,
        globals_mutable,
        reference_globals,
        memory_max_pages,
        data_offsets,
        resolved_functions: Vec::new(),
        resolved_imports: Vec::new(),
        table_targets: Vec::new(),
        instance: InstanceTemplate {
            memory_template,
            memory_capacity: None,
            globals_template,
            table_template,
            table_info,
            passive_data,
            data_lengths,
        },
    })
}

fn load_dependencies<'a>(
    index: usize,
    modules: &mut Vec<ParsedModule<'a>>,
    providers: &HashMap<&str, &'a [u8]>,
    namespaces: &mut HashMap<String, usize>,
    visiting: &mut Vec<bool>,
    postorder: &mut Vec<usize>,
) -> Result<(), Error> {
    visiting[index] = true;
    for (namespace, name) in modules[index].imports.clone() {
        let provider = if let Some(&provider) = namespaces.get(&namespace) {
            if visiting[provider] {
                return Err(Error::Unsupported(format!(
                    "cyclic provider dependency through {namespace:?}"
                )));
            }
            provider
        } else {
            let bytes = providers.get(namespace.as_str()).ok_or_else(|| {
                Error::Unsupported(format!("missing Wasm provider {namespace:?}"))
            })?;
            let provider = modules.len();
            modules.push(parse_module(bytes)?);
            namespaces.insert(namespace.clone(), provider);
            visiting.push(false);
            load_dependencies(
                provider, modules, providers, namespaces, visiting, postorder,
            )?;
            provider
        };
        let &(kind, function) = modules[provider].exports.get(&name).ok_or_else(|| {
            Error::Unsupported(format!("provider {namespace:?} has no export {name:?}"))
        })?;
        if kind != ExternalKind::Func {
            return Err(Error::Unsupported(format!(
                "provider export {namespace}.{name} is not a function"
            )));
        }
        let import_index = modules[index].resolved_imports.len();
        if modules[index].function_type(import_index)?
            != modules[provider].function_type(function)?
        {
            return Err(Error::Unsupported(format!(
                "signature mismatch for import {namespace}.{name}"
            )));
        }
        modules[index].resolved_imports.push((provider, function));
    }
    visiting[index] = false;
    postorder.push(index);
    Ok(())
}

fn remap_reference(value: u32, functions: &[usize]) -> Result<u32, Error> {
    if value == 0 {
        return Ok(0);
    }
    let index = *functions
        .get(value as usize - 1)
        .ok_or_else(|| Error::Unsupported("function reference is out of range".into()))?;
    u32::try_from(index)
        .ok()
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| Error::Unsupported("too many linked functions".into()))
}

fn translate(
    wasm: &[u8],
    export_name: &str,
    providers: &[(&str, &[u8])],
) -> Result<Translation, Error> {
    let mut provider_map = HashMap::new();
    for &(namespace, bytes) in providers {
        if provider_map.insert(namespace, bytes).is_some() {
            return Err(Error::Unsupported(format!(
                "duplicate Wasm provider {namespace:?}"
            )));
        }
    }
    let mut modules = vec![parse_module(wasm)?];
    let mut postorder = Vec::new();
    load_dependencies(
        0,
        &mut modules,
        &provider_map,
        &mut HashMap::new(),
        &mut vec![false],
        &mut postorder,
    )?;
    let mut functions = Vec::new();
    let mut global_types = Vec::new();
    for (module_index, module) in modules.iter_mut().enumerate() {
        module.resolved_functions = vec![usize::MAX; module.function_types.len()];
        for local in module.imports.len()..module.function_types.len() {
            module.resolved_functions[local] = functions.len();
            functions.push((module_index, local));
            global_types.push(module.function_type(local)?.clone());
        }
    }
    for &module_index in &postorder {
        for (local, (provider, exported)) in modules[module_index]
            .resolved_imports
            .clone()
            .into_iter()
            .enumerate()
        {
            modules[module_index].resolved_functions[local] =
                modules[provider].resolved_functions[exported];
        }
    }
    let &(kind, local_export) = modules[0]
        .exports
        .get(export_name)
        .ok_or_else(|| Error::Unsupported(format!("export {export_name:?} is not a function")))?;
    if kind != ExternalKind::Func {
        return Err(Error::Unsupported(format!(
            "export {export_name:?} is not a function"
        )));
    }
    let index = modules[0].resolved_functions[local_export];
    let ty = &global_types[index];
    if ty
        .params()
        .iter()
        .chain(ty.results())
        .any(|ty| !is_numeric(*ty))
    {
        return Err(Error::Unsupported(
            "exported signatures must contain numeric values".into(),
        ));
    }
    // Mutable tables can receive references returned by functions in other modules.
    let mut referenced = Vec::new();
    for module in &mut modules {
        let instance = &mut module.instance;
        if let Some(table) = &mut instance.table_template {
            for value in table {
                *value = remap_reference(*value, &module.resolved_functions)?;
                if *value != 0 {
                    referenced.push((*value - 1) as usize);
                }
            }
        }
        for segment in &mut instance.table_info.elements {
            for value in segment {
                *value = remap_reference(*value, &module.resolved_functions)?;
                if *value != 0 {
                    referenced.push((*value - 1) as usize);
                }
            }
        }
        for &global in &module.reference_globals {
            let value = remap_reference(
                instance.globals_template[global] as u32,
                &module.resolved_functions,
            )?;
            instance.globals_template[global] = u64::from(value);
            if value != 0 {
                referenced.push((value - 1) as usize);
            }
        }
        for body in &module.bodies {
            for op in body.get_operators_reader()? {
                if let Operator::RefFunc { function_index } = op? {
                    referenced.push(module.resolved_functions[function_index as usize]);
                }
            }
        }
    }
    referenced.sort_unstable();
    referenced.dedup();
    let starts: Vec<_> = postorder
        .iter()
        .filter_map(|&module| {
            modules[module]
                .start_index
                .map(|local| modules[module].resolved_functions[local])
        })
        .collect();
    for module in &mut modules {
        module.instance.table_info.mutable = false;
        module.table_targets = module
            .instance
            .table_template
            .iter()
            .flatten()
            .copied()
            .filter(|&v| v != 0)
            .map(|v| (v - 1) as usize)
            .collect();
        module.table_targets.sort_unstable();
        module.table_targets.dedup();
    }
    // A newly reachable table mutation can expose more indirect call targets.
    // Repeat reachability until those targets no longer change.
    loop {
        let mut reachable = vec![0u8; functions.len()];
        let mut ignored_order = Vec::new();
        let mut ignored_traps = false;
        let mut ignored_growth = vec![false; modules.len()];
        let mut ignored_recursion = false;
        for function in std::iter::once(&index).chain(&starts) {
            visit_function(
                *function,
                &modules,
                &functions,
                &global_types,
                &mut reachable,
                &mut ignored_order,
                &mut ignored_traps,
                &mut ignored_growth,
                &mut ignored_recursion,
            )?;
        }
        let mut mutates = vec![false; modules.len()];
        let mut grows = vec![false; modules.len()];
        for (function, &state) in reachable.iter().enumerate() {
            if state != 2 {
                continue;
            }
            let (module_index, local) = functions[function];
            let module = &modules[module_index];
            for op in module.bodies[local - module.imports.len()].get_operators_reader()? {
                match op? {
                    Operator::TableSet { .. }
                    | Operator::TableFill { .. }
                    | Operator::TableCopy { .. }
                    | Operator::TableInit { .. } => mutates[module_index] = true,
                    Operator::TableGrow { .. } => {
                        mutates[module_index] = true;
                        grows[module_index] = true;
                    }
                    _ => {}
                }
            }
        }
        let mut changed = false;
        for (module_index, module) in modules.iter_mut().enumerate() {
            if mutates[module_index] && !module.instance.table_info.mutable {
                module.instance.table_info.mutable = true;
                module.table_targets = referenced.clone();
                changed = true;
            }
        }
        if !changed {
            for (module_index, module) in modules.iter_mut().enumerate() {
                if !grows[module_index] {
                    module.instance.table_info.capacity =
                        module.instance.table_template.as_ref().map_or(0, Vec::len);
                }
            }
            break;
        }
    }
    let mut recursive = false;
    let mut order = Vec::new();
    let mut visit_state = vec![0u8; functions.len()];
    let mut can_trap = modules.iter().any(|m| m.instance.memory_template.is_some());
    let mut uses_growth = vec![false; modules.len()];
    visit_function(
        index,
        &modules,
        &functions,
        &global_types,
        &mut visit_state,
        &mut order,
        &mut can_trap,
        &mut uses_growth,
        &mut recursive,
    )?;
    for &start in &starts {
        visit_function(
            start,
            &modules,
            &functions,
            &global_types,
            &mut visit_state,
            &mut order,
            &mut can_trap,
            &mut uses_growth,
            &mut recursive,
        )?;
    }
    can_trap |= recursive;
    let mut slots = Vec::new();
    let mut next_slot = 4u64;
    for (module_index, module) in modules.iter_mut().enumerate() {
        let instance = &mut module.instance;
        instance.memory_capacity = instance.memory_template.as_ref().map(|template| {
            let initial_pages = template.len() / 65536;
            let wanted = initial_pages
                + if uses_growth[module_index] {
                    MAX_GROWTH_PAGES
                } else {
                    0
                };
            wanted
                .min(
                    module
                        .memory_max_pages
                        .unwrap_or(u64::MAX)
                        .min(usize::MAX as u64) as usize,
                )
                .min(MAX_MEMORY_BYTES_PER_INSTANCE / 65536)
                * 65536
        });
        let mut allocate = |needed: bool| {
            needed.then(|| {
                let slot = next_slot;
                next_slot += 1;
                slot
            })
        };
        slots.push(BufferSlots {
            memory: allocate(instance.memory_template.is_some()),
            pages: allocate(instance.memory_template.is_some()),
            globals: allocate(!instance.globals_template.is_empty()),
            table: allocate(instance.table_template.is_some()),
            passive_data: allocate(!instance.data_lengths.is_empty()),
        });
    }
    if next_slot > 31 {
        return Err(Error::Unsupported(
            "linked module state exceeds Metal's 31 buffer slots".into(),
        ));
    }
    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(include_str!("softfloat64.metal"));
    source.push_str(include_str!("wasm_float.metal"));
    source.push_str("struct ModuleState { device uchar* memory; device uint* pages; device ulong* globals; device uint* table; device const uchar* passive_data; uint table_size; thread uint* element_lengths; thread uint* data_lengths; };\n");
    for (module_index, module) in modules.iter().enumerate() {
        for (element, values) in module.instance.table_info.elements.iter().enumerate() {
            source.push_str(&format!(
                "constant uint m{module_index}_element_{element}[{}] = {{{}}};\n",
                values.len().max(1),
                if values.is_empty() {
                    "0u".into()
                } else {
                    values
                        .iter()
                        .map(|v| format!("{v}u"))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            ));
        }
    }
    for &function in &order {
        source.push_str(&function_signature(
            function,
            &global_types[function],
            recursive,
        )?);
        source.push_str(";\n");
    }
    for &function in &order {
        let (module_index, local) = functions[function];
        source.push_str(&translate_body(
            &modules[module_index],
            module_index,
            function,
            local,
            &global_types,
            recursive,
        )?);
    }
    let fast_i32 = ty.params() == [ValType::I32] && ty.results() == [ValType::I32];
    let input_ty = if fast_i32 { "uint" } else { "ulong" };
    source.push_str(&format!("kernel void run_wasm(device const {input_ty}* input [[buffer(0)]], device {input_ty}* output [[buffer(1)]], constant uint& count [[buffer(2)]]"));
    if can_trap {
        source.push_str(", device uint* traps [[buffer(3)]]");
    }
    for (module_index, slot) in slots.iter().enumerate() {
        for (slot, ty, field) in [
            (slot.memory, "uchar", "memory"),
            (slot.pages, "uint", "pages"),
            (slot.globals, "ulong", "globals"),
            (slot.table, "uint", "table"),
            (slot.passive_data, "const uchar", "passive_data"),
        ] {
            if let Some(slot) = slot {
                source.push_str(&format!(
                    ", device {ty}* m{module_index}_{field} [[buffer({slot})]]"
                ));
            }
        }
    }
    source.push_str(
        ", uint id [[thread_position_in_grid]]) {\nif (id >= count) return;\nuint trap = 0u;\n",
    );
    source.push_str(&format!("ModuleState states[{}] = {{}};\n", modules.len()));
    for (module_index, module) in modules.iter().enumerate() {
        let instance = &module.instance;
        if let Some(capacity) = instance.memory_capacity {
            source.push_str(&format!("states[{module_index}].memory = m{module_index}_memory + ulong(id) * {}ul;\nstates[{module_index}].pages = m{module_index}_pages + id;\n", capacity.max(1)));
        }
        if !instance.globals_template.is_empty() {
            source.push_str(&format!(
                "states[{module_index}].globals = m{module_index}_globals + ulong(id) * {}ul;\n",
                instance.globals_template.len()
            ));
        }
        if let Some(table) = &instance.table_template {
            let offset = if instance.table_info.mutable {
                format!(" + ulong(id) * {}ul", instance.table_info.capacity.max(1))
            } else {
                String::new()
            };
            source.push_str(&format!("states[{module_index}].table = m{module_index}_table{offset};\nstates[{module_index}].table_size = {}u;\n", table.len()));
        }
        for (field, lengths) in [
            ("element_lengths", &instance.table_info.lengths),
            ("data_lengths", &instance.data_lengths),
        ] {
            if !lengths.is_empty() {
                source.push_str(&format!("uint m{module_index}_{field}[{}] = {{{}}};\nstates[{module_index}].{field} = m{module_index}_{field};\n", lengths.len(), lengths.iter().map(|v| format!("{v}u")).collect::<Vec<_>>().join(", ")));
            }
        }
        if !instance.data_lengths.is_empty() {
            source.push_str(&format!(
                "states[{module_index}].passive_data = m{module_index}_passive_data;\n"
            ));
        }
    }
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
    let context = if recursive {
        "states, trap, 0u"
    } else {
        "states, trap"
    };
    for start in starts {
        source.push_str(&format!("if (trap == 0u) f{start}({context});\n"));
    }
    args.push(context.into());
    source.push_str(&format!("if (trap == 0u) f{index}({});\n", args.join(", ")));
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
        instances: modules.into_iter().map(|m| m.instance).collect(),
        slots,
        recursive,
    })
}

fn is_numeric(ty: ValType) -> bool {
    matches!(
        ty,
        ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64
    )
}

fn is_supported_value(ty: ValType) -> bool {
    is_numeric(ty) || ty == ValType::Ref(RefType::FUNCREF)
}

fn eval_const(expr: &ConstExpr<'_>, globals: &[u64]) -> Result<u64, Error> {
    let mut operators = expr.get_operators_reader().into_iter();
    let value = match operators.next().transpose()? {
        Some(Operator::I32Const { value }) => value as u32 as u64,
        Some(Operator::I64Const { value }) => value as u64,
        Some(Operator::F32Const { value }) => value.bits() as u64,
        Some(Operator::F64Const { value }) => value.bits(),
        Some(Operator::RefNull {
            hty:
                wasmparser::HeapType::Abstract {
                    ty: wasmparser::AbstractHeapType::Func,
                    ..
                },
        }) => 0,
        Some(Operator::RefFunc { function_index }) => u64::from(function_index) + 1,
        Some(Operator::GlobalGet { global_index }) => *globals
            .get(global_index as usize)
            .ok_or_else(|| Error::Unsupported("constant global index is out of range".into()))?,
        _ => return Err(Error::Unsupported("unsupported scalar initializer".into())),
    };
    if !matches!(operators.next().transpose()?, Some(Operator::End)) || operators.next().is_some() {
        return Err(Error::Unsupported("unsupported scalar initializer".into()));
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
    modules: &[ParsedModule<'_>],
    functions: &[(usize, usize)],
    global_types: &[FuncType],
    state: &mut [u8],
    order: &mut Vec<usize>,
    can_trap: &mut bool,
    uses_growth: &mut [bool],
    recursive: &mut bool,
) -> Result<(), Error> {
    match state.get(index).copied() {
        Some(2) => return Ok(()),
        Some(1) => {
            *recursive = true;
            return Ok(());
        }
        Some(0) => {}
        _ => return Err(Error::Unsupported("function index is out of range".into())),
    }
    state[index] = 1;
    let (module_index, local) = functions[index];
    let module = &modules[module_index];
    let mut previous_constant = None;
    for operator in module.bodies[local - module.imports.len()].get_operators_reader()? {
        let operator = operator?;
        let current_constant = match &operator {
            Operator::I32Const { value } => Some(*value as u32),
            _ => None,
        };
        match operator {
            Operator::Call { function_index } => {
                visit_function(
                    module.resolved_functions[function_index as usize],
                    modules,
                    functions,
                    global_types,
                    state,
                    order,
                    can_trap,
                    uses_growth,
                    recursive,
                )?;
            }
            Operator::CallIndirect { type_index, .. } => {
                *can_trap = true;
                let expected = &module.types[type_index as usize];
                for &target in &module.table_targets {
                    if !module.instance.table_info.mutable {
                        if let Some(slot) = previous_constant {
                            if module
                                .instance
                                .table_template
                                .as_ref()
                                .and_then(|t| t.get(slot as usize))
                                .copied()
                                != Some(target as u32 + 1)
                            {
                                continue;
                            }
                        }
                    }
                    if &global_types[target] == expected {
                        visit_function(
                            target,
                            modules,
                            functions,
                            global_types,
                            state,
                            order,
                            can_trap,
                            uses_growth,
                            recursive,
                        )?;
                    }
                }
            }
            Operator::TableGet { .. }
            | Operator::TableSet { .. }
            | Operator::TableFill { .. }
            | Operator::TableCopy { .. }
            | Operator::TableInit { .. } => *can_trap = true,
            Operator::MemoryGrow { .. } => uses_growth[module_index] = true,
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

fn function_signature(index: usize, ty: &FuncType, recursive: bool) -> Result<String, Error> {
    if ty
        .params()
        .iter()
        .chain(ty.results())
        .any(|ty| !is_supported_value(*ty))
    {
        return Err(Error::Unsupported(
            "only numeric and funcref function signatures are supported".into(),
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
    args.push("thread ModuleState* states".into());
    args.push("thread uint& trap".into());
    if recursive {
        args.push("uint depth".into());
    }
    Ok(format!("void f{index}({})", args.join(", ")))
}

fn translate_body(
    module: &ParsedModule<'_>,
    module_index: usize,
    function_index: usize,
    local_function: usize,
    global_types: &[FuncType],
    recursive: bool,
) -> Result<String, Error> {
    let body = &module.bodies[local_function - module.imports.len()];
    let ty = module.function_type(local_function)?;
    let types = module.types.as_slice();
    let function_types = module.function_types.as_slice();
    let memory_bytes = module.instance.memory_template.as_ref().map(Vec::len);
    let memory_capacity = module.instance.memory_capacity;
    let globals_mutable = module.globals_mutable.as_slice();
    let table_values = module.instance.table_template.as_deref();
    let table_info = &module.instance.table_info;
    let table_targets = module.table_targets.as_slice();
    let data_offsets = module.data_offsets.as_slice();
    let mut source = function_signature(function_index, ty, recursive)?;
    source.push_str(" {\n");
    if recursive {
        source.push_str(&format!(
            "if (depth >= {MAX_CALL_DEPTH}u) {{ trap = 8u; return; }}\n"
        ));
    }
    for (exists, ty, field) in [
        (memory_bytes.is_some(), "device uchar*", "memory"),
        (memory_bytes.is_some(), "device uint*", "pages"),
        (!globals_mutable.is_empty(), "device ulong*", "globals"),
        (
            table_values.is_some(),
            if table_info.mutable {
                "device uint*"
            } else {
                "device const uint*"
            },
            "table",
        ),
        (
            !table_info.lengths.is_empty(),
            "thread uint*",
            "element_lengths",
        ),
        (
            !data_offsets.is_empty(),
            "device const uchar*",
            "passive_data",
        ),
        (!data_offsets.is_empty(), "thread uint*", "data_lengths"),
        (table_info.mutable, "thread uint&", "table_size"),
    ] {
        if exists {
            source.push_str(&format!("{ty} {field} = states[{module_index}].{field};\n"));
        }
    }
    for index in 0..ty.params().len() {
        source.push_str(&format!("ulong l{index} = a{index};\n"));
    }
    let mut local_count = ty.params().len();
    for local in body.get_locals_reader()? {
        let (count, ty) = local?;
        if !is_supported_value(ty) {
            return Err(Error::Unsupported(
                "only numeric and funcref locals are supported".into(),
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
        module_index,
        resolved_functions: &module.resolved_functions,
        global_types,
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
        table_info,
        table_targets,
        data_offsets,
        recursive,
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
    module_index: usize,
    resolved_functions: &'a [usize],
    global_types: &'a [FuncType],
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
    table_info: &'a TableInfo,
    table_targets: &'a [usize],
    data_offsets: &'a [usize],
    recursive: bool,
    frames: Vec<Frame>,
}

impl Compiler<'_> {
    fn table_size(&self) -> Result<String, Error> {
        let initial = self
            .table_len
            .ok_or_else(|| Error::Unsupported("table is missing".into()))?;
        Ok(if self.table_info.mutable {
            "ulong(table_size)".into()
        } else {
            format!("{initial}ul")
        })
    }

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
        self.source.push_str(&format!("{value} = 0ul;\n"));
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
            BlockType::Type(ty) if is_supported_value(ty) => (0, 1),
            BlockType::FuncType(index) => {
                let ty = self
                    .types
                    .get(index as usize)
                    .ok_or_else(|| Error::Unsupported("block type index is out of range".into()))?;
                if ty
                    .params()
                    .iter()
                    .chain(ty.results())
                    .any(|ty| !is_supported_value(*ty))
                {
                    return Err(Error::Unsupported(
                        "only numeric and funcref block signatures are supported".into(),
                    ));
                }
                (ty.params().len(), ty.results().len())
            }
            _ => {
                return Err(Error::Unsupported(
                    "only numeric and funcref block signatures are supported".into(),
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
            Operator::RefNull {
                hty:
                    wasmparser::HeapType::Abstract {
                        ty: wasmparser::AbstractHeapType::Func,
                        ..
                    },
            } => {
                stack.push("0ul".into());
            }
            Operator::RefFunc { function_index } => {
                stack.push(format!(
                    "{}ul",
                    self.resolved_functions[*function_index as usize] + 1
                ));
            }
            Operator::RefIsNull => {
                let value = pop(stack)?;
                self.emit_value(stack, format!("uint({value} == 0ul)"));
            }
            Operator::TableGet { table: 0 } => {
                let index = pop(stack)?;
                let size = self.table_size()?;
                self.source.push_str(&format!(
                    "if ({index} >= {size}) {{ trap = 6u; return; }}\n"
                ));
                self.emit_value(stack, format!("table[{index}]"));
            }
            Operator::TableSet { table: 0 } => {
                let value = pop(stack)?;
                let index = pop(stack)?;
                let size = self.table_size()?;
                self.source.push_str(&format!("if ({index} >= {size}) {{ trap = 6u; return; }}\ntable[{index}] = uint({value});\n"));
            }
            Operator::TableSize { table: 0 } => {
                let size = self.table_size()?;
                self.emit_value(stack, size);
            }
            Operator::TableGrow { table: 0 } => {
                let count = pop(stack)?;
                let value = pop(stack)?;
                self.table_size()?;
                let result = self.fresh_value();
                let slot = self.fresh_value();
                self.source.push_str(&format!(
                    "{result} = 0xfffffffful;\nif (ulong(table_size) + {count} <= {}ul) {{\n{result} = table_size;\nfor ({slot} = table_size; {slot} < ulong(table_size) + {count}; ++{slot}) table[{slot}] = uint({value});\ntable_size += uint({count});\n}}\n", self.table_info.capacity));
                stack.push(result);
            }
            Operator::TableFill { table: 0 } => {
                let count = pop(stack)?;
                let value = pop(stack)?;
                let destination = pop(stack)?;
                let size = self.table_size()?;
                let slot = self.fresh_value();
                self.source.push_str(&format!(
                    "if ({destination} + {count} > {size}) {{ trap = 6u; return; }}\nfor ({slot} = 0ul; {slot} < {count}; ++{slot}) table[{destination} + {slot}] = uint({value});\n"));
            }
            Operator::TableCopy {
                dst_table: 0,
                src_table: 0,
            } => {
                let count = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let size = self.table_size()?;
                let slot = self.fresh_value();
                self.source.push_str(&format!(
                    "if ({destination} + {count} > {size} || {source} + {count} > {size}) {{ trap = 6u; return; }}\nif ({destination} > {source}) {{ for ({slot} = {count}; {slot} != 0ul; --{slot}) table[{destination} + {slot} - 1ul] = table[{source} + {slot} - 1ul]; }} else {{ for ({slot} = 0ul; {slot} < {count}; ++{slot}) table[{destination} + {slot}] = table[{source} + {slot}]; }}\n"));
            }
            Operator::TableInit {
                elem_index,
                table: 0,
            } => {
                self.table_info
                    .elements
                    .get(*elem_index as usize)
                    .ok_or_else(|| Error::Unsupported("element index is out of range".into()))?;
                let count = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let size = self.table_size()?;
                let slot = self.fresh_value();
                self.source.push_str(&format!(
                    "if ({destination} + {count} > {size} || {source} + {count} > ulong(element_lengths[{elem_index}u])) {{ trap = 6u; return; }}\nfor ({slot} = 0ul; {slot} < {count}; ++{slot}) table[{destination} + {slot}] = m{}_element_{elem_index}[{source} + {slot}];\n", self.module_index));
            }
            Operator::ElemDrop { elem_index } => {
                self.table_info
                    .elements
                    .get(*elem_index as usize)
                    .ok_or_else(|| Error::Unsupported("element index is out of range".into()))?;
                self.source
                    .push_str(&format!("element_lengths[{elem_index}u] = 0u;\n"));
            }
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
            Operator::MemoryFill { mem: 0 } => {
                self.memory_bytes
                    .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
                let length = pop(stack)?;
                let value = pop(stack)?;
                let destination = pop(stack)?;
                let offset = self.fresh_value();
                self.source.push_str(&format!(
                    "if (ulong({destination}) + ulong({length}) > ulong(pages[0]) * 65536ul) {{ trap = 4u; return; }}\nfor ({offset} = 0ul; {offset} < ulong({length}); ++{offset}) memory[ulong({destination}) + {offset}] = uchar({value});\n"
                ));
            }
            Operator::MemoryCopy {
                dst_mem: 0,
                src_mem: 0,
            } => {
                self.memory_bytes
                    .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
                let length = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let offset = self.fresh_value();
                self.source.push_str(&format!(
                    "if (ulong({destination}) + ulong({length}) > ulong(pages[0]) * 65536ul || ulong({source}) + ulong({length}) > ulong(pages[0]) * 65536ul) {{ trap = 4u; return; }}\nif ({destination} <= {source}) {{\nfor ({offset} = 0ul; {offset} < ulong({length}); ++{offset}) memory[ulong({destination}) + {offset}] = memory[ulong({source}) + {offset}];\n}} else {{\nfor ({offset} = ulong({length}); {offset} > 0ul; --{offset}) memory[ulong({destination}) + {offset} - 1ul] = memory[ulong({source}) + {offset} - 1ul];\n}}\n"
                ));
            }
            Operator::MemoryInit { data_index, mem: 0 } => {
                self.memory_bytes
                    .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
                let offset = *self.data_offsets.get(*data_index as usize).ok_or_else(|| {
                    Error::Unsupported("data segment index is out of range".into())
                })?;
                let length = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let byte = self.fresh_value();
                self.source.push_str(&format!(
                    "if (ulong({destination}) + ulong({length}) > ulong(pages[0]) * 65536ul || ulong({source}) + ulong({length}) > ulong(data_lengths[{data_index}u])) {{ trap = 4u; return; }}\nfor ({byte} = 0ul; {byte} < ulong({length}); ++{byte}) memory[ulong({destination}) + {byte}] = passive_data[{offset}ul + ulong({source}) + {byte}];\n"
                ));
            }
            Operator::DataDrop { data_index } => {
                self.data_offsets.get(*data_index as usize).ok_or_else(|| {
                    Error::Unsupported("data segment index is out of range".into())
                })?;
                self.source
                    .push_str(&format!("data_lengths[{data_index}u] = 0u;\n"));
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
                    .any(|ty| !is_supported_value(*ty))
                {
                    return Err(Error::Unsupported(
                        "only numeric and funcref function signatures are supported".into(),
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
                call_args.push("states".into());
                call_args.push("trap".into());
                if self.recursive {
                    call_args.push("depth + 1u".into());
                }
                let index = self.resolved_functions[index];
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
                    .any(|ty| !is_supported_value(*ty))
                {
                    return Err(Error::Unsupported(
                        "only numeric and funcref indirect signatures are supported".into(),
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
                let table_len = if self.table_info.mutable {
                    "table_size".into()
                } else {
                    format!("{table_len}ul")
                };
                self.source.push_str(&format!(
                    "if ({element} >= {table_len}) {{ trap = 6u; return; }}\n"
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
                    if !self.table_info.mutable
                        && constant_slot.is_some()
                        && constant_target != Some(index as u32 + 1)
                    {
                        continue;
                    }
                    let target_ty = &self.global_types[index];
                    if target_ty != ty {
                        continue;
                    }
                    let mut call_args = args.clone();
                    call_args.extend(results.iter().cloned());
                    call_args.push("states".into());
                    call_args.push("trap".into());
                    if self.recursive {
                        call_args.push("depth + 1u".into());
                    }
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
                ty:
                    ValType::I32
                    | ValType::I64
                    | ValType::F32
                    | ValType::F64
                    | ValType::Ref(RefType::FUNCREF),
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

    fn linked_matches_wasmtime(root: &str, providers: &[(&str, &str)], inputs: &[u32]) {
        let wasm = wat::parse_str(root).unwrap();
        let providers: Vec<_> = providers
            .iter()
            .map(|(name, source)| (*name, wat::parse_str(source).unwrap()))
            .collect();
        // Registration order must not determine dependency initialization order.
        let registrations: Vec<_> = providers
            .iter()
            .rev()
            .map(|(name, bytes)| (*name, bytes.as_slice()))
            .collect();
        let kernel = Kernel::new_with_imports(&wasm, "f", &registrations).unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let providers: Vec<_> = providers
            .iter()
            .map(|(name, bytes)| (*name, wasmtime::Module::new(&engine, bytes).unwrap()))
            .collect();
        let expected: Vec<_> = inputs
            .iter()
            .map(|&input| {
                let mut store = wasmtime::Store::new(&engine, ());
                let mut linker = wasmtime::Linker::new(&engine);
                for (name, module) in &providers {
                    let instance = linker.instantiate(&mut store, module).unwrap();
                    linker.instance(&mut store, name, instance).unwrap();
                }
                let instance = linker.instantiate(&mut store, &module).unwrap();
                instance
                    .get_typed_func::<i32, i32>(&mut store, "f")
                    .unwrap()
                    .call(&mut store, input as i32)
                    .unwrap() as u32
            })
            .collect();
        assert_eq!(kernel.run(inputs).unwrap(), expected);
        assert_eq!(kernel.run(inputs).unwrap(), expected);
    }

    #[test]
    fn wasm_imports_own_state_and_share_provider_aliases_per_lane() {
        let provider = r#"(module
            (type $t (func (param i32) (result i32)))
            (memory 1) (table 1 funcref)
            (global $g (mut i32) (i32.const 2))
            (data $initial "\09\00\00\00")
            (func $compute (type $t)
                local.get 0 global.get $g i32.add i32.const 0 i32.load i32.add)
            (elem $functions func $compute)
            (func $start
                i32.const 0 i32.const 0 i32.const 4 memory.init $initial data.drop $initial
                i32.const 5 global.set $g
                i32.const 0 i32.const 0 i32.const 1 table.init $functions elem.drop $functions)
            (start $start)
            (func (export "add") (type $t)
                global.get $g i32.const 1 i32.add global.set $g
                i32.const 0 local.get 0 i32.store
                local.get 0 i32.const 0 call_indirect (type $t))
            (func (export "read") (result i32)
                i32.const 0 i32.load global.get $g i32.add))"#;
        let root = r#"(module
            (import "p" "add" (func $add (param i32) (result i32)))
            (import "p" "add" (func $alias (param i32) (result i32)))
            (import "p" "read" (func $read (result i32)))
            (import "q" "read" (func $other (result i32)))
            (memory 1) (data (i32.const 0) "\64\00\00\00")
            (global $g (mut i32) (i32.const 3))
            (func (export "f") (param i32) (result i32)
                local.get 0 call $add drop
                local.get 0 i32.const 1 i32.add call $alias drop
                call $read call $other i32.add
                i32.const 0 i32.load i32.add global.get $g i32.add))"#;
        linked_matches_wasmtime(
            root,
            &[("p", provider), ("q", provider)],
            &[0, 1, 7, 15, u32::MAX],
        );
    }

    #[test]
    fn wasm_imports_preserve_references_and_module_type_indices() {
        let provider = r#"(module
            (type $result (func (result i32)))
            (type $target (func (param i32) (result i32)))
            (global $g (mut i32) (i32.const 0))
            (func $start i32.const 23 global.set $g) (start $start)
            (func $transform (export "transform") (type $target)
                block (type $result) local.get 0 global.get $g i32.add end)
            (elem declare func $transform)
            (func (export "reference") (result funcref) ref.func $transform))"#;
        let root = r#"(module
            (type $unused (func (param f64) (result f64)))
            (type $target (func (param i32) (result i32)))
            (import "p" "transform" (func $transform (type $target)))
            (import "p" "reference" (func $reference (result funcref)))
            (table 2 funcref)
            (global $saved funcref (ref.func $transform))
            (elem (i32.const 0) $transform)
            (func (export "f") (param i32) (result i32)
                i32.const 1 call $reference table.set
                local.get 0 i32.const 1 call_indirect (type $target)
                local.get 0 i32.const 0 call_indirect (type $target) i32.add
                i32.const 0 global.get $saved table.set
                local.get 0 i32.const 0 call_indirect (type $target) i32.add
                i32.const 1 ref.func $transform table.set
                local.get 0 call $transform i32.add))"#;
        linked_matches_wasmtime(root, &[("p", provider)], &[0, 1, 17, u32::MAX]);
    }

    #[test]
    fn wasm_imports_initialize_dependencies_and_resolve_reexports() {
        let base = r#"(module
            (global $g (mut i32) (i32.const 0))
            (func $start i32.const 11 global.set $g) (start $start)
            (func (export "bump") (result i32)
                global.get $g i32.const 1 i32.add global.set $g global.get $g))"#;
        let middle = r#"(module
            (import "base" "bump" (func $bump (result i32)))
            (export "again" (func $bump))
            (global $g (mut i32) (i32.const 0))
            (func $start call $bump global.set $g) (start $start)
            (func (export "read") (result i32)
                global.get $g i32.const 100 i32.mul call $bump i32.add))"#;
        let root = r#"(module
            (import "mid" "read" (func $read (result i32)))
            (import "mid" "again" (func $again (result i32)))
            (global $g (mut i32) (i32.const 0))
            (func $start call $read global.set $g) (start $start)
            (func (export "f") (param i32) (result i32)
                global.get $g call $again i32.add call $read i32.add local.get 0 i32.add))"#;
        linked_matches_wasmtime(root, &[("base", base), ("mid", middle)], &[0, 1, 42]);
        linked_matches_wasmtime(
            r#"(module (import "p" "f" (func $f (param i32) (result i32))) (export "f" (func $f)))"#,
            &[(
                "p",
                r#"(module (func (export "f") (param i32) (result i32) local.get 0 i32.const 7 i32.mul))"#,
            )],
            &[0, 1, 42],
        );
    }

    #[test]
    fn wasm_imports_reject_missing_mismatched_and_cyclic_providers() {
        let root = wat::parse_str(
            r#"(module
            (import "p" "f" (func $f (param i32) (result i32)))
            (export "f" (func $f)))"#,
        )
        .unwrap();
        let bad = |result: Result<Kernel, Error>, expected: &str| {
            assert!(
                matches!(result, Err(Error::Unsupported(ref reason)) if reason.contains(expected)),
                "expected {expected}"
            );
        };
        bad(Kernel::new(&root, "f"), "missing Wasm provider");
        for (provider, expected) in [
            (
                r#"(module (func (export "f") (param i64) (result i32) i32.const 0))"#,
                "signature mismatch",
            ),
            (r#"(module (memory (export "f") 1))"#, "not a function"),
            (r#"(module)"#, "has no export"),
            (
                r#"(module (import "p" "f" (func $f (param i32) (result i32))) (export "f" (func $f)))"#,
                "cyclic provider",
            ),
        ] {
            let provider = wat::parse_str(provider).unwrap();
            bad(
                Kernel::new_with_imports(&root, "f", &[("p", &provider)]),
                expected,
            );
        }
        let provider =
            wat::parse_str(r#"(module (func (export "f") (param i32) (result i32) local.get 0))"#)
                .unwrap();
        bad(
            Kernel::new_with_imports(&root, "f", &[("p", &provider), ("p", &provider)]),
            "duplicate Wasm provider",
        );
        let resource_import =
            wat::parse_str(r#"(module (import "p" "m" (memory 1)) (func (export "f")))"#).unwrap();
        bad(
            Kernel::new_with_imports(&resource_import, "f", &[("p", &provider)]),
            "not a function",
        );
    }

    #[test]
    fn wasm_imports_link_general_signatures_and_runtime_traps() {
        let provider = wat::parse_str(
            r#"(module
            (func (export "mixed") (param i64 f64) (result i64 f64)
                local.get 0 i64.const 7 i64.add
                local.get 1 f64.const 2 f64.mul))"#,
        )
        .unwrap();
        let root = wat::parse_str(
            r#"(module
            (import "p" "mixed" (func $mixed (param i64 f64) (result i64 f64)))
            (func (export "f") (param i64 f64) (result i64 f64)
                local.get 0 local.get 1 call $mixed))"#,
        )
        .unwrap();
        let kernel = Kernel::new_with_imports(&root, "f", &[("p", &provider)]).unwrap();
        assert_eq!(
            kernel
                .run_values(&[
                    vec![Value::I64(5), Value::F64(1.25f64.to_bits())],
                    vec![Value::I64(u64::MAX), Value::F64((-3.0f64).to_bits())],
                ])
                .unwrap(),
            [
                vec![Value::I64(12), Value::F64(2.5f64.to_bits())],
                vec![Value::I64(6), Value::F64((-6.0f64).to_bits())],
            ]
        );
        let provider = wat::parse_str(
            r#"(module
            (func (export "f") (param i32) (result i32) i32.const 100 local.get 0 i32.div_u))"#,
        )
        .unwrap();
        let root = wat::parse_str(
            r#"(module
            (import "p" "f" (func $f (param i32) (result i32))) (export "f" (func $f)))"#,
        )
        .unwrap();
        let kernel = Kernel::new_with_imports(&root, "f", &[("p", &provider)]).unwrap();
        assert!(matches!(
            kernel.run(&[2, 0]),
            Err(Error::Trap {
                input_index: 1,
                reason: "integer divide by zero"
            })
        ));
        let p = wat::parse_str(
            r#"(module
            (import "q" "f" (func $f (param i32) (result i32))) (export "f" (func $f)))"#,
        )
        .unwrap();
        let q = root.clone();
        assert!(
            matches!(Kernel::new_with_imports(&root, "f", &[("p", &p), ("q", &q)]),
            Err(Error::Unsupported(ref reason)) if reason.contains("cyclic provider"))
        );
    }

    #[test]
    fn wasm_imports_propagate_start_traps_and_limit_buffer_slots() {
        let root = wat::parse_str(
            r#"(module
            (import "p" "f" (func $f (param i32) (result i32))) (export "f" (func $f)))"#,
        )
        .unwrap();
        let provider = wat::parse_str(
            r#"(module (func $start unreachable) (start $start)
            (func (export "f") (param i32) (result i32) local.get 0))"#,
        )
        .unwrap();
        let kernel = Kernel::new_with_imports(&root, "f", &[("p", &provider)]).unwrap();
        assert!(matches!(
            kernel.run(&[0]),
            Err(Error::Trap {
                reason: "unreachable executed",
                ..
            })
        ));
        let provider = wat::parse_str(
            r#"(module (global i32 (i32.const 0))
            (func (export "f") (result i32) global.get 0))"#,
        )
        .unwrap();
        let names: Vec<_> = (0..28).map(|index| format!("p{index}")).collect();
        let registrations: Vec<_> = names
            .iter()
            .map(|name| (name.as_str(), provider.as_slice()))
            .collect();
        let imports = names
            .iter()
            .map(|name| format!("(import \"{name}\" \"f\" (func (result i32)))"))
            .collect::<String>();
        let root = wat::parse_str(format!(
            "(module {imports} (func (export \"f\") (result i32) i32.const 0))"
        ))
        .unwrap();
        assert!(
            matches!(Kernel::new_with_imports(&root, "f", &registrations), Err(Error::Unsupported(ref reason)) if reason.contains("31 buffer slots"))
        );
    }

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
    fn start_initializes_each_lane_before_export() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (global $g (mut i32) (i32.const 2))
                (func $start
                    i32.const 0 i32.const 11 i32.store
                    i32.const 7 global.set $g)
                (start $start)
                (func (export "f") (param i32) (result i32)
                    local.get 0 i32.const 0 i32.load i32.add global.get $g i32.add))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 1, 2]).unwrap(), [18, 19, 20]);
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
    fn bulk_memory_fill_and_overlapping_copy_match_wasmtime() {
        let wat = r#"(module
            (memory 1)
            (func (export "f") (param i32) (result i32)
                i32.const 0 local.get 0 i32.const 8 memory.fill
                i32.const 2 i32.const 0 i32.const 6 memory.copy
                i32.const 2 i32.load))"#;
        matches_wasmtime(wat, &[0, 1, 17, 255]);
        let wasm = wat::parse_str(wat).unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[17]).unwrap(), [0x1111_1111]);
    }

    #[test]
    fn memory_copy_preserves_distinct_bytes_in_both_overlap_directions() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (data (i32.const 0) "\01\02\03\04\05\06\07\08")
                (func (export "f") (param i32) (result i32)
                    local.get 0
                    if
                        i32.const 0 i32.const 2 i32.const 6 memory.copy
                    else
                        i32.const 2 i32.const 0 i32.const 6 memory.copy
                    end
                    i32.const 0 i32.load))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 1]).unwrap(), [0x0201_0201, 0x0605_0403]);
    }

    #[test]
    fn memory_fill_checks_zero_length_bounds() {
        let wasm = wat::parse_str(
            r#"(module (memory 1)
                (func (export "f") (param i32) (result i32)
                    i32.const 65536 i32.const 17 local.get 0 memory.fill
                    i32.const 3))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0]).unwrap(), [3]);
        assert!(matches!(
            kernel.run(&[1]),
            Err(Error::Trap {
                reason: "out of bounds memory access",
                ..
            })
        ));
    }

    #[test]
    fn passive_data_is_initialized_and_dropped_per_lane() {
        let wasm = wat::parse_str(
            r#"(module
                (memory 1)
                (data "ABCD")
                (func (export "f") (param i32) (result i32)
                    i32.const 0 i32.const 0 i32.const 4 memory.init 0
                    data.drop 0
                    i32.const 4 i32.const 0 local.get 0 memory.init 0
                    i32.const 0 i32.load))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 0]).unwrap(), [0x4443_4241; 2]);
        assert!(matches!(
            kernel.run(&[0, 1]),
            Err(Error::Trap {
                input_index: 1,
                reason: "out of bounds memory access"
            })
        ));
    }

    #[test]
    fn repeated_loads_in_a_loop_discard_prior_bits() {
        let wasm = wat::parse_str(
            r#"(module
            (memory 1)
            (data (i32.const 0) "\ff\00")
            (func (export "f") (param i32) (result i32) (local i32 i32)
                i32.const 0 local.set 1
                loop
                    local.get 1 i32.load8_u local.set 2
                    local.get 1 i32.const 1 i32.add local.tee 1
                    i32.const 2 i32.ne br_if 0
                end
                local.get 2))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 1]).unwrap(), [0, 0]);
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
    fn mutable_tables_keep_references_and_mutations_per_lane() {
        let wat = r#"(module
            (type $t (func (param i32) (result i32)))
            (table 2 4 funcref)
            (global $r (mut funcref) (ref.null func))
            (func $a (type $t) local.get 0 i32.const 11 i32.add)
            (func $b (type $t) local.get 0 i32.const 7 i32.mul)
            (elem declare func $a $b)
            (func $identity (param funcref) (result funcref) local.get 0)
            (func (export "f") (param i32) (result i32) (local funcref)
                local.get 0 i32.const 1 i32.and
                if (result funcref) ref.func $a else ref.func $b end
                call $identity local.set 1
                local.get 1 global.set $r
                i32.const 0 global.get $r table.set
                i32.const 1 i32.const 0 table.get table.set
                local.get 0 i32.const 1 call_indirect (type $t)))"#;
        matches_wasmtime(wat, &[0, 1, 2, 3, 4, 17, u32::MAX]);
        let kernel = Kernel::new(&wat::parse_str(wat).unwrap(), "f").unwrap();
        assert_eq!(kernel.run(&[1, 2, 3]).unwrap(), [12, 14, 14]);
        assert_eq!(kernel.run(&[1, 2, 3]).unwrap(), [12, 14, 14]);
    }

    #[test]
    fn table_bulk_operations_and_growth_match_wasmtime() {
        let wat = r#"(module
            (type $t (func (result i32)))
            (table 3 6 funcref)
            (func $a (type $t) i32.const 13)
            (func $b (type $t) i32.const 29)
            (elem $passive func $a $b)
            (func (export "f") (param i32) (result i32) (local i32)
                i32.const 0 i32.const 0 i32.const 2 table.init $passive
                elem.drop $passive elem.drop $passive
                i32.const 3 i32.const 0 i32.const 0 table.init $passive
                ref.func $b i32.const 3 table.grow local.set 1
                i32.const 1 i32.const 0 i32.const 2 table.copy
                i32.const 3 ref.func $a i32.const 2 table.fill
                i32.const 4 i32.const 0 i32.const 2 table.copy
                local.get 0 call_indirect (type $t)
                table.size i32.add local.get 1 i32.add
                ref.null func i32.const 1 table.grow i32.add))"#;
        // Each instance drops its passive segment. Compare fresh Wasmtime instances.
        let wasm = wat::parse_str(wat).unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let inputs = [0, 1, 2, 3, 4, 5];
        let expected: Vec<_> = inputs
            .iter()
            .map(|&input| {
                let mut store = wasmtime::Store::new(&engine, ());
                let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
                instance
                    .get_typed_func::<i32, i32>(&mut store, "f")
                    .unwrap()
                    .call(&mut store, input as i32)
                    .unwrap() as u32
            })
            .collect();
        assert_eq!(kernel.run(&inputs).unwrap(), expected);
        assert_eq!(kernel.run(&inputs).unwrap(), expected);
    }

    #[test]
    fn table_bounds_and_dropped_elements_trap() {
        for body in [
            "i32.const 2 table.get drop",
            "i32.const 2 ref.null func table.set",
            "i32.const 3 ref.null func i32.const 0 table.fill",
            "i32.const 0 i32.const 3 i32.const 0 table.copy",
            "i32.const 2 i32.const 0 i32.const 1 table.init $e",
            "elem.drop $e i32.const 0 i32.const 0 i32.const 1 table.init $e",
            "elem.drop $e i32.const 0 i32.const 1 i32.const 0 table.init $e",
            "i32.const -1 ref.null func i32.const 2 table.fill",
        ] {
            let wat = format!(
                r#"(module (table 2 funcref) (func $a) (elem $e func $a)
                (func (export "f") (param i32) (result i32) {body} i32.const 0))"#
            );
            let wasm = wat::parse_str(wat).unwrap();
            let kernel = Kernel::new(&wasm, "f").unwrap();
            assert!(
                matches!(
                    kernel.run(&[0]),
                    Err(Error::Trap {
                        reason: "undefined or uninitialized table element",
                        ..
                    })
                ),
                "{body}"
            );
            let engine = wasmtime::Engine::default();
            let module = wasmtime::Module::new(&engine, &wasm).unwrap();
            let mut store = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            assert!(
                instance
                    .get_typed_func::<i32, i32>(&mut store, "f")
                    .unwrap()
                    .call(&mut store, 0)
                    .is_err(),
                "{body}"
            );
        }
    }

    #[test]
    fn unreachable_table_mutation_keeps_shared_table_storage() {
        let wasm = wat::parse_str(
            r#"(module
                (table 65536 funcref)
                (func $unused
                    i32.const 0 ref.null func table.set
                    ref.null func i32.const 1 table.grow drop)
                (func (export "f") (param i32) (result i32)
                    table.size local.get 0 i32.add))"#,
        )
        .unwrap();
        let translation = translate(&wasm, "f", &[]).unwrap();
        assert!(!translation.instances[0].table_info.mutable);
        assert_eq!(translation.instances[0].table_info.capacity, 65536);
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 1, 2]).unwrap(), [65536, 65537, 65538]);
    }

    #[test]
    fn null_references_and_dropped_active_elements_match_wasmtime() {
        matches_wasmtime(
            r#"(module
            (table 0 0 funcref)
            (elem $active (i32.const 0) func)
            (elem $declared declare func)
            (func (export "f") (param i32) (result i32) (local funcref)
                elem.drop $declared
                i32.const 0 i32.const 0 i32.const 0 table.init $active
                i32.const 0 ref.null func i32.const 0 table.fill
                i32.const 0 i32.const 0 i32.const 0 table.copy
                ref.null func i32.const 0 table.grow drop
                local.get 1 ref.is_null
                ref.null func ref.null func local.get 0 select (result funcref)
                ref.is_null i32.add))"#,
            &[0, 1],
        );
        matches_wasmtime(
            r#"(module (elem $e declare func)
            (func (export "f") (param i32) (result i32)
                elem.drop $e local.get 0))"#,
            &[0, 1],
        );
    }

    #[test]
    fn table_start_mutation_and_reference_results_match_wasmtime() {
        matches_wasmtime(
            r#"(module
            (type $get (func (result funcref)))
            (table 2 funcref)
            (func $value (result i32) i32.const 31)
            (func $get (type $get) ref.func $value)
            (elem declare func $get $value)
            (func $start i32.const 0 ref.func $get table.set)
            (start $start)
            (func (export "f") (param i32) (result i32)
                i32.const 1 i32.const 0 call_indirect (type $get) table.set
                i32.const 1 call_indirect (result i32)))"#,
            &[0, 1, 3],
        );
        matches_wasmtime(
            r#"(module (table 1 funcref)
            (func (export "f") (param i32) (result i32)
                i32.const 0 table.get ref.is_null table.size i32.add))"#,
            &[0, 1],
        );
    }

    #[test]
    fn table_growth_respects_resource_cap() {
        let wasm = wat::parse_str(
            r#"(module (table 0 funcref)
            (func (export "f") (param i32) (result i32)
                ref.null func local.get 0 table.grow))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(
            kernel.run(&[0, 1, 256, 257, u32::MAX]).unwrap(),
            [0, 0, 0, u32::MAX, u32::MAX]
        );
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
    fn indirect_calls_check_type_and_bound_recursion() {
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
        let kernel = Kernel::new(&wat::parse_str(recursive).unwrap(), "f").unwrap();
        assert!(matches!(
            kernel.run(&[1]),
            Err(Error::Trap {
                reason: "call depth limit exceeded",
                ..
            })
        ));
    }

    #[test]
    fn direct_recursion_matches_wasmtime_within_depth_limit() {
        let wat = r#"(module
            (func $factorial (export "f") (param i32) (result i32)
                local.get 0 i32.eqz
                if (result i32)
                    i32.const 1
                else
                    local.get 0
                    local.get 0 i32.const 1 i32.sub call $factorial
                    i32.mul
                end))"#;
        matches_wasmtime(wat, &[0, 1, 2, 5, 10]);
        let kernel = Kernel::new(&wat::parse_str(wat).unwrap(), "f").unwrap();
        assert!(matches!(
            kernel.run(&[32]),
            Err(Error::Trap {
                reason: "call depth limit exceeded",
                ..
            })
        ));
    }

    #[test]
    fn mutual_recursion_matches_wasmtime_and_traps_at_limit() {
        let wat = r#"(module
            (func $even (export "f") (param i32) (result i32)
                local.get 0 i32.eqz
                if (result i32)
                    i32.const 1
                else
                    local.get 0 i32.const 1 i32.sub call $odd
                end)
            (func $odd (param i32) (result i32)
                local.get 0 i32.eqz
                if (result i32)
                    i32.const 0
                else
                    local.get 0 i32.const 1 i32.sub call $even
                end))"#;
        matches_wasmtime(wat, &[0, 1, 2, 7, 31]);
        let kernel = Kernel::new(&wat::parse_str(wat).unwrap(), "f").unwrap();
        assert!(matches!(
            kernel.run(&[32]),
            Err(Error::Trap {
                reason: "call depth limit exceeded",
                ..
            })
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

    #[test]
    fn varied_float_bits_match_wasmtime() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut pairs64 = Vec::new();
        let mut pairs32 = Vec::new();
        let mut near_f32_zero = Vec::new();
        for _ in 0..128 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let a = state;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let b = state;
            pairs64.push(vec![Value::F64(a), Value::F64(b)]);
            pairs32.push(vec![Value::F32(a as u32), Value::F32(b as u32)]);
            near_f32_zero.push(vec![Value::F64((873u64 << 52) | (a & ((1u64 << 52) - 1)))]);
        }
        for ty in ["f32", "f64"] {
            let rows = if ty == "f32" { &pairs32 } else { &pairs64 };
            for op in ["add", "sub", "mul", "div"] {
                compare_values_to_wasmtime(
                    &format!(
                        "(module (func (export \"f\") (param {ty} {ty}) (result {ty}) local.get 0 local.get 1 {ty}.{op}))"
                    ),
                    rows,
                );
            }
            let unary: Vec<_> = rows.iter().map(|row| vec![row[0]]).collect();
            compare_values_to_wasmtime(
                &format!(
                    "(module (func (export \"f\") (param {ty}) (result {ty}) local.get 0 {ty}.sqrt))"
                ),
                &unary,
            );
        }
        compare_values_to_wasmtime(
            "(module (func (export \"f\") (param f64) (result f32) local.get 0 f32.demote_f64))",
            &near_f32_zero,
        );
    }
}
