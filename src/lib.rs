//! Execute a supported WebAssembly function once per Metal thread.

use std::any::Any;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error as StdError;
use std::ffi::CStr;
use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use metal::objc::{
    Message,
    rc::autoreleasepool,
    runtime::{Object, Sel},
};
use metal::{
    CompileOptions, ComputePipelineDescriptor, ComputePipelineState, Device,
    MTLCommandBufferStatus, MTLResourceOptions, MTLSize,
};
use wasmparser::types::CoreTypeId;
use wasmparser::{
    AbstractHeapType, BlockType, Catch, CompositeInnerType, ConstExpr, DataKind, ElementItems,
    ElementKind, ExternalKind, FuncType, FunctionBody, GlobalType, HeapType, MemoryType, Operator,
    Parser, Payload, RefType, StorageType, TableInit, TableType, TryTable, TypeRef, UnpackedIndex,
    ValType, Validator,
};

mod store;
pub use store::{Instance, Store};

mod gc;
mod gc_translate;
use gc::GcRecord;
pub use gc::GcRef;
use gc_translate::gc_type_source;
use gc_translate::root_word;

static NEXT_KERNEL_ID: AtomicU64 = AtomicU64::new(1);

const MAX_MEMORY_BYTES_PER_INSTANCE: usize = 16 * 1024 * 1024;
const MAX_GROWTH_PAGES: usize = 4;
const MAX_TABLE_ELEMENTS: usize = 65536;
const MAX_TABLE_GROWTH: usize = 256;
const MAX_CALL_DEPTH: usize = 32;
const MAX_BATCH_BYTES: usize = 64 * 1024 * 1024;

/// Per-lane resource limits reserved when compiling a kernel.
#[derive(Clone, Copy, Debug)]
pub struct KernelLimits {
    pub memory_bytes: usize,
    pub memory_growth_pages: usize,
    pub table_growth_elements: usize,
    pub call_depth: usize,
    /// Fuel per lane and dispatch: one unit per function entry and loop iteration.
    /// None disables metering. Kernel starts and the exported call share this budget.
    pub fuel: Option<u64>,
    /// Maximum live exception records per lane. Serialized Kernel input records
    /// remain pinned for that invocation; unreachable new records can be reused.
    pub exception_records: usize,
    /// Bounded GC heap size in 64-bit words per lane.
    pub gc_heap_words: usize,
}

impl Default for KernelLimits {
    fn default() -> Self {
        Self {
            memory_bytes: MAX_MEMORY_BYTES_PER_INSTANCE,
            memory_growth_pages: MAX_GROWTH_PAGES,
            table_growth_elements: MAX_TABLE_GROWTH,
            call_depth: MAX_CALL_DEPTH,
            fuel: Some(1_000_000),
            exception_records: 64,
            gc_heap_words: 1024,
        }
    }
}

/// An opaque function identity owned by one kernel or store.
/// Handles cannot cross owner boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunctionRef {
    kernel: u64,
    index: u32,
    ty: RefType,
}

/// An opaque host value retained by its kernel, or by an active store lease.
/// Metal carries the identity, but cannot access the host value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalRef {
    generation: u64,
    kernel: u64,
    index: u64,
}

/// An immutable exception retained by its kernel, or by an active store lease.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExceptionRef {
    generation: u64,
    kernel: u64,
    index: u32,
}

#[derive(Clone)]
struct ExceptionRecord {
    tag: u32,
    payload: Vec<Value>,
}

#[derive(Clone, Copy)]
struct ExceptionIo {
    records: usize,
    payload_words: usize,
    input: bool,
    output: bool,
}

impl ExceptionIo {
    fn words(self) -> Result<usize, Error> {
        self.payload_words
            .checked_add(1)
            .and_then(|n| n.checked_mul(self.records))
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Unsupported("exception transfer size overflow".into()))
    }
}

/// One WebAssembly value. Floating-point variants hold their bit patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    I32(u32),
    I64(u64),
    F32(u32),
    F64(u64),
    V128([u8; 16]),
    /// A nullable function reference obtained from its kernel or store.
    FuncRef(Option<FunctionRef>),
    ExternRef(Option<ExternalRef>),
    ExnRef(Option<ExceptionRef>),
    /// An unboxed 31-bit signed integer reference.
    I31(u32),
    /// A nullable kernel snapshot or live store reference.
    GcRef(Option<GcRef>),
}

impl Value {
    fn ty(self) -> ValType {
        match self {
            Self::I32(_) => ValType::I32,
            Self::I64(_) => ValType::I64,
            Self::F32(_) => ValType::F32,
            Self::F64(_) => ValType::F64,
            Self::V128(_) => ValType::V128,
            Self::FuncRef(value) => ValType::Ref(value.map_or(RefType::FUNCREF, |f| f.ty)),
            Self::ExternRef(_) => ValType::Ref(RefType::EXTERNREF),
            Self::ExnRef(_) => ValType::Ref(RefType::EXNREF),
            Self::I31(_) => ValType::Ref(RefType::I31REF.as_non_null()),
            Self::GcRef(value) => ValType::Ref(value.map_or(RefType::ANYREF, |r| r.ty)),
        }
    }

    fn bits(self) -> u64 {
        match self {
            Self::I32(bits) | Self::F32(bits) => bits as u64,
            Self::I64(bits) | Self::F64(bits) => bits,
            Self::V128(_) => unreachable!("v128 has two words"),
            Self::FuncRef(value) => value.map_or(0, |f| u64::from(f.index)),
            Self::ExternRef(value) => value.map_or(0, |f| f.index),
            Self::ExnRef(value) => value.map_or(0, |f| u64::from(f.index)),
            Self::I31(value) => 0x100000000 | u64::from(value & 0x7fffffff),
            Self::GcRef(value) => value.map_or(0, |r| r.index),
        }
    }

    fn words(self) -> impl Iterator<Item = u64> {
        let (low, high) = match self {
            Self::V128(bytes) => (
                u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                Some(u64::from_le_bytes(bytes[8..].try_into().unwrap())),
            ),
            _ => (self.bits(), None),
        };
        std::iter::once(low).chain(high)
    }
}

/// A compiled WebAssembly export, run once per Metal thread.
pub struct Kernel {
    id: u64,
    external_values: Mutex<Vec<Arc<dyn Any + Send + Sync>>>,
    exception_values: Mutex<Vec<ExceptionRecord>>,
    gc_values: Mutex<Vec<GcRecord>>,
    exception_tags: HashMap<String, u32>,
    tag_payload_types: Vec<Vec<ValType>>,
    exception_io: Option<ExceptionIo>,
    function_reference_types: Vec<RefType>,
    exported_functions: HashMap<String, u32>,
    reference_supertypes: ReferenceTypes,
    gc: Option<GcLayout>,
    gc_types: Vec<GcType>,
    device: Device,
    queue: metal::CommandQueue,
    pipeline: ComputePipelineState,
    can_trap: bool,
    params: Vec<ValType>,
    results: Vec<ValType>,
    instances: Vec<InstanceBuffers>,
    memory_exports: HashMap<String, (usize, usize)>,
}

struct Translation {
    persistent_metadata: Option<store::MetadataLayout>,
    exception_tags: HashMap<String, u32>,
    tag_payload_types: Vec<Vec<ValType>>,
    exception_io: Option<ExceptionIo>,
    function_reference_types: Vec<RefType>,
    exported_functions: HashMap<String, u32>,
    reference_supertypes: ReferenceTypes,
    gc: Option<GcLayout>,
    gc_types: Vec<GcType>,
    source: String,
    can_trap: bool,
    params: Vec<ValType>,
    results: Vec<ValType>,
    instances: Vec<InstanceTemplate>,
    slots: Vec<BufferSlots>,
    recursive: bool,
    memory_exports: HashMap<String, (usize, usize)>,
}

struct InstanceTemplate {
    memories: Vec<MemoryTemplate>,
    globals_template: Vec<u64>,
    tables: Vec<TableTemplate>,
    element_info: TableInfo,
    passive_data: Vec<u8>,
    data_lengths: Vec<u32>,
}

struct MemoryTemplate {
    bytes: Vec<u8>,
    capacity: usize,
    max_pages: Option<u64>,
}

struct TableTemplate {
    function_refs: bool,
    values: Vec<u64>,
    info: TableInfo,
    targets: Vec<usize>,
}

#[derive(Default)]
struct BufferSlots {
    memories: Vec<(u64, u64)>,
    globals: Option<u64>,
    tables: Vec<u64>,
    passive_data: Option<u64>,
}

struct InstanceBuffers {
    template: InstanceTemplate,
    slots: BufferSlots,
    tables: Vec<Option<metal::Buffer>>,
    passive_data: Option<metal::Buffer>,
}

struct MemoryInput<'a> {
    offset: usize,
    width: usize,
    rows: &'a [Vec<u8>],
}

#[derive(Default)]
struct TableInfo {
    mutable: bool,
    capacity: usize,
    elements: Vec<Vec<u64>>,
    function_elements: Vec<bool>,
    element_types: Vec<RefType>,
    lengths: Vec<u32>,
}

/// An error while validating, compiling, or running a WebAssembly export.
#[derive(Debug)]
pub enum Error {
    InvalidWasm(wasmparser::BinaryReaderError),
    Unsupported(String),
    Metal(String),
    UncaughtException {
        input_index: usize,
    },
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

    /// Compile an export with custom per-lane resource limits.
    pub fn new_with_limits(wasm: &[u8], export: &str, limits: KernelLimits) -> Result<Self, Error> {
        Self::new_with_imports_and_limits(wasm, export, &[], limits)
    }

    /// Compile an export with Wasm modules registered as import providers.
    /// A provider namespace owns one independent instance per lane, shared by all
    /// imports from that namespace. Provider dependencies initialize before importers.
    /// Host callbacks and cyclic provider dependencies are rejected.
    pub fn new_with_imports(
        wasm: &[u8],
        export: &str,
        providers: &[(&str, &[u8])],
    ) -> Result<Self, Error> {
        Self::new_with_imports_and_limits(wasm, export, providers, KernelLimits::default())
    }

    /// Compile an export with Wasm providers and custom per-lane resource limits.
    pub fn new_with_imports_and_limits(
        wasm: &[u8],
        export: &str,
        providers: &[(&str, &[u8])],
        limits: KernelLimits,
    ) -> Result<Self, Error> {
        if !limits.memory_bytes.is_multiple_of(65536)
            || limits.memory_bytes / 65536 > 65536
            || limits.exception_records == 0
            || limits.exception_records > u32::MAX as usize
            || limits.gc_heap_words < 4
            || limits.gc_heap_words > u32::MAX as usize
            || limits.call_depth == 0
            || limits.call_depth > u32::MAX as usize
        {
            return Err(Error::Unsupported("invalid kernel resource limits".into()));
        }
        let translation = translate_with_limits(wasm, export, providers, limits)?;
        Self::from_translation(translation, limits)
    }

    fn from_translation(translation: Translation, limits: KernelLimits) -> Result<Self, Error> {
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
            descriptor.set_max_call_stack_depth(limits.call_depth as u64 + 2);
            device.new_compute_pipeline_state(&descriptor)
        } else {
            device.new_compute_pipeline_state_with_function(&function)
        }
        .map_err(Error::Metal)?;
        let queue = device.new_command_queue();
        let mut instances = Vec::new();
        for (template, slots) in translation.instances.into_iter().zip(translation.slots) {
            let tables = template
                .tables
                .iter()
                .map(|table| {
                    (!table.info.mutable)
                        .then(|| make_buffer(&device, &table.values))
                        .transpose()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let passive_data = (!template.data_lengths.is_empty())
                .then(|| make_buffer(&device, &template.passive_data))
                .transpose()?;
            instances.push(InstanceBuffers {
                template,
                slots,
                tables,
                passive_data,
            });
        }
        Ok(Self {
            id: NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed),
            external_values: Mutex::new(Vec::new()),
            exception_values: Mutex::new(Vec::new()),
            gc_values: Mutex::new(Vec::new()),
            gc: translation.gc,
            gc_types: translation.gc_types,
            exception_tags: translation.exception_tags,
            tag_payload_types: translation.tag_payload_types,
            exception_io: translation.exception_io,
            function_reference_types: translation.function_reference_types,
            exported_functions: translation.exported_functions,
            reference_supertypes: translation.reference_supertypes,
            device,
            queue,
            pipeline,
            can_trap: translation.can_trap,
            params: translation.params,
            results: translation.results,
            instances,
            memory_exports: translation.memory_exports,
        })
    }

    /// Obtain a function handle for an export of the root module.
    pub fn function_reference(&self, export: &str) -> Result<FunctionRef, Error> {
        let index = *self
            .exported_functions
            .get(export)
            .ok_or_else(|| Error::Unsupported(format!("export {export:?} is not a function")))?;
        Ok(FunctionRef {
            kernel: self.id,
            index,
            ty: self.function_reference_types[index as usize - 1],
        })
    }

    fn accepts_value(&self, value: Value, expected: ValType) -> bool {
        match (value, expected) {
            (Value::ExnRef(None), ValType::Ref(ty)) => ty.is_nullable() && is_exception_ref(ty),
            (Value::ExnRef(Some(reference)), ValType::Ref(ty)) => {
                reference.kernel == self.id
                    && matches!(
                        ty.heap_type(),
                        HeapType::Abstract {
                            shared: false,
                            ty: AbstractHeapType::Exn
                        }
                    )
            }
            (Value::ExternRef(None), ValType::Ref(ty)) => ty.is_nullable() && is_external_ref(ty),
            (Value::ExternRef(Some(reference)), ValType::Ref(ty)) => {
                self.validate_external_reference(reference)
                    && matches!(
                        ty.heap_type(),
                        HeapType::Abstract {
                            shared: false,
                            ty: AbstractHeapType::Extern
                        }
                    )
            }
            (Value::GcRef(None), ValType::Ref(ty)) => {
                ty.is_nullable() && is_gc_ref(ty, &self.reference_supertypes)
            }
            (Value::GcRef(Some(r)), ValType::Ref(ty)) => {
                self.validate_gc_reference(r) && ref_subtype(r.ty, ty, &self.reference_supertypes)
            }
            (Value::I31(_), ValType::Ref(ty)) => ref_subtype(
                RefType::I31REF.as_non_null(),
                ty,
                &self.reference_supertypes,
            ),
            (Value::FuncRef(None), ValType::Ref(ty)) => {
                ty.is_nullable() && is_function_ref_in(ty, &self.reference_supertypes)
            }
            (Value::FuncRef(Some(reference)), ValType::Ref(ty)) => {
                reference.kernel == self.id
                    && ref_subtype(reference.ty, ty, &self.reference_supertypes)
            }
            _ => value.ty() == expected,
        }
    }

    /// Retain a host value and obtain an opaque reference for this kernel.
    pub fn external_reference<T: Any + Send + Sync>(&self, value: T) -> Result<ExternalRef, Error> {
        let mut values = self
            .external_values
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let index = u32::try_from(values.len())
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Unsupported("too many external references".into()))?;
        values.push(Arc::new(value));
        Ok(ExternalRef {
            generation: 0,
            kernel: self.id,
            index: u64::from(index),
        })
    }

    /// Retrieve a retained host value without exposing it to the shader.
    pub fn external_reference_value<T: Any + Send + Sync>(
        &self,
        reference: ExternalRef,
    ) -> Result<Arc<T>, Error> {
        if reference.kernel != self.id {
            return Err(Error::Unsupported(
                "external reference belongs to another kernel".into(),
            ));
        }
        self.external_values
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(reference.index as usize - 1)
            .cloned()
            .ok_or_else(|| Error::Unsupported("unknown external reference".into()))?
            .downcast::<T>()
            .map_err(|_| Error::Unsupported("external reference host type mismatch".into()))
    }

    /// Create an exception for an exported tag. Payload references must belong to this kernel.
    pub fn exception_reference(
        &self,
        tag_export: &str,
        payload: &[Value],
    ) -> Result<ExceptionRef, Error> {
        let tag = *self
            .exception_tags
            .get(tag_export)
            .ok_or_else(|| Error::Unsupported(format!("export {tag_export:?} is not a tag")))?;
        let types = &self.tag_payload_types[tag as usize];
        if payload.len() != types.len()
            || payload
                .iter()
                .zip(types)
                .any(|(&value, &ty)| !self.accepts_value(value, ty))
        {
            return Err(Error::Unsupported(
                "exception payload does not match the tag signature".into(),
            ));
        }
        self.retain_exception(ExceptionRecord {
            tag,
            payload: payload.to_vec(),
        })
    }

    fn retain_exception(&self, record: ExceptionRecord) -> Result<ExceptionRef, Error> {
        let mut records = self
            .exception_values
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let index = u32::try_from(records.len())
            .ok()
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| Error::Unsupported("too many retained exceptions".into()))?;
        records.push(record);
        Ok(ExceptionRef {
            generation: 0,
            kernel: self.id,
            index,
        })
    }

    /// Read an immutable payload, including any retained reference values.
    pub fn exception_payload(&self, reference: ExceptionRef) -> Result<Vec<Value>, Error> {
        if reference.kernel != self.id {
            return Err(Error::Unsupported(
                "exception reference belongs to another kernel".into(),
            ));
        }
        self.exception_values
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(reference.index as usize - 1)
            .map(|record| record.payload.clone())
            .ok_or_else(|| Error::Unsupported("unknown exception reference".into()))
    }

    /// Run one independent invocation for each input. Values are raw i32 bit patterns.
    pub fn run(&self, inputs: &[u32]) -> Result<Vec<u32>, Error> {
        self.run_timed(inputs).map(|(output, _)| output)
    }

    /// Run the batch and return summed GPU execution time when available.
    /// Large batches use multiple dispatches. This excludes host-side buffer
    /// copies and command submission.
    pub fn run_timed(&self, inputs: &[u32]) -> Result<(Vec<u32>, Option<Duration>), Error> {
        if self.params != [ValType::I32] || self.results != [ValType::I32] {
            return Err(Error::Unsupported(
                "run requires type (param i32) (result i32); use run_values".into(),
            ));
        }
        if inputs.is_empty() {
            return Ok((Vec::new(), None));
        }
        let batch_size = self
            .batch_limit(size_of::<u32>(), size_of::<u32>())?
            .min(inputs.len());
        let instance_buffers = self.prepare_batch(batch_size as u32)?;
        let mut values = Vec::with_capacity(inputs.len());
        let mut gpu_time = Some(Duration::ZERO);
        for chunk in inputs.chunks(batch_size) {
            if !values.is_empty() {
                self.reset_batch(&instance_buffers, chunk.len() as u32);
            }
            let bytes = size_of_val(chunk);
            let input = unsafe { std::slice::from_raw_parts(chunk.as_ptr().cast::<u8>(), bytes) };
            let (output, elapsed) = self
                .dispatch(input, bytes, chunk.len() as u32, &instance_buffers, None, 0)
                .map_err(|error| shift_trap(error, values.len()))?;
            let result =
                unsafe { std::slice::from_raw_parts(output.contents().cast::<u32>(), chunk.len()) };
            values.extend_from_slice(result);
            gpu_time = gpu_time.and_then(|total| elapsed.and_then(|time| total.checked_add(time)));
        }
        Ok((values, gpu_time))
    }

    /// Run one independent invocation per row. Each row must match the export's parameters.
    pub fn run_values(&self, inputs: &[Vec<Value>]) -> Result<Vec<Vec<Value>>, Error> {
        self.run_values_impl(inputs, None, None)
            .map(|(values, _)| values)
    }

    /// Run each invocation and copy a byte range from an exported memory for each lane.
    /// The range must fit the memory after every invocation, including any growth.
    pub fn run_values_with_memory(
        &self,
        inputs: &[Vec<Value>],
        memory_export: &str,
        range: Range<usize>,
    ) -> Result<(Vec<Vec<Value>>, Vec<Vec<u8>>), Error> {
        let owner = *self.memory_exports.get(memory_export).ok_or_else(|| {
            Error::Unsupported(format!("export {memory_export:?} is not a memory"))
        })?;
        if range.start > range.end {
            return Err(Error::Unsupported("invalid memory range".into()));
        }
        let (values, memory) = self.run_values_impl(inputs, Some((owner, range)), None)?;
        Ok((values, memory))
    }

    /// Copy one equally sized byte slice into each lane's exported memory after
    /// its data segments and start functions run, then invoke the export.
    /// The input range must fit the memory's initial size. An empty output range
    /// returns values without copying memory back.
    pub fn run_values_with_memory_io(
        &self,
        inputs: &[Vec<Value>],
        memory_export: &str,
        input_offset: usize,
        memory_inputs: &[Vec<u8>],
        output_range: Range<usize>,
    ) -> Result<(Vec<Vec<Value>>, Vec<Vec<u8>>), Error> {
        let owner = *self.memory_exports.get(memory_export).ok_or_else(|| {
            Error::Unsupported(format!("export {memory_export:?} is not a memory"))
        })?;
        if output_range.start > output_range.end {
            return Err(Error::Unsupported("invalid memory range".into()));
        }
        if memory_inputs.len() != inputs.len() {
            return Err(Error::Unsupported(
                "memory input count does not match invocation count".into(),
            ));
        }
        let width = memory_inputs.first().map_or(0, Vec::len);
        if memory_inputs.iter().any(|row| row.len() != width) {
            return Err(Error::Unsupported(
                "memory inputs must have equal lengths".into(),
            ));
        }
        let initial_size = self.instances[owner.0].template.memories[owner.1]
            .bytes
            .len();
        if input_offset
            .checked_add(width)
            .is_none_or(|end| end > initial_size)
        {
            return Err(Error::Unsupported(
                "memory input range exceeds initial memory size".into(),
            ));
        }
        self.run_values_impl(
            inputs,
            Some((owner, output_range)),
            Some(MemoryInput {
                offset: input_offset,
                width,
                rows: memory_inputs,
            }),
        )
    }

    fn encode_exceptions(
        &self,
        row: &[Value],
        io: ExceptionIo,
    ) -> Result<(Vec<u64>, Vec<u64>, Vec<ExceptionRef>), Error> {
        let records = self
            .exception_values
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let mut pending: Vec<_> = row
            .iter()
            .filter_map(|value| match value {
                Value::ExnRef(Some(reference)) => Some(*reference),
                _ => None,
            })
            .collect();
        let mut needed = HashSet::new();
        while let Some(reference) = pending.pop() {
            if !needed.insert(reference.index) {
                continue;
            }
            if needed.len() > io.records {
                return Err(Error::Trap {
                    input_index: 0,
                    reason: "exception record limit exceeded",
                });
            }
            let record = records
                .get(reference.index as usize - 1)
                .ok_or_else(|| Error::Unsupported("unknown exception reference".into()))?;
            pending.extend(record.payload.iter().filter_map(|value| match value {
                Value::ExnRef(Some(reference)) => Some(*reference),
                _ => None,
            }));
        }
        // Immutable records refer only to older records. Increasing host IDs give
        // a topological order without recursion or a host stack-depth limit.
        let mut ids: Vec<_> = needed.into_iter().collect();
        ids.sort_unstable();
        let mapping: HashMap<_, _> = ids
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id, slot as u64 + 1))
            .collect();
        let encode = |value: Value| match value {
            Value::ExnRef(Some(reference)) => vec![mapping[&reference.index]],
            _ => value.words().collect(),
        };
        let mut arena = vec![0; io.words()?];
        arena[0] = ids.len() as u64;
        for (slot, &id) in ids.iter().enumerate() {
            let record = &records[id as usize - 1];
            let offset = 1 + slot * (io.payload_words + 1);
            arena[offset] = u64::from(record.tag);
            let payload: Vec<_> = record.payload.iter().copied().flat_map(&encode).collect();
            arena[offset + 1..offset + 1 + payload.len()].copy_from_slice(&payload);
        }
        let arguments = row.iter().copied().flat_map(encode).collect();
        let handles = ids
            .into_iter()
            .map(|index| ExceptionRef {
                generation: 0,
                kernel: self.id,
                index,
            })
            .collect();
        Ok((arguments, arena, handles))
    }

    fn decode_value(
        &self,
        ty: ValType,
        words: &[u64],
        exceptions: &[Option<ExceptionRef>],
    ) -> Result<Value, Error> {
        let low = words[0];
        Ok(match ty {
            ValType::I32 => Value::I32(low as u32),
            ValType::I64 => Value::I64(low),
            ValType::F32 => Value::F32(low as u32),
            ValType::F64 => Value::F64(low),
            ValType::V128 => {
                let mut bytes = [0; 16];
                bytes[..8].copy_from_slice(&low.to_le_bytes());
                bytes[8..].copy_from_slice(&words[1].to_le_bytes());
                Value::V128(bytes)
            }
            ValType::Ref(ty) if is_function_ref_in(ty, &self.reference_supertypes) => {
                Value::FuncRef(if low == 0 {
                    None
                } else {
                    Some(FunctionRef {
                        kernel: self.id,
                        index: low as u32,
                        ty: *self
                            .function_reference_types
                            .get(low as usize - 1)
                            .ok_or_else(|| {
                                Error::Metal("invalid function reference result".into())
                            })?,
                    })
                })
            }
            ValType::Ref(ty) if is_external_ref(ty) => {
                Value::ExternRef((low != 0).then_some(ExternalRef {
                    generation: 0,
                    kernel: self.id,
                    index: low,
                }))
            }
            ValType::Ref(ty) if is_exception_ref(ty) => Value::ExnRef(if low == 0 {
                None
            } else {
                Some(
                    exceptions
                        .get(low as usize - 1)
                        .copied()
                        .flatten()
                        .ok_or_else(|| Error::Metal("invalid exception reference result".into()))?,
                )
            }),
            _ => {
                return Err(Error::Unsupported(
                    "unsupported reference result type".into(),
                ));
            }
        })
    }

    fn decode_exceptions(
        &self,
        io: ExceptionIo,
        arena: &[u64],
        results: &[u64],
        existing: &[ExceptionRef],
    ) -> Result<Vec<Option<ExceptionRef>>, Error> {
        let count = usize::try_from(arena[0])
            .ok()
            .filter(|&n| n <= io.records && n >= existing.len())
            .ok_or_else(|| Error::Metal("invalid exception record count".into()))?;
        let mut handles = vec![None; count];
        for (slot, &reference) in existing.iter().enumerate() {
            handles[slot] = Some(reference);
        }
        let mut pending = Vec::new();
        let mut word = 0;
        for &ty in &self.results {
            if matches!(ty, ValType::Ref(ty) if is_exception_ref(ty)) && results[word] != 0 {
                pending.push(results[word]);
            }
            word += value_words(ty);
        }
        let mut needed = HashSet::new();
        while let Some(id) = pending.pop() {
            let slot = usize::try_from(id)
                .ok()
                .filter(|&n| n > 0 && n <= count)
                .ok_or_else(|| Error::Metal("invalid exception reference result".into()))?
                - 1;
            if handles[slot].is_some() || !needed.insert(slot) {
                continue;
            }
            let offset = 1 + slot * (io.payload_words + 1);
            let tag = usize::try_from(arena[offset])
                .ok()
                .filter(|&n| n != 0)
                .and_then(|n| self.tag_payload_types.get(n))
                .ok_or_else(|| Error::Metal("invalid exception tag result".into()))?;
            let mut word = offset + 1;
            for &ty in tag {
                if matches!(ty, ValType::Ref(ty) if is_exception_ref(ty)) && arena[word] != 0 {
                    if arena[word] > slot as u64 {
                        return Err(Error::Metal(
                            "exception payload does not refer to an older record".into(),
                        ));
                    }
                    pending.push(arena[word]);
                }
                word += value_words(ty);
            }
        }
        let mut needed: Vec<_> = needed.into_iter().collect();
        needed.sort_unstable();
        for slot in needed {
            let offset = 1 + slot * (io.payload_words + 1);
            let tag = arena[offset] as u32;
            let mut word = offset + 1;
            let mut payload = Vec::new();
            for &ty in &self.tag_payload_types[tag as usize] {
                payload.push(self.decode_value(ty, &arena[word..], &handles)?);
                word += value_words(ty);
            }
            handles[slot] = Some(self.retain_exception(ExceptionRecord { tag, payload })?);
        }
        Ok(handles)
    }

    fn run_values_impl(
        &self,
        inputs: &[Vec<Value>],
        capture: Option<((usize, usize), Range<usize>)>,
        memory_input: Option<MemoryInput<'_>>,
    ) -> Result<(Vec<Vec<Value>>, Vec<Vec<u8>>), Error> {
        for row in inputs {
            if row.len() != self.params.len()
                || row
                    .iter()
                    .zip(&self.params)
                    .any(|(value, ty)| !self.accepts_value(*value, *ty))
            {
                return Err(Error::Unsupported(
                    "input values do not match the export signature".into(),
                ));
            }
        }
        if inputs.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if capture.is_none()
            && memory_input.is_none()
            && self.params == [ValType::I32]
            && self.results == [ValType::I32]
        {
            let flat: Vec<_> = inputs
                .iter()
                .map(|row| match row[0] {
                    Value::I32(bits) => bits,
                    _ => unreachable!(),
                })
                .collect();
            return Ok((
                self.run(&flat)?
                    .into_iter()
                    .map(|bits| vec![Value::I32(bits)])
                    .collect(),
                Vec::new(),
            ));
        }
        let fast_i32 = self.params == [ValType::I32] && self.results == [ValType::I32];
        let word_size = if fast_i32 {
            size_of::<u32>()
        } else {
            size_of::<u64>()
        };
        let argument_stride = self
            .params
            .iter()
            .map(|ty| value_words(*ty))
            .sum::<usize>()
            .checked_mul(word_size)
            .ok_or_else(|| Error::Unsupported("input buffer size overflow".into()))?;
        let exception_words = self
            .exception_io
            .map(ExceptionIo::words)
            .transpose()?
            .unwrap_or(0);
        let exception_bytes = exception_words
            .checked_mul(size_of::<u64>())
            .ok_or_else(|| Error::Unsupported("exception transfer size overflow".into()))?;
        let input_stride = argument_stride
            .checked_add(memory_input.as_ref().map_or(0, |input| input.width))
            .and_then(|stride| {
                stride.checked_add(if self.exception_io.is_some_and(|io| io.input) {
                    exception_bytes.checked_add(8)?
                } else {
                    0
                })
            })
            .ok_or_else(|| Error::Unsupported("input buffer size overflow".into()))?;
        let output_stride = self
            .results
            .iter()
            .map(|ty| value_words(*ty))
            .sum::<usize>()
            .checked_mul(word_size)
            .ok_or_else(|| Error::Unsupported("output buffer size overflow".into()))?;
        let transfer_output_stride = output_stride
            .checked_add(if self.exception_io.is_some_and(|io| io.output) {
                exception_bytes
            } else {
                0
            })
            .ok_or_else(|| Error::Unsupported("output buffer size overflow".into()))?;
        let batch_size = self
            .batch_limit(input_stride, transfer_output_stride)?
            .min(inputs.len());
        let instance_buffers = self.prepare_batch(batch_size as u32)?;
        let mut rows = Vec::with_capacity(inputs.len());
        let mut memories = Vec::with_capacity(if capture.is_some() { inputs.len() } else { 0 });
        for start in (0..inputs.len()).step_by(batch_size) {
            let end = (start + batch_size).min(inputs.len());
            if start != 0 {
                self.reset_batch(&instance_buffers, (end - start) as u32);
            }
            let encoded32;
            let encoded64;
            let mut exception_arenas = Vec::new();
            let mut exception_handles = Vec::new();
            let input_bytes;
            let input = if fast_i32 {
                encoded32 = inputs[start..end]
                    .iter()
                    .map(|row| row[0].bits() as u32)
                    .collect::<Vec<_>>();
                input_bytes = size_of_val(encoded32.as_slice());
                encoded32.as_ptr().cast::<u8>()
            } else {
                if let Some(gc) = self.gc {
                    let buffer = &instance_buffers.last().unwrap()[0].1;
                    let stride = gc.heap_words + gc.root_words;
                    let heap_words = unsafe {
                        std::slice::from_raw_parts_mut(
                            buffer.contents().cast::<u64>(),
                            stride * (end - start),
                        )
                    };
                    let mut arguments = Vec::new();
                    for (index, row) in inputs[start..end].iter().enumerate() {
                        let encoded = self
                            .encode_gc_row(row, self.exception_io)
                            .map_err(|e| shift_trap(e, start + index))?;
                        arguments.extend(encoded.arguments);
                        heap_words[index * stride..index * stride + gc.heap_words]
                            .copy_from_slice(&encoded.heap);
                        if self.exception_io.is_some_and(|io| io.input) {
                            exception_arenas.extend(encoded.exceptions);
                        }
                        exception_handles.push(encoded.existing_exceptions);
                    }
                    encoded64 = arguments;
                } else if let Some(io) = self.exception_io.filter(|io| io.input) {
                    let mut arguments = Vec::new();
                    for (index, row) in inputs[start..end].iter().enumerate() {
                        let (args, arena, handles) = self
                            .encode_exceptions(row, io)
                            .map_err(|error| shift_trap(error, start + index))?;
                        arguments.extend(args);
                        exception_arenas.extend(arena);
                        exception_handles.push(handles);
                    }
                    encoded64 = arguments;
                } else {
                    encoded64 = inputs[start..end]
                        .iter()
                        .flat_map(|row| row.iter().flat_map(|value| value.words()))
                        .collect::<Vec<_>>();
                }
                input_bytes = size_of_val(encoded64.as_slice());
                encoded64.as_ptr().cast::<u8>()
            };
            let arguments = unsafe { std::slice::from_raw_parts(input, input_bytes) };
            let mut exception_input_offset = 0;
            let packed = (memory_input.is_some() || !exception_arenas.is_empty()).then(|| {
                let mut packed = Vec::with_capacity((end - start) * input_stride);
                packed.extend_from_slice(arguments);
                if let Some(memory_input) = &memory_input {
                    for row in &memory_input.rows[start..end] {
                        packed.extend_from_slice(row);
                    }
                }
                if !exception_arenas.is_empty() {
                    // Memory input slices can have arbitrary byte lengths.
                    exception_input_offset = packed.len().next_multiple_of(size_of::<u64>());
                    packed.resize(exception_input_offset, 0);
                    for word in &exception_arenas {
                        packed.extend_from_slice(&word.to_le_bytes());
                    }
                }
                packed
            });
            let input = packed.as_deref().unwrap_or(arguments);
            let (output, _) = self
                .dispatch(
                    input,
                    (end - start) * transfer_output_stride,
                    (end - start) as u32,
                    &instance_buffers,
                    memory_input
                        .as_ref()
                        .map(|input| (capture.as_ref().unwrap().0, input.offset, input.width)),
                    exception_input_offset,
                )
                .map_err(|error| shift_trap(error, start))?;
            if fast_i32 {
                let values = unsafe {
                    std::slice::from_raw_parts(output.contents().cast::<u32>(), end - start)
                };
                rows.extend(values.iter().map(|&value| vec![Value::I32(value)]));
            } else {
                let values = unsafe {
                    std::slice::from_raw_parts(
                        output.contents().cast::<u64>(),
                        (end - start) * (transfer_output_stride / size_of::<u64>()),
                    )
                };
                let result_words = output_stride / size_of::<u64>();
                for index in 0..end - start {
                    let result = &values[index * result_words..(index + 1) * result_words];
                    if let Some(gc) = self.gc {
                        let buffer = &instance_buffers.last().unwrap()[0].1;
                        let stride = gc.heap_words + gc.root_words;
                        let heaps = unsafe {
                            std::slice::from_raw_parts(
                                buffer.contents().cast::<u64>(),
                                stride * (end - start),
                            )
                        };
                        let arena = if self.exception_io.is_some_and(|io| io.output) {
                            let offset = (end - start) * result_words + index * exception_words;
                            &values[offset..offset + exception_words]
                        } else {
                            &[]
                        };
                        rows.push(
                            self.decode_gc_row(
                                result,
                                &heaps[index * stride..index * stride + gc.heap_words],
                                arena,
                                self.exception_io,
                                exception_handles
                                    .get(index)
                                    .map(Vec::as_slice)
                                    .unwrap_or(&[]),
                            )?,
                        );
                        continue;
                    }
                    let handles = if let Some(io) = self.exception_io.filter(|io| io.output) {
                        let offset = (end - start) * result_words + index * exception_words;
                        self.decode_exceptions(
                            io,
                            &values[offset..offset + exception_words],
                            result,
                            exception_handles
                                .get(index)
                                .map(Vec::as_slice)
                                .unwrap_or(&[]),
                        )?
                    } else {
                        Vec::new()
                    };
                    let mut row = Vec::new();
                    let mut word = 0;
                    for &ty in &self.results {
                        row.push(self.decode_value(ty, &result[word..], &handles)?);
                        word += value_words(ty);
                    }
                    rows.push(row);
                }
            }
            if let Some(((owner, resource), range)) = &capture {
                let instance = &self.instances[*owner];
                let buffers = &instance_buffers[*owner];
                let buffer = |slot| &buffers.iter().find(|(index, _)| *index == slot).unwrap().1;
                let pages = unsafe {
                    std::slice::from_raw_parts(
                        buffer(instance.slots.memories[*resource].1)
                            .contents()
                            .cast::<u32>(),
                        end - start,
                    )
                };
                let stride = instance.template.memories[*resource].capacity;
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        buffer(instance.slots.memories[*resource].0)
                            .contents()
                            .cast::<u8>(),
                        stride * (end - start),
                    )
                };
                for (lane, &page_count) in pages.iter().enumerate() {
                    if range.end > page_count as usize * 65536 {
                        return Err(Error::Unsupported(format!(
                            "memory range exceeds memory size at input {}",
                            start + lane
                        )));
                    }
                    memories.push(
                        bytes[lane * stride + range.start..lane * stride + range.end].to_vec(),
                    );
                }
            }
        }
        Ok((rows, memories))
    }

    fn batch_limit(&self, input_stride: usize, output_stride: usize) -> Result<usize, Error> {
        let max_buffer = usize::try_from(self.device.max_buffer_length()).unwrap_or(usize::MAX);
        let mut limit = u32::MAX as usize;
        let mut per_lane = 0usize;
        let mut include = |stride: usize| -> Result<(), Error> {
            if stride > max_buffer {
                return Err(Error::Unsupported(
                    "single invocation exceeds the Metal buffer limit".into(),
                ));
            }
            if stride != 0 {
                limit = limit.min(max_buffer / stride);
                per_lane = per_lane.saturating_add(stride);
            }
            Ok(())
        };
        include(input_stride.max(4))?;
        include(output_stride.max(4))?;
        if self.can_trap {
            include(size_of::<u32>())?;
        }
        for instance in &self.instances {
            let template = &instance.template;
            for memory in &template.memories {
                include(memory.capacity.max(1))?;
                include(size_of::<u32>())?;
            }
            if instance.slots.globals.is_some() {
                include(template.globals_template.len() * size_of::<u64>())?;
            }
            for table in &template.tables {
                if table.info.mutable {
                    include(table.info.capacity.max(1) * size_of::<u64>())?;
                }
            }
        }
        if let Some(gc) = self.gc {
            include((gc.heap_words + gc.root_words) * 8)?;
        }
        Ok(limit.min((MAX_BATCH_BYTES / per_lane).max(1)))
    }

    fn prepare_batch(&self, count: u32) -> Result<Vec<Vec<(u64, metal::Buffer)>>, Error> {
        let mut buffers = self
            .instances
            .iter()
            .map(|instance| instance.for_batch(&self.device, count))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(gc) = self.gc {
            buffers.push(vec![(
                gc.slot,
                lane_buffer(
                    &self.device,
                    &[gc.heap_words as u64],
                    gc.heap_words + gc.root_words,
                    count,
                )?,
            )]);
        }
        Ok(buffers)
    }

    fn reset_batch(&self, buffers: &[Vec<(u64, metal::Buffer)>], count: u32) {
        for (instance, buffers) in self.instances.iter().zip(buffers) {
            instance.reset_batch(buffers, count);
        }
        if let Some(gc) = self.gc {
            reset_lane_buffer(
                &buffers.last().unwrap()[0].1,
                &[gc.heap_words as u64],
                gc.heap_words + gc.root_words,
                count,
            );
        }
    }

    fn dispatch(
        &self,
        input_data: &[u8],
        output_bytes: usize,
        count: u32,
        instance_buffers: &[Vec<(u64, metal::Buffer)>],
        memory_input: Option<((usize, usize), usize, usize)>,
        exception_input_offset: usize,
    ) -> Result<(metal::Buffer, Option<Duration>), Error> {
        autoreleasepool(|| {
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
            let ((owner, resource), offset, width) = memory_input.unwrap_or(((0, 0), 0, 0));
            let config = [
                count as u64,
                offset as u64,
                width as u64,
                owner as u64,
                resource as u64,
                exception_input_offset as u64,
            ];
            encoder.set_bytes(2, size_of_val(&config) as u64, config.as_ptr().cast());
            if let Some(traps) = &traps {
                encoder.set_buffer(3, Some(traps), 0);
            }
            for buffers in instance_buffers {
                for (slot, buffer) in buffers {
                    encoder.set_buffer(*slot, Some(buffer), 0);
                }
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
                let detail = unsafe {
                    let error: *mut Object = command
                        .send_message(Sel::register("error"), ())
                        .unwrap_or(std::ptr::null_mut());
                    if error.is_null() {
                        String::new()
                    } else {
                        let description: *mut Object = (&*error)
                            .send_message(Sel::register("localizedDescription"), ())
                            .unwrap_or(std::ptr::null_mut());
                        if description.is_null() {
                            String::new()
                        } else {
                            let bytes: *const std::ffi::c_char = (&*description)
                                .send_message(Sel::register("UTF8String"), ())
                                .unwrap_or(std::ptr::null());
                            if bytes.is_null() {
                                String::new()
                            } else {
                                format!(": {}", CStr::from_ptr(bytes).to_string_lossy())
                            }
                        }
                    }
                };
                return Err(Error::Metal(format!(
                    "Metal command failed: {:?}{detail}",
                    command.status(),
                )));
            }
            if let Some(traps) = &traps {
                let codes = unsafe {
                    std::slice::from_raw_parts(traps.contents().cast::<u32>(), count as usize)
                };
                if let Some((input_index, code)) =
                    codes.iter().enumerate().find(|(_, code)| **code != 0)
                {
                    if *code == 11 {
                        return Err(Error::UncaughtException { input_index });
                    }
                    let reason = match code {
                        1 => "integer divide by zero",
                        2 => "integer overflow",
                        3 => "unreachable executed",
                        4 => "out of bounds memory access",
                        5 => "invalid conversion to integer",
                        6 => "undefined or uninitialized table element",
                        7 => "indirect call type mismatch",
                        8 => "call depth limit exceeded",
                        9 => "null reference",
                        10 => "exception record limit exceeded",
                        15 => "GC resource limit exceeded",
                        16 => "out of bounds array access",
                        17 => "fuel exhausted",
                        12 => "null function reference",
                        13 => "cast failure",
                        14 => "out of bounds table access",
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
        })
    }
}

fn shift_trap(error: Error, offset: usize) -> Error {
    match error {
        Error::UncaughtException { input_index } => Error::UncaughtException {
            input_index: offset + input_index,
        },
        Error::Trap {
            input_index,
            reason,
        } => Error::Trap {
            input_index: offset + input_index,
            reason,
        },
        error => error,
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
    let bytes = bytes.max(4) as u64;
    if bytes > device.max_buffer_length() {
        return Err(Error::Unsupported(
            "batch exceeds the Metal buffer limit".into(),
        ));
    }
    let buffer = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    reset_lane_buffer(&buffer, template, capacity, count);
    Ok(buffer)
}

fn reset_lane_buffer<T: Copy + Default>(
    buffer: &metal::Buffer,
    template: &[T],
    capacity: usize,
    count: u32,
) {
    let stride = capacity.max(1);
    // The buffer was allocated for at least `count` lanes with this stride.
    let data = unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().cast::<T>(), stride * count as usize)
    };
    for lane in data.chunks_mut(stride) {
        lane.fill(T::default());
        lane[..template.len()].copy_from_slice(template);
    }
}

impl InstanceBuffers {
    fn for_batch(&self, device: &Device, count: u32) -> Result<Vec<(u64, metal::Buffer)>, Error> {
        let mut buffers = Vec::new();
        let template = &self.template;
        for (memory, &(slot, pages)) in template.memories.iter().zip(&self.slots.memories) {
            let initial = &memory.bytes;
            buffers.push((slot, lane_buffer(device, initial, memory.capacity, count)?));
            buffers.push((
                pages,
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
        for (index, (table, &slot)) in template.tables.iter().zip(&self.slots.tables).enumerate() {
            let buffer = if table.info.mutable {
                lane_buffer(device, &table.values, table.info.capacity, count)?
            } else {
                self.tables[index].as_ref().unwrap().clone()
            };
            buffers.push((slot, buffer));
        }
        if let Some(slot) = self.slots.passive_data {
            buffers.push((slot, self.passive_data.as_ref().unwrap().clone()));
        }
        Ok(buffers)
    }

    fn reset_batch(&self, buffers: &[(u64, metal::Buffer)], count: u32) {
        let buffer = |slot| &buffers.iter().find(|(index, _)| *index == slot).unwrap().1;
        let template = &self.template;
        for (memory, &(slot, pages)) in template.memories.iter().zip(&self.slots.memories) {
            let initial = &memory.bytes;
            reset_lane_buffer(buffer(slot), initial, memory.capacity, count);
            reset_lane_buffer(buffer(pages), &[initial.len() as u32 / 65536], 1, count);
        }
        if let Some(slot) = self.slots.globals {
            reset_lane_buffer(
                buffer(slot),
                &template.globals_template,
                template.globals_template.len(),
                count,
            );
        }
        for (table, &slot) in template.tables.iter().zip(&self.slots.tables) {
            if table.info.mutable {
                reset_lane_buffer(buffer(slot), &table.values, table.info.capacity, count);
            }
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWasm(error) => write!(formatter, "invalid WebAssembly: {error}"),
            Self::Unsupported(reason) => write!(formatter, "unsupported WebAssembly: {reason}"),
            Self::Metal(reason) => write!(formatter, "Metal error: {reason}"),
            Self::UncaughtException { input_index } => write!(
                formatter,
                "uncaught WebAssembly exception at input {input_index}"
            ),
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

// Reuse the validator's canonical interner across providers. Recursive group
// identity and declared function subtyping must survive module-local indices.
#[derive(Default)]
struct TypeContext {
    validator: Validator,
    next_tag: u32,
}

fn normalize_ref(ty: RefType, ids: &[CoreTypeId]) -> RefType {
    match ty.heap_type() {
        HeapType::Concrete(UnpackedIndex::Module(index)) => RefType::new(
            ty.is_nullable(),
            HeapType::Concrete(UnpackedIndex::Id(ids[index as usize])),
        )
        .unwrap(),
        _ => ty,
    }
}

fn normalize_value(ty: ValType, ids: &[CoreTypeId]) -> ValType {
    match ty {
        ValType::Ref(ty) => ValType::Ref(normalize_ref(ty, ids)),
        _ => ty,
    }
}

fn ref_subtype(actual: RefType, expected: RefType, parents: &ReferenceTypes) -> bool {
    if actual.is_nullable() && !expected.is_nullable() {
        return false;
    }
    if actual.heap_type() == expected.heap_type() {
        return true;
    }
    match (actual.heap_type(), expected.heap_type()) {
        (
            HeapType::Concrete(UnpackedIndex::Id(mut id)),
            HeapType::Concrete(UnpackedIndex::Id(target)),
        ) => {
            while let Some((Some(parent), _)) = parents.get(&id) {
                if *parent == target {
                    return true;
                }
                id = *parent;
            }
            false
        }
        (
            _,
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Func,
            },
        ) => is_function_ref_in(actual, parents),
        (
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::NoFunc,
            },
            _,
        ) => is_function_ref_in(expected, parents),
        (
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::NoExtern,
            },
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Extern,
            },
        ) => true,
        (
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::NoExn,
            },
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Exn,
            },
        ) => true,
        (
            actual,
            HeapType::Abstract {
                shared: false,
                ty: expected,
            },
        ) => {
            let actual = match actual {
                HeapType::Concrete(UnpackedIndex::Id(id)) => {
                    parents.get(&id).map(|(_, kind)| *kind)
                }
                HeapType::Abstract { shared: false, ty } => Some(ty),
                _ => None,
            };
            match (actual, expected) {
                (Some(actual), expected) if actual == expected => true,
                (
                    Some(AbstractHeapType::None),
                    AbstractHeapType::Any
                    | AbstractHeapType::Eq
                    | AbstractHeapType::I31
                    | AbstractHeapType::Struct
                    | AbstractHeapType::Array,
                ) => true,
                (
                    Some(
                        AbstractHeapType::Eq
                        | AbstractHeapType::I31
                        | AbstractHeapType::Struct
                        | AbstractHeapType::Array,
                    ),
                    AbstractHeapType::Any,
                ) => true,
                (
                    Some(
                        AbstractHeapType::I31 | AbstractHeapType::Struct | AbstractHeapType::Array,
                    ),
                    AbstractHeapType::Eq,
                ) => true,
                _ => false,
            }
        }
        (
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::None,
            },
            HeapType::Concrete(UnpackedIndex::Id(id)),
        ) => parents
            .get(&id)
            .is_some_and(|(_, kind)| *kind != AbstractHeapType::Func),
        _ => false,
    }
}

fn is_function_ref(ty: RefType) -> bool {
    matches!(
        ty.heap_type(),
        HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Func | AbstractHeapType::NoFunc
        }
    )
}

fn is_function_ref_in(ty: RefType, kinds: &ReferenceTypes) -> bool {
    is_function_ref(ty)
        || matches!(ty.heap_type(), HeapType::Concrete(UnpackedIndex::Id(id)) if kinds.get(&id).is_some_and(|(_, k)| *k == AbstractHeapType::Func))
}
fn is_gc_ref(ty: RefType, kinds: &ReferenceTypes) -> bool {
    !is_function_ref_in(ty, kinds) && !is_external_ref(ty) && !is_exception_ref(ty)
}

fn is_external_ref(ty: RefType) -> bool {
    matches!(
        ty.heap_type(),
        HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Extern | AbstractHeapType::NoExtern
        }
    )
}

fn is_exception_ref(ty: RefType) -> bool {
    matches!(
        ty.heap_type(),
        HeapType::Abstract {
            shared: false,
            ty: AbstractHeapType::Exn | AbstractHeapType::NoExn
        }
    )
}

type ReferenceTypes = HashMap<CoreTypeId, (Option<CoreTypeId>, AbstractHeapType)>;

#[derive(Clone)]
struct GcType {
    fields: Vec<StorageType>,
    array: bool,
    reference: RefType,
}
impl GcType {
    fn words(&self) -> usize {
        self.fields.iter().map(storage_words).sum()
    }
}
fn storage_words(ty: &StorageType) -> usize {
    match ty {
        StorageType::Val(ValType::V128) => 2,
        _ => 1,
    }
}
fn storage_is_ref(ty: &StorageType) -> bool {
    matches!(ty, StorageType::Val(ValType::Ref(r)) if !is_function_ref(*r))
}
#[derive(Clone, Copy)]
struct GcLayout {
    slot: u64,
    heap_words: usize,
    root_words: usize,
}

struct ParsedModule<'a> {
    uses_exceptions: bool,
    types: Vec<FuncType>,
    type_ids: Vec<CoreTypeId>,
    supertypes: ReferenceTypes,
    gc_types: Vec<Option<GcType>>,
    gc_ids: Vec<u32>,
    tag_types: Vec<u32>,
    tag_bindings: Vec<u32>,
    function_types: Vec<u32>,
    bodies: Vec<FunctionBody<'a>>,
    exports: HashMap<String, (ExternalKind, usize)>,
    imports: Vec<(String, String)>,
    all_imports: Vec<(String, String, TypeRef)>,
    resolved_imports: Vec<(usize, usize)>,
    resolved_functions: Vec<usize>,
    memory_types: Vec<MemoryType>,
    memory_bindings: Vec<Option<(usize, usize)>>,
    table_types: Vec<TableType>,
    table_bindings: Vec<Option<(usize, usize)>>,
    global_types: Vec<GlobalType>,
    global_bindings: Vec<(usize, usize)>,
    local_global_slots: Vec<usize>,
    dynamic_globals: Vec<(usize, ValType, ConstExpr<'a>)>,
    active_elements: Vec<(usize, usize, ConstExpr<'a>)>,
    active_data: Vec<(usize, ConstExpr<'a>, usize, usize)>,
    table_init_exprs: Vec<(usize, ConstExpr<'a>)>,
    dynamic_elements: Vec<(usize, usize, ConstExpr<'a>)>,
    start_index: Option<usize>,
    globals_mutable: Vec<bool>,
    reference_globals: Vec<usize>,
    data_offsets: Vec<usize>,
    instance: InstanceTemplate,
}

impl ParsedModule<'_> {
    fn accepts_function(&self, expected: usize, actual: &Self, function: usize) -> bool {
        let mut id = actual.type_ids[actual.function_types[function] as usize];
        loop {
            if id == self.type_ids[expected] {
                return true;
            }
            match actual.supertypes.get(&id).and_then(|(p, _)| *p) {
                Some(parent) => id = parent,
                None => return false,
            }
        }
    }

    fn function_type(&self, index: usize) -> Result<&FuncType, Error> {
        self.function_types
            .get(index)
            .and_then(|index| self.types.get(*index as usize))
            .ok_or_else(|| Error::Unsupported("function type is out of range".into()))
    }
}

fn parse_module<'a>(
    wasm: &'a [u8],
    limits: KernelLimits,
    context: &mut TypeContext,
) -> Result<ParsedModule<'a>, Error> {
    context.validator.reset();
    let validated = context.validator.validate_all(wasm)?;
    let validated = validated.as_ref();
    let type_ids: Vec<_> = (0..validated.core_type_count_in_module())
        .map(|i| validated.core_type_at_in_module(i))
        .collect();
    let mut supertypes = HashMap::new();
    let mut types = Vec::new();
    let mut gc_types = Vec::new();
    for &id in &type_ids {
        let subtype = &validated[id];
        if subtype.composite_type.shared {
            return Err(Error::Unsupported("shared types are unsupported".into()));
        }
        let reference = RefType::new(false, HeapType::Concrete(UnpackedIndex::Id(id))).unwrap();
        let (function, gc, kind) = match &subtype.composite_type.inner {
            CompositeInnerType::Func(ty) => (ty.clone(), None, AbstractHeapType::Func),
            // Preserve module type indices while storing GC layouts separately.
            CompositeInnerType::Struct(ty) => (
                FuncType::new([], []),
                Some(GcType {
                    fields: ty.fields.iter().map(|f| f.element_type).collect(),
                    array: false,
                    reference,
                }),
                AbstractHeapType::Struct,
            ),
            CompositeInnerType::Array(ty) => (
                FuncType::new([], []),
                Some(GcType {
                    fields: vec![ty.0.element_type],
                    array: true,
                    reference,
                }),
                AbstractHeapType::Array,
            ),
            _ => {
                return Err(Error::Unsupported(
                    "continuation types are unsupported".into(),
                ));
            }
        };
        types.push(function);
        gc_types.push(gc);
        supertypes.insert(id, (validated.supertype_of(id), kind));
    }
    let mut tag_types = Vec::new();
    let mut tag_bindings = Vec::new();
    let mut function_types = Vec::new();
    let mut bodies = Vec::new();
    let mut exports = HashMap::new();
    let mut imports = Vec::new();
    let mut all_imports = Vec::new();
    let mut start_index = None;
    let mut memories = Vec::new();
    let mut memory_types = Vec::new();
    let mut memory_bindings = Vec::new();
    let mut globals_template = Vec::new();
    let mut globals_mutable = Vec::new();
    let mut global_types = Vec::new();
    let mut local_global_slots = Vec::new();
    let mut imported_globals = 0;
    let mut dynamic_globals = Vec::new();
    let mut active_elements = Vec::new();
    let mut active_data = Vec::new();
    let mut table_init_exprs = Vec::new();
    let mut dynamic_elements = Vec::new();
    let mut reference_globals = Vec::new();
    let mut tables = Vec::new();
    let mut table_types = Vec::new();
    let mut table_bindings = Vec::new();
    let mut element_info = TableInfo::default();
    let mut passive_data = Vec::new();
    let mut data_offsets = Vec::new();
    let mut data_lengths = Vec::new();
    for payload in Parser::new(0).parse_all(wasm) {
        match payload? {
            Payload::Version { .. } | Payload::CustomSection(_) | Payload::End(_) => {}
            Payload::TypeSection(_) => {}
            Payload::ImportSection(section) => {
                for import in section.into_imports() {
                    let mut import = import?;
                    import.ty = match import.ty {
                        TypeRef::Global(mut ty) => {
                            ty.content_type = normalize_value(ty.content_type, &type_ids);
                            TypeRef::Global(ty)
                        }
                        TypeRef::Table(mut ty) => {
                            ty.element_type = normalize_ref(ty.element_type, &type_ids);
                            TypeRef::Table(ty)
                        }
                        ty => ty,
                    };
                    all_imports.push((import.module.to_owned(), import.name.to_owned(), import.ty));
                    match import.ty {
                        TypeRef::Func(ty) => {
                            function_types.push(ty);
                            imports.push((import.module.to_owned(), import.name.to_owned()));
                        }
                        TypeRef::Memory(ty) => {
                            if ty.shared || ty.page_size_log2() != 16 {
                                return Err(Error::Unsupported(
                                    "only unshared 64 KiB page memories are supported".into(),
                                ));
                            }
                            memory_types.push(ty);
                            memory_bindings.push(None);
                        }
                        TypeRef::Table(ty) => {
                            if !is_supported_value(ValType::Ref(ty.element_type)) || ty.shared {
                                return Err(Error::Unsupported(
                                    "unsupported table reference type".into(),
                                ));
                            }
                            table_types.push(ty);
                            table_bindings.push(None);
                        }
                        TypeRef::Tag(ty) => {
                            tag_types.push(ty.func_type_idx);
                            tag_bindings.push(0);
                        }
                        TypeRef::Global(ty) => {
                            if !is_supported_value(ty.content_type) || ty.shared {
                                return Err(Error::Unsupported(
                                    "only unshared numeric, function, external, and exception reference globals are supported"
                                        .into(),
                                ));
                            }
                            global_types.push(ty);
                            imported_globals += 1;
                            globals_mutable.push(ty.mutable);
                        }
                        _ => {
                            return Err(Error::Unsupported(format!(
                                "import {}.{} has an unsupported type",
                                import.module, import.name
                            )));
                        }
                    }
                }
            }
            Payload::FunctionSection(section) => {
                function_types.extend(section.into_iter().collect::<Result<Vec<_>, _>>()?);
            }
            Payload::TableSection(section) => {
                for table in section {
                    let mut table = table?;
                    table.ty.element_type = normalize_ref(table.ty.element_type, &type_ids);
                    if !is_supported_value(ValType::Ref(table.ty.element_type)) || table.ty.shared {
                        return Err(Error::Unsupported(
                            "unsupported table reference type".into(),
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
                        TableInit::Expr(expr) if imported_globals == 0 && !expr_uses_gc(&expr) => {
                            eval_const(&expr, &globals_template, &local_global_slots)?
                        }
                        TableInit::Expr(expr) => {
                            table_init_exprs.push((tables.len(), expr));
                            0
                        }
                    };
                    let capacity = size.saturating_add(limits.table_growth_elements).min(
                        table
                            .ty
                            .maximum
                            .unwrap_or(u64::MAX)
                            .min(MAX_TABLE_ELEMENTS as u64) as usize,
                    );
                    table_types.push(table.ty);
                    table_bindings.push(None);
                    tables.push(TableTemplate {
                        function_refs: is_function_ref_in(table.ty.element_type, &supertypes),
                        values: vec![value; size],
                        info: TableInfo {
                            capacity,
                            ..Default::default()
                        },
                        targets: Vec::new(),
                    });
                }
            }
            Payload::MemorySection(section) => {
                for memory in section {
                    let memory = memory?;
                    if memory.shared || memory.page_size_log2() != 16 {
                        return Err(Error::Unsupported(
                            "only unshared 64 KiB page memories are supported".into(),
                        ));
                    }
                    let bytes = memory
                        .initial
                        .checked_mul(65536)
                        .and_then(|bytes| usize::try_from(bytes).ok())
                        .ok_or_else(|| Error::Unsupported("memory size overflow".into()))?;
                    if bytes > limits.memory_bytes
                        || memories
                            .iter()
                            .map(|memory: &MemoryTemplate| memory.bytes.len())
                            .sum::<usize>()
                            > limits.memory_bytes - bytes
                    {
                        return Err(Error::Unsupported(
                            "initial memory exceeds the configured per-lane limit".into(),
                        ));
                    }
                    memory_types.push(memory);
                    memory_bindings.push(None);
                    memories.push(MemoryTemplate {
                        bytes: vec![0; bytes],
                        capacity: 0,
                        max_pages: memory.maximum,
                    });
                }
            }
            Payload::GlobalSection(section) => {
                for global in section {
                    let mut global = global?;
                    global.ty.content_type = normalize_value(global.ty.content_type, &type_ids);
                    if !is_supported_value(global.ty.content_type) || global.ty.shared {
                        return Err(Error::Unsupported(
                            "only unshared numeric, function, external, and exception reference globals are supported".into(),
                        ));
                    }
                    let slot = globals_template.len();
                    local_global_slots.push(slot);
                    if imported_globals == 0
                        && dynamic_globals.is_empty()
                        && !matches!(global.ty.content_type, ValType::Ref(r) if is_gc_ref(r, &supertypes))
                        && !expr_uses_gc(&global.init_expr)
                    {
                        if global.ty.content_type == ValType::V128 {
                            globals_template.extend(eval_v128_const(
                                &global.init_expr,
                                &globals_template,
                                &local_global_slots,
                            )?);
                        } else {
                            globals_template.push(eval_const(
                                &global.init_expr,
                                &globals_template,
                                &local_global_slots,
                            )?);
                        }
                    } else {
                        globals_template.push(0);
                        if global.ty.content_type == ValType::V128 {
                            globals_template.push(0);
                        }
                        dynamic_globals.push((slot, global.ty.content_type, global.init_expr));
                    }
                    globals_mutable.push(global.ty.mutable);
                    global_types.push(global.ty);
                    if imported_globals == 0
                        && matches!(global.ty.content_type, ValType::Ref(ty) if is_function_ref_in(ty, &supertypes))
                    {
                        reference_globals.push(slot);
                    }
                }
            }
            Payload::ElementSection(section) => {
                for element in section {
                    let element = element?;
                    element_info.element_types.push(match &element.items {
                        ElementItems::Functions(_) => RefType::FUNCREF,
                        ElementItems::Expressions(ty, _) => *ty,
                    });
                    let function_elements = match &element.items {
                        ElementItems::Functions(_) => true,
                        ElementItems::Expressions(ty, _) => {
                            is_function_ref_in(normalize_ref(*ty, &type_ids), &supertypes)
                        }
                    };
                    let values: Vec<u64> = match element.items {
                        ElementItems::Functions(indices) => indices
                            .into_iter()
                            .map(|index| {
                                u64::from(index?).checked_add(1).ok_or_else(|| {
                                    Error::Unsupported("function index overflow".into())
                                })
                            })
                            .collect::<Result<_, Error>>()?,
                        ElementItems::Expressions(ty, expressions)
                            if is_supported_value(ValType::Ref(ty)) =>
                        {
                            let mut values = Vec::new();
                            for expr in expressions {
                                let expr = expr?;
                                if expr_uses_gc(&expr)
                                    || (imported_globals != 0 || !dynamic_globals.is_empty())
                                        && expr
                                            .get_operators_reader()
                                            .into_iter()
                                            .any(|op| matches!(op, Ok(Operator::GlobalGet { .. })))
                                {
                                    dynamic_elements.push((
                                        element_info.elements.len(),
                                        values.len(),
                                        expr,
                                    ));
                                    values.push(0);
                                } else {
                                    values.push(eval_const(
                                        &expr,
                                        &globals_template,
                                        &local_global_slots,
                                    )?);
                                }
                            }
                            values
                        }
                        _ => {
                            return Err(Error::Unsupported(
                                "unsupported element reference type".into(),
                            ));
                        }
                    };
                    let length = match element.kind {
                        ElementKind::Active {
                            table_index,
                            offset_expr,
                        } => {
                            let table_index = table_index.unwrap_or(0) as usize;
                            let table_type = table_types.get(table_index).ok_or_else(|| {
                                Error::Unsupported("element table is out of range".into())
                            })?;
                            let mut initialized = false;
                            let imported = table_types.len() - tables.len();
                            if let Some(table) = table_index
                                .checked_sub(imported)
                                .and_then(|index| tables.get_mut(index))
                                .filter(|_| {
                                    imported_globals == 0
                                        && dynamic_globals.is_empty()
                                        && !dynamic_elements.iter().any(|(segment, _, _)| {
                                            *segment == element_info.elements.len()
                                        })
                                })
                            {
                                let offset = eval_const(
                                    &offset_expr,
                                    &globals_template,
                                    &local_global_slots,
                                )?;
                                let offset = if table_type.table64 {
                                    offset
                                } else {
                                    offset as u32 as u64
                                };
                                if let Some(end) = usize::try_from(offset).ok().and_then(|offset| {
                                    offset
                                        .checked_add(values.len())
                                        .filter(|&end| end <= table.values.len())
                                }) {
                                    table.values[end - values.len()..end].copy_from_slice(&values);
                                    initialized = true;
                                }
                            }
                            if !initialized {
                                active_elements.push((
                                    element_info.elements.len(),
                                    table_index,
                                    offset_expr,
                                ));
                            }
                            0
                        }
                        ElementKind::Passive => values.len() as u32,
                        ElementKind::Declared => 0,
                    };
                    element_info.lengths.push(length);
                    element_info.function_elements.push(function_elements);
                    element_info.elements.push(values);
                }
            }
            Payload::DataSection(section) => {
                for data in section {
                    let data = data?;
                    match data.kind {
                        DataKind::Active {
                            memory_index,
                            offset_expr,
                        } => {
                            let memory_index = memory_index as usize;
                            let memory_type = memory_types.get(memory_index).ok_or_else(|| {
                                Error::Unsupported("data memory is out of range".into())
                            })?;
                            let mut initialized = false;
                            let imported = memory_types.len() - memories.len();
                            if let Some(memory) = memory_index
                                .checked_sub(imported)
                                .and_then(|index| memories.get_mut(index))
                                .filter(|_| imported_globals == 0 && dynamic_globals.is_empty())
                            {
                                let offset = eval_const(
                                    &offset_expr,
                                    &globals_template,
                                    &local_global_slots,
                                )?;
                                let offset = if memory_type.memory64 {
                                    offset
                                } else {
                                    offset as u32 as u64
                                };
                                if let Some(end) = usize::try_from(offset).ok().and_then(|offset| {
                                    offset
                                        .checked_add(data.data.len())
                                        .filter(|&end| end <= memory.bytes.len())
                                }) {
                                    memory.bytes[end - data.data.len()..end]
                                        .copy_from_slice(data.data);
                                    initialized = true;
                                }
                            }
                            if initialized {
                                data_offsets.push(0);
                            } else {
                                let offset = passive_data.len();
                                active_data.push((
                                    memory_index,
                                    offset_expr,
                                    offset,
                                    data.data.len(),
                                ));
                                data_offsets.push(offset);
                                passive_data.extend_from_slice(data.data);
                            }
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
                    }
                }
            }
            Payload::DataCountSection { .. } => {}
            Payload::TagSection(section) => {
                for tag in section {
                    tag_types.push(tag?.func_type_idx);
                    context.next_tag = context
                        .next_tag
                        .checked_add(1)
                        .ok_or_else(|| Error::Unsupported("too many tags".into()))?;
                    tag_bindings.push(context.next_tag);
                }
            }
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

    let uses_exceptions = !tag_types.is_empty()
        || types.iter().any(|ty| {
            ty.params()
                .iter()
                .chain(ty.results())
                .any(|ty| matches!(ty, ValType::Ref(ty) if is_exception_ref(*ty)))
        })
        || bodies.iter().any(|b| {
            b.get_operators_reader()
                .unwrap()
                .into_iter()
                .any(|op| matches!(op, Ok(Operator::ThrowRef | Operator::TryTable { .. })))
        });
    Ok(ParsedModule {
        uses_exceptions,
        types,
        gc_ids: vec![0; type_ids.len()],
        gc_types,
        type_ids,
        supertypes,
        tag_types,
        tag_bindings,
        function_types,
        bodies,
        exports,
        imports,
        all_imports,
        start_index,
        globals_mutable,
        global_types,
        global_bindings: Vec::new(),
        local_global_slots,
        dynamic_globals,
        active_elements,
        active_data,
        table_init_exprs,
        dynamic_elements,
        reference_globals,
        memory_types,
        memory_bindings,
        table_types,
        table_bindings,
        data_offsets,
        resolved_functions: Vec::new(),
        resolved_imports: Vec::new(),
        instance: InstanceTemplate {
            memories,
            globals_template,
            tables,
            element_info,
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
    limits: KernelLimits,
    context: &mut TypeContext,
) -> Result<(), Error> {
    visiting[index] = true;
    let imported_globals = modules[index]
        .all_imports
        .iter()
        .filter(|(_, _, ty)| matches!(ty, TypeRef::Global(_)))
        .count();
    modules[index].global_bindings = vec![(usize::MAX, usize::MAX); imported_globals];
    let local_slots = modules[index].local_global_slots.clone();
    modules[index]
        .global_bindings
        .extend(local_slots.into_iter().map(|slot| (index, slot)));
    let local_memories = modules[index].instance.memories.len();
    let memory_imports = modules[index].memory_types.len() - local_memories;
    for local in 0..local_memories {
        modules[index].memory_bindings[memory_imports + local] = Some((index, local));
    }
    let local_tables = modules[index].instance.tables.len();
    let table_imports = modules[index].table_types.len() - local_tables;
    for local in 0..local_tables {
        modules[index].table_bindings[table_imports + local] = Some((index, local));
    }
    let links = modules[index].all_imports.clone();
    let mut global_import = 0;
    let mut memory_import = 0;
    let mut table_import = 0;
    let mut tag_import = 0;
    for (namespace, name, ty) in links {
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
            modules.push(parse_module(bytes, limits, context)?);
            namespaces.insert(namespace.clone(), provider);
            visiting.push(false);
            load_dependencies(
                provider, modules, providers, namespaces, visiting, postorder, limits, context,
            )?;
            provider
        };
        let &(kind, exported) = modules[provider].exports.get(&name).ok_or_else(|| {
            Error::Unsupported(format!("provider {namespace:?} has no export {name:?}"))
        })?;
        match ty {
            TypeRef::Func(_) if kind == ExternalKind::Func => {
                let import_index = modules[index].resolved_imports.len();
                let (mut owner, mut function) = (provider, exported);
                while function < modules[owner].imports.len() {
                    (owner, function) = modules[owner].resolved_imports[function];
                }
                if !modules[index].accepts_function(
                    modules[index].function_types[import_index] as usize,
                    &modules[owner],
                    function,
                ) {
                    return Err(Error::Unsupported(format!(
                        "signature mismatch for import {namespace}.{name}"
                    )));
                }
                modules[index].resolved_imports.push((provider, exported));
            }
            TypeRef::Memory(expected) if kind == ExternalKind::Memory => {
                let (owner, local) = modules[provider].memory_bindings[exported]
                    .expect("resolved memory definition");
                let owner = &modules[owner];
                let imported = owner.memory_types.len() - owner.instance.memories.len();
                let actual = owner.memory_types[imported + local];
                if !limits_match(
                    actual.initial,
                    actual.maximum,
                    expected.initial,
                    expected.maximum,
                ) || actual.memory64 != expected.memory64
                    || actual.shared != expected.shared
                    || actual.page_size_log2() != expected.page_size_log2()
                {
                    return Err(Error::Unsupported(format!(
                        "memory type mismatch for import {namespace}.{name}"
                    )));
                }
                modules[index].memory_bindings[memory_import] =
                    modules[provider].memory_bindings[exported];
                memory_import += 1;
            }
            TypeRef::Table(expected) if kind == ExternalKind::Table => {
                let (owner, local) =
                    modules[provider].table_bindings[exported].expect("resolved table definition");
                let owner = &modules[owner];
                let imported = owner.table_types.len() - owner.instance.tables.len();
                let actual = owner.table_types[imported + local];
                if !limits_match(
                    actual.initial,
                    actual.maximum,
                    expected.initial,
                    expected.maximum,
                ) || actual.element_type != expected.element_type
                    || actual.table64 != expected.table64
                    || actual.shared != expected.shared
                {
                    return Err(Error::Unsupported(format!(
                        "table type mismatch for import {namespace}.{name}"
                    )));
                }
                modules[index].table_bindings[table_import] =
                    modules[provider].table_bindings[exported];
                table_import += 1;
            }
            TypeRef::Global(expected) if kind == ExternalKind::Global => {
                let binding = modules[provider].global_bindings[exported];
                let owner = &modules[binding.0];
                let definition = owner
                    .global_bindings
                    .iter()
                    .position(|&candidate| candidate == binding)
                    .expect("resolved global definition");
                let actual = owner.global_types[definition];
                let matches = actual.mutable == expected.mutable
                    && actual.shared == expected.shared
                    && (actual.content_type == expected.content_type
                        || (!expected.mutable
                            && match (actual.content_type, expected.content_type) {
                                (ValType::Ref(actual), ValType::Ref(expected)) => {
                                    ref_subtype(actual, expected, &owner.supertypes)
                                }
                                _ => false,
                            }));
                if !matches {
                    return Err(Error::Unsupported(format!(
                        "global type mismatch for import {namespace}.{name}"
                    )));
                }
                modules[index].global_bindings[global_import] = binding;
                global_import += 1;
            }
            TypeRef::Tag(expected) if kind == ExternalKind::Tag => {
                if modules[index].type_ids[expected.func_type_idx as usize]
                    != modules[provider].type_ids[modules[provider].tag_types[exported] as usize]
                {
                    return Err(Error::Unsupported(format!(
                        "tag type mismatch for import {namespace}.{name}"
                    )));
                }
                modules[index].tag_bindings[tag_import] = modules[provider].tag_bindings[exported];
                tag_import += 1;
            }
            TypeRef::Func(_) => {
                return Err(Error::Unsupported(format!(
                    "provider export {namespace}.{name} is not a function"
                )));
            }
            _ => {
                return Err(Error::Unsupported(format!(
                    "provider export {namespace}.{name} has the wrong kind"
                )));
            }
        }
    }
    debug_assert_eq!(global_import, imported_globals);
    visiting[index] = false;
    postorder.push(index);
    Ok(())
}

fn limits_match(
    actual_min: u64,
    actual_max: Option<u64>,
    expected_min: u64,
    expected_max: Option<u64>,
) -> bool {
    actual_min >= expected_min
        && expected_max.is_none_or(|max| actual_max.is_some_and(|actual| actual <= max))
}

fn remap_reference(value: u64, functions: &[usize]) -> Result<u64, Error> {
    if value == 0 {
        return Ok(0);
    }
    let index = *functions
        .get(value as usize - 1)
        .ok_or_else(|| Error::Unsupported("function reference is out of range".into()))?;
    u64::try_from(index)
        .ok()
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| Error::Unsupported("too many linked functions".into()))
}

fn translate_with_limits(
    wasm: &[u8],
    export_name: &str,
    providers: &[(&str, &[u8])],
    limits: KernelLimits,
) -> Result<Translation, Error> {
    let mut provider_map = HashMap::new();
    for &(namespace, bytes) in providers {
        if provider_map.insert(namespace, bytes).is_some() {
            return Err(Error::Unsupported(format!(
                "duplicate Wasm provider {namespace:?}"
            )));
        }
    }
    let mut context = TypeContext::default();
    let mut modules = vec![parse_module(wasm, limits, &mut context)?];
    let mut postorder = Vec::new();
    load_dependencies(
        0,
        &mut modules,
        &provider_map,
        &mut HashMap::new(),
        &mut vec![false],
        &mut postorder,
        limits,
        &mut context,
    )?;
    translate_modules(
        modules,
        postorder,
        context.next_tag as usize + 1,
        0,
        Some(export_name),
        limits,
        false,
    )
}

fn translate_modules(
    mut modules: Vec<ParsedModule<'_>>,
    postorder: Vec<usize>,
    tag_count: usize,
    root: usize,
    export_name: Option<&str>,
    limits: KernelLimits,
    persistent: bool,
) -> Result<Translation, Error> {
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
    let mut gc_types = Vec::new();
    let mut gc_type_ids = HashMap::new();
    for module in &mut modules {
        for (index, ty) in module.gc_types.iter().enumerate() {
            if let Some(ty) = ty {
                let next = gc_types.len() as u32 + 1;
                let id = *gc_type_ids
                    .entry(module.type_ids[index])
                    .or_insert_with(|| {
                        gc_types.push(ty.clone());
                        next
                    });
                module.gc_ids[index] = id;
            }
        }
    }
    let uses_gc = modules.iter().any(|m| m.uses_exceptions)
        || !gc_types.is_empty()
        || modules.iter().any(|m| {
            let gc_ref = |r| is_gc_ref(normalize_ref(r, &m.type_ids), &m.supertypes);
            m.types
                .iter()
                .flat_map(|t| t.params().iter().chain(t.results()))
                .chain(m.global_types.iter().map(|g| &g.content_type))
                .any(|t| matches!(t, ValType::Ref(r) if gc_ref(*r)))
                || m.table_types.iter().any(|t| gc_ref(t.element_type))
                || m.bodies.iter().any(|b| {
                    b.get_locals_reader()
                        .unwrap()
                        .into_iter()
                        .any(|local| matches!(local, Ok((_,ValType::Ref(r))) if gc_ref(r)))
                        || b.get_operators_reader()
                            .unwrap()
                            .into_iter()
                            .any(|op| match op {
                                Ok(
                                    Operator::RefI31
                                    | Operator::RefEq
                                    | Operator::AnyConvertExtern
                                    | Operator::ExternConvertAny,
                                ) => true,
                                Ok(
                                    Operator::RefNull { hty }
                                    | Operator::RefTestNonNull { hty }
                                    | Operator::RefTestNullable { hty }
                                    | Operator::RefCastNonNull { hty }
                                    | Operator::RefCastNullable { hty },
                                ) => gc_ref(RefType::new(true, hty).unwrap()),
                                Ok(
                                    Operator::BrOnCast { to_ref_type, .. }
                                    | Operator::BrOnCastFail { to_ref_type, .. },
                                ) => gc_ref(to_ref_type),
                                _ => false,
                            })
                })
        });
    let index = if let Some(export_name) = export_name {
        let &(kind, local) = modules[root].exports.get(export_name).ok_or_else(|| {
            Error::Unsupported(format!("export {export_name:?} is not a function"))
        })?;
        if kind != ExternalKind::Func {
            return Err(Error::Unsupported(format!(
                "export {export_name:?} is not a function"
            )));
        }
        Some(modules[root].resolved_functions[local])
    } else {
        None
    };
    let ty = index
        .map(|i| global_types[i].clone())
        .unwrap_or_else(|| FuncType::new([], []));
    if ty
        .params()
        .iter()
        .chain(ty.results())
        .any(|ty| !is_supported_value(*ty))
    {
        return Err(Error::Unsupported(
            "unsupported exported signature value type".into(),
        ));
    }
    // Mutable tables can receive references returned by functions in other modules.
    let mut referenced = Vec::new();
    // The host can inject exported handles into tables through reference arguments.
    // Include only functions whose nominal type is accepted by an input parameter.
    let mut input_types = ty.params().to_vec();
    if input_types
        .iter()
        .any(|ty| matches!(ty, ValType::Ref(ty) if is_exception_ref(*ty)))
    {
        input_types.extend(modules.iter().flat_map(|module| {
            module
                .tag_types
                .iter()
                .flat_map(|&t| module.types[t as usize].params().iter().copied())
        }));
    }
    for &(kind, local) in modules[root].exports.values() {
        if kind != ExternalKind::Func {
            continue;
        }
        let function = modules[root].resolved_functions[local];
        let (owner, local) = functions[function];
        let actual = &modules[owner];
        let reference = RefType::new(
            false,
            HeapType::Concrete(UnpackedIndex::Id(
                actual.type_ids[actual.function_types[local] as usize],
            )),
        )
        .unwrap();
        if input_types.iter().any(|ty| matches!(ty, ValType::Ref(expected) if ref_subtype(reference, *expected, &actual.supertypes))) {
            referenced.push(function);
        }
    }
    let wasm_reference_start = referenced.len();
    for module in &mut modules {
        let instance = &mut module.instance;
        for table in &mut instance.tables {
            if !table.function_refs {
                continue;
            }
            for value in &mut table.values {
                *value = remap_reference(*value, &module.resolved_functions)?;
                if *value != 0 {
                    referenced.push((*value - 1) as usize);
                }
            }
        }
        for (index, segment) in instance.element_info.elements.iter_mut().enumerate() {
            if !instance.element_info.function_elements[index] {
                continue;
            }
            for value in segment {
                *value = remap_reference(*value, &module.resolved_functions)?;
                if *value != 0 {
                    referenced.push((*value - 1) as usize);
                }
            }
        }
        for &global in &module.reference_globals {
            let value = remap_reference(
                instance.globals_template[global],
                &module.resolved_functions,
            )?;
            instance.globals_template[global] = value;
            if value != 0 {
                referenced.push((value - 1) as usize);
            }
        }
        for (_, _, expr) in &module.dynamic_globals {
            for op in expr.get_operators_reader() {
                if let Operator::RefFunc { function_index } = op? {
                    referenced.push(module.resolved_functions[function_index as usize]);
                }
            }
        }
        for (_, expr) in &module.table_init_exprs {
            for op in expr.get_operators_reader() {
                if let Operator::RefFunc { function_index } = op? {
                    referenced.push(module.resolved_functions[function_index as usize]);
                }
            }
        }
        for (_, _, expr) in &module.dynamic_elements {
            for op in expr.get_operators_reader() {
                if let Operator::RefFunc { function_index } = op? {
                    referenced.push(module.resolved_functions[function_index as usize]);
                }
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
    let mut wasm_referenced = referenced[wasm_reference_start..].to_vec();
    wasm_referenced.sort_unstable();
    wasm_referenced.dedup();
    let mut private_tables: Vec<Vec<bool>> = modules
        .iter()
        .map(|module| vec![persistent; module.instance.tables.len()])
        .collect();
    if persistent {
        // An unexported table cannot acquire a new alias in a later instance.
        // Check every body, including currently unreachable functions, before
        // excluding host-injected identities from its possible callees.
        for module in &modules {
            for &(kind, index) in module.exports.values() {
                if kind == ExternalKind::Table {
                    let (owner, local) = module.table_bindings[index].unwrap();
                    private_tables[owner][local] = false;
                }
            }
            for body in &module.bodies {
                for operator in body.get_operators_reader()? {
                    let table = match operator? {
                        Operator::TableSet { table }
                        | Operator::TableFill { table }
                        | Operator::TableInit { table, .. }
                        | Operator::TableGrow { table } => table,
                        Operator::TableCopy { dst_table, .. } => dst_table,
                        _ => continue,
                    };
                    let (owner, local) = module.table_bindings[table as usize].unwrap();
                    private_tables[owner][local] = false;
                }
            }
        }
        referenced.extend(0..functions.len());
    }
    referenced.sort_unstable();
    referenced.dedup();
    let starts: Vec<_> = postorder
        .iter()
        .filter_map(|&module| {
            if persistent && (export_name.is_some() || module != root) {
                return None;
            }
            modules[module]
                .start_index
                .map(|local| modules[module].resolved_functions[local])
        })
        .collect();
    for module in &mut modules {
        for table in &mut module.instance.tables {
            table.info.mutable = false;
            table.targets = if table.function_refs {
                table
                    .values
                    .iter()
                    .copied()
                    .filter(|&v| v != 0)
                    .map(|v| (v - 1) as usize)
                    .collect()
            } else {
                Vec::new()
            };
            table.targets.sort_unstable();
            table.targets.dedup();
        }
    }
    // A newly reachable table mutation can expose more indirect call targets.
    // Repeat reachability until those targets no longer change.
    loop {
        let mut reachable = vec![0u8; functions.len()];
        let mut ignored_order = Vec::new();
        let mut ignored_traps = false;
        let mut ignored_growth: Vec<Vec<bool>> = modules
            .iter()
            .map(|module| vec![false; module.instance.memories.len()])
            .collect();
        let mut ignored_recursion = false;
        for function in index.iter().chain(&starts) {
            visit_function(
                *function,
                &modules,
                &functions,
                &mut reachable,
                &mut ignored_order,
                &mut ignored_traps,
                &mut ignored_growth,
                &mut ignored_recursion,
            )?;
        }
        let mut mutates: Vec<Vec<bool>> = modules
            .iter()
            .map(|module| vec![false; module.instance.tables.len()])
            .collect();
        let mut grows = mutates.clone();
        if persistent {
            for row in &mut mutates {
                row.fill(true);
            }
            for row in &mut grows {
                row.fill(true);
            }
        }
        for module in &modules {
            for &(_, table_index, _) in &module.active_elements {
                let (owner, local) = module.table_bindings[table_index].unwrap();
                mutates[owner][local] = true;
            }
            for &(local, _) in &module.table_init_exprs {
                if local < module.instance.tables.len() {
                    let (owner, table) = module.table_bindings
                        [module.table_types.len() - module.instance.tables.len() + local]
                        .unwrap();
                    mutates[owner][table] = true;
                }
            }
        }
        for (function, &state) in reachable.iter().enumerate() {
            if state != 2 {
                continue;
            }
            let (module_index, local) = functions[function];
            let module = &modules[module_index];
            for op in module.bodies[local - module.imports.len()].get_operators_reader()? {
                match op? {
                    Operator::TableSet { table }
                    | Operator::TableFill { table }
                    | Operator::TableInit { table, .. } => {
                        let (owner, local) = module.table_bindings[table as usize].unwrap();
                        mutates[owner][local] = true;
                    }
                    Operator::TableCopy { dst_table, .. } => {
                        let (owner, local) = module.table_bindings[dst_table as usize].unwrap();
                        mutates[owner][local] = true;
                    }
                    Operator::TableGrow { table } => {
                        let (owner, local) = module.table_bindings[table as usize].unwrap();
                        mutates[owner][local] = true;
                        grows[owner][local] = true;
                    }
                    _ => {}
                }
            }
        }
        let mut changed = false;
        for (module_index, module) in modules.iter_mut().enumerate() {
            for (table_index, table) in module.instance.tables.iter_mut().enumerate() {
                if mutates[module_index][table_index] && !table.info.mutable {
                    table.info.mutable = true;
                    if table.function_refs {
                        // Lane storage stays mutable. A private table with no
                        // Wasm writes can only contain initializer references.
                        table.targets = if private_tables[module_index][table_index] {
                            wasm_referenced.clone()
                        } else {
                            referenced.clone()
                        };
                    }
                    changed = true;
                }
            }
        }
        if !changed {
            for (module_index, module) in modules.iter_mut().enumerate() {
                for (table_index, table) in module.instance.tables.iter_mut().enumerate() {
                    if !grows[module_index][table_index] {
                        table.info.capacity = table.values.len();
                    }
                }
            }
            break;
        }
    }
    let mut recursive = false;
    let mut order = Vec::new();
    let mut visit_state = vec![0u8; functions.len()];
    let mut can_trap = limits.fuel.is_some()
        || modules.iter().any(|m| {
            !m.instance.memories.is_empty()
                || !m.active_data.is_empty()
                || !m.active_elements.is_empty()
        });
    let mut uses_growth: Vec<Vec<bool>> = modules
        .iter()
        .map(|module| vec![false; module.instance.memories.len()])
        .collect();
    // Persisted resources have whole-instance capacities, while a pipeline only
    // needs this entry and its possible callees. Provider starts already ran.
    let entries = index.iter().copied().chain(starts.iter().copied());
    for entry in entries {
        visit_function(
            entry,
            &modules,
            &functions,
            &mut visit_state,
            &mut order,
            &mut can_trap,
            &mut uses_growth,
            &mut recursive,
        )?;
    }
    let tail_calls = order.iter().any(|&function| {
        let (module_index, local) = functions[function];
        modules[module_index].bodies[local - modules[module_index].imports.len()]
            .get_operators_reader()
            .unwrap()
            .into_iter()
            .any(|op| {
                matches!(
                    op,
                    Ok(Operator::ReturnCall { .. }
                        | Operator::ReturnCallIndirect { .. }
                        | Operator::ReturnCallRef { .. })
                )
            })
    });
    recursive |= tail_calls;
    can_trap |= recursive || modules.iter().any(|m| m.uses_exceptions);
    let mut slots = Vec::new();
    let mut next_slot = 4u64;
    for (module_index, module) in modules.iter_mut().enumerate() {
        let instance = &mut module.instance;
        let initial_total: usize = instance
            .memories
            .iter()
            .map(|memory| memory.bytes.len())
            .sum();
        let growth_budget_pages = (limits.memory_bytes - initial_total) / 65536;
        for (memory_index, memory) in instance.memories.iter_mut().enumerate() {
            let initial_pages = memory.bytes.len() / 65536;
            let wanted = initial_pages.saturating_add(
                if persistent || uses_growth[module_index][memory_index] {
                    limits.memory_growth_pages
                } else {
                    0
                },
            );
            let max_pages = memory.max_pages.unwrap_or(u64::MAX).min(usize::MAX as u64) as usize;
            let extra_pages = wanted
                .min(max_pages)
                .saturating_sub(initial_pages)
                .min(growth_budget_pages);
            memory.capacity = memory.bytes.len() + extra_pages * 65536;
        }
        let mut allocate = |needed: bool| {
            needed.then(|| {
                let slot = next_slot;
                next_slot += 1;
                slot
            })
        };
        let memories = (0..instance.memories.len())
            .map(|_| (allocate(true).unwrap(), allocate(true).unwrap()))
            .collect();
        let tables = (0..instance.tables.len())
            .map(|_| allocate(true).unwrap())
            .collect();
        slots.push(BufferSlots {
            memories,
            globals: allocate(!instance.globals_template.is_empty()),
            tables,
            passive_data: allocate(!instance.data_lengths.is_empty()),
        });
    }
    let init_roots = modules
        .iter()
        .map(|m| {
            m.dynamic_globals
                .iter()
                .map(|(_, _, e)| e)
                .chain(m.dynamic_elements.iter().map(|(_, _, e)| e))
                .chain(m.table_init_exprs.iter().map(|(_, e)| e))
                .map(|e| e.get_operators_reader().into_iter().count())
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap_or(0)
        + ty.params().len()
        + 1;
    let mut gc = uses_gc.then_some(GcLayout {
        slot: next_slot,
        heap_words: limits.gc_heap_words,
        root_words: 0,
    });
    if uses_gc {
        next_slot += 1;
        can_trap = true;
    }
    let persistent_slot = persistent.then_some(6);
    if persistent {
        can_trap = true;
    }
    if persistent {
        if let Some(gc) = &mut gc {
            gc.slot = 5;
        }
    }
    if !persistent && next_slot > 31 {
        return Err(Error::Unsupported(
            "linked module state exceeds Metal's 31 buffer slots".into(),
        ));
    }
    let mut source = String::from("#include <metal_stdlib>\nusing namespace metal;\n");
    source.push_str(include_str!("softfloat64.metal"));
    source.push_str(include_str!("wasm_float.metal"));
    source.push_str(include_str!("wasm_simd.metal"));
    if uses_gc {
        source.push_str(include_str!("wasm_gc.metal"));
    }
    let max_memories = modules
        .iter()
        .map(|m| m.instance.memories.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let max_tables = modules
        .iter()
        .map(|m| m.instance.tables.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let max_elements = modules
        .iter()
        .map(|m| m.instance.element_info.elements.len())
        .max()
        .unwrap_or(0)
        .max(1);
    let exceptions = modules.iter().any(|m| m.uses_exceptions);
    let exception_payload_words = modules
        .iter()
        .flat_map(|m| {
            m.tag_types.iter().map(|&t| {
                m.types[t as usize]
                    .params()
                    .iter()
                    .map(|ty| value_words(*ty))
                    .sum::<usize>()
            })
        })
        .max()
        .unwrap_or(0)
        .max(1);
    let gc_boundary = uses_gc
        && ty
            .params()
            .iter()
            .chain(ty.results())
            .any(|t| matches!(t, ValType::Ref(_)));
    let needs_exception_input = gc_boundary && exceptions
        || ty
            .params()
            .iter()
            .any(|ty| matches!(ty, ValType::Ref(ty) if is_exception_ref(*ty)));
    let needs_exception_output = gc_boundary && exceptions
        || ty
            .results()
            .iter()
            .any(|ty| matches!(ty, ValType::Ref(ty) if is_exception_ref(*ty)));
    let exception_io = (!persistent && (needs_exception_input || needs_exception_output))
        .then_some(ExceptionIo {
            records: limits.exception_records,
            payload_words: exception_payload_words,
            input: needs_exception_input,
            output: needs_exception_output,
        });
    if exceptions {
        let words = exception_payload_words;
        source.push_str(&format!("constant uint EXCEPTION_RECORDS = {}u; constant uint EXCEPTION_WORDS = {words}u;\nstruct ExceptionState {{ uint pending; uint count; uint pinned_count; uint marks[{}]; uint tags[{}]; ulong values[{}]; }};\n", limits.exception_records, limits.exception_records, limits.exception_records, limits.exception_records.checked_mul(words).ok_or_else(|| Error::Unsupported("exception arena size overflow".into()))?));
    }
    let gc_field = if uses_gc { "thread GcState* gc;" } else { "" };
    let exception_field = if exceptions {
        "thread ExceptionState* exceptions;"
    } else {
        ""
    };
    let fuel_field = if limits.fuel.is_some() {
        "thread ulong* fuel;"
    } else {
        ""
    };
    source.push_str(&format!("struct ModuleState {{ {fuel_field} {exception_field} {gc_field} device uchar* memory[{max_memories}]; device uint* pages[{max_memories}]; device ulong* globals; device ulong* table[{max_tables}]; device const uchar* passive_data; uint table_size[{max_tables}]; thread uint* element_lengths; thread uint* data_lengths; thread ulong* element_values[{max_elements}]; }};\n"));
    if uses_gc {
        source.push_str(&gc_type_source(&gc_types, &modules, persistent));
    }
    if tail_calls {
        let max_args = order
            .iter()
            .map(|&f| {
                global_types[f]
                    .params()
                    .iter()
                    .map(|ty| value_words(*ty))
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
            .max(1);
        source.push_str(&format!(
            "struct TailFrame {{ uint next; ulong args[{max_args}]; }};\n"
        ));
        source.push_str("void invoke(uint target, thread ulong* input_args, thread ulong* output_results, thread ModuleState* states, thread uint& trap, uint depth);\n");
    }
    for (module_index, module) in modules.iter().enumerate() {
        for (element, values) in module.instance.element_info.elements.iter().enumerate() {
            source.push_str(&format!(
                "constant ulong m{module_index}_element_{element}[{}] = {{{}}};\n",
                values.len().max(1),
                if values.is_empty() {
                    "0u".into()
                } else {
                    values
                        .iter()
                        .map(|v| format!("{v}ul"))
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
            tail_calls,
        )?);
        source.push_str(";\n");
    }
    let mut max_gc_roots = 1;
    for &function in &order {
        let (module_index, local) = functions[function];
        let (body_source, roots) = translate_body(
            &modules,
            module_index,
            function,
            local,
            &global_types,
            recursive,
            tail_calls,
            limits.call_depth,
            limits.memory_bytes,
            &order,
            uses_gc,
            limits.fuel.is_some(),
        )?;
        source.push_str(&body_source);
        max_gc_roots = max_gc_roots.max(roots);
    }
    if let Some(layout) = &mut gc {
        let helper_roots = gc_types
            .iter()
            .map(|t| t.fields.len().max(1))
            .max()
            .unwrap_or(1);
        let depth = if recursive {
            limits.call_depth
        } else {
            order.len().max(1)
        };
        layout.root_words = max_gc_roots
            .checked_mul(depth)
            .and_then(|n| n.checked_add(init_roots))
            .and_then(|n| n.checked_add(helper_roots))
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or_else(|| Error::Unsupported("GC root capacity overflow".into()))?;
    }
    if tail_calls {
        source.push_str(&tail_dispatcher(
            &order,
            &global_types,
            limits.call_depth,
            exceptions,
        ));
    }
    let fast_i32 = ty.params() == [ValType::I32] && ty.results() == [ValType::I32];
    let input_ty = if fast_i32 { "uint" } else { "ulong" };
    source.push_str(&format!("kernel void run_wasm(device const {input_ty}* input [[buffer(0)]], device {input_ty}* output [[buffer(1)]], constant ulong* config [[buffer(2)]]"));
    if can_trap {
        source.push_str(", device uint* traps [[buffer(3)]]");
    }
    if persistent {
        source.push_str(", device uchar* resource_buffer [[buffer(4)]]");
    } else {
        for (module_index, slot) in slots.iter().enumerate() {
            for (index, &(memory, pages)) in slot.memories.iter().enumerate() {
                source.push_str(&format!(", device uchar* m{module_index}_memory{index} [[buffer({memory})]], device uint* m{module_index}_pages{index} [[buffer({pages})]]"));
            }
            for (index, &table) in slot.tables.iter().enumerate() {
                source.push_str(&format!(
                    ", device ulong* m{module_index}_table{index} [[buffer({table})]]"
                ));
            }
            for (slot, ty, field) in [
                (slot.globals, "ulong", "globals"),
                (slot.passive_data, "const uchar", "passive_data"),
            ] {
                if let Some(slot) = slot {
                    source.push_str(&format!(
                        ", device {ty}* m{module_index}_{field} [[buffer({slot})]]"
                    ));
                }
            }
        }
    }
    if let Some(layout) = gc {
        source.push_str(&format!(
            ", device ulong* gc_buffer [[buffer({})]]",
            layout.slot
        ));
    }
    if let Some(slot) = persistent_slot {
        source.push_str(&format!(
            ", device ulong* persistent_state [[buffer({slot})]]"
        ));
    }
    source.push_str(
        ", uint id [[thread_position_in_grid]]) {\nulong count = config[0];\nif (id >= count) return;\nuint trap = 0u;\n",
    );
    if persistent {
        source.push_str(&store::resource_source(&modules));
    }
    source.push_str(&format!("ModuleState states[{}] = {{}};\n", modules.len()));
    if let Some(fuel) = limits.fuel {
        source.push_str(&format!("ulong fuel = {fuel}ul;\n"));
        for module in 0..modules.len() {
            source.push_str(&format!("states[{module}].fuel = &fuel;\n"));
        }
    }
    if let Some(layout) = gc {
        let stride = layout.heap_words
            + layout.root_words
            + if persistent {
                layout.heap_words + limits.exception_records + 1
            } else {
                0
            };
        source.push_str(&format!("GcState gc = {{ gc_buffer + ulong(id) * {stride}ul, {}u, gc_buffer + ulong(id) * {stride}ul + {}ul, {}u, 0u, states, nullptr, 0u }};\nGcRootFrame init_frame(gc, {init_roots}u, trap);\n", layout.heap_words, layout.heap_words, layout.root_words));
        if persistent {
            source.push_str(&format!("gc.host_roots = gc_buffer + ulong(id)*{stride}ul + {}ul; gc.host_count = uint(gc.host_roots[{}]);\n", layout.heap_words + layout.root_words, layout.heap_words + limits.exception_records));
        }
        for m in 0..modules.len() {
            source.push_str(&format!("states[{m}].gc = &gc;\n"));
        }
    }
    if exceptions {
        source.push_str("ExceptionState exceptions = {};\n");
        if let Some(io) = exception_io.filter(|io| io.input) {
            let stride = io.words()?;
            let record_words = io.payload_words + 1;
            source.push_str(&format!("device const ulong* exception_input = reinterpret_cast<device const ulong*>(reinterpret_cast<device const uchar*>(input) + config[5]) + ulong(id) * {stride}ul;\nexceptions.count = uint(exception_input[0]); exceptions.pinned_count = exceptions.count;\nfor (uint j = 0u; j < exceptions.count; ++j) {{ exceptions.tags[j] = uint(exception_input[1ul + ulong(j) * {record_words}ul]); for (uint k = 0u; k < EXCEPTION_WORDS; ++k) exceptions.values[j * EXCEPTION_WORDS + k] = exception_input[2ul + ulong(j) * {record_words}ul + k]; }}\n"));
        }
        for module in 0..modules.len() {
            source.push_str(&format!("states[{module}].exceptions = &exceptions;\n"));
        }
    }
    for (module_index, module) in modules.iter().enumerate() {
        let instance = &module.instance;
        for (index, memory) in instance.memories.iter().enumerate() {
            source.push_str(&format!("states[{module_index}].memory[{index}] = m{module_index}_memory{index} + ulong(id) * {}ul;\nstates[{module_index}].pages[{index}] = m{module_index}_pages{index} + id;\n", memory.capacity.max(1)));
        }
        if !instance.globals_template.is_empty() {
            source.push_str(&format!(
                "states[{module_index}].globals = m{module_index}_globals + ulong(id) * {}ul;\n",
                instance.globals_template.len()
            ));
        }
        for (index, table) in instance.tables.iter().enumerate() {
            let offset = if table.info.mutable {
                format!(" + ulong(id) * {}ul", table.info.capacity.max(1))
            } else {
                String::new()
            };
            source.push_str(&format!("states[{module_index}].table[{index}] = m{module_index}_table{index}{offset};\nstates[{module_index}].table_size[{index}] = {}u;\n", table.values.len()));
        }
        for (field, lengths) in [
            ("element_lengths", &instance.element_info.lengths),
            ("data_lengths", &instance.data_lengths),
        ] {
            if !lengths.is_empty() {
                source.push_str(&format!("uint m{module_index}_{field}[{}] = {{{}}};\nstates[{module_index}].{field} = m{module_index}_{field};\n", lengths.len(), lengths.iter().map(|v| format!("{v}u")).collect::<Vec<_>>().join(", ")));
            }
        }
        for (segment, values) in instance.element_info.elements.iter().enumerate() {
            if module
                .dynamic_elements
                .iter()
                .any(|(index, _, _)| *index == segment)
            {
                source.push_str(&format!("ulong m{module_index}_dynamic_element_{segment}[{}] = {{{}}};\nstates[{module_index}].element_values[{segment}] = m{module_index}_dynamic_element_{segment};\n", values.len().max(1), if values.is_empty() { "0u".into() } else { values.iter().map(|v| format!("{v}ul")).collect::<Vec<_>>().join(", ") }));
            }
        }
        if !instance.data_lengths.is_empty() {
            source.push_str(&format!(
                "states[{module_index}].passive_data = m{module_index}_passive_data;\n"
            ));
        }
    }
    let metadata = persistent.then(|| {
        store::MetadataLayout::new(
            &modules,
            exceptions.then_some((limits.exception_records, exception_payload_words)),
        )
    });
    if let Some(layout) = &metadata {
        source.push_str(&layout.load_source());
    }
    let mut args = Vec::new();
    let param_words: usize = ty.params().iter().map(|ty| value_words(*ty)).sum();
    let result_words: usize = ty.results().iter().map(|ty| value_words(*ty)).sum();
    let mut param_word = 0;
    for (param, param_ty) in ty.params().iter().enumerate() {
        let offset = if fast_i32 {
            "id".into()
        } else {
            format!("id * {param_words}u + {param_word}u")
        };
        if *param_ty == ValType::V128 {
            args.push(format!("ulong2(input[{offset}], input[({offset}) + 1u])"));
        } else {
            args.push(format!("ulong(input[{offset}])"));
        }
        let _ = param;
        param_word += value_words(*param_ty);
    }
    for (result, result_ty) in ty.results().iter().enumerate() {
        source.push_str(&format!(
            "{} result{result} = {};\n",
            if *result_ty == ValType::V128 {
                "ulong2"
            } else {
                "ulong"
            },
            if *result_ty == ValType::V128 {
                "ulong2(0ul)"
            } else {
                "0ul"
            }
        ));
        args.push(format!("result{result}"));
    }
    let context = if recursive {
        "states, trap, 0u"
    } else {
        "states, trap"
    };
    let ready = if exceptions {
        "trap == 0u && exceptions.pending == 0u"
    } else {
        "trap == 0u"
    };
    if uses_gc {
        for (i, arg) in args.iter().take(ty.params().len()).enumerate() {
            if let ValType::Ref(r) = ty.params()[i] {
                source.push_str(&format!(
                    "gc.roots[{}u] = {};\n",
                    init_roots - ty.params().len() + i,
                    root_word(arg, r)
                ));
            }
        }
    }
    for &module_index in &postorder {
        if persistent && (export_name.is_some() || module_index != root) {
            continue;
        }
        let module = &modules[module_index];
        // Active/declarative element expressions remain roots until this
        // module finishes initialization, even though their saved length is zero.
        for (segment, values) in module.instance.element_info.elements.iter().enumerate() {
            if module
                .dynamic_elements
                .iter()
                .any(|(s, _, _)| *s == segment)
            {
                source.push_str(&format!(
                    "states[{module_index}].element_lengths[{segment}] = {}u;\n",
                    values.len()
                ));
            }
        }
        for (slot, ty, expr) in &module.dynamic_globals {
            let value = const_source(expr, module, uses_gc)?;
            if *ty == ValType::V128 {
                source.push_str(&format!("if ({ready}) {{ ulong2 value = {value}; states[{module_index}].globals[{slot}u] = value.x; states[{module_index}].globals[{}u] = value.y; }}\n", slot + 1));
            } else {
                source.push_str(&format!(
                    "if ({ready}) states[{module_index}].globals[{slot}u] = {value};\n"
                ));
            }
        }
        for &(segment, item, ref expr) in &module.dynamic_elements {
            let value = const_source(expr, module, uses_gc)?;
            source.push_str(&format!("if ({ready}) states[{module_index}].element_values[{segment}][{item}] = ulong({value});\n"));
        }
        for &(local, ref expr) in &module.table_init_exprs {
            let value = const_source(expr, module, uses_gc)?;
            let count = module.instance.tables[local].values.len();
            source.push_str(&format!("if ({ready}) {{ for (ulong j = 0ul; j < {count}ul; ++j) states[{module_index}].table[{local}][j] = ulong({value}); }}\n"));
        }
        for &(segment, table_index, ref expr) in &module.active_elements {
            let offset = const_source(expr, module, uses_gc)?;
            let (owner, local) = module.table_bindings[table_index].unwrap();
            let count = module.instance.element_info.elements[segment].len();
            let offset = if module.table_types[table_index].table64 {
                format!("ulong({offset})")
            } else {
                format!("ulong(uint({offset}))")
            };
            let values = if module
                .dynamic_elements
                .iter()
                .any(|(index, _, _)| *index == segment)
            {
                format!("states[{module_index}].element_values[{segment}]")
            } else {
                format!("m{module_index}_element_{segment}")
            };
            source.push_str(&format!("if ({ready}) {{ ulong offset = {offset}; if (offset > states[{owner}].table_size[{local}] || {count}ul > states[{owner}].table_size[{local}] - offset) trap = 14u; else {{ for (ulong j = 0ul; j < {count}ul; ++j) states[{owner}].table[{local}][offset + j] = {values}[j]; }} }}\n"));
        }
        for (memory_index, expr, data_offset, count) in &module.active_data {
            let offset = const_source(expr, module, uses_gc)?;
            let (owner, local) = module.memory_bindings[*memory_index].unwrap();
            let offset = if module.memory_types[*memory_index].memory64 {
                format!("ulong({offset})")
            } else {
                format!("ulong(uint({offset}))")
            };
            source.push_str(&format!("if ({ready}) {{ ulong offset = {offset}; ulong size = ulong(states[{owner}].pages[{local}][0]) * 65536ul; if (offset > size || {count}ul > size - offset) trap = 4u; else {{ for (ulong j = 0ul; j < {count}ul; ++j) states[{owner}].memory[{local}][offset + j] = states[{module_index}].passive_data[{data_offset}ul + j]; }} }}\n"));
        }
        if uses_gc {
            for (segment, values) in module.instance.element_info.elements.iter().enumerate() {
                if module.instance.element_info.lengths[segment] == 0
                    && module
                        .dynamic_elements
                        .iter()
                        .any(|(s, _, _)| *s == segment)
                {
                    source.push_str(&format!("states[{module_index}].element_lengths[{segment}] = 0u; for(uint j=0u;j<{}u;++j) states[{module_index}].element_values[{segment}][j]=0ul;\n",values.len()));
                }
            }
        }
        if let Some(local) = module.start_index {
            let start = module.resolved_functions[local];
            if tail_calls {
                source.push_str(&format!("if ({ready}) {{ ulong start_args[1] = {{0ul}}; ulong start_results[1] = {{0ul}}; invoke({start}u, start_args, start_results, states, trap, 0u); }}\n"));
            } else {
                source.push_str(&format!("if ({ready}) f{start}({context});\n"));
            }
        }
    }
    if uses_gc {
        source.push_str(&format!(
            "for(uint j=0u;j<{init_roots}u;++j) gc.roots[j]=0ul;\n"
        ));
    }
    if !modules[root].memory_bindings.is_empty() {
        let argument_bytes = param_words * if fast_i32 { 4 } else { 8 };
        source.push_str(&format!("if (config[2] != 0ul && {ready}) {{\n device const uchar* memory_input = reinterpret_cast<device const uchar*>(input) + count * {argument_bytes}ul + ulong(id) * config[2];\n"));
        for (owner, local) in modules[root].memory_bindings.iter().flatten() {
            source.push_str(&format!("if (config[3] == {owner}ul && config[4] == {local}ul) for (ulong j = 0ul; j < config[2]; ++j) states[{owner}].memory[{local}][config[1] + j] = memory_input[j];\n"));
        }
        source.push_str("}\n");
    }
    if let Some(index) = index {
        if tail_calls {
            source.push_str(&format!(
                "ulong top_args[{}] = {{}}; ulong top_results[{}] = {{}};\n",
                param_words.max(1),
                result_words.max(1)
            ));
            let mut word = 0;
            for (arg, ty) in args.iter().take(ty.params().len()).zip(ty.params()) {
                if *ty == ValType::V128 {
                    source.push_str(&format!(
                        "top_args[{word}] = ({arg}).x; top_args[{}] = ({arg}).y;\n",
                        word + 1
                    ));
                } else {
                    source.push_str(&format!("top_args[{word}] = {arg};\n"));
                }
                word += value_words(*ty);
            }
            source.push_str(&format!(
                "if ({ready}) invoke({index}u, top_args, top_results, states, trap, 0u);\n"
            ));
            let mut word = 0;
            for (result, ty) in ty.results().iter().enumerate() {
                if *ty == ValType::V128 {
                    source.push_str(&format!(
                        "result{result} = ulong2(top_results[{word}], top_results[{}]);\n",
                        word + 1
                    ));
                } else {
                    source.push_str(&format!("result{result} = top_results[{word}];\n"));
                }
                word += value_words(*ty);
            }
        } else {
            args.push(context.into());
            source.push_str(&format!("if ({ready}) f{index}({});\n", args.join(", ")));
        }
    }
    if let Some(layout) = &metadata {
        source.push_str(&layout.save_source());
    }
    if exceptions {
        source.push_str("if (trap == 0u && exceptions.pending != 0u) trap = 11u;\n");
    }
    if can_trap {
        source.push_str("traps[id] = trap;\n");
    }
    let mut result_word = 0;
    for (result, result_ty) in ty.results().iter().enumerate() {
        let offset = if fast_i32 {
            "id".into()
        } else {
            format!("id * {result_words}u + {result_word}u")
        };
        if *result_ty == ValType::V128 {
            source.push_str(&format!("output[{offset}] = result{result}.x;\noutput[({offset}) + 1u] = result{result}.y;\n"));
        } else {
            source.push_str(&format!("output[{offset}] = {input_ty}(result{result});\n"));
        }
        result_word += value_words(*result_ty);
    }
    if let Some(io) = exception_io.filter(|io| io.output) {
        let stride = io.words()?;
        let record_words = io.payload_words + 1;
        source.push_str(&format!("device ulong* exception_output = output + count * {result_words}ul + ulong(id) * {stride}ul;\nexception_output[0] = exceptions.count;\nfor (uint j = 0u; j < exceptions.count; ++j) {{ exception_output[1ul + ulong(j) * {record_words}ul] = exceptions.tags[j]; for (uint k = 0u; k < EXCEPTION_WORDS; ++k) exception_output[2ul + ulong(j) * {record_words}ul + k] = exceptions.values[j * EXCEPTION_WORDS + k]; }}\n"));
    }
    source.push_str("}\n");
    let memory_exports = modules[root]
        .exports
        .iter()
        .filter_map(|(name, &(kind, index))| {
            (kind == ExternalKind::Memory)
                .then(|| (name.clone(), modules[root].memory_bindings[index].unwrap()))
        })
        .collect();
    let function_reference_types = functions
        .iter()
        .map(|&(module, local)| {
            RefType::new(
                false,
                HeapType::Concrete(UnpackedIndex::Id(
                    modules[module].type_ids[modules[module].function_types[local] as usize],
                )),
            )
            .unwrap()
        })
        .collect();
    let exported_functions = modules[root]
        .exports
        .iter()
        .filter_map(|(name, &(kind, local))| {
            (kind == ExternalKind::Func).then(|| {
                (
                    name.clone(),
                    modules[root].resolved_functions[local] as u32 + 1,
                )
            })
        })
        .collect();
    let reference_supertypes = modules
        .iter()
        .flat_map(|m| m.supertypes.iter().map(|(&id, &parent)| (id, parent)))
        .collect();
    let exception_tags = modules[root]
        .exports
        .iter()
        .filter_map(|(name, &(kind, local))| {
            (kind == ExternalKind::Tag).then(|| (name.clone(), modules[root].tag_bindings[local]))
        })
        .collect();
    let mut tag_payload_types = vec![Vec::new(); tag_count];
    for module in &modules {
        for (&tag, &ty) in module.tag_bindings.iter().zip(&module.tag_types) {
            tag_payload_types[tag as usize] = module.types[ty as usize].params().to_vec();
        }
    }
    Ok(Translation {
        persistent_metadata: metadata,
        exception_tags,
        tag_payload_types,
        exception_io,
        function_reference_types,
        exported_functions,
        reference_supertypes,
        gc,
        gc_types,
        source,
        can_trap,
        params: ty.params().to_vec(),
        results: ty.results().to_vec(),
        instances: modules.into_iter().map(|m| m.instance).collect(),
        slots,
        recursive,
        memory_exports,
    })
}

fn is_numeric(ty: ValType) -> bool {
    matches!(
        ty,
        ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 | ValType::V128
    )
}

fn value_words(ty: ValType) -> usize {
    if ty == ValType::V128 { 2 } else { 1 }
}

fn tail_dispatcher(
    order: &[usize],
    types: &[FuncType],
    call_depth: usize,
    exceptions: bool,
) -> String {
    let mut source = format!(
        "void invoke(uint target, thread ulong* input_args, thread ulong* output_results, thread ModuleState* states, thread uint& trap, uint depth) {{\nif (depth >= {call_depth}u) {{ trap = 8u; return; }}\nTailFrame tail = {{}};\nswitch (target) {{\n"
    );
    for &function in order {
        let words: usize = types[function]
            .params()
            .iter()
            .map(|ty| value_words(*ty))
            .sum();
        source.push_str(&format!("case {function}u: for (uint i = 0u; i < {words}u; ++i) tail.args[i] = input_args[i]; break;\n"));
    }
    source.push_str(
        "default: trap = 7u; return;\n}\nfor (;;) {\ntail.next = 0u;\nswitch (target) {\n",
    );
    for &function in order {
        let ty = &types[function];
        source.push_str(&format!("case {function}u: {{\n"));
        let mut args = Vec::new();
        let mut word = 0;
        for param in ty.params() {
            if *param == ValType::V128 {
                args.push(format!(
                    "ulong2(tail.args[{word}], tail.args[{}])",
                    word + 1
                ));
            } else {
                args.push(format!("tail.args[{word}]"));
            }
            word += value_words(*param);
        }
        for (result, result_ty) in ty.results().iter().enumerate() {
            source.push_str(&format!(
                "{} result{result} = {};\n",
                if *result_ty == ValType::V128 {
                    "ulong2"
                } else {
                    "ulong"
                },
                if *result_ty == ValType::V128 {
                    "ulong2(0ul)"
                } else {
                    "0ul"
                }
            ));
            args.push(format!("result{result}"));
        }
        args.extend([
            "states".into(),
            "trap".into(),
            "tail".into(),
            "depth".into(),
        ]);
        source.push_str(&format!("f{function}({});\n", args.join(", ")));
        source.push_str(if exceptions { "if (trap != 0u || states[0].exceptions->pending != 0u) return;\nif (tail.next == 0u) {\n" } else { "if (trap != 0u) return;\nif (tail.next == 0u) {\n" });
        let mut word = 0;
        for (result, result_ty) in ty.results().iter().enumerate() {
            if *result_ty == ValType::V128 {
                source.push_str(&format!("output_results[{word}] = result{result}.x; output_results[{}] = result{result}.y;\n", word + 1));
            } else {
                source.push_str(&format!("output_results[{word}] = result{result};\n"));
            }
            word += value_words(*result_ty);
        }
        source.push_str("return;\n}\nbreak;\n}\n");
    }
    source.push_str("default: trap = 7u; return;\n}\ntarget = tail.next - 1u;\n}\n}\n");
    source
}

fn is_supported_value(ty: ValType) -> bool {
    is_numeric(ty)
        || matches!(ty, ValType::Ref(r) if matches!(r.heap_type(), HeapType::Concrete(_) | HeapType::Abstract { shared:false, ty:AbstractHeapType::Any | AbstractHeapType::Eq | AbstractHeapType::I31 | AbstractHeapType::Struct | AbstractHeapType::Array | AbstractHeapType::None | AbstractHeapType::Func | AbstractHeapType::NoFunc | AbstractHeapType::Extern | AbstractHeapType::NoExtern | AbstractHeapType::Exn | AbstractHeapType::NoExn }))
}

fn const_source(
    expr: &ConstExpr<'_>,
    module: &ParsedModule<'_>,
    uses_gc: bool,
) -> Result<String, Error> {
    let mut operators = expr.get_operators_reader().into_iter();
    let mut stack: Vec<String> = Vec::new();
    for (root_index, operator) in operators.by_ref().enumerate() {
        let operator = operator?;
        let reference = match &operator {
            Operator::GlobalGet { global_index } => {
                match module.global_types[*global_index as usize].content_type {
                    ValType::Ref(r) => Some(r),
                    _ => None,
                }
            }
            Operator::StructNew { .. }
            | Operator::StructNewDefault { .. }
            | Operator::ArrayNew { .. }
            | Operator::ArrayNewDefault { .. }
            | Operator::ArrayNewFixed { .. }
            | Operator::AnyConvertExtern => Some(RefType::ANYREF),
            Operator::ExternConvertAny => Some(RefType::EXTERNREF),
            _ => None,
        };
        let value = match operator {
            Operator::I32Const { value } => format!("{}ul", value as u32),
            Operator::I64Const { value } => format!("{}ul", value as u64),
            Operator::F32Const { value } => format!("{}ul", value.bits()),
            Operator::F64Const { value } => format!("{}ul", value.bits()),
            Operator::V128Const { value } => {
                let bytes = value.bytes();
                format!(
                    "ulong2({}ul, {}ul)",
                    u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                    u64::from_le_bytes(bytes[8..].try_into().unwrap())
                )
            }
            Operator::RefNull { .. } => "0ul".into(),
            Operator::RefFunc { function_index } => {
                format!(
                    "{}ul",
                    module.resolved_functions[function_index as usize] + 1
                )
            }
            Operator::GlobalGet { global_index } => {
                let &(owner, slot) = module
                    .global_bindings
                    .get(global_index as usize)
                    .ok_or_else(|| {
                        Error::Unsupported("constant global index is out of range".into())
                    })?;
                if module.global_types[global_index as usize].content_type == ValType::V128 {
                    format!(
                        "ulong2(states[{owner}].globals[{slot}u], states[{owner}].globals[{}u])",
                        slot + 1
                    )
                } else {
                    format!("states[{owner}].globals[{slot}u]")
                }
            }
            op @ (Operator::I32Add
            | Operator::I32Sub
            | Operator::I32Mul
            | Operator::I64Add
            | Operator::I64Sub
            | Operator::I64Mul) => {
                let rhs = stack
                    .pop()
                    .ok_or_else(|| Error::Unsupported("invalid initializer stack".into()))?;
                let lhs = stack
                    .pop()
                    .ok_or_else(|| Error::Unsupported("invalid initializer stack".into()))?;
                let symbol = match op {
                    Operator::I32Add | Operator::I64Add => "+",
                    Operator::I32Sub | Operator::I64Sub => "-",
                    _ => "*",
                };
                if matches!(op, Operator::I32Add | Operator::I32Sub | Operator::I32Mul) {
                    format!("ulong(uint({lhs}) {symbol} uint({rhs}))")
                } else {
                    format!("({lhs} {symbol} {rhs})")
                }
            }
            Operator::RefI31 => {
                let v = pop(&mut stack)?;
                format!("(0x100000000ul | (ulong(uint({v})) & 0x7ffffffful))")
            }
            Operator::AnyConvertExtern => {
                let v = pop(&mut stack)?;
                format!("gc_internalize({v})")
            }
            Operator::ExternConvertAny => {
                let v = pop(&mut stack)?;
                format!("gc_externalize({v})")
            }
            op @ (Operator::StructNew {
                struct_type_index: index,
            }
            | Operator::StructNewDefault {
                struct_type_index: index,
            }) => {
                let t = module.gc_types[index as usize].as_ref().unwrap();
                let args = if matches!(op, Operator::StructNewDefault { .. }) {
                    t.fields
                        .iter()
                        .map(|f| {
                            if storage_words(f) == 2 {
                                "ulong2(0ul)".into()
                            } else {
                                "0ul".into()
                            }
                        })
                        .collect::<Vec<_>>()
                } else {
                    stack.split_off(stack.len() - t.fields.len())
                };
                format!(
                    "gc_struct_{}({}gc,trap)",
                    module.gc_ids[index as usize],
                    if args.is_empty() {
                        String::new()
                    } else {
                        format!("{},", args.join(","))
                    }
                )
            }
            op @ (Operator::ArrayNew {
                array_type_index: index,
            }
            | Operator::ArrayNewDefault {
                array_type_index: index,
            }) => {
                let n = pop(&mut stack)?;
                let t = module.gc_types[index as usize].as_ref().unwrap();
                let v = if matches!(op, Operator::ArrayNewDefault { .. }) {
                    if storage_words(&t.fields[0]) == 2 {
                        "ulong2(0ul)".into()
                    } else {
                        "0ul".into()
                    }
                } else {
                    pop(&mut stack)?
                };
                format!(
                    "gc_array_{}({v},{n},gc,trap)",
                    module.gc_ids[index as usize]
                )
            }
            Operator::ArrayNewFixed {
                array_type_index: index,
                array_size,
            } => {
                let args = stack.split_off(stack.len() - array_size as usize);
                format!(
                    "gc_fixed_{}_{}({}gc,trap)",
                    module.gc_ids[index as usize],
                    array_size,
                    if args.is_empty() {
                        String::new()
                    } else {
                        format!("{},", args.join(","))
                    }
                )
            }
            Operator::End => break,
            _ => return Err(Error::Unsupported("unsupported scalar initializer".into())),
        };
        stack.push(if let Some(r) = reference.filter(|_| uses_gc) {
            format!(
                "gc_keep(gc, {root_index}u, {value}, {}u)",
                gc_translate::root_kind(r)
            )
        } else {
            value
        });
    }
    if stack.len() != 1 || operators.next().is_some() {
        return Err(Error::Unsupported("unsupported scalar initializer".into()));
    }
    Ok(stack.pop().unwrap())
}

fn eval_const(expr: &ConstExpr<'_>, globals: &[u64], slots: &[usize]) -> Result<u64, Error> {
    let mut operators = expr.get_operators_reader().into_iter();
    let mut stack: Vec<u64> = Vec::new();
    for operator in operators.by_ref() {
        let value = match operator? {
            Operator::I32Const { value } => value as u32 as u64,
            Operator::I64Const { value } => value as u64,
            Operator::F32Const { value } => value.bits() as u64,
            Operator::F64Const { value } => value.bits(),
            Operator::RefNull { .. } => 0,
            Operator::RefFunc { function_index } => u64::from(function_index) + 1,
            Operator::GlobalGet { global_index } => *globals
                .get(*slots.get(global_index as usize).ok_or_else(|| {
                    Error::Unsupported("constant global index is out of range".into())
                })?)
                .ok_or_else(|| {
                    Error::Unsupported("constant global index is out of range".into())
                })?,
            op @ (Operator::I32Add
            | Operator::I32Sub
            | Operator::I32Mul
            | Operator::I64Add
            | Operator::I64Sub
            | Operator::I64Mul) => {
                let rhs = stack
                    .pop()
                    .ok_or_else(|| Error::Unsupported("invalid initializer stack".into()))?;
                let lhs = stack
                    .pop()
                    .ok_or_else(|| Error::Unsupported("invalid initializer stack".into()))?;
                match op {
                    Operator::I32Add => (lhs as u32).wrapping_add(rhs as u32) as u64,
                    Operator::I32Sub => (lhs as u32).wrapping_sub(rhs as u32) as u64,
                    Operator::I32Mul => (lhs as u32).wrapping_mul(rhs as u32) as u64,
                    Operator::I64Add => lhs.wrapping_add(rhs),
                    Operator::I64Sub => lhs.wrapping_sub(rhs),
                    _ => lhs.wrapping_mul(rhs),
                }
            }
            Operator::End => break,
            _ => return Err(Error::Unsupported("unsupported scalar initializer".into())),
        };
        stack.push(value);
    }
    if stack.len() != 1 || operators.next().is_some() {
        return Err(Error::Unsupported("unsupported scalar initializer".into()));
    }
    Ok(stack.pop().unwrap())
}

fn eval_v128_const(
    expr: &ConstExpr<'_>,
    globals: &[u64],
    slots: &[usize],
) -> Result<[u64; 2], Error> {
    let mut operators = expr.get_operators_reader().into_iter();
    let value = match operators.next().transpose()? {
        Some(Operator::V128Const { value }) => {
            let bytes = value.bytes();
            [
                u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                u64::from_le_bytes(bytes[8..].try_into().unwrap()),
            ]
        }
        Some(Operator::GlobalGet { global_index }) => {
            let slot = *slots.get(global_index as usize).ok_or_else(|| {
                Error::Unsupported("constant global index is out of range".into())
            })?;
            [
                *globals.get(slot).ok_or_else(|| {
                    Error::Unsupported("constant global index is out of range".into())
                })?,
                *globals.get(slot + 1).ok_or_else(|| {
                    Error::Unsupported("constant global index is out of range".into())
                })?,
            ]
        }
        _ => return Err(Error::Unsupported("unsupported v128 initializer".into())),
    };
    if !matches!(operators.next().transpose()?, Some(Operator::End)) || operators.next().is_some() {
        return Err(Error::Unsupported("unsupported v128 initializer".into()));
    }
    Ok(value)
}

fn visit_function(
    index: usize,
    modules: &[ParsedModule<'_>],
    functions: &[(usize, usize)],
    state: &mut [u8],
    order: &mut Vec<usize>,
    can_trap: &mut bool,
    uses_growth: &mut [Vec<bool>],
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
    let mut previous_reference = None;
    for operator in module.bodies[local - module.imports.len()].get_operators_reader()? {
        let operator = operator?;
        let current_reference = match operator {
            Operator::RefFunc { function_index } => {
                Some(module.resolved_functions[function_index as usize] as u32 + 1)
            }
            Operator::RefNull { .. } => Some(0),
            Operator::GlobalGet { global_index }
                if !module.global_types[global_index as usize].mutable =>
            {
                let (owner, slot) = module.global_bindings[global_index as usize];
                let provider = &modules[owner];
                if matches!(module.global_types[global_index as usize].content_type, ValType::Ref(ty) if is_function_ref_in(ty, &module.supertypes))
                    && !provider.dynamic_globals.iter().any(|(s, _, _)| *s == slot)
                {
                    Some(provider.instance.globals_template[slot] as u32)
                } else {
                    None
                }
            }
            _ => None,
        };
        let current_constant = match &operator {
            Operator::I32Const { value } => Some(*value as u32),
            _ => None,
        };
        match operator {
            Operator::Call { function_index } | Operator::ReturnCall { function_index } => {
                visit_function(
                    module.resolved_functions[function_index as usize],
                    modules,
                    functions,
                    state,
                    order,
                    can_trap,
                    uses_growth,
                    recursive,
                )?;
            }
            Operator::CallIndirect {
                type_index,
                table_index: _,
            }
            | Operator::ReturnCallIndirect {
                type_index,
                table_index: _,
            }
            | Operator::CallRef { type_index }
            | Operator::ReturnCallRef { type_index } => {
                *can_trap = true;
                let expected = type_index as usize;
                let targets: Vec<usize> = match operator {
                    Operator::CallRef { .. } | Operator::ReturnCallRef { .. } => {
                        match previous_reference {
                            Some(0) => Vec::new(),
                            Some(reference) => vec![reference as usize - 1],
                            None => (0..functions.len()).collect(),
                        }
                    }
                    Operator::CallIndirect { table_index, .. }
                    | Operator::ReturnCallIndirect { table_index, .. } => {
                        let (owner, local) = module.table_bindings[table_index as usize]
                            .ok_or_else(|| Error::Unsupported("table is missing".into()))?;
                        modules[owner].instance.tables[local].targets.clone()
                    }
                    _ => unreachable!(),
                };
                let owner = match operator {
                    Operator::CallIndirect { table_index, .. }
                    | Operator::ReturnCallIndirect { table_index, .. } => {
                        let (owner, local) = module.table_bindings[table_index as usize].unwrap();
                        Some(&modules[owner].instance.tables[local])
                    }
                    _ => None,
                };
                for target in targets {
                    if let Some(owner) = owner
                        && !matches!(
                            operator,
                            Operator::CallRef { .. } | Operator::ReturnCallRef { .. }
                        )
                    {
                        if !owner.info.mutable {
                            if let Some(slot) = previous_constant {
                                if owner.values.get(slot as usize).copied()
                                    != Some(target as u64 + 1)
                                {
                                    continue;
                                }
                            }
                        }
                    }
                    if module.accepts_function(
                        expected,
                        &modules[functions[target].0],
                        functions[target].1,
                    ) {
                        visit_function(
                            target,
                            modules,
                            functions,
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
            Operator::MemoryGrow { mem } => {
                let (owner, local) = module.memory_bindings[mem as usize].unwrap();
                uses_growth[owner][local] = true;
            }
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
            | Operator::RefAsNonNull
            | Operator::RefCastNonNull { .. }
            | Operator::RefCastNullable { .. }
            | Operator::Throw { .. }
            | Operator::ThrowRef
            | Operator::TryTable { .. }
            | Operator::Unreachable => *can_trap = true,
            _ => {}
        }
        previous_constant = current_constant;
        previous_reference = current_reference;
    }
    state[index] = 2;
    order.push(index);
    Ok(())
}

fn function_signature(
    index: usize,
    ty: &FuncType,
    recursive: bool,
    tail_calls: bool,
) -> Result<String, Error> {
    if ty
        .params()
        .iter()
        .chain(ty.results())
        .any(|ty| !is_supported_value(*ty))
    {
        return Err(Error::Unsupported(
            "unsupported function signature value type".into(),
        ));
    }
    let mut args: Vec<_> = ty
        .params()
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            format!(
                "{} a{index}",
                if *ty == ValType::V128 {
                    "ulong2"
                } else {
                    "ulong"
                }
            )
        })
        .collect();
    args.extend(ty.results().iter().enumerate().map(|(index, ty)| {
        format!(
            "thread {}& r{index}",
            if *ty == ValType::V128 {
                "ulong2"
            } else {
                "ulong"
            }
        )
    }));
    args.push("thread ModuleState* states".into());
    args.push("thread uint& trap".into());
    if tail_calls {
        args.push("thread TailFrame& tail".into());
    }
    if recursive {
        args.push("uint depth".into());
    }
    Ok(format!("void f{index}({})", args.join(", ")))
}

fn translate_body(
    modules: &[ParsedModule<'_>],
    module_index: usize,
    function_index: usize,
    local_function: usize,
    global_types: &[FuncType],
    recursive: bool,
    tail_calls: bool,
    call_depth: usize,
    memory_bytes: usize,
    order: &[usize],
    uses_gc: bool,
    metered: bool,
) -> Result<(String, usize), Error> {
    let module = &modules[module_index];
    let body = &module.bodies[local_function - module.imports.len()];
    let ty = module.function_type(local_function)?;
    let cast_targets: Vec<HashSet<usize>> = (0..module.types.len())
        .map(|expected| {
            modules
                .iter()
                .flat_map(|actual| {
                    (actual.imports.len()..actual.function_types.len()).filter_map(move |local| {
                        module
                            .accepts_function(expected, actual, local)
                            .then_some(actual.resolved_functions[local])
                    })
                })
                .collect()
        })
        .collect();
    let type_targets = cast_targets
        .iter()
        .map(|targets| {
            targets
                .iter()
                .copied()
                .filter(|f| order.contains(f))
                .collect()
        })
        .collect();
    let types = module.types.as_slice();
    let function_types = module.function_types.as_slice();
    let memories: Vec<_> = module
        .memory_bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            let (owner, local) = binding.unwrap();
            let memory = &modules[owner].instance.memories[local];
            MemoryState {
                owner,
                local,
                capacity: memory.capacity,
                owner_count: modules[owner].instance.memories.len(),
                is64: module.memory_types[index].memory64,
            }
        })
        .collect();
    let tables: Vec<_> = module
        .table_bindings
        .iter()
        .enumerate()
        .map(|(index, binding)| {
            let (owner, local) = binding.unwrap();
            let table = &modules[owner].instance.tables[local];
            TableState {
                owner,
                local,
                values: &table.values,
                info: &table.info,
                targets: &table.targets,
                is64: module.table_types[index].table64,
                element_type: module.table_types[index].element_type,
            }
        })
        .collect();
    let globals_mutable = module.globals_mutable.as_slice();
    let data_offsets = module.data_offsets.as_slice();
    let exceptions = modules.iter().any(|m| m.uses_exceptions);
    let mut source = function_signature(function_index, ty, recursive, tail_calls)?;
    source.push_str(" {\n");
    if metered {
        source
            .push_str("if (*states[0].fuel == 0ul) { trap = 17u; return; }\n--*states[0].fuel;\n");
    }
    if exceptions {
        source.push_str("thread ExceptionState& exceptions = *states[0].exceptions;\n");
    }
    if recursive && !tail_calls {
        source.push_str(&format!(
            "if (depth >= {call_depth}u) {{ trap = 8u; return; }}\n"
        ));
    }
    if !memories.is_empty() {
        source.push_str("device uchar* memory = nullptr; device uint* pages = nullptr;\n");
    }
    if !tables.is_empty() {
        source.push_str("device ulong* table = nullptr; thread uint* table_size = nullptr;\n");
    }
    for (exists, ty, field) in [
        (
            !module.instance.element_info.lengths.is_empty(),
            "thread uint*",
            "element_lengths",
        ),
        (
            !data_offsets.is_empty(),
            "device const uchar*",
            "passive_data",
        ),
        (!data_offsets.is_empty(), "thread uint*", "data_lengths"),
    ] {
        if exists {
            source.push_str(&format!("{ty} {field} = states[{module_index}].{field};\n"));
        }
    }
    let mut vector_names = HashSet::new();
    let mut local_types = ty.params().to_vec();
    for (index, ty) in ty.params().iter().enumerate() {
        if *ty == ValType::V128 {
            vector_names.insert(format!("l{index}"));
        }
        source.push_str(&format!(
            "{} l{index} = a{index};\n",
            if *ty == ValType::V128 {
                "ulong2"
            } else {
                "ulong"
            }
        ));
    }
    let mut local_count = ty.params().len();
    for local in body.get_locals_reader()? {
        let (count, ty) = local?;
        if !is_supported_value(ty) {
            return Err(Error::Unsupported(
                "only numeric, function, external, and exception reference locals are supported"
                    .into(),
            ));
        }
        let end = local_count
            .checked_add(count as usize)
            .ok_or_else(|| Error::Unsupported("too many locals".into()))?;
        for index in local_count..end {
            if ty == ValType::V128 {
                vector_names.insert(format!("l{index}"));
            }
            source.push_str(&format!(
                "{} l{index} = {};\n",
                if ty == ValType::V128 {
                    "ulong2"
                } else {
                    "ulong"
                },
                if ty == ValType::V128 {
                    "ulong2(0ul)"
                } else {
                    "0ul"
                }
            ));
        }
        local_types.extend(std::iter::repeat_n(ty, count as usize));
        local_count = end;
    }

    let mut operators = body.get_operators_reader()?.into_iter();
    let (instructions, end) = read_instructions(&mut operators)?;
    if end != End::Block || operators.next().is_some() {
        return Err(Error::Unsupported("invalid function body".into()));
    }
    if uses_gc {
        source.push_str("thread GcState& gc = *states[0].gc;\nGcRootFrame gc_frame(gc, GC_FRAME_SLOTSu, trap);\nif (trap != 0u) return;\n");
    }
    let declarations_at = source.len();
    source.push_str("uint flow = 0u;\n");
    let mut compiler = Compiler {
        source,
        uses_gc,
        metered,
        gc_types: &module.gc_types,
        gc_ids: &module.gc_ids,
        max_gc_roots: 0,
        module_index,
        resolved_functions: &module.resolved_functions,
        global_types,
        type_targets,
        cast_targets,
        tag_bindings: &module.tag_bindings,
        tag_types: &module.tag_types,
        exceptions,
        declarations: String::new(),
        local_count,
        local_types,
        vector_names,
        reference_names: HashMap::new(),
        next_value: 0,
        next_frame: 1,
        types,
        function_types,
        memories,
        memory_limit_pages: memory_bytes / 65536,
        globals_mutable,
        global_value_types: &module.global_types,
        global_bindings: &module.global_bindings,
        tables,
        element_info: &module.instance.element_info,
        dynamic_elements: &module.dynamic_elements,
        data_offsets,
        recursive,
        tail_calls,
        frames: vec![Frame {
            id: 1,
            params: Vec::new(),
            results: (0..ty.results().len())
                .map(|index| format!("r{index}"))
                .collect(),
            loop_label: false,
        }],
    };
    for (index, ty) in ty.results().iter().enumerate() {
        if *ty == ValType::V128 {
            compiler.vector_names.insert(format!("r{index}"));
        }
    }
    let mut stack = Vec::new();
    let reachable = compiler.emit_instructions(&instructions, &mut stack)?;
    let results = compiler.frames[0].results.clone();
    compiler.finish_path(reachable, &mut stack, 0, &results)?;
    compiler.source.push_str("}\n");
    compiler
        .source
        .insert_str(declarations_at, &compiler.declarations);
    Ok((
        compiler.source.replace(
            "GC_FRAME_SLOTSu",
            &format!("{}u", compiler.max_gc_roots.max(1)),
        ),
        compiler.max_gc_roots.max(1),
    ))
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
    TryTable(TryTable, Vec<Instruction<'a>>),
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
            Operator::TryTable { try_table } => {
                let (body, end) = read_instructions(operators)?;
                if end != End::Block {
                    return Err(Error::Unsupported("else outside if".into()));
                }
                instructions.push(Instruction::TryTable(try_table, body));
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

struct MemoryState {
    owner: usize,
    local: usize,
    capacity: usize,
    owner_count: usize,
    is64: bool,
}

struct TableState<'a> {
    element_type: RefType,
    owner: usize,
    local: usize,
    values: &'a [u64],
    info: &'a TableInfo,
    targets: &'a [usize],
    is64: bool,
}

struct Compiler<'a> {
    uses_gc: bool,
    metered: bool,
    gc_types: &'a [Option<GcType>],
    gc_ids: &'a [u32],
    max_gc_roots: usize,
    module_index: usize,
    resolved_functions: &'a [usize],
    global_types: &'a [FuncType],
    type_targets: Vec<HashSet<usize>>,
    cast_targets: Vec<HashSet<usize>>,
    tag_bindings: &'a [u32],
    tag_types: &'a [u32],
    exceptions: bool,
    source: String,
    declarations: String,
    local_count: usize,
    local_types: Vec<ValType>,
    vector_names: HashSet<String>,
    reference_names: HashMap<String, RefType>,
    next_value: usize,
    next_frame: usize,
    types: &'a [FuncType],
    function_types: &'a [u32],
    memories: Vec<MemoryState>,
    memory_limit_pages: usize,
    globals_mutable: &'a [bool],
    global_value_types: &'a [GlobalType],
    global_bindings: &'a [(usize, usize)],
    tables: Vec<TableState<'a>>,
    element_info: &'a TableInfo,
    dynamic_elements: &'a [(usize, usize, ConstExpr<'a>)],
    data_offsets: &'a [usize],
    recursive: bool,
    tail_calls: bool,
    frames: Vec<Frame>,
}

impl Compiler<'_> {
    fn reference_test(&self, value: &str, ty: RefType) -> Result<String, Error> {
        let nonnull = match ty.heap_type() {
            HeapType::Concrete(UnpackedIndex::Module(index))
                if self.gc_ids[index as usize] != 0 =>
            {
                format!(
                    "(({value} >> 32u) == 3ul && gc_type_match(uint(gc.heap[uint({value})]), {}u))",
                    self.gc_ids[index as usize]
                )
            }
            HeapType::Concrete(UnpackedIndex::Module(index)) => {
                let targets = self
                    .cast_targets
                    .get(index as usize)
                    .ok_or_else(|| Error::Unsupported("cast type is out of range".into()))?;
                let mut targets: Vec<_> = targets.iter().copied().collect();
                targets.sort_unstable();
                if targets.is_empty() {
                    "false".into()
                } else {
                    targets
                        .iter()
                        .map(|target| format!("{value} == {}ul", target + 1))
                        .collect::<Vec<_>>()
                        .join(" || ")
                }
            }
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Func | AbstractHeapType::Extern | AbstractHeapType::Exn,
            } => "true".into(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::NoFunc | AbstractHeapType::NoExtern | AbstractHeapType::NoExn,
            } => "false".into(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Any,
            } => "true".into(),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Eq,
            } => format!("(({value}>>32u)==1ul || ({value}>>32u)==3ul)"),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::I31,
            } => format!("({value}>>32u)==1ul"),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::Struct | AbstractHeapType::Array,
            } => format!(
                "(({value}>>32u)==3ul && gc_kind(uint(gc.heap[uint({value})]))=={}u)",
                if matches!(
                    ty.heap_type(),
                    HeapType::Abstract {
                        ty: AbstractHeapType::Struct,
                        ..
                    }
                ) {
                    1
                } else {
                    2
                }
            ),
            HeapType::Abstract {
                shared: false,
                ty: AbstractHeapType::None,
            } => "false".into(),
            _ => return Err(Error::Unsupported("unsupported reference cast type".into())),
        };
        Ok(if ty.is_nullable() {
            format!("({value} == 0ul || ({nonnull}))")
        } else {
            format!("({value} != 0ul && ({nonnull}))")
        })
    }

    fn flow_guard(&self) -> &'static str {
        if self.exceptions {
            "if (flow == 0u && exceptions.pending == 0u) {\n"
        } else {
            "if (flow == 0u) {\n"
        }
    }

    fn emit_throw(&mut self, tag: u32, stack: &mut Vec<String>) -> Result<(), Error> {
        let params = self.types[self.tag_types[tag as usize] as usize]
            .params()
            .to_vec();
        if stack.len() < params.len() {
            return Err(Error::Unsupported("missing exception payload".into()));
        }
        self.publish_gc_roots(stack);
        let args = stack.split_off(stack.len() - params.len());
        self.source.push_str(&format!("exceptions.pending = exception_alloc(gc, trap); if (trap != 0u) return;\nexceptions.tags[exceptions.pending - 1u] = {}u;\n", self.tag_bindings[tag as usize]));
        let mut word = 0;
        for (arg, ty) in args.iter().zip(params) {
            if ty == ValType::V128 {
                self.source.push_str(&format!("exceptions.values[(exceptions.pending - 1u) * EXCEPTION_WORDS + {word}u] = ({arg}).x;\nexceptions.values[(exceptions.pending - 1u) * EXCEPTION_WORDS + {}u] = ({arg}).y;\n", word + 1));
            } else {
                self.source.push_str(&format!("exceptions.values[(exceptions.pending - 1u) * EXCEPTION_WORDS + {word}u] = {arg};\n"));
            }
            word += value_words(ty);
        }
        Ok(())
    }

    fn emit_try_table(
        &mut self,
        table: &TryTable,
        body: &[Instruction<'_>],
        stack: &mut Vec<String>,
    ) -> Result<(), Error> {
        self.emit_block(table.ty, body, stack, false)?;
        if table.catches.is_empty() {
            return Ok(());
        }
        self.source.push_str("if (exceptions.pending != 0u) {\n");
        for (index, catch) in table.catches.iter().enumerate() {
            let (tag, label, reference) = match *catch {
                Catch::One { tag, label } => (Some(tag), label, false),
                Catch::OneRef { tag, label } => (Some(tag), label, true),
                Catch::All { label } => (None, label, false),
                Catch::AllRef { label } => (None, label, true),
            };
            let condition = tag.map_or_else(
                || "true".into(),
                |tag| {
                    format!(
                        "exceptions.tags[exceptions.pending - 1u] == {}u",
                        self.tag_bindings[tag as usize]
                    )
                },
            );
            self.source.push_str(&format!(
                "{}if ({condition}) {{\n",
                if index == 0 { "" } else { "else " }
            ));
            let mut args = Vec::new();
            if let Some(tag) = tag {
                let mut word = 0;
                for ty in self.types[self.tag_types[tag as usize] as usize].params() {
                    let value = format!(
                        "exceptions.values[(exceptions.pending - 1u) * EXCEPTION_WORDS + {word}u]"
                    );
                    args.push(if *ty == ValType::V128 { format!("ulong2({value}, exceptions.values[(exceptions.pending - 1u) * EXCEPTION_WORDS + {}u])", word + 1) } else { value });
                    word += value_words(*ty);
                }
            }
            if reference {
                args.push("ulong(exceptions.pending)".into());
            }
            self.emit_branch(label, &args, None)?;
            self.source.push_str("exceptions.pending = 0u;\n}\n");
        }
        self.source.push_str("}\n");
        Ok(())
    }

    fn emit_dispatch_call(
        &mut self,
        target: &str,
        ty: &FuncType,
        args: Vec<String>,
        stack: &mut Vec<String>,
    ) {
        let id = self.next_value;
        self.next_value += 1;
        let input_words: usize = ty.params().iter().map(|ty| value_words(*ty)).sum();
        let output_words: usize = ty.results().iter().map(|ty| value_words(*ty)).sum();
        self.source.push_str(&format!(
            "ulong call_args{id}[{}] = {{}}; ulong call_results{id}[{}] = {{}};\n",
            input_words.max(1),
            output_words.max(1)
        ));
        let mut word = 0;
        for (arg, param) in args.iter().zip(ty.params()) {
            if *param == ValType::V128 {
                self.source.push_str(&format!(
                    "call_args{id}[{word}] = ({arg}).x; call_args{id}[{}] = ({arg}).y;\n",
                    word + 1
                ));
            } else {
                self.source
                    .push_str(&format!("call_args{id}[{word}] = {arg};\n"));
            }
            word += value_words(*param);
        }
        self.source.push_str(&format!("invoke(uint({target}), call_args{id}, call_results{id}, states, trap, depth + 1u);\nif (trap != 0u) return;\n"));
        let mut word = 0;
        for result_ty in ty.results() {
            if *result_ty == ValType::V128 {
                self.emit_vector(
                    stack,
                    format!(
                        "ulong2(call_results{id}[{word}], call_results{id}[{}])",
                        word + 1
                    ),
                );
            } else {
                self.emit_typed(stack, format!("call_results{id}[{word}]"), *result_ty);
            }
            word += value_words(*result_ty);
        }
    }

    fn emit_tail_call(
        &mut self,
        operator: &Operator<'_>,
        stack: &mut Vec<String>,
    ) -> Result<(), Error> {
        let (ty, target) = match operator {
            Operator::ReturnCall { function_index } => {
                let local = *function_index as usize;
                let ty = self.types[self.function_types[local] as usize].clone();
                (ty, format!("{}u", self.resolved_functions[local] + 1))
            }
            Operator::ReturnCallIndirect {
                type_index,
                table_index,
            } => {
                let ty = self.types[*type_index as usize].clone();
                let element = pop(stack)?;
                let size = self.table_size(*table_index)?;
                self.source.push_str(&format!(
                    "if ({element} >= {size}) {{ trap = 6u; return; }}\n"
                ));
                let target = self.fresh_value();
                self.source
                    .push_str(&format!("{target} = table[uint({element})];\n"));
                (ty, target)
            }
            Operator::ReturnCallRef { type_index } => {
                let ty = self.types[*type_index as usize].clone();
                (ty, pop(stack)?)
            }
            _ => return Err(Error::Unsupported("unsupported tail call".into())),
        };
        if stack.len() < ty.params().len() {
            return Err(Error::Unsupported("tail-call argument is missing".into()));
        }
        let args = stack.split_off(stack.len() - ty.params().len());
        if !matches!(operator, Operator::ReturnCall { .. }) {
            let null_trap = if matches!(operator, Operator::ReturnCallRef { .. }) {
                12
            } else {
                6
            };
            self.source.push_str(&format!(
                "if ({target} == 0ul) {{ trap = {null_trap}u; return; }}\nswitch ({target}) {{\n"
            ));
            for index in 0..self.global_types.len() {
                if self.type_targets[match operator {
                    Operator::ReturnCallIndirect { type_index, .. }
                    | Operator::ReturnCallRef { type_index } => *type_index as usize,
                    _ => unreachable!(),
                }]
                .contains(&index)
                {
                    self.source
                        .push_str(&format!("case {}u: break;\n", index + 1));
                }
            }
            self.source.push_str("default: trap = 7u; return;\n}\n");
        }
        let mut word = 0;
        for (arg, param) in args.iter().zip(ty.params()) {
            if *param == ValType::V128 {
                self.source.push_str(&format!(
                    "tail.args[{word}] = ({arg}).x; tail.args[{}] = ({arg}).y;\n",
                    word + 1
                ));
            } else {
                self.source
                    .push_str(&format!("tail.args[{word}] = {arg};\n"));
            }
            word += value_words(*param);
        }
        self.source
            .push_str(&format!("tail.next = uint({target}); return;\n"));
        Ok(())
    }

    fn table_size(&mut self, index: u32) -> Result<String, Error> {
        let table_state = self
            .tables
            .get(index as usize)
            .ok_or_else(|| Error::Unsupported("table is missing".into()))?;
        self.source.push_str(&format!(
            "table = states[{}].table[{}]; table_size = &states[{}].table_size[{}];\n",
            table_state.owner, table_state.local, table_state.owner, table_state.local
        ));
        Ok(if table_state.info.mutable {
            format!(
                "ulong(states[{}].table_size[{}])",
                table_state.owner, table_state.local
            )
        } else {
            format!("{}ul", table_state.values.len())
        })
    }

    fn select_memory(&mut self, index: u32) -> Result<&MemoryState, Error> {
        let memory = self
            .memories
            .get(index as usize)
            .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
        self.source.push_str(&format!(
            "memory = states[{}].memory[{}]; pages = states[{}].pages[{}];\n",
            memory.owner, memory.local, memory.owner, memory.local
        ));
        Ok(memory)
    }

    fn fresh_value(&mut self) -> String {
        let name = format!("v{}", self.next_value);
        self.next_value += 1;
        self.declarations
            .push_str(&format!("ulong {name} = 0ul;\n"));
        name
    }

    fn fresh_vector(&mut self) -> String {
        let name = format!("v{}", self.next_value);
        self.next_value += 1;
        self.declarations
            .push_str(&format!("ulong2 {name} = ulong2(0ul);\n"));
        self.vector_names.insert(name.clone());
        name
    }

    fn fresh_typed(&mut self, ty: ValType) -> String {
        if ty == ValType::V128 {
            self.fresh_vector()
        } else {
            let name = self.fresh_value();
            if let ValType::Ref(r) = ty {
                self.reference_names.insert(name.clone(), r);
            }
            name
        }
    }
    fn emit_typed(&mut self, stack: &mut Vec<String>, expression: String, ty: ValType) {
        let name = self.fresh_typed(ty);
        self.source.push_str(&format!("{name} = {expression};\n"));
        stack.push(name);
    }

    fn emit_vector(&mut self, stack: &mut Vec<String>, expression: String) {
        let name = self.fresh_vector();
        self.source.push_str(&format!("{name} = {expression};\n"));
        stack.push(name);
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
        let is64 = self.select_memory(memarg.memory)?.is64;
        let index = self.fresh_value();
        let address = if is64 {
            format!("ulong({address})")
        } else {
            format!("ulong(uint({address}))")
        };
        self.source.push_str(&format!(
            "if ({address} > ulong(pages[0]) * 65536ul || {}ul > ulong(pages[0]) * 65536ul - {address} || {width}ul > ulong(pages[0]) * 65536ul - {address} - {}ul) {{ trap = 4u; return; }}\n",
            memarg.offset, memarg.offset,
        ));
        self.source
            .push_str(&format!("{index} = {address} + {}ul;\n", memarg.offset));
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
            BlockType::Empty => (Vec::new(), Vec::new()),
            BlockType::Type(ty) if is_supported_value(ty) => (Vec::new(), vec![ty]),
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
                        "unsupported block signature value type".into(),
                    ));
                }
                (ty.params().to_vec(), ty.results().to_vec())
            }
            _ => {
                return Err(Error::Unsupported(
                    "unsupported block signature value type".into(),
                ));
            }
        };
        self.next_frame += 1;
        Ok(Frame {
            id: self.next_frame,
            params: params.into_iter().map(|ty| self.fresh_typed(ty)).collect(),
            results: results.into_iter().map(|ty| self.fresh_typed(ty)).collect(),
            loop_label,
        })
    }

    fn assign_slots(&mut self, slots: &[String], values: &[String]) {
        if values.iter().any(|value| slots.contains(value)) {
            // Loop branches can swap parameter slots; read all values before writing any.
            let snapshots: Vec<_> = values
                .iter()
                .map(|value| {
                    let name = if self.vector_names.contains(value) {
                        self.fresh_vector()
                    } else {
                        self.fresh_value()
                    };
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
            self.source.push_str(self.flow_guard());
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
                    Operator::Throw { tag_index } => {
                        self.emit_throw(*tag_index, stack)?;
                        reachable = false;
                        break;
                    }
                    Operator::ThrowRef => {
                        let reference = pop(stack)?;
                        self.source.push_str(&format!("if ({reference} == 0ul) {{ trap = 9u; return; }}\nexceptions.pending = uint({reference});\n"));
                        reachable = false;
                        break;
                    }
                    Operator::BrOnNull { relative_depth } => {
                        let reference = pop(stack)?;
                        self.emit_branch(
                            *relative_depth,
                            stack,
                            Some(format!("({reference} == 0ul)")),
                        )?;
                        stack.push(reference);
                        true
                    }
                    Operator::BrOnNonNull { relative_depth } => {
                        let reference = stack
                            .last()
                            .cloned()
                            .ok_or_else(|| Error::Unsupported("missing reference".into()))?;
                        self.emit_branch(*relative_depth, stack, Some(reference))?;
                        pop(stack)?;
                        true
                    }
                    Operator::BrOnCast {
                        relative_depth,
                        to_ref_type,
                        ..
                    }
                    | Operator::BrOnCastFail {
                        relative_depth,
                        to_ref_type,
                        ..
                    } => {
                        let value = stack
                            .last()
                            .ok_or_else(|| Error::Unsupported("missing reference".into()))?;
                        let test = self.reference_test(value, *to_ref_type)?;
                        let condition = if matches!(operator, Operator::BrOnCastFail { .. }) {
                            format!("!{test}")
                        } else {
                            test
                        };
                        self.emit_branch(*relative_depth, stack, Some(condition))?;
                        true
                    }
                    Operator::BrIf { relative_depth } => {
                        let condition = pop(stack)?;
                        self.emit_branch(*relative_depth, stack, Some(condition))?;
                        true
                    }
                    Operator::BrTable { targets } => {
                        let index = pop(stack)?;
                        // Group identical destinations. A long chain of nested
                        // conditionals can exhaust the Metal compiler's stack.
                        let mut cases: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
                        for (case, depth) in targets.targets().enumerate() {
                            let depth = depth?;
                            if depth != targets.default() {
                                cases.entry(depth).or_default().push(case);
                            }
                        }
                        self.source
                            .push_str(&format!("switch (uint({index})) {{\n"));
                        for (depth, indices) in cases {
                            for case in indices {
                                self.source.push_str(&format!("case {case}u:\n"));
                            }
                            self.emit_branch(depth, stack, None)?;
                            self.source.push_str("break;\n");
                        }
                        self.source.push_str("default:\n");
                        self.emit_branch(targets.default(), stack, None)?;
                        self.source.push_str("break;\n}\n");
                        reachable = false;
                        break;
                    }
                    Operator::Return => {
                        self.emit_branch((self.frames.len() - 1) as u32, stack, None)?;
                        reachable = false;
                        break;
                    }
                    Operator::ReturnCall { .. }
                    | Operator::ReturnCallIndirect { .. }
                    | Operator::ReturnCallRef { .. } => {
                        self.emit_tail_call(operator, stack)?;
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
                        self.exceptions
                            && matches!(
                                operator,
                                Operator::Call { .. }
                                    | Operator::CallIndirect { .. }
                                    | Operator::CallRef { .. }
                            )
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
                Instruction::TryTable(table, body) => {
                    self.emit_try_table(table, body, stack)?;
                    true
                }
                Instruction::If(blockty, then_body, else_body) => {
                    self.emit_if(*blockty, then_body, else_body, stack)?;
                    true
                }
            };
            if guard {
                self.source.push_str(self.flow_guard());
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
            if self.metered {
                self.source.push_str(
                    "if (*states[0].fuel == 0ul) { trap = 17u; return; }\n--*states[0].fuel;\n",
                );
            }
        } else {
            self.source.push_str("{\n");
        }
        let reachable = self.emit_instructions(body, &mut inner_stack)?;
        self.finish_path(reachable, &mut inner_stack, base, &frame.results)?;
        if loop_label {
            if self.exceptions {
                self.source
                    .push_str("if (exceptions.pending != 0u) break;\n");
            }
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
        if self.uses_gc
            && matches!(
                operator,
                Operator::Call { .. }
                    | Operator::CallRef { .. }
                    | Operator::CallIndirect { .. }
                    | Operator::StructNew { .. }
                    | Operator::StructNewDefault { .. }
                    | Operator::ArrayNew { .. }
                    | Operator::ArrayNewDefault { .. }
                    | Operator::ArrayNewFixed { .. }
                    | Operator::ArrayNewData { .. }
                    | Operator::ArrayNewElem { .. }
            )
        {
            self.publish_gc_roots(stack);
        }
        if self.emit_gc_operator(operator, stack)? {
            return Ok(());
        }
        match operator {
            Operator::Nop => {}
            Operator::I32Const { value } => stack.push(format!("{}ul", *value as u32)),
            Operator::I64Const { value } => stack.push(format!("{}ul", *value as u64)),
            Operator::F32Const { value } => stack.push(format!("{}ul", value.bits())),
            Operator::F64Const { value } => stack.push(format!("{}ul", value.bits())),
            Operator::V128Const { value } => {
                let bytes = value.bytes();
                self.emit_vector(
                    stack,
                    format!(
                        "ulong2({}ul, {}ul)",
                        u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                        u64::from_le_bytes(bytes[8..].try_into().unwrap())
                    ),
                );
            }
            Operator::RefNull { .. } => {
                stack.push("0ul".into());
            }
            Operator::RefFunc { function_index } => {
                stack.push(format!(
                    "{}ul",
                    self.resolved_functions[*function_index as usize] + 1
                ));
            }
            Operator::RefAsNonNull => {
                let reference = pop(stack)?;
                self.source.push_str(&format!(
                    "if ({reference} == 0ul) {{ trap = 9u; return; }}\n"
                ));
                stack.push(reference);
            }
            Operator::RefTestNonNull { hty }
            | Operator::RefTestNullable { hty }
            | Operator::RefCastNonNull { hty }
            | Operator::RefCastNullable { hty } => {
                let value = pop(stack)?;
                let nullable = matches!(
                    operator,
                    Operator::RefTestNullable { .. } | Operator::RefCastNullable { .. }
                );
                let test = self.reference_test(&value, RefType::new(nullable, *hty).unwrap())?;
                if matches!(
                    operator,
                    Operator::RefTestNonNull { .. } | Operator::RefTestNullable { .. }
                ) {
                    self.emit_value(stack, format!("uint({test})"));
                } else {
                    self.source
                        .push_str(&format!("if (!{test}) {{ trap = 13u; return; }}\n"));
                    stack.push(value);
                }
            }
            Operator::RefIsNull => {
                let value = pop(stack)?;
                self.emit_value(stack, format!("uint({value} == 0ul)"));
            }
            Operator::TableGet { table } => {
                let index = pop(stack)?;
                let size = self.table_size(*table)?;
                self.source.push_str(&format!(
                    "if ({index} >= {size}) {{ trap = 14u; return; }}\n"
                ));
                self.emit_typed(
                    stack,
                    format!("table[{index}]"),
                    ValType::Ref(self.tables[*table as usize].element_type),
                );
            }
            Operator::TableSet { table } => {
                let value = pop(stack)?;
                let index = pop(stack)?;
                let size = self.table_size(*table)?;
                self.source.push_str(&format!("if ({index} >= {size}) {{ trap = 14u; return; }}\ntable[{index}] = ulong({value});\n"));
            }
            Operator::TableSize { table } => {
                let size = self.table_size(*table)?;
                self.emit_value(stack, size);
            }
            Operator::TableGrow { table } => {
                let count = pop(stack)?;
                let value = pop(stack)?;
                self.table_size(*table)?;
                let table_state = &self.tables[*table as usize];
                let capacity = table_state.info.capacity;
                let failure = if table_state.is64 {
                    "0xfffffffffffffffful"
                } else {
                    "0xfffffffful"
                };
                let result = self.fresh_value();
                let slot = self.fresh_value();
                self.source.push_str(&format!(
                    "{result} = {failure};\nif ({count} <= {capacity}ul - ulong(*table_size)) {{\n{result} = *table_size;\nfor ({slot} = *table_size; {slot} < ulong(*table_size) + {count}; ++{slot}) table[{slot}] = ulong({value});\n*table_size += uint({count});\n}}\n"));
                stack.push(result);
            }
            Operator::TableFill { table } => {
                let count = pop(stack)?;
                let value = pop(stack)?;
                let destination = pop(stack)?;
                let size = self.table_size(*table)?;
                let slot = self.fresh_value();
                self.source.push_str(&format!(
                    "if ({destination} > {size} || {count} > {size} - {destination}) {{ trap = 14u; return; }}\nfor ({slot} = 0ul; {slot} < {count}; ++{slot}) table[{destination} + {slot}] = ulong({value});\n"));
            }
            Operator::TableCopy {
                dst_table,
                src_table,
            } => {
                let count = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let dst_size = self.table_size(*dst_table)?;
                let dst = &self.tables[*dst_table as usize];
                let dst_ptr = format!("states[{}].table[{}]", dst.owner, dst.local);
                let src_size = self.table_size(*src_table)?;
                let src = &self.tables[*src_table as usize];
                let src_ptr = format!("states[{}].table[{}]", src.owner, src.local);
                let slot = self.fresh_value();
                self.source.push_str(&format!(
                    "if ({destination} > {dst_size} || {count} > {dst_size} - {destination} || {source} > {src_size} || {count} > {src_size} - {source}) {{ trap = 14u; return; }}\nif ({dst_ptr} == {src_ptr} && {destination} > {source}) {{ for ({slot} = {count}; {slot} != 0ul; --{slot}) {dst_ptr}[{destination} + {slot} - 1ul] = {src_ptr}[{source} + {slot} - 1ul]; }} else {{ for ({slot} = 0ul; {slot} < {count}; ++{slot}) {dst_ptr}[{destination} + {slot}] = {src_ptr}[{source} + {slot}]; }}\n"));
            }
            Operator::TableInit { elem_index, table } => {
                self.element_info
                    .elements
                    .get(*elem_index as usize)
                    .ok_or_else(|| Error::Unsupported("element index is out of range".into()))?;
                let count = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let size = self.table_size(*table)?;
                let slot = self.fresh_value();
                let values = if self
                    .dynamic_elements
                    .iter()
                    .any(|(segment, _, _)| *segment == *elem_index as usize)
                {
                    format!("states[{}].element_values[{elem_index}]", self.module_index)
                } else {
                    format!("m{}_element_{elem_index}", self.module_index)
                };
                self.source.push_str(&format!(
                    "if ({destination} > {size} || {count} > {size} - {destination} || {source} > ulong(element_lengths[{elem_index}u]) || {count} > ulong(element_lengths[{elem_index}u]) - {source}) {{ trap = 14u; return; }}\nfor ({slot} = 0ul; {slot} < {count}; ++{slot}) table[{destination} + {slot}] = {values}[{source} + {slot}];\n"));
            }
            Operator::ElemDrop { elem_index } => {
                self.element_info
                    .elements
                    .get(*elem_index as usize)
                    .ok_or_else(|| Error::Unsupported("element index is out of range".into()))?;
                self.source
                    .push_str(&format!("element_lengths[{elem_index}u] = 0u;\n"));
                if self.uses_gc
                    && self
                        .dynamic_elements
                        .iter()
                        .any(|(s, _, _)| *s == *elem_index as usize)
                {
                    let length = self.element_info.elements[*elem_index as usize].len();
                    self.source.push_str(&format!("for(uint j=0u;j<{length}u;++j) states[{}].element_values[{elem_index}][j]=0ul;\n",self.module_index));
                }
            }
            Operator::GlobalGet { global_index } => {
                let index = *global_index as usize;
                if index >= self.globals_mutable.len() {
                    return Err(Error::Unsupported("global index is out of range".into()));
                }
                let (owner, slot) = self.global_bindings[index];
                if self.global_value_types[index].content_type == ValType::V128 {
                    self.emit_vector(stack, format!("ulong2(states[{owner}].globals[{slot}u], states[{owner}].globals[{}u])", slot + 1));
                } else {
                    self.emit_typed(
                        stack,
                        format!("states[{owner}].globals[{slot}u]"),
                        self.global_value_types[index].content_type,
                    );
                }
            }
            Operator::GlobalSet { global_index } => {
                let index = *global_index as usize;
                if !self.globals_mutable.get(index).copied().unwrap_or(false) {
                    return Err(Error::Unsupported("global is immutable or missing".into()));
                }
                let value = pop(stack)?;
                let (owner, slot) = self.global_bindings[index];
                if self.global_value_types[index].content_type == ValType::V128 {
                    self.source.push_str(&format!("states[{owner}].globals[{slot}u] = {value}.x;\nstates[{owner}].globals[{}u] = {value}.y;\n", slot + 1));
                } else {
                    self.source
                        .push_str(&format!("states[{owner}].globals[{slot}u] = {value};\n"));
                }
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
            Operator::V128Load { memarg } => {
                let address = pop(stack)?;
                let index = self.memory_address(&address, memarg, 16)?;
                self.emit_vector(stack, format!("wasm_simd_load(memory, {index})"));
            }
            Operator::V128Load8x8S { memarg }
            | Operator::V128Load8x8U { memarg }
            | Operator::V128Load16x4S { memarg }
            | Operator::V128Load16x4U { memarg }
            | Operator::V128Load32x2S { memarg }
            | Operator::V128Load32x2U { memarg } => {
                let source_bits = match operator {
                    Operator::V128Load8x8S { .. } | Operator::V128Load8x8U { .. } => 8,
                    Operator::V128Load16x4S { .. } | Operator::V128Load16x4U { .. } => 16,
                    _ => 32,
                };
                let signed = matches!(
                    operator,
                    Operator::V128Load8x8S { .. }
                        | Operator::V128Load16x4S { .. }
                        | Operator::V128Load32x2S { .. }
                );
                let address = pop(stack)?;
                let index = self.memory_address(&address, memarg, 8)?;
                self.emit_vector(
                    stack,
                    format!("wasm_simd_load_extend(memory, {index}, {source_bits}u, {signed})"),
                );
            }
            Operator::V128Load8Splat { memarg }
            | Operator::V128Load16Splat { memarg }
            | Operator::V128Load32Splat { memarg }
            | Operator::V128Load64Splat { memarg }
            | Operator::V128Load32Zero { memarg }
            | Operator::V128Load64Zero { memarg } => {
                let bits = match operator {
                    Operator::V128Load8Splat { .. } => 8,
                    Operator::V128Load16Splat { .. } => 16,
                    Operator::V128Load32Splat { .. } | Operator::V128Load32Zero { .. } => 32,
                    _ => 64,
                };
                let address = pop(stack)?;
                let index = self.memory_address(&address, memarg, bits / 8)?;
                let part = format!("wasm_simd_read(memory, {index}, {}u)", bits / 8);
                let expression = if matches!(
                    operator,
                    Operator::V128Load32Zero { .. } | Operator::V128Load64Zero { .. }
                ) {
                    format!("ulong2({part}, 0ul)")
                } else {
                    format!("wasm_simd_splat({part}, {bits}u)")
                };
                self.emit_vector(stack, expression);
            }
            Operator::V128Store { memarg } => {
                let value = pop(stack)?;
                let address = pop(stack)?;
                let index = self.memory_address(&address, memarg, 16)?;
                self.source
                    .push_str(&format!("wasm_simd_store(memory, {index}, {value});\n"));
            }
            Operator::V128Load8Lane { memarg, lane }
            | Operator::V128Load16Lane { memarg, lane }
            | Operator::V128Load32Lane { memarg, lane }
            | Operator::V128Load64Lane { memarg, lane } => {
                let bits = match operator {
                    Operator::V128Load8Lane { .. } => 8,
                    Operator::V128Load16Lane { .. } => 16,
                    Operator::V128Load32Lane { .. } => 32,
                    _ => 64,
                };
                let value = pop(stack)?;
                let address = pop(stack)?;
                let index = self.memory_address(&address, memarg, bits / 8)?;
                self.emit_vector(stack, format!("wasm_simd_replace({value}, {bits}u, {lane}u, wasm_simd_read(memory, {index}, {}u))", bits / 8));
            }
            Operator::V128Store8Lane { memarg, lane }
            | Operator::V128Store16Lane { memarg, lane }
            | Operator::V128Store32Lane { memarg, lane }
            | Operator::V128Store64Lane { memarg, lane } => {
                let bits = match operator {
                    Operator::V128Store8Lane { .. } => 8,
                    Operator::V128Store16Lane { .. } => 16,
                    Operator::V128Store32Lane { .. } => 32,
                    _ => 64,
                };
                let value = pop(stack)?;
                let address = pop(stack)?;
                let index = self.memory_address(&address, memarg, bits / 8)?;
                self.source.push_str(&format!("wasm_simd_write(memory, {index}, {}u, wasm_simd_lane({value}, {bits}u, {lane}u));\n", bits / 8));
            }
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
            Operator::MemorySize { mem } => {
                self.select_memory(*mem)?;
                self.emit_value(stack, "pages[0]".into());
            }
            Operator::MemoryGrow { mem } => {
                let memory = self.select_memory(*mem)?;
                let capacity = memory.capacity;
                let owner = memory.owner;
                let owner_count = memory.owner_count;
                let failure = if memory.is64 {
                    "0xfffffffffffffffful"
                } else {
                    "0xfffffffful"
                };
                let delta = pop(stack)?;
                let prior = self.fresh_value();
                let total = self.fresh_value();
                let pages = (0..owner_count)
                    .map(|index| format!("ulong(states[{owner}].pages[{index}][0])"))
                    .collect::<Vec<_>>()
                    .join(" + ");
                self.source.push_str(&format!(
                    "{prior} = pages[0];\n{total} = {pages};\nif ({delta} <= ulong({}u - pages[0]) && {total} <= {}ul && {delta} <= {}ul - {total}) pages[0] += uint({delta}); else {prior} = {failure};\n",
                    capacity / 65536,
                    self.memory_limit_pages,
                    self.memory_limit_pages
                ));
                stack.push(prior);
            }
            Operator::MemoryFill { mem } => {
                self.select_memory(*mem)?;
                let length = pop(stack)?;
                let value = pop(stack)?;
                let destination = pop(stack)?;
                let offset = self.fresh_value();
                self.source.push_str(&format!(
                    "if (ulong({destination}) > ulong(pages[0]) * 65536ul || ulong({length}) > ulong(pages[0]) * 65536ul - ulong({destination})) {{ trap = 4u; return; }}\nfor ({offset} = 0ul; {offset} < ulong({length}); ++{offset}) memory[ulong({destination}) + {offset}] = uchar({value});\n"
                ));
            }
            Operator::MemoryCopy { dst_mem, src_mem } => {
                let dst = self
                    .memories
                    .get(*dst_mem as usize)
                    .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
                let dst_ptr = format!("states[{}].memory[{}]", dst.owner, dst.local);
                let dst_pages = format!("states[{}].pages[{}][0]", dst.owner, dst.local);
                let src = self
                    .memories
                    .get(*src_mem as usize)
                    .ok_or_else(|| Error::Unsupported("memory is missing".into()))?;
                let src_ptr = format!("states[{}].memory[{}]", src.owner, src.local);
                let src_pages = format!("states[{}].pages[{}][0]", src.owner, src.local);
                let length = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let offset = self.fresh_value();
                self.source.push_str(&format!(
                    "if (ulong({destination}) > ulong({dst_pages}) * 65536ul || ulong({length}) > ulong({dst_pages}) * 65536ul - ulong({destination}) || ulong({source}) > ulong({src_pages}) * 65536ul || ulong({length}) > ulong({src_pages}) * 65536ul - ulong({source})) {{ trap = 4u; return; }}\nif ({dst_ptr} == {src_ptr} && {destination} > {source}) {{\nfor ({offset} = ulong({length}); {offset} > 0ul; --{offset}) {dst_ptr}[ulong({destination}) + {offset} - 1ul] = {src_ptr}[ulong({source}) + {offset} - 1ul];\n}} else {{\nfor ({offset} = 0ul; {offset} < ulong({length}); ++{offset}) {dst_ptr}[ulong({destination}) + {offset}] = {src_ptr}[ulong({source}) + {offset}];\n}}\n"
                ));
            }
            Operator::MemoryInit { data_index, mem } => {
                self.select_memory(*mem)?;
                let offset = *self.data_offsets.get(*data_index as usize).ok_or_else(|| {
                    Error::Unsupported("data segment index is out of range".into())
                })?;
                let length = pop(stack)?;
                let source = pop(stack)?;
                let destination = pop(stack)?;
                let byte = self.fresh_value();
                self.source.push_str(&format!(
                    "if (ulong({destination}) > ulong(pages[0]) * 65536ul || ulong({length}) > ulong(pages[0]) * 65536ul - ulong({destination}) || ulong({source}) > ulong(data_lengths[{data_index}u]) || ulong({length}) > ulong(data_lengths[{data_index}u]) - ulong({source})) {{ trap = 4u; return; }}\nfor ({byte} = 0ul; {byte} < ulong({length}); ++{byte}) memory[ulong({destination}) + {byte}] = passive_data[{offset}ul + ulong({source}) + {byte}];\n"
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
                        "unsupported function signature value type".into(),
                    ));
                }
                if stack.len() < ty.params().len() {
                    return Err(Error::Unsupported("call argument is missing".into()));
                }
                let args = stack.split_off(stack.len() - ty.params().len());
                if self.tail_calls {
                    let ty = ty.clone();
                    let target = self.resolved_functions[index];
                    self.emit_dispatch_call(&format!("{target}u"), &ty, args, stack);
                    return Ok(());
                }
                let result_types = ty.results().to_vec();
                let results: Vec<_> = result_types
                    .into_iter()
                    .map(|ty| self.fresh_typed(ty))
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
            Operator::CallRef { type_index } => {
                let ty = self.types[*type_index as usize].clone();
                let target = pop(stack)?;
                if stack.len() < ty.params().len() {
                    return Err(Error::Unsupported("call_ref argument is missing".into()));
                }
                let args = stack.split_off(stack.len() - ty.params().len());
                self.source.push_str(&format!(
                    "if ({target} == 0ul) {{ trap = 12u; return; }}\nswitch ({target}) {{\n"
                ));
                let results: Vec<_> = ty
                    .results()
                    .iter()
                    .map(|ty| self.fresh_typed(*ty))
                    .collect();
                let mut targets: Vec<_> = self.type_targets[*type_index as usize]
                    .iter()
                    .copied()
                    .collect();
                targets.sort_unstable();
                for index in targets {
                    self.source.push_str(&format!("case {}u: {{\n", index + 1));
                    if self.tail_calls {
                        let mut returned = Vec::new();
                        self.emit_dispatch_call(
                            &format!("{index}u"),
                            &ty,
                            args.clone(),
                            &mut returned,
                        );
                        self.assign_slots(&results, &returned);
                    } else {
                        let mut call_args = args.clone();
                        call_args.extend(results.iter().cloned());
                        call_args.extend(["states".into(), "trap".into()]);
                        if self.recursive {
                            call_args.push("depth + 1u".into());
                        }
                        self.source
                            .push_str(&format!("f{index}({});\n", call_args.join(", ")));
                    }
                    self.source.push_str("break;\n}\n");
                }
                self.source
                    .push_str("default: trap = 7u; return;\n}\nif (trap != 0u) return;\n");
                stack.extend(results);
            }
            Operator::CallIndirect {
                type_index,
                table_index,
            } => {
                self.tables
                    .get(*table_index as usize)
                    .ok_or_else(|| Error::Unsupported("table is missing".into()))?;
                let table_mutable = self.tables[*table_index as usize].info.mutable;
                let table_values = self.tables[*table_index as usize].values;
                let table_targets = self.tables[*table_index as usize].targets;
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
                        "unsupported indirect signature value type".into(),
                    ));
                }
                let element = pop(stack)?;
                if stack.len() < ty.params().len() {
                    return Err(Error::Unsupported(
                        "indirect call argument is missing".into(),
                    ));
                }
                let args = stack.split_off(stack.len() - ty.params().len());
                let size = self.table_size(*table_index)?;
                if self.tail_calls {
                    let ty = ty.clone();
                    self.source.push_str(&format!(
                        "if ({element} >= {size}) {{ trap = 6u; return; }}\n"
                    ));
                    let target = self.fresh_value();
                    self.source.push_str(&format!("{target} = table[uint({element})];\nif ({target} == 0ul) {{ trap = 6u; return; }}\nswitch ({target}) {{\n"));
                    for index in 0..self.global_types.len() {
                        if self.type_targets[*type_index as usize].contains(&index) {
                            self.source
                                .push_str(&format!("case {}u: break;\n", index + 1));
                        }
                    }
                    self.source.push_str("default: trap = 7u; return;\n}\n");
                    self.emit_dispatch_call(&format!("{target} - 1ul"), &ty, args, stack);
                    return Ok(());
                }
                let result_types = ty.results().to_vec();
                let results: Vec<_> = result_types
                    .into_iter()
                    .map(|ty| self.fresh_typed(ty))
                    .collect();
                self.source.push_str(&format!(
                    "if ({element} >= {size}) {{ trap = 6u; return; }}\n"
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
                let constant_target =
                    constant_slot.and_then(|slot| table_values.get(slot).copied());
                for &index in table_targets {
                    if !table_mutable
                        && constant_slot.is_some()
                        && constant_target != Some(index as u64 + 1)
                    {
                        continue;
                    }
                    if !self.type_targets[*type_index as usize].contains(&index) {
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
                if self.local_types[*local_index as usize] == ValType::V128 {
                    self.emit_vector(stack, format!("l{local_index}"));
                } else {
                    self.emit_typed(
                        stack,
                        format!("l{local_index}"),
                        self.local_types[*local_index as usize],
                    );
                }
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
            Operator::Select | Operator::TypedSelect { .. } => {
                if let Operator::TypedSelect { ty } = operator
                    && !is_supported_value(*ty)
                {
                    return Err(Error::Unsupported("unsupported select value type".into()));
                }
                let condition = pop(stack)?;
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                let expression = format!("({condition} != 0u ? {lhs} : {rhs})");
                if self.vector_names.contains(&lhs) {
                    self.emit_vector(stack, expression);
                } else {
                    self.emit_typed(
                        stack,
                        expression,
                        match operator {
                            Operator::TypedSelect { ty } => *ty,
                            _ => ValType::I64,
                        },
                    );
                }
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
            _ => {
                if !self.emit_simd_operator(operator, stack)? {
                    return Err(Error::Unsupported(format!("instruction {operator:?}")));
                }
            }
        }
        Ok(())
    }

    fn emit_simd_operator(
        &mut self,
        operator: &Operator<'_>,
        stack: &mut Vec<String>,
    ) -> Result<bool, Error> {
        if let Some((bits, op)) = simd_binary_op(operator) {
            let rhs = pop(stack)?;
            let lhs = pop(stack)?;
            let expression = if bits == 32 && op <= 2 {
                let symbol = ["+", "-", "*"][op as usize];
                format!("as_type<ulong2>(as_type<uint4>({lhs}) {symbol} as_type<uint4>({rhs}))")
            } else {
                format!("wasm_simd_binary({lhs}, {rhs}, {bits}u, {op}u)")
            };
            self.emit_vector(stack, expression);
            return Ok(true);
        }
        if let Some((bits, op)) = simd_shift_op(operator) {
            let amount = pop(stack)?;
            let vector = pop(stack)?;
            self.emit_vector(
                stack,
                format!("wasm_simd_shift({vector}, {amount}, {bits}u, {op}u)"),
            );
            return Ok(true);
        }
        if let Some((bits, op)) = simd_unary_op(operator) {
            let value = pop(stack)?;
            self.emit_vector(stack, format!("wasm_simd_unary({value}, {bits}u, {op}u)"));
            return Ok(true);
        }
        if let Some((bits, op)) = simd_float_unary_op(operator) {
            let value = pop(stack)?;
            self.emit_vector(
                stack,
                format!("wasm_simd_float_unary({value}, {bits}u, {op}u)"),
            );
            return Ok(true);
        }
        if let Some((bits, op)) = simd_float_binary_op(operator) {
            let rhs = pop(stack)?;
            let lhs = pop(stack)?;
            self.emit_vector(
                stack,
                format!("wasm_simd_float_binary({lhs}, {rhs}, {bits}u, {op}u)"),
            );
            return Ok(true);
        }
        if let Some(op) = simd_convert_op(operator) {
            let value = pop(stack)?;
            self.emit_vector(stack, format!("wasm_simd_convert({value}, {op}u)"));
            return Ok(true);
        }
        match operator {
            Operator::I8x16RelaxedSwizzle => {
                let indices = pop(stack)?;
                let value = pop(stack)?;
                self.emit_vector(stack, format!("wasm_simd_swizzle({value}, {indices})"));
            }
            Operator::I8x16RelaxedLaneselect
            | Operator::I16x8RelaxedLaneselect
            | Operator::I32x4RelaxedLaneselect
            | Operator::I64x2RelaxedLaneselect => {
                let mask = pop(stack)?;
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_vector(stack, format!("({lhs} & {mask}) | ({rhs} & ~{mask})"));
            }
            Operator::F32x4RelaxedMadd
            | Operator::F32x4RelaxedNmadd
            | Operator::F64x2RelaxedMadd
            | Operator::F64x2RelaxedNmadd => {
                let c = pop(stack)?;
                let b = pop(stack)?;
                let a = pop(stack)?;
                let bits = if matches!(
                    operator,
                    Operator::F32x4RelaxedMadd | Operator::F32x4RelaxedNmadd
                ) {
                    32
                } else {
                    64
                };
                let negate = matches!(
                    operator,
                    Operator::F32x4RelaxedNmadd | Operator::F64x2RelaxedNmadd
                );
                self.emit_vector(
                    stack,
                    format!("wasm_simd_relaxed_madd({a}, {b}, {c}, {bits}u, {negate})"),
                );
            }
            Operator::I16x8RelaxedQ15mulrS => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_vector(stack, format!("wasm_simd_q15mulr({lhs}, {rhs})"));
            }
            Operator::I16x8RelaxedDotI8x16I7x16S => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_vector(stack, format!("wasm_simd_relaxed_dot16({lhs}, {rhs})"));
            }
            Operator::I32x4RelaxedDotI8x16I7x16AddS => {
                let c = pop(stack)?;
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_vector(
                    stack,
                    format!("wasm_simd_relaxed_dot_add({lhs}, {rhs}, {c})"),
                );
            }
            Operator::V128Not => {
                let value = pop(stack)?;
                self.emit_vector(stack, format!("~{value}"));
            }
            Operator::V128And | Operator::V128AndNot | Operator::V128Or | Operator::V128Xor => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                let expression = match operator {
                    Operator::V128And => format!("{lhs} & {rhs}"),
                    Operator::V128AndNot => format!("{lhs} & ~{rhs}"),
                    Operator::V128Or => format!("{lhs} | {rhs}"),
                    _ => format!("{lhs} ^ {rhs}"),
                };
                self.emit_vector(stack, expression);
            }
            Operator::V128Bitselect => {
                let mask = pop(stack)?;
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_vector(stack, format!("({lhs} & {mask}) | ({rhs} & ~{mask})"));
            }
            Operator::V128AnyTrue => {
                let value = pop(stack)?;
                self.emit_value(
                    stack,
                    format!("ulong({value}.x != 0ul || {value}.y != 0ul)"),
                );
            }
            Operator::I8x16AllTrue
            | Operator::I16x8AllTrue
            | Operator::I32x4AllTrue
            | Operator::I64x2AllTrue => {
                let value = pop(stack)?;
                let bits = match operator {
                    Operator::I8x16AllTrue => 8,
                    Operator::I16x8AllTrue => 16,
                    Operator::I32x4AllTrue => 32,
                    _ => 64,
                };
                self.emit_value(stack, format!("wasm_simd_all_true({value}, {bits}u)"));
            }
            Operator::I8x16Bitmask
            | Operator::I16x8Bitmask
            | Operator::I32x4Bitmask
            | Operator::I64x2Bitmask => {
                let value = pop(stack)?;
                let bits = match operator {
                    Operator::I8x16Bitmask => 8,
                    Operator::I16x8Bitmask => 16,
                    Operator::I32x4Bitmask => 32,
                    _ => 64,
                };
                self.emit_value(stack, format!("wasm_simd_bitmask({value}, {bits}u)"));
            }
            Operator::I8x16Popcnt => {
                let value = pop(stack)?;
                self.emit_vector(stack, format!("wasm_simd_popcnt({value})"));
            }
            Operator::I8x16Swizzle => {
                let indices = pop(stack)?;
                let value = pop(stack)?;
                self.emit_vector(stack, format!("wasm_simd_swizzle({value}, {indices})"));
            }
            Operator::I8x16Shuffle { lanes } => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                let result = self.fresh_vector();
                self.source.push_str(&format!("{result} = ulong2(0ul);\n"));
                for (index, lane) in lanes.iter().enumerate() {
                    let (value, lane) = if *lane < 16 {
                        (&lhs, *lane)
                    } else {
                        (&rhs, *lane - 16)
                    };
                    self.source.push_str(&format!("{result} = wasm_simd_replace({result}, 8u, {index}u, wasm_simd_lane({value}, 8u, {lane}u));\n"));
                }
                stack.push(result);
            }
            Operator::I8x16NarrowI16x8S
            | Operator::I8x16NarrowI16x8U
            | Operator::I16x8NarrowI32x4S
            | Operator::I16x8NarrowI32x4U => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                let bits = if matches!(
                    operator,
                    Operator::I8x16NarrowI16x8S | Operator::I8x16NarrowI16x8U
                ) {
                    16
                } else {
                    32
                };
                let signed = matches!(
                    operator,
                    Operator::I8x16NarrowI16x8S | Operator::I16x8NarrowI32x4S
                );
                self.emit_vector(
                    stack,
                    format!("wasm_simd_narrow({lhs}, {rhs}, {bits}u, {signed})"),
                );
            }
            Operator::I16x8ExtAddPairwiseI8x16S
            | Operator::I16x8ExtAddPairwiseI8x16U
            | Operator::I32x4ExtAddPairwiseI16x8S
            | Operator::I32x4ExtAddPairwiseI16x8U => {
                let value = pop(stack)?;
                let bits = if matches!(
                    operator,
                    Operator::I16x8ExtAddPairwiseI8x16S | Operator::I16x8ExtAddPairwiseI8x16U
                ) {
                    8
                } else {
                    16
                };
                let signed = matches!(
                    operator,
                    Operator::I16x8ExtAddPairwiseI8x16S | Operator::I32x4ExtAddPairwiseI16x8S
                );
                self.emit_vector(
                    stack,
                    format!("wasm_simd_pairwise({value}, {bits}u, {signed})"),
                );
            }
            Operator::I16x8ExtendLowI8x16S
            | Operator::I16x8ExtendHighI8x16S
            | Operator::I16x8ExtendLowI8x16U
            | Operator::I16x8ExtendHighI8x16U
            | Operator::I32x4ExtendLowI16x8S
            | Operator::I32x4ExtendHighI16x8S
            | Operator::I32x4ExtendLowI16x8U
            | Operator::I32x4ExtendHighI16x8U
            | Operator::I64x2ExtendLowI32x4S
            | Operator::I64x2ExtendHighI32x4S
            | Operator::I64x2ExtendLowI32x4U
            | Operator::I64x2ExtendHighI32x4U => {
                let value = pop(stack)?;
                let bits = if matches!(
                    operator,
                    Operator::I16x8ExtendLowI8x16S
                        | Operator::I16x8ExtendHighI8x16S
                        | Operator::I16x8ExtendLowI8x16U
                        | Operator::I16x8ExtendHighI8x16U
                ) {
                    8
                } else if matches!(
                    operator,
                    Operator::I32x4ExtendLowI16x8S
                        | Operator::I32x4ExtendHighI16x8S
                        | Operator::I32x4ExtendLowI16x8U
                        | Operator::I32x4ExtendHighI16x8U
                ) {
                    16
                } else {
                    32
                };
                let high = matches!(
                    operator,
                    Operator::I16x8ExtendHighI8x16S
                        | Operator::I16x8ExtendHighI8x16U
                        | Operator::I32x4ExtendHighI16x8S
                        | Operator::I32x4ExtendHighI16x8U
                        | Operator::I64x2ExtendHighI32x4S
                        | Operator::I64x2ExtendHighI32x4U
                );
                let signed = matches!(
                    operator,
                    Operator::I16x8ExtendLowI8x16S
                        | Operator::I16x8ExtendHighI8x16S
                        | Operator::I32x4ExtendLowI16x8S
                        | Operator::I32x4ExtendHighI16x8S
                        | Operator::I64x2ExtendLowI32x4S
                        | Operator::I64x2ExtendHighI32x4S
                );
                self.emit_vector(
                    stack,
                    format!("wasm_simd_extend({value}, {bits}u, {high}, {signed})"),
                );
            }
            Operator::I16x8ExtMulLowI8x16S
            | Operator::I16x8ExtMulHighI8x16S
            | Operator::I16x8ExtMulLowI8x16U
            | Operator::I16x8ExtMulHighI8x16U
            | Operator::I32x4ExtMulLowI16x8S
            | Operator::I32x4ExtMulHighI16x8S
            | Operator::I32x4ExtMulLowI16x8U
            | Operator::I32x4ExtMulHighI16x8U
            | Operator::I64x2ExtMulLowI32x4S
            | Operator::I64x2ExtMulHighI32x4S
            | Operator::I64x2ExtMulLowI32x4U
            | Operator::I64x2ExtMulHighI32x4U => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                let bits = if matches!(
                    operator,
                    Operator::I16x8ExtMulLowI8x16S
                        | Operator::I16x8ExtMulHighI8x16S
                        | Operator::I16x8ExtMulLowI8x16U
                        | Operator::I16x8ExtMulHighI8x16U
                ) {
                    8
                } else if matches!(
                    operator,
                    Operator::I32x4ExtMulLowI16x8S
                        | Operator::I32x4ExtMulHighI16x8S
                        | Operator::I32x4ExtMulLowI16x8U
                        | Operator::I32x4ExtMulHighI16x8U
                ) {
                    16
                } else {
                    32
                };
                let high = matches!(
                    operator,
                    Operator::I16x8ExtMulHighI8x16S
                        | Operator::I16x8ExtMulHighI8x16U
                        | Operator::I32x4ExtMulHighI16x8S
                        | Operator::I32x4ExtMulHighI16x8U
                        | Operator::I64x2ExtMulHighI32x4S
                        | Operator::I64x2ExtMulHighI32x4U
                );
                let signed = matches!(
                    operator,
                    Operator::I16x8ExtMulLowI8x16S
                        | Operator::I16x8ExtMulHighI8x16S
                        | Operator::I32x4ExtMulLowI16x8S
                        | Operator::I32x4ExtMulHighI16x8S
                        | Operator::I64x2ExtMulLowI32x4S
                        | Operator::I64x2ExtMulHighI32x4S
                );
                self.emit_vector(
                    stack,
                    format!("wasm_simd_extmul({lhs}, {rhs}, {bits}u, {high}, {signed})"),
                );
            }
            Operator::I32x4DotI16x8S => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_vector(stack, format!("wasm_simd_dot({lhs}, {rhs})"));
            }
            Operator::I16x8Q15MulrSatS => {
                let rhs = pop(stack)?;
                let lhs = pop(stack)?;
                self.emit_vector(stack, format!("wasm_simd_q15mulr({lhs}, {rhs})"));
            }
            Operator::I8x16Splat
            | Operator::I16x8Splat
            | Operator::I32x4Splat
            | Operator::I64x2Splat
            | Operator::F32x4Splat
            | Operator::F64x2Splat => {
                let value = pop(stack)?;
                let bits = match operator {
                    Operator::I8x16Splat => 8,
                    Operator::I16x8Splat => 16,
                    Operator::I32x4Splat | Operator::F32x4Splat => 32,
                    _ => 64,
                };
                self.emit_vector(stack, format!("wasm_simd_splat({value}, {bits}u)"));
            }
            Operator::I8x16ExtractLaneS { lane }
            | Operator::I8x16ExtractLaneU { lane }
            | Operator::I16x8ExtractLaneS { lane }
            | Operator::I16x8ExtractLaneU { lane }
            | Operator::I32x4ExtractLane { lane }
            | Operator::I64x2ExtractLane { lane }
            | Operator::F32x4ExtractLane { lane }
            | Operator::F64x2ExtractLane { lane } => {
                let value = pop(stack)?;
                let bits = match operator {
                    Operator::I8x16ExtractLaneS { .. } | Operator::I8x16ExtractLaneU { .. } => 8,
                    Operator::I16x8ExtractLaneS { .. } | Operator::I16x8ExtractLaneU { .. } => 16,
                    Operator::I32x4ExtractLane { .. } | Operator::F32x4ExtractLane { .. } => 32,
                    _ => 64,
                };
                let lane_value = format!("wasm_simd_lane({value}, {bits}u, {lane}u)");
                let expression = if matches!(
                    operator,
                    Operator::I8x16ExtractLaneS { .. } | Operator::I16x8ExtractLaneS { .. }
                ) {
                    format!("ulong(uint(wasm_simd_signed({lane_value}, {bits}u)))")
                } else {
                    lane_value
                };
                self.emit_value(stack, expression);
            }
            Operator::I8x16ReplaceLane { lane }
            | Operator::I16x8ReplaceLane { lane }
            | Operator::I32x4ReplaceLane { lane }
            | Operator::I64x2ReplaceLane { lane }
            | Operator::F32x4ReplaceLane { lane }
            | Operator::F64x2ReplaceLane { lane } => {
                let part = pop(stack)?;
                let value = pop(stack)?;
                let bits = match operator {
                    Operator::I8x16ReplaceLane { .. } => 8,
                    Operator::I16x8ReplaceLane { .. } => 16,
                    Operator::I32x4ReplaceLane { .. } | Operator::F32x4ReplaceLane { .. } => 32,
                    _ => 64,
                };
                self.emit_vector(
                    stack,
                    format!("wasm_simd_replace({value}, {bits}u, {lane}u, {part})"),
                );
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

fn simd_binary_op(operator: &Operator<'_>) -> Option<(u32, u32)> {
    use Operator::*;
    Some(match operator {
        I8x16Add => (8, 0),
        I8x16Sub => (8, 1),
        I8x16Eq => (8, 3),
        I8x16Ne => (8, 4),
        I8x16LtS => (8, 5),
        I8x16LtU => (8, 6),
        I8x16GtS => (8, 7),
        I8x16GtU => (8, 8),
        I8x16LeS => (8, 9),
        I8x16LeU => (8, 10),
        I8x16GeS => (8, 11),
        I8x16GeU => (8, 12),
        I8x16MinS => (8, 13),
        I8x16MinU => (8, 14),
        I8x16MaxS => (8, 15),
        I8x16MaxU => (8, 16),
        I8x16AvgrU => (8, 17),
        I8x16AddSatS => (8, 18),
        I8x16AddSatU => (8, 19),
        I8x16SubSatS => (8, 20),
        I8x16SubSatU => (8, 21),
        I16x8Add => (16, 0),
        I16x8Sub => (16, 1),
        I16x8Mul => (16, 2),
        I16x8Eq => (16, 3),
        I16x8Ne => (16, 4),
        I16x8LtS => (16, 5),
        I16x8LtU => (16, 6),
        I16x8GtS => (16, 7),
        I16x8GtU => (16, 8),
        I16x8LeS => (16, 9),
        I16x8LeU => (16, 10),
        I16x8GeS => (16, 11),
        I16x8GeU => (16, 12),
        I16x8MinS => (16, 13),
        I16x8MinU => (16, 14),
        I16x8MaxS => (16, 15),
        I16x8MaxU => (16, 16),
        I16x8AvgrU => (16, 17),
        I16x8AddSatS => (16, 18),
        I16x8AddSatU => (16, 19),
        I16x8SubSatS => (16, 20),
        I16x8SubSatU => (16, 21),
        I32x4Add => (32, 0),
        I32x4Sub => (32, 1),
        I32x4Mul => (32, 2),
        I32x4Eq => (32, 3),
        I32x4Ne => (32, 4),
        I32x4LtS => (32, 5),
        I32x4LtU => (32, 6),
        I32x4GtS => (32, 7),
        I32x4GtU => (32, 8),
        I32x4LeS => (32, 9),
        I32x4LeU => (32, 10),
        I32x4GeS => (32, 11),
        I32x4GeU => (32, 12),
        I32x4MinS => (32, 13),
        I32x4MinU => (32, 14),
        I32x4MaxS => (32, 15),
        I32x4MaxU => (32, 16),
        I64x2Add => (64, 0),
        I64x2Sub => (64, 1),
        I64x2Mul => (64, 2),
        I64x2Eq => (64, 3),
        I64x2Ne => (64, 4),
        I64x2LtS => (64, 5),
        I64x2GtS => (64, 7),
        I64x2LeS => (64, 9),
        I64x2GeS => (64, 11),
        _ => return None,
    })
}

fn simd_shift_op(operator: &Operator<'_>) -> Option<(u32, u32)> {
    use Operator::*;
    Some(match operator {
        I8x16Shl => (8, 0),
        I8x16ShrS => (8, 1),
        I8x16ShrU => (8, 2),
        I16x8Shl => (16, 0),
        I16x8ShrS => (16, 1),
        I16x8ShrU => (16, 2),
        I32x4Shl => (32, 0),
        I32x4ShrS => (32, 1),
        I32x4ShrU => (32, 2),
        I64x2Shl => (64, 0),
        I64x2ShrS => (64, 1),
        I64x2ShrU => (64, 2),
        _ => return None,
    })
}

fn simd_unary_op(operator: &Operator<'_>) -> Option<(u32, u32)> {
    use Operator::*;
    Some(match operator {
        I8x16Abs => (8, 0),
        I8x16Neg => (8, 1),
        I16x8Abs => (16, 0),
        I16x8Neg => (16, 1),
        I32x4Abs => (32, 0),
        I32x4Neg => (32, 1),
        I64x2Abs => (64, 0),
        I64x2Neg => (64, 1),
        _ => return None,
    })
}

fn simd_float_unary_op(operator: &Operator<'_>) -> Option<(u32, u32)> {
    use Operator::*;
    Some(match operator {
        F32x4Abs => (32, 0),
        F32x4Neg => (32, 1),
        F32x4Sqrt => (32, 2),
        F32x4Ceil => (32, 3),
        F32x4Floor => (32, 4),
        F32x4Trunc => (32, 5),
        F32x4Nearest => (32, 6),
        F64x2Abs => (64, 0),
        F64x2Neg => (64, 1),
        F64x2Sqrt => (64, 2),
        F64x2Ceil => (64, 3),
        F64x2Floor => (64, 4),
        F64x2Trunc => (64, 5),
        F64x2Nearest => (64, 6),
        _ => return None,
    })
}

fn simd_float_binary_op(operator: &Operator<'_>) -> Option<(u32, u32)> {
    use Operator::*;
    Some(match operator {
        F32x4Add => (32, 0),
        F32x4Sub => (32, 1),
        F32x4Mul => (32, 2),
        F32x4Div => (32, 3),
        F32x4Min => (32, 4),
        F32x4Max => (32, 5),
        F32x4Eq => (32, 6),
        F32x4Ne => (32, 7),
        F32x4Lt => (32, 8),
        F32x4Gt => (32, 9),
        F32x4Le => (32, 10),
        F32x4Ge => (32, 11),
        F32x4PMin => (32, 12),
        F32x4PMax => (32, 13),
        F32x4RelaxedMin => (32, 4),
        F32x4RelaxedMax => (32, 5),
        F64x2Add => (64, 0),
        F64x2Sub => (64, 1),
        F64x2Mul => (64, 2),
        F64x2Div => (64, 3),
        F64x2Min => (64, 4),
        F64x2Max => (64, 5),
        F64x2Eq => (64, 6),
        F64x2Ne => (64, 7),
        F64x2Lt => (64, 8),
        F64x2Gt => (64, 9),
        F64x2Le => (64, 10),
        F64x2Ge => (64, 11),
        F64x2PMin => (64, 12),
        F64x2PMax => (64, 13),
        F64x2RelaxedMin => (64, 4),
        F64x2RelaxedMax => (64, 5),
        _ => return None,
    })
}

fn simd_convert_op(operator: &Operator<'_>) -> Option<u32> {
    use Operator::*;
    Some(match operator {
        I32x4TruncSatF32x4S => 0,
        I32x4TruncSatF32x4U => 1,
        I32x4RelaxedTruncF32x4S => 0,
        I32x4RelaxedTruncF32x4U => 1,
        F32x4ConvertI32x4S => 2,
        F32x4ConvertI32x4U => 3,
        I32x4TruncSatF64x2SZero => 4,
        I32x4TruncSatF64x2UZero => 5,
        I32x4RelaxedTruncF64x2SZero => 4,
        I32x4RelaxedTruncF64x2UZero => 5,
        F64x2ConvertLowI32x4S => 6,
        F64x2ConvertLowI32x4U => 7,
        F32x4DemoteF64x2Zero => 8,
        F64x2PromoteLowF32x4 => 9,
        _ => return None,
    })
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
    fn linked_module_memory_is_counted_and_reset_across_batches() {
        let provider = wat::parse_str(
            r#"(module (memory 1)
                (global $g (mut i32) (i32.const 7))
                (func (export "add") (param i32) (result i32)
                    global.get $g local.get 0 i32.add
                    global.get $g i32.const 1 i32.add global.set $g))"#,
        )
        .unwrap();
        let root = wat::parse_str(
            r#"(module
                (import "p" "add" (func $add (param i32) (result i32)))
                (memory 1)
                (func (export "f") (param i32) (result i32)
                    local.get 0 call $add))"#,
        )
        .unwrap();
        let kernel = Kernel::new_with_imports(&root, "f", &[("p", &provider)]).unwrap();
        let limit = kernel.batch_limit(4, 4).unwrap();
        assert!(
            limit < 600,
            "both module memories must count toward the budget"
        );
        let inputs: Vec<u32> = (0..limit as u32 + 5).collect();
        let expected: Vec<u32> = inputs.iter().map(|value| value + 7).collect();
        assert_eq!(kernel.run(&inputs).unwrap(), expected);
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
    fn imported_resources_share_per_lane_state_and_initialize_in_order() {
        let provider = r#"(module
            (type $t (func (param i32) (result i32)))
            (memory (export "mem") 1 3)
            (table (export "tab") 2 5 funcref)
            (global $g (export "g") (mut i32) (i32.const 1))
            (func $old (type $t) i32.const 100)
            (elem (i32.const 0) func $old)
            (func $start
                i32.const 0 i32.const 7 i32.store
                i32.const 10 global.set $g)
            (start $start)
            (func (export "read") (result i32)
                i32.const 0 i32.load global.get $g i32.add))"#;
        let root = r#"(module
            (type $t (func (param i32) (result i32)))
            (import "p" "read" (func $read (result i32)))
            (import "p" "mem" (memory 1 3))
            (import "p" "tab" (table 2 5 funcref))
            (import "p" "g" (global $g (mut i32)))
            (export "mem" (memory 0))
            (func $new (type $t) local.get 0 i32.const 2 i32.mul)
            (elem (i32.const 0) func $new)
            (data (i32.const 0) "\05\00\00\00")
            (func (export "f") (param i32) (result i32)
                global.get $g local.get 0 i32.add global.set $g
                local.get 0 i32.const 0 call_indirect (type $t)
                call $read i32.add))"#;
        linked_matches_wasmtime(root, &[("p", provider)], &[0, 1, 7, 42]);
        let wasm = wat::parse_str(root).unwrap();
        let provider = wat::parse_str(provider).unwrap();
        let kernel = Kernel::new_with_imports(&wasm, "f", &[("p", &provider)]).unwrap();
        let inputs = [0, 1, 7].map(|input| vec![Value::I32(input)]);
        let (values, memory) = kernel.run_values_with_memory(&inputs, "mem", 0..4).unwrap();
        assert_eq!(
            values,
            [
                vec![Value::I32(15)],
                vec![Value::I32(18)],
                vec![Value::I32(36)]
            ]
        );
        assert_eq!(memory, vec![vec![5, 0, 0, 0]; 3]);
    }

    #[test]
    fn passive_segments_initialize_imported_memory_and_table() {
        let provider = r#"(module
            (memory (export "mem") 1)
            (table (export "tab") 1 funcref))"#;
        let root = r#"(module
            (type $t (func (param i32) (result i32)))
            (import "p" "mem" (memory 1))
            (import "p" "tab" (table 1 funcref))
            (data $d "\05\00\00\00")
            (func $inc (type $t) local.get 0 i32.const 7 i32.add)
            (elem $e func $inc)
            (func (export "f") (param i32) (result i32)
                i32.const 0 i32.const 0 i32.const 4 memory.init $d
                data.drop $d
                i32.const 0 i32.const 0 i32.const 1 table.init $e
                elem.drop $e
                i32.const 0 i32.load
                local.get 0 i32.const 0 call_indirect (type $t)
                i32.add))"#;
        linked_matches_wasmtime(root, &[("p", provider)], &[0, 1, 42]);
    }

    #[test]
    fn imported_memory_and_globals_work_through_reexports() {
        let base = r#"(module
            (memory (export "mem") 1 2)
            (global (export "offset") i32 (i32.const 12))
            (func $start i32.const 12 i32.const 5 i32.store8)
            (start $start))"#;
        let middle = r#"(module
            (import "base" "mem" (memory 1 2))
            (import "base" "offset" (global $offset i32))
            (export "mem" (memory 0))
            (export "offset" (global $offset)))"#;
        let root = r#"(module
            (import "mid" "mem" (memory 1 2))
            (import "mid" "offset" (global $offset i32))
            (export "mem" (memory 0))
            (global $local i32 (global.get $offset))
            (data (global.get $offset) "\2a")
            (func (export "f") (param i32) (result i32)
                global.get $local i32.load8_u drop
                global.get $local i32.load8_u local.get 0 i32.add))"#;
        linked_matches_wasmtime(root, &[("base", base), ("mid", middle)], &[0, 1, 42]);
        let wasm = wat::parse_str(root).unwrap();
        let base = wat::parse_str(base).unwrap();
        let middle = wat::parse_str(middle).unwrap();
        let kernel =
            Kernel::new_with_imports(&wasm, "f", &[("mid", &middle), ("base", &base)]).unwrap();
        let (values, memory) = kernel
            .run_values_with_memory(&[vec![Value::I32(3)]], "mem", 12..13)
            .unwrap();
        assert_eq!(values, [vec![Value::I32(45)]]);
        assert_eq!(memory, [vec![42]]);
    }

    #[test]
    fn resource_imports_reject_type_mismatches() {
        for (root, provider, expected) in [
            (
                r#"(module (import "p" "x" (memory 2 3)) (func (export "f")))"#,
                r#"(module (memory (export "x") 1 3))"#,
                "memory type mismatch",
            ),
            (
                r#"(module (import "p" "x" (table 2 3 funcref)) (func (export "f")))"#,
                r#"(module (table (export "x") 1 3 funcref))"#,
                "table type mismatch",
            ),
            (
                r#"(module (import "p" "x" (global i64)) (func (export "f")))"#,
                r#"(module (global (export "x") i32 (i32.const 0)))"#,
                "global type mismatch",
            ),
        ] {
            let root = wat::parse_str(root).unwrap();
            let provider = wat::parse_str(provider).unwrap();
            assert!(
                matches!(Kernel::new_with_imports(&root, "f", &[("p", &provider)]),
                Err(Error::Unsupported(reason)) if reason.contains(expected))
            );
        }
    }

    #[test]
    fn out_of_bounds_active_segments_trap_at_invocation() {
        for (wat, reason) in [
            (
                r#"(module (memory 1) (data (i32.const 65536) "x") (func (export "f")))"#,
                "out of bounds memory access",
            ),
            (
                r#"(module (table 1 funcref) (func $target)
                    (elem (i32.const 1) func $target) (func (export "f")))"#,
                "out of bounds table access",
            ),
        ] {
            let wasm = wat::parse_str(wat).unwrap();
            let kernel = Kernel::new(&wasm, "f").unwrap();
            assert!(matches!(kernel.run_values(&[vec![]]),
                Err(Error::Trap { input_index: 0, reason: actual }) if actual == reason));
            let engine = wasmtime::Engine::default();
            let module = wasmtime::Module::new(&engine, &wasm).unwrap();
            let mut store = wasmtime::Store::new(&engine, ());
            assert!(wasmtime::Instance::new(&mut store, &module, &[]).is_err());
        }
    }

    #[test]
    fn imported_resource_growth_updates_provider_state() {
        let provider = wat::parse_str(
            r#"(module
            (memory (export "mem") 1 20)
            (table (export "tab") 1 500 funcref)
            (func (export "pages") (result i32) memory.size)
            (func (export "slots") (result i32) table.size))"#,
        )
        .unwrap();
        let root = wat::parse_str(
            r#"(module
            (import "p" "mem" (memory 1 20))
            (import "p" "tab" (table 1 500 funcref))
            (import "p" "pages" (func $pages (result i32)))
            (import "p" "slots" (func $slots (result i32)))
            (func (export "f") (param i32) (result i32 i32)
                local.get 0 memory.grow drop
                ref.null func local.get 0 table.grow drop
                call $pages call $slots))"#,
        )
        .unwrap();
        let limits = KernelLimits {
            memory_bytes: 16 * 65536,
            memory_growth_pages: 16,
            table_growth_elements: 512,
            ..KernelLimits::default()
        };
        let kernel =
            Kernel::new_with_imports_and_limits(&root, "f", &[("p", &provider)], limits).unwrap();
        let inputs = [vec![Value::I32(8)], vec![Value::I32(3)]];
        assert_eq!(
            kernel.run_values(&inputs).unwrap(),
            [
                vec![Value::I32(9), Value::I32(9)],
                vec![Value::I32(4), Value::I32(4)],
            ]
        );
        let engine = wasmtime::Engine::default();
        let provider_module = wasmtime::Module::new(&engine, &provider).unwrap();
        let root_module = wasmtime::Module::new(&engine, &root).unwrap();
        for (input, expected) in [(8, 9), (3, 4)] {
            let mut store = wasmtime::Store::new(&engine, ());
            let mut linker = wasmtime::Linker::new(&engine);
            let instance = linker.instantiate(&mut store, &provider_module).unwrap();
            linker.instance(&mut store, "p", instance).unwrap();
            let instance = linker.instantiate(&mut store, &root_module).unwrap();
            let actual = instance
                .get_typed_func::<i32, (i32, i32)>(&mut store, "f")
                .unwrap()
                .call(&mut store, input)
                .unwrap();
            assert_eq!(actual, (expected, expected));
        }
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
            let reason = match result {
                Err(Error::Unsupported(reason)) => reason,
                Err(error) => panic!("expected unsupported {expected}, got {error}"),
                Ok(_) => panic!("expected unsupported {expected}, got a kernel"),
            };
            assert!(
                reason.contains(expected),
                "expected {expected}, got {reason}"
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
            "has no export",
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
    fn memory_batches_preserve_results_and_global_trap_indices() {
        let wasm = wat::parse_str(
            r#"(module (memory 1)
                (global $g (mut i32) (i32.const 7))
                (func (export "f") (param i32) (result i32)
                    local.get 0 i32.load drop
                    global.get $g local.get 0 i32.add
                    global.get $g i32.const 1 i32.add global.set $g))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let limit = kernel.batch_limit(4, 4).unwrap();
        assert!(limit < 65_000);
        let mut inputs: Vec<u32> = (0..limit as u32 + 9).collect();
        let expected: Vec<u32> = inputs.iter().map(|value| value + 7).collect();
        assert_eq!(kernel.run(&inputs).unwrap(), expected);
        assert_eq!(kernel.run(&inputs).unwrap(), expected);
        inputs[limit + 3] = 65_536;
        assert!(matches!(
            kernel.run(&inputs),
            Err(Error::Trap {
                input_index,
                reason: "out of bounds memory access"
            }) if input_index == limit + 3
        ));
    }

    #[test]
    fn value_batches_preserve_multiple_results_and_trap_indices() {
        let wasm = wat::parse_str(
            r#"(module (memory 1)
                (func (export "f") (param i64 i32) (result i64 i32)
                    local.get 1 local.get 1 i32.store
                    local.get 0 i64.const 9 i64.add
                    local.get 1 i32.load))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let limit = kernel.batch_limit(16, 16).unwrap();
        assert!(limit < 65_000);
        let mut inputs: Vec<_> = (0..limit as u32 + 7)
            .map(|value| vec![Value::I64(value as u64), Value::I32(value)])
            .collect();
        let expected: Vec<_> = (0..inputs.len() as u64)
            .map(|value| vec![Value::I64(value + 9), Value::I32(value as u32)])
            .collect();
        let actual = kernel.run_values(&inputs).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual, expected, "input index {index}");
        }
        inputs[limit + 2][1] = Value::I32(65_536);
        assert!(matches!(
            kernel.run_values(&inputs),
            Err(Error::Trap {
                input_index,
                reason: "out of bounds memory access"
            }) if input_index == limit + 2
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
    fn exported_memory_bytes_follow_lanes_across_batches() {
        let wasm = wat::parse_str(
            r#"(module (memory (export "mem") 1)
            (func (export "f") (param i32) (result i32)
                i32.const 0 local.get 0 i32.store
                local.get 0))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let limit = kernel.batch_limit(4, 4).unwrap();
        let inputs: Vec<_> = (0..limit + 3)
            .map(|index| vec![Value::I32(index as u32)])
            .collect();
        for _ in 0..2 {
            let (values, memories) = kernel.run_values_with_memory(&inputs, "mem", 0..4).unwrap();
            for index in [0, limit - 1, limit, limit + 2] {
                assert_eq!(values[index], [Value::I32(index as u32)]);
                assert_eq!(memories[index], (index as u32).to_le_bytes());
            }
        }
        assert!(
            matches!(kernel.run_values_with_memory(&inputs[..1], "mem", 65536..65537),
            Err(Error::Unsupported(reason)) if reason.contains("memory range exceeds"))
        );
    }

    #[test]
    fn per_lane_memory_input_follows_start_and_resets_across_batches() {
        let wasm = wat::parse_str(
            r#"(module
                (memory (export "mem") 1)
                (data (i32.const 0) "\05\06\07\08")
                (func $start i32.const 0 i32.const 9 i32.store8)
                (start $start)
                (func (export "f") (param i32) (result i32) (local i32)
                    local.get 0
                    i32.const 0 i32.load8_u i32.add
                    i32.const 1 i32.load8_u i32.add
                    local.set 1
                    i32.const 3 local.get 1 i32.store8
                    local.get 1))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let limit = kernel.batch_limit(8, 4).unwrap();
        let inputs: Vec<_> = (0..limit + 3)
            .map(|index| vec![Value::I32(index as u32)])
            .collect();
        let memory_inputs: Vec<_> = (0..inputs.len())
            .map(|index| vec![index as u8, 2, 3, 4])
            .collect();
        for _ in 0..2 {
            let (values, memories) = kernel
                .run_values_with_memory_io(&inputs, "mem", 0, &memory_inputs, 0..4)
                .unwrap();
            for index in [0, limit - 1, limit, limit + 2] {
                let expected = index as u32 + index as u8 as u32 + 2;
                assert_eq!(values[index], [Value::I32(expected)]);
                assert_eq!(memories[index], [index as u8, 2, 3, expected as u8]);
            }
        }
        assert!(matches!(
            kernel.run_values_with_memory_io(&inputs[..1], "mem", 0, &[], 0..4),
            Err(Error::Unsupported(reason)) if reason.contains("count")));
        assert!(matches!(
            kernel.run_values_with_memory_io(&inputs[..2], "mem", 0, &[vec![1], vec![1, 2]], 0..4),
            Err(Error::Unsupported(reason)) if reason.contains("equal lengths")));
        assert!(matches!(
            kernel.run_values_with_memory_io(&inputs[..1], "mem", 65535, &[vec![1, 2]], 0..4),
            Err(Error::Unsupported(reason)) if reason.contains("initial memory size")));
    }

    #[test]
    fn per_lane_memory_input_reaches_imported_memory() {
        let provider = wat::parse_str(r#"(module (memory (export "mem") 1))"#).unwrap();
        let wasm = wat::parse_str(
            r#"(module
                (import "p" "mem" (memory 1))
                (export "mem" (memory 0))
                (func (export "f") (result i32)
                    i32.const 3 i32.load8_u))"#,
        )
        .unwrap();
        let kernel = Kernel::new_with_imports(&wasm, "f", &[("p", &provider)]).unwrap();
        let (values, memories) = kernel
            .run_values_with_memory_io(
                &[vec![], vec![]],
                "mem",
                2,
                &[vec![10, 11], vec![20, 21]],
                2..4,
            )
            .unwrap();
        assert_eq!(values, [vec![Value::I32(11)], vec![Value::I32(21)]]);
        assert_eq!(memories, [vec![10, 11], vec![20, 21]]);
    }

    #[test]
    fn per_lane_memory_input_preserves_general_numeric_arguments() {
        let wasm = wat::parse_str(
            r#"(module
                (memory (export "mem") 1)
                (func (export "f") (param i32 i64) (result i64)
                    local.get 1
                    local.get 0
                    i32.const 0 i32.load8_u
                    i32.add
                    i64.extend_i32_u
                    i64.add))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let (values, memories) = kernel
            .run_values_with_memory_io(
                &[
                    vec![Value::I32(3), Value::I64(100)],
                    vec![Value::I32(7), Value::I64(200)],
                ],
                "mem",
                0,
                &[vec![5], vec![9]],
                0..0,
            )
            .unwrap();
        assert_eq!(values, [vec![Value::I64(108)], vec![Value::I64(216)]]);
        assert_eq!(memories, [vec![], vec![]]);
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
                    Value::V128(bits) => wasmtime::Val::V128(u128::from_le_bytes(*bits).into()),
                    Value::FuncRef(_)
                    | Value::ExternRef(_)
                    | Value::ExnRef(_)
                    | Value::I31(_)
                    | Value::GcRef(_) => {
                        panic!("numeric differential helper received a reference")
                    }
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
                    ValType::V128 => wasmtime::Val::V128(0u128.into()),
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
                    (Value::V128(a), wasmtime::Val::V128(b)) => {
                        u128::from_le_bytes(a) == b.as_u128()
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
                        reason: "out of bounds table access",
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
        let translation = translate_with_limits(&wasm, "f", &[], KernelLimits::default()).unwrap();
        assert!(!translation.instances[0].tables[0].info.mutable);
        assert_eq!(translation.instances[0].tables[0].info.capacity, 65536);
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert_eq!(kernel.run(&[0, 1, 2]).unwrap(), [65536, 65537, 65538]);
    }

    #[test]
    fn mutating_one_table_keeps_other_table_fixed() {
        let source = r#"(module (type $t (func (param i32) (result i32)))
                (table $fixed 1 funcref) (table $mutable 1 2 funcref)
                (func $safe (type $t) local.get 0 i32.const 1 i32.add)
                (func $other (type $t) local.get 0 i32.const 7 i32.add)
                (elem (table $fixed) (i32.const 0) func $safe)
                (elem declare func $other)
                (func (export "f") (param i32) (result i32)
                    i32.const 0 ref.func $other table.set $mutable
                    ref.null func i32.const 1 table.grow $mutable drop
                    local.get 0 i32.const 0 call_indirect $fixed (type $t)))"#;
        let wasm = wat::parse_str(source).unwrap();
        let translation = translate_with_limits(&wasm, "f", &[], KernelLimits::default()).unwrap();
        assert!(!translation.instances[0].tables[0].info.mutable);
        assert_eq!(translation.instances[0].tables[0].info.capacity, 1);
        assert_eq!(translation.instances[0].tables[0].targets, [0]);
        assert!(translation.instances[0].tables[1].info.mutable);
        assert_eq!(translation.instances[0].tables[1].info.capacity, 2);
        matches_wasmtime(source, &[0, 37]);
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
    fn custom_limits_allow_more_memory_and_table_growth() {
        let memory = wat::parse_str(
            r#"(module (memory (export "mem") 1 40)
            (func (export "f") (param i32) (result i32 i32)
                local.get 0 memory.grow memory.size))"#,
        )
        .unwrap();
        let input = [vec![Value::I32(8)]];
        assert_eq!(
            Kernel::new(&memory, "f")
                .unwrap()
                .run_values(&input)
                .unwrap(),
            [vec![Value::I32(u32::MAX), Value::I32(1)]]
        );
        let limits = KernelLimits {
            memory_bytes: 16 * 65536,
            memory_growth_pages: 16,
            table_growth_elements: 512,
            ..KernelLimits::default()
        };
        let kernel = Kernel::new_with_limits(&memory, "f", limits).unwrap();
        let (values, bytes) = kernel
            .run_values_with_memory(&input, "mem", 8 * 65536..8 * 65536 + 4)
            .unwrap();
        assert_eq!(values, [vec![Value::I32(1), Value::I32(9)]]);
        assert_eq!(bytes, [vec![0, 0, 0, 0]]);
        let table = wat::parse_str(
            r#"(module (table 0 1000 funcref)
            (func (export "f") (param i32) (result i32)
                ref.null func local.get 0 table.grow))"#,
        )
        .unwrap();
        assert_eq!(
            Kernel::new(&table, "f").unwrap().run(&[300]).unwrap(),
            [u32::MAX]
        );
        assert_eq!(
            Kernel::new_with_limits(&table, "f", limits)
                .unwrap()
                .run(&[300])
                .unwrap(),
            [0]
        );
    }

    #[test]
    fn table_size_is_fresh_when_state_buffers_are_reused() {
        let wasm = wat::parse_str(
            r#"(module (memory 1) (table 1 2 funcref)
                (func (export "f") (param i32) (result i32)
                    ref.null func i32.const 1 table.grow drop
                    table.size))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let limit = kernel.batch_limit(4, 4).unwrap();
        assert!(limit < 65_000);
        assert_eq!(kernel.run(&vec![0; limit + 3]).unwrap(), vec![2; limit + 3]);
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
    fn custom_call_depth_allows_deeper_recursion() {
        let wasm = wat::parse_str(
            r#"(module
            (func $f (export "f") (param i32) (result i32)
                local.get 0 i32.eqz
                if (result i32) i32.const 0
                else local.get 0 i32.const 1 i32.sub call $f i32.const 1 i32.add end))"#,
        )
        .unwrap();
        assert!(matches!(
            Kernel::new(&wasm, "f").unwrap().run(&[40]),
            Err(Error::Trap {
                reason: "call depth limit exceeded",
                ..
            })
        ));
        let limits = KernelLimits {
            call_depth: 64,
            ..KernelLimits::default()
        };
        assert_eq!(
            Kernel::new_with_limits(&wasm, "f", limits)
                .unwrap()
                .run(&[40])
                .unwrap(),
            [40]
        );
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

    #[test]
    fn simd_values_calls_globals_and_memory_match_wasmtime() {
        let a = Value::V128([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let b = Value::V128([255, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0]);
        compare_values_to_wasmtime(
            "(module (func (export \"f\") (param v128 v128) (result v128) local.get 0 local.get 1 i32x4.add))",
            &[vec![a, b]],
        );
        compare_values_to_wasmtime(
            "(module (func (export \"f\") (param i32 v128) (result v128 i32)
                local.get 1 local.get 0 i32x4.splat i32x4.add local.get 0))",
            &[vec![Value::I32(19), a], vec![Value::I32(u32::MAX), b]],
        );
        compare_values_to_wasmtime(
            "(module (global $g (mut v128) (v128.const i32x4 1 2 3 4))
                (func $add (param v128) (result v128) local.get 0 global.get $g i32x4.add)
                (func (export \"f\") (param v128) (result v128)
                  (local v128) local.get 0 call $add local.tee 1 global.set $g local.get 1))",
            &[vec![a], vec![b]],
        );
        compare_values_to_wasmtime(
            "(module (memory 1) (func (export \"f\") (param v128) (result v128)
                i32.const 1 local.get 0 v128.store
                i32.const 1 v128.load))",
            &[vec![a], vec![b]],
        );
        compare_values_to_wasmtime(
            "(module (memory 1) (global $zero (mut v128) (v128.const i32x4 0 0 0 0))
                (func (export \"f\") (param i32 v128) (result i64) (local i64)
                    local.get 0 local.get 1 v128.store16_lane 6
                    local.get 0 i64.load local.set 2
                    local.get 0 global.get $zero v128.store
                    local.get 2))",
            &[vec![
                Value::I32(6),
                Value::V128([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 7, 0, 0]),
            ]],
        );
    }

    #[test]
    fn extended_constant_expressions_match_wasmtime() {
        compare_values_to_wasmtime(
            "(module
                (global $base i32 (i32.const 40))
                (global $answer i32 (i32.add (global.get $base) (i32.const 2)))
                (memory 1)
                (data (i32.add (i32.const 1) (i32.const 1)) \"\\2a\")
                (func (export \"f\") (param i32) (result i32)
                    global.get $answer
                    i32.const 2 i32.load8_u i32.add
                    local.get 0 i32.add))",
            &[vec![Value::I32(0)], vec![Value::I32(7)]],
        );
    }

    #[test]
    fn tail_calls_reuse_the_call_frame() {
        compare_values_to_wasmtime(
            "(module
                (func $sum (param i32 i32) (result i32)
                    local.get 0 i32.eqz
                    if (result i32) local.get 1
                    else
                        local.get 0 i32.const 1 i32.sub
                        local.get 1 local.get 0 i32.add
                        return_call $sum
                    end)
                (func (export \"f\") (param i32) (result i32)
                    local.get 0 i32.const 0 return_call $sum))",
            &[
                vec![Value::I32(1)],
                vec![Value::I32(10)],
                vec![Value::I32(1_000)],
            ],
        );
        compare_values_to_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (table 1 funcref) (elem (i32.const 0) $inc)
                (func $inc (type $t) local.get 0 i32.const 1 i32.add)
                (func (export \"f\") (param i32) (result i32)
                    local.get 0 i32.const 0 return_call_indirect (type $t)))",
            &[vec![Value::I32(0)], vec![Value::I32(100)]],
        );
        compare_values_to_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (func $inc (type $t) local.get 0 i32.const 1 i32.add)
                (elem declare func $inc)
                (func (export \"f\") (param i32) (result i32)
                    local.get 0 ref.func $inc return_call_ref $t))",
            &[vec![Value::I32(0)], vec![Value::I32(100)]],
        );
    }

    #[test]
    fn multiple_memories_and_memory64_match_wasmtime() {
        compare_values_to_wasmtime(
            "(module (memory $a 1 2) (memory $b 1 2)
                (data (memory $a) (i32.const 0) \"\\05\")
                (data (memory $b) (i32.const 0) \"\\07\")
                (func (export \"f\") (param i32) (result i32)
                    (memory.copy $b $a (i32.const 2) (i32.const 0) (i32.const 1))
                    (i32.store8 $a (i32.const 1) (local.get 0))
                    (i32.add (i32.load8_u $a (i32.const 1)) (i32.load8_u $b (i32.const 2)))))",
            &[vec![Value::I32(0)], vec![Value::I32(37)]],
        );
        compare_values_to_wasmtime(
            "(module (memory $a i64 1 2) (memory $b i64 1 2)
                (data (memory $b) (i64.const 2) \"\\0b\")
                (func (export \"f\") (param i64) (result i64)
                    (i64.store8 $a (i64.const 1) (local.get 0))
                    (memory.copy $a $b (i64.const 3) (i64.const 2) (i64.const 1))
                    (i64.add (i64.load8_u $a (i64.const 1)) (i64.load8_u $a (i64.const 3)))))",
            &[vec![Value::I64(0)], vec![Value::I64(37)]],
        );
    }

    #[test]
    fn multiple_tables_and_table64_match_wasmtime() {
        compare_values_to_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (table $a 2 funcref) (table $b 2 funcref)
                (func $inc (type $t) local.get 0 i32.const 1 i32.add)
                (elem (table $a) (i32.const 1) func $inc)
                (func (export \"f\") (param i32) (result i32)
                    (table.copy $b $a (i32.const 0) (i32.const 1) (i32.const 1))
                    (call_indirect $b (type $t) (local.get 0) (i32.const 0))))",
            &[vec![Value::I32(0)], vec![Value::I32(37)]],
        );
        compare_values_to_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (table $a i64 1 2 funcref)
                (func $inc (type $t) local.get 0 i32.const 1 i32.add)
                (elem (table $a) (i64.const 0) func $inc)
                (func (export \"f\") (param i32) (result i32)
                    (call_indirect $a (type $t) (local.get 0) (i64.const 0))))",
            &[vec![Value::I32(0)], vec![Value::I32(37)]],
        );
    }

    #[test]
    fn funcref_global_initializers_match_wasmtime() {
        compare_values_to_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (func $inc (type $t) local.get 0 i32.const 1 i32.add)
                (global $r funcref (ref.func $inc))
                (table 1 funcref)
                (elem (i32.const 0) funcref (global.get $r))
                (func (export \"f\") (param i32) (result i32)
                    (call_indirect (type $t) (local.get 0) (i32.const 0))))",
            &[vec![Value::I32(0)], vec![Value::I32(37)]],
        );
    }

    #[test]
    fn imported_funcref_element_expressions_match_wasmtime() {
        let provider = "(module (type $t (func (param i32) (result i32)))
            (func $inc (type $t) local.get 0 i32.const 7 i32.add)
            (global (export \"ref\") funcref (ref.func $inc)))";
        linked_matches_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (import \"p\" \"ref\" (global $r funcref))
                (table 1 funcref (global.get $r))
                (func (export \"f\") (param i32) (result i32)
                    (call_indirect (type $t) (local.get 0) (i32.const 0))))",
            &[("p", provider)],
            &[0, 1, 37],
        );
        linked_matches_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (import \"p\" \"ref\" (global $r funcref))
                (table 1 funcref)
                (elem (i32.const 0) funcref (global.get $r))
                (func (export \"f\") (param i32) (result i32)
                    (call_indirect (type $t) (local.get 0) (i32.const 0))))",
            &[("p", provider)],
            &[0, 1, 37],
        );
        linked_matches_wasmtime(
            "(module (type $t (func (param i32) (result i32)))
                (import \"p\" \"ref\" (global $r funcref))
                (table 1 funcref)
                (elem $e funcref (global.get $r))
                (func (export \"f\") (param i32) (result i32)
                    (table.init $e (i32.const 0) (i32.const 0) (i32.const 1))
                    (call_indirect (type $t) (local.get 0) (i32.const 0))))",
            &[("p", provider)],
            &[0, 1, 37],
        );
    }

    #[test]
    fn second_memory_export_supports_per_lane_io() {
        let wasm = wat::parse_str(
            "(module (memory $a (export \"first\") 1) (memory $b (export \"second\") 1)
                (func (export \"f\") (param i32) (result i32)
                    (i32.load8_u $b (local.get 0))))",
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        let (values, memories) = kernel
            .run_values_with_memory_io(
                &[vec![Value::I32(1)], vec![Value::I32(1)]],
                "second",
                1,
                &[vec![11], vec![23]],
                0..3,
            )
            .unwrap();
        assert_eq!(values, [vec![Value::I32(11)], vec![Value::I32(23)]]);
        assert_eq!(memories, [vec![0, 11, 0], vec![0, 23, 0]]);
    }

    #[test]
    fn multiple_imported_memories_and_tables_keep_provider_state() {
        let provider = r#"(module
            (type $t (func (param i32) (result i32)))
            (memory $a (export "a") 1) (memory $b (export "b") 1)
            (table $ta (export "ta") 1 funcref)
            (table $tb (export "tb") 1 funcref)
            (func $inc (type $t) local.get 0 i32.const 1 i32.add)
            (elem (table $tb) (i32.const 0) func $inc)
            (data (memory $b) (i32.const 0) "\09"))"#;
        let root = r#"(module
            (type $t (func (param i32) (result i32)))
            (import "p" "a" (memory $a 1))
            (import "p" "b" (memory $b 1))
            (import "p" "ta" (table $ta 1 funcref))
            (import "p" "tb" (table $tb 1 funcref))
            (func (export "f") (param i32) (result i32)
                (i32.store8 $a (i32.const 0) (local.get 0))
                (call_indirect $tb (type $t)
                    (i32.add (i32.load8_u $a (i32.const 0))
                             (i32.load8_u $b (i32.const 0)))
                    (i32.const 0))))"#;
        linked_matches_wasmtime(root, &[("p", provider)], &[0, 1, 25]);
    }

    #[test]
    fn table64_growth_bulk_operations_and_failure_match_wasmtime() {
        compare_values_to_wasmtime(
            r#"(module (type $t (func (param i32) (result i32)))
                (table $table i64 1 2 funcref)
                (func $inc (type $t) local.get 0 i32.const 1 i32.add)
                (elem $segment func $inc)
                (func (export "f") (param i32) (result i32 i64)
                    ref.null func i64.const 1 table.grow $table drop
                    i64.const 0 i32.const 0 i32.const 1 table.init $table $segment
                    i64.const 1 i64.const 0 i64.const 1 table.copy $table $table
                    i64.const 0 ref.func $inc table.set $table
                    i64.const 0 ref.func $inc i64.const 1 table.fill $table
                    local.get 0 i64.const 1 call_indirect $table (type $t)
                    ref.null func i64.const 1 table.grow $table) )"#,
            &[vec![Value::I32(0)], vec![Value::I32(9)]],
        );
    }

    #[test]
    fn memory64_and_second_memory_growth_are_independent() {
        compare_values_to_wasmtime(
            r#"(module (memory $small 1 2) (memory $wide i64 1 2)
                (func (export "f") (param i32) (result i32 i64 i32 i64)
                    local.get 0 memory.grow $small
                    i64.const 1 memory.grow $wide
                    memory.size $small
                    memory.size $wide))"#,
            &[
                vec![Value::I32(0)],
                vec![Value::I32(1)],
                vec![Value::I32(u32::MAX)],
            ],
        );
    }

    #[test]
    fn growth_reserve_follows_the_selected_memory() {
        let wasm = wat::parse_str(
            r#"(module (memory $first 1 2) (memory $second 1 2)
                (func (export "f") (param i32) (result i32 i32 i32)
                    local.get 0 memory.grow $second
                    memory.size $first
                    memory.size $second))"#,
        )
        .unwrap();
        let limits = KernelLimits {
            memory_bytes: 3 * 65536,
            memory_growth_pages: 1,
            ..KernelLimits::default()
        };
        let actual = Kernel::new_with_limits(&wasm, "f", limits)
            .unwrap()
            .run_values(&[vec![Value::I32(1)], vec![Value::I32(2)]])
            .unwrap();
        assert_eq!(
            actual,
            [
                vec![Value::I32(1), Value::I32(1), Value::I32(2)],
                vec![Value::I32(u32::MAX), Value::I32(1), Value::I32(1)],
            ]
        );
    }

    #[test]
    fn growable_memories_share_the_runtime_page_limit() {
        let wasm = wat::parse_str(
            r#"(module (memory $first 1 2) (memory $second 1 2)
                (func (export "f") (param i32) (result i32 i32)
                    local.get 0
                    if
                        i32.const 1 memory.grow $second drop
                    else
                        i32.const 1 memory.grow $first drop
                    end
                    memory.size $first
                    memory.size $second)
                (func (export "both") (result i32 i32)
                    i32.const 1 memory.grow $first
                    i32.const 1 memory.grow $second))"#,
        )
        .unwrap();
        let limits = KernelLimits {
            memory_bytes: 3 * 65536,
            memory_growth_pages: 1,
            ..KernelLimits::default()
        };
        let actual = Kernel::new_with_limits(&wasm, "f", limits)
            .unwrap()
            .run_values(&[vec![Value::I32(0)], vec![Value::I32(1)]])
            .unwrap();
        assert_eq!(
            actual,
            [
                vec![Value::I32(2), Value::I32(1)],
                vec![Value::I32(1), Value::I32(2)],
            ]
        );
        let both = Kernel::new_with_limits(&wasm, "both", limits).unwrap();
        assert_eq!(
            both.run_values(&[vec![]]).unwrap(),
            [vec![Value::I32(1), Value::I32(u32::MAX)]]
        );
    }

    #[test]
    fn table_copy_between_32_and_64_bit_tables_matches_wasmtime() {
        compare_values_to_wasmtime(
            r#"(module (type $t (func (param i32) (result i32)))
                (table $small 1 funcref) (table $wide i64 1 funcref)
                (func $inc (type $t) local.get 0 i32.const 1 i32.add)
                (elem (table $wide) (i64.const 0) func $inc)
                (func (export "f") (param i32) (result i32)
                    (table.copy $small $wide (i32.const 0) (i64.const 0) (i32.const 1))
                    (call_indirect $small (type $t) (local.get 0) (i32.const 0))))"#,
            &[vec![Value::I32(0)], vec![Value::I32(17)]],
        );
    }

    #[test]
    fn memory_copy_between_32_and_64_bit_memories_matches_wasmtime() {
        compare_values_to_wasmtime(
            r#"(module (memory $small 1) (memory $wide i64 1)
                (data (memory $wide) (i64.const 0) "\2a")
                (func (export "f") (param i32) (result i32)
                    (memory.copy $small $wide (i32.const 1) (i64.const 0) (i32.const 1))
                    (i32.add (i32.load8_u $small (i32.const 1)) (local.get 0))))"#,
            &[vec![Value::I32(0)], vec![Value::I32(17)]],
        );
    }
}

fn expr_uses_gc(expr: &ConstExpr<'_>) -> bool {
    expr.get_operators_reader().into_iter().any(|op| {
        matches!(
            op,
            Ok(Operator::StructNew { .. }
                | Operator::StructNewDefault { .. }
                | Operator::ArrayNew { .. }
                | Operator::ArrayNewDefault { .. }
                | Operator::ArrayNewFixed { .. }
                | Operator::RefI31
                | Operator::AnyConvertExtern
                | Operator::ExternConvertAny)
        )
    })
}
