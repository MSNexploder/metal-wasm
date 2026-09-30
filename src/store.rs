use super::*;

/// A store-scoped module instance. Each instance has one independent state per lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Instance {
    store: u64,
    index: usize,
}

struct Module {
    wasm: Vec<u8>,
    imports: HashMap<String, usize>,
    visible: bool,
    exports: HashMap<String, (ExternalKind, usize)>,
    memories: Vec<Option<(usize, usize)>>,
    tables: Vec<Option<(usize, usize)>>,
    globals: Vec<(usize, usize)>,
    global_types: Vec<GlobalType>,
    tags: Vec<u32>,
}

#[derive(Clone)]
struct Resource {
    name: String,
    ty: &'static str,
    per_lane: usize,
    fixed: usize,
    size: usize,
    shared: bool,
}
impl Resource {
    fn offset(&self, lanes: usize) -> usize {
        // allocate validates every offset and end before installing this layout.
        self.per_lane * lanes + self.fixed
    }
}

fn resources<'a>(templates: impl Iterator<Item = &'a InstanceTemplate>) -> Vec<Vec<Resource>> {
    let (mut per_lane, mut fixed) = (0, 0);
    templates
        .enumerate()
        .map(|(m, t)| {
            let mut result = Vec::new();
            let mut add = |field: String, ty, size: usize, shared| {
                result.push(Resource {
                    name: format!("m{m}_{field}"),
                    ty,
                    per_lane,
                    fixed,
                    size,
                    shared,
                });
                let aligned = size.max(1).div_ceil(8) * 8;
                if shared {
                    fixed += aligned;
                } else {
                    per_lane += aligned;
                }
            };
            for (i, memory) in t.memories.iter().enumerate() {
                add(format!("memory{i}"), "uchar", memory.capacity.max(1), false);
                add(format!("pages{i}"), "uint", 4, false);
            }
            for (i, table) in t.tables.iter().enumerate() {
                add(
                    format!("table{i}"),
                    "ulong",
                    table.info.capacity.max(1) * 8,
                    false,
                );
            }
            if !t.globals_template.is_empty() {
                add(
                    "globals".into(),
                    "ulong",
                    t.globals_template.len() * 8,
                    false,
                );
            }
            if !t.data_lengths.is_empty() {
                add(
                    "passive_data".into(),
                    "const uchar",
                    t.passive_data.len(),
                    true,
                );
            }
            result
        })
        .collect()
}

pub(super) fn resource_source(modules: &[ParsedModule<'_>]) -> String {
    resources(modules.iter().map(|m| &m.instance)).into_iter().flatten().map(|r|
        format!("device {}* {} = reinterpret_cast<device {}*>(resource_buffer + count * {}ul + {}ul);\n",r.ty,r.name,r.ty,r.per_lane,r.fixed)
    ).collect()
}

pub(super) struct MetadataLayout {
    fields: Vec<(String, u64)>,
    module_ranges: Vec<Range<usize>>,
    exceptions: Option<(usize, usize)>,
    global_roots: Vec<(usize, usize, RefType)>,
    table_roots: Vec<(usize, usize, RefType)>,
    element_roots: Vec<(usize, usize, usize, RefType)>,
}
impl MetadataLayout {
    pub(super) fn new(modules: &[ParsedModule<'_>], exceptions: Option<(usize, usize)>) -> Self {
        let mut fields = Vec::new();
        let mut global_roots = Vec::new();
        let mut table_roots = Vec::new();
        let mut element_roots = Vec::new();
        let mut module_ranges = Vec::new();
        for (m, module) in modules.iter().enumerate() {
            let start = fields.len();
            let imported = module.global_types.len() - module.local_global_slots.len();
            for (i, &slot) in module.local_global_slots.iter().enumerate() {
                if let ValType::Ref(r) = module.global_types[imported + i].content_type {
                    global_roots.push((m, slot, r));
                }
            }
            let imported = module.table_types.len() - module.instance.tables.len();
            for i in 0..module.instance.tables.len() {
                table_roots.push((m, i, module.table_types[imported + i].element_type));
            }
            for (t, table) in module.instance.tables.iter().enumerate() {
                fields.push((
                    format!("states[{m}].table_size[{t}]"),
                    table.values.len() as u64,
                ));
            }
            for (field, lengths) in [
                ("element_lengths", &module.instance.element_info.lengths),
                ("data_lengths", &module.instance.data_lengths),
            ] {
                for (i, &len) in lengths.iter().enumerate() {
                    fields.push((format!("states[{m}].{field}[{i}]"), u64::from(len)));
                }
            }
            for (i, values) in module.instance.element_info.elements.iter().enumerate() {
                if module.dynamic_elements.iter().any(|(s, _, _)| *s == i) {
                    element_roots.push((
                        fields.len(),
                        start + module.instance.tables.len() + i,
                        values.len(),
                        module.instance.element_info.element_types[i],
                    ));
                    for (j, &value) in values.iter().enumerate() {
                        fields.push((format!("states[{m}].element_values[{i}][{j}]"), value));
                    }
                }
            }
            module_ranges.push(start..fields.len());
        }
        Self {
            fields,
            module_ranges,
            exceptions,
            global_roots,
            table_roots,
            element_roots,
        }
    }
    fn stride(&self) -> usize {
        (self.fields.len() + self.exceptions.map_or(0, |(n, w)| 1 + n * (w + 1))).max(1)
    }
    fn initial(&self) -> Vec<u64> {
        let mut v = vec![0; self.stride()];
        for (i, (_, value)) in self.fields.iter().enumerate() {
            v[i] = *value;
        }
        v
    }
    pub(super) fn load_source(&self) -> String {
        let mut s = format!(
            "device ulong* saved = persistent_state + ulong(id)*{}ul;\n",
            self.stride()
        );
        for (i, (field, _)) in self.fields.iter().enumerate() {
            s.push_str(&format!("{field} = saved[{i}];\n"));
        }
        if let Some((_, w)) = self.exceptions {
            let o = self.fields.len();
            s.push_str(&format!("exceptions.count = uint(saved[{o}]); for(uint j=0u;j<exceptions.count;++j) {{ exceptions.tags[j]=uint(saved[{}ul+ulong(j)*{}ul]); for(uint k=0u;k<{w}u;++k) exceptions.values[j*{w}u+k]=saved[{}ul+ulong(j)*{}ul+k]; }}\n",o+1,w+1,o+2,w+1));
        }
        s
    }
    pub(super) fn save_source(&self) -> String {
        let mut s = String::new();
        for (i, (field, _)) in self.fields.iter().enumerate() {
            s.push_str(&format!("saved[{i}] = {field};\n"));
        }
        if let Some((_, w)) = self.exceptions {
            let o = self.fields.len();
            s.push_str(&format!("saved[{o}]=exceptions.count; for(uint j=0u;j<exceptions.count;++j) {{ saved[{}ul+ulong(j)*{}ul]=exceptions.tags[j]; for(uint k=0u;k<{w}u;++k) saved[{}ul+ulong(j)*{}ul+k]=exceptions.values[j*{w}u+k]; }}\n",o+1,w+1,o+2,w+1));
        }
        s
    }
}

#[derive(Clone, Copy)]
struct LiveGc {
    lane: usize,
    word: u64,
    ty: RefType,
}
#[derive(Clone)]
enum LiveException {
    Lane { lane: usize, slot: usize },
    Host(ExceptionRecord),
}

// The lease generation belongs to the public handle. The stored identity stays
// available to Wasm and other retained graphs after that public lease is released.
#[derive(Clone)]
struct Lease<T> {
    generation: u64,
    retained: bool,
    value: Option<T>,
}
impl<T> Lease<T> {
    fn valid(&self, generation: u64) -> bool {
        self.retained && self.generation == generation && self.value.is_some()
    }
    fn release(&mut self, generation: u64) -> Result<(), Error> {
        if !self.valid(generation) {
            return Err(Error::Unsupported(
                "invalid or released reference for this store".into(),
            ));
        }
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::Unsupported("reference generation exhausted".into()))?;
        self.retained = false;
        Ok(())
    }
}
fn lease_insert<T>(values: &mut Vec<Lease<T>>, value: T) -> Result<usize, Error> {
    if let Some(index) = values.iter().position(|v| v.value.is_none()) {
        values[index].value = Some(value);
        values[index].retained = true;
        return Ok(index);
    }
    if values.len() >= u32::MAX as usize {
        return Err(Error::Unsupported("too many retained references".into()));
    }
    values.push(Lease {
        generation: 1,
        retained: true,
        value: Some(value),
    });
    Ok(values.len() - 1)
}
#[derive(Default)]
struct Reachable {
    gc: HashSet<(usize, u64)>,
    exceptions: HashSet<(usize, usize)>,
    host_exceptions: HashSet<usize>,
    externals: HashSet<usize>,
}
enum TraceReference {
    Host(Value),
    Guest(usize, RefType, u64),
}

/// A fixed collection of lanes with persistent WebAssembly state.
/// Calls, initialization, and traps retain memory, table, global, and heap writes.
/// GC and exception handles obtained from a lane can only be passed to that lane.
/// Host values and function handles are shared by all lanes in the same store.
/// Host handles retain their graphs until explicitly released. Copied handles and
/// conversion aliases share one lease. Collection preserves all reachable Wasm state.
/// A failed Metal command makes the store unusable.
pub struct Store {
    id: u64,
    lanes: usize,
    limits: KernelLimits,
    device: Device,
    modules: Vec<Module>,
    entries: HashMap<(usize, String), Kernel>,
    resources: Vec<Vec<Resource>>,
    resource_buffer: Option<metal::Buffer>,
    metadata: MetadataLayout,
    metadata_buffer: Option<metal::Buffer>,
    heap: Option<(metal::Buffer, usize)>,
    gc_values: Mutex<Vec<Lease<LiveGc>>>,
    exceptions: Mutex<Vec<Lease<LiveException>>>,
    host_exception_slots: HashMap<(usize, u32), usize>,
    externals: Mutex<Vec<Lease<Arc<dyn Any + Send + Sync>>>>,
    function_types: Vec<RefType>,
    reference_types: ReferenceTypes,
    gc_types: Vec<GcType>,
    tag_types: Vec<Vec<ValType>>,
    poisoned: bool,
}

impl Store {
    pub fn new(lanes: usize) -> Result<Self, Error> {
        Self::new_with_limits(lanes, KernelLimits::default())
    }
    pub fn new_with_limits(lanes: usize, limits: KernelLimits) -> Result<Self, Error> {
        if lanes == 0 || lanes > u32::MAX as usize {
            return Err(Error::Unsupported(
                "store lane count must be between 1 and u32::MAX".into(),
            ));
        }
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
        Ok(Self {
            id: NEXT_KERNEL_ID.fetch_add(1, Ordering::Relaxed),
            lanes,
            limits,
            device: Device::system_default()
                .ok_or_else(|| Error::Metal("no Metal device is available".into()))?,
            modules: Vec::new(),
            entries: HashMap::new(),
            resources: Vec::new(),
            resource_buffer: None,
            metadata: MetadataLayout {
                fields: Vec::new(),
                module_ranges: Vec::new(),
                exceptions: None,
                global_roots: Vec::new(),
                table_roots: Vec::new(),
                element_roots: Vec::new(),
            },
            metadata_buffer: None,
            heap: None,
            gc_values: Mutex::new(Vec::new()),
            exceptions: Mutex::new(Vec::new()),
            host_exception_slots: HashMap::new(),
            externals: Mutex::new(Vec::new()),
            function_types: Vec::new(),
            reference_types: HashMap::new(),
            gc_types: Vec::new(),
            tag_types: Vec::new(),
            poisoned: false,
        })
    }
    fn check(&self) -> Result<(), Error> {
        if self.poisoned {
            Err(Error::Metal(
                "store is unusable after a failed Metal command".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn instance(&self, instance: Instance) -> Result<usize, Error> {
        self.check()?;
        if instance.store != self.id || self.modules.get(instance.index).is_none_or(|m| !m.visible)
        {
            return Err(Error::Unsupported("invalid instance for this store".into()));
        }
        Ok(instance.index)
    }
    fn export(
        &self,
        instance: Instance,
        name: &str,
        kind: ExternalKind,
    ) -> Result<(usize, usize), Error> {
        let m = self.instance(instance)?;
        let &(actual, index) = self.modules[m]
            .exports
            .get(name)
            .ok_or_else(|| Error::Unsupported(format!("unknown export {name:?}")))?;
        if actual != kind {
            return Err(Error::Unsupported(format!(
                "export {name:?} has the wrong kind"
            )));
        }
        Ok((m, index))
    }
    fn parse(&self) -> Result<(Vec<ParsedModule<'_>>, usize), Error> {
        let mut context = TypeContext::default();
        let mut modules = self
            .modules
            .iter()
            .map(|m| parse_module(&m.wasm, self.limits, &mut context))
            .collect::<Result<Vec<_>, _>>()?;
        for i in 0..modules.len() {
            link_module(i, &mut modules, &self.modules[i].imports)?;
            // Import matching uses the smallest current size, so every lane must satisfy it.
            for local in 0..modules[i].memory_types.len() {
                if let Some((owner, index)) = modules[i].memory_bindings[local] {
                    if owner < self.resources.len() {
                        modules[i].memory_types[local].initial =
                            self.memory_pages(owner, index).into_iter().min().unwrap() as u64;
                    }
                }
            }
            for local in 0..modules[i].table_types.len() {
                if let Some((owner, index)) = modules[i].table_bindings[local] {
                    if owner < self.resources.len() {
                        modules[i].table_types[local].initial = (0..self.lanes)
                            .map(|lane| self.table_size(owner, index, lane) as u64)
                            .min()
                            .unwrap();
                    }
                }
            }
        }
        Ok((modules, context.next_tag as usize + 1))
    }
    fn compile(
        &self,
        root: usize,
        export: Option<&str>,
    ) -> Result<(Kernel, MetadataLayout), Error> {
        let (modules, tags) = self.parse()?;
        let count = modules.len();
        let mut translation = translate_modules(
            modules,
            (0..count).collect(),
            tags,
            root,
            export,
            self.limits,
            true,
        )?;
        let metadata = translation.persistent_metadata.take().unwrap();
        Ok((
            Kernel::from_translation(translation, self.limits)?,
            metadata,
        ))
    }
    pub fn instantiate(
        &mut self,
        wasm: &[u8],
        imports: &[(&str, Instance)],
    ) -> Result<Instance, Error> {
        self.check()?;
        let mut map = HashMap::new();
        for &(name, instance) in imports {
            let index = self.instance(instance)?;
            if map.insert(name.to_owned(), index).is_some() {
                return Err(Error::Unsupported(format!(
                    "duplicate Wasm provider {name:?}"
                )));
            }
        }
        let index = self.modules.len();
        self.modules.push(Module {
            wasm: wasm.to_vec(),
            imports: map,
            visible: false,
            exports: HashMap::new(),
            memories: Vec::new(),
            tables: Vec::new(),
            globals: Vec::new(),
            global_types: Vec::new(),
            tags: Vec::new(),
        });
        let preparation = (|| {
            let (parsed, _) = self.parse()?;
            let module = &parsed[index];
            let info = (
                module.exports.clone(),
                module.memory_bindings.clone(),
                module.table_bindings.clone(),
                module.global_bindings.clone(),
                module.global_types.clone(),
                module.tag_bindings.clone(),
            );
            let (kernel, metadata) = self.compile(index, None)?;
            Ok::<_, Error>((info, kernel, metadata))
        })();
        let (info, kernel, metadata) = match preparation {
            Ok(v) => v,
            Err(e) => {
                self.modules.pop();
                return Err(e);
            }
        };
        if let Err(e) = self.allocate(&kernel, metadata) {
            self.modules.pop();
            return Err(e);
        }
        let module = &mut self.modules[index];
        (
            module.exports,
            module.memories,
            module.tables,
            module.globals,
            module.global_types,
            module.tags,
        ) = info;
        self.function_types = kernel.function_reference_types.clone();
        self.reference_types = kernel.reference_supertypes.clone();
        self.gc_types = kernel.gc_types.clone();
        self.tag_types = kernel.tag_payload_types.clone();
        self.entries.clear();
        self.execute(&kernel, &vec![Vec::new(); self.lanes])?;
        self.modules[index].visible = true;
        Ok(Instance {
            store: self.id,
            index,
        })
    }
    fn resource(&self, module: usize, field: &str) -> &Resource {
        let name = format!("m{module}_{field}");
        self.resources[module]
            .iter()
            .find(|r| r.name == name)
            .unwrap()
    }
    fn bytes(&self, r: &Resource) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.resource_buffer
                    .as_ref()
                    .unwrap()
                    .contents()
                    .cast::<u8>()
                    .add(r.offset(self.lanes)),
                r.size * if r.shared { 1 } else { self.lanes },
            )
        }
    }
    fn memory_pages(&self, module: usize, index: usize) -> Vec<u32> {
        self.bytes(self.resource(module, &format!("pages{index}")))
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    }
    fn metadata_words(&self) -> &[u64] {
        unsafe {
            std::slice::from_raw_parts(
                self.metadata_buffer.as_ref().unwrap().contents().cast(),
                self.metadata.stride() * self.lanes,
            )
        }
    }
    fn table_size(&self, module: usize, index: usize, lane: usize) -> usize {
        self.metadata_words()
            [lane * self.metadata.stride() + self.metadata.module_ranges[module].start + index]
            as usize
    }
    fn allocate(&mut self, kernel: &Kernel, metadata: MetadataLayout) -> Result<(), Error> {
        let layout = resources(kernel.instances.iter().map(|i| &i.template));
        let total = layout
            .iter()
            .flatten()
            .try_fold(4usize, |total, r| {
                let offset = r.per_lane.checked_mul(self.lanes)?.checked_add(r.fixed)?;
                let bytes = r
                    .size
                    .max(1)
                    .div_ceil(8)
                    .checked_mul(8)?
                    .checked_mul(if r.shared { 1 } else { self.lanes })?;
                Some(total.max(offset.checked_add(bytes)?))
            })
            .ok_or_else(|| Error::Unsupported("store resource size overflow".into()))?;
        let metadata_bytes = metadata
            .stride()
            .checked_mul(self.lanes)
            .and_then(|words| words.checked_mul(size_of::<u64>()))
            .ok_or_else(|| Error::Unsupported("store metadata size overflow".into()))?;
        if total.max(metadata_bytes) > isize::MAX as usize
            || total.max(metadata_bytes) as u64 > self.device.max_buffer_length()
        {
            return Err(Error::Unsupported(
                "store exceeds the Metal buffer limit".into(),
            ));
        }
        let buffer = self
            .device
            .new_buffer(total as u64, MTLResourceOptions::StorageModeShared);
        unsafe {
            std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, total);
        }
        if let Some(old) = &self.resource_buffer {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    old.contents().cast::<u8>(),
                    buffer.contents().cast::<u8>(),
                    old.length() as usize,
                );
            }
        }
        for m in self.resources.len()..layout.len() {
            let t = &kernel.instances[m].template;
            let write = |field: &str, bytes: &[u8]| {
                let name = format!("m{m}_{field}");
                let r = layout[m].iter().find(|r| r.name == name).unwrap();
                for lane in 0..if r.shared { 1 } else { self.lanes } {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes.as_ptr(),
                            buffer
                                .contents()
                                .cast::<u8>()
                                .add(r.offset(self.lanes) + lane * r.size),
                            bytes.len(),
                        );
                    }
                }
            };
            for (i, memory) in t.memories.iter().enumerate() {
                write(&format!("memory{i}"), &memory.bytes);
                write(
                    &format!("pages{i}"),
                    &(memory.bytes.len() as u32 / 65536).to_ne_bytes(),
                );
            }
            for (i, table) in t.tables.iter().enumerate() {
                write(&format!("table{i}"), words_bytes(&table.values));
            }
            if !t.globals_template.is_empty() {
                write("globals", words_bytes(&t.globals_template));
            }
            if !t.data_lengths.is_empty() {
                write("passive_data", &t.passive_data);
            }
        }
        let mut data = metadata.initial().repeat(self.lanes);
        if self.metadata_buffer.is_some() {
            let old = self.metadata_words();
            for lane in 0..self.lanes {
                for m in 0..self.metadata.module_ranges.len() {
                    let a = self.metadata.module_ranges[m].clone();
                    let b = metadata.module_ranges[m].clone();
                    data[lane * metadata.stride() + b.start..lane * metadata.stride() + b.end]
                        .copy_from_slice(
                            &old[lane * self.metadata.stride() + a.start
                                ..lane * self.metadata.stride() + a.end],
                        );
                }
                if let Some((_, old_width)) = self.metadata.exceptions {
                    let (_, new_width) = metadata.exceptions.unwrap();
                    let src = lane * self.metadata.stride() + self.metadata.fields.len();
                    let dst = lane * metadata.stride() + metadata.fields.len();
                    let count = old[src] as usize;
                    data[dst] = count as u64;
                    for record in 0..count {
                        let a = src + 1 + record * (old_width + 1);
                        let b = dst + 1 + record * (new_width + 1);
                        data[b..b + old_width + 1].copy_from_slice(&old[a..a + old_width + 1]);
                    }
                }
            }
        }
        let metadata_buffer = make_buffer(&self.device, &data)?;
        self.resource_buffer = Some(buffer);
        self.resources = layout;
        self.metadata = metadata;
        self.metadata_buffer = Some(metadata_buffer);
        Ok(())
    }
    pub fn call(
        &mut self,
        instance: Instance,
        export: &str,
        inputs: &[Vec<Value>],
    ) -> Result<Vec<Vec<Value>>, Error> {
        self.call_timed(instance, export, inputs, false)
            .map(|(v, _)| v)
    }
    pub fn call_i32(
        &mut self,
        instance: Instance,
        export: &str,
        inputs: &[u32],
    ) -> Result<Vec<u32>, Error> {
        self.call_i32_timed(instance, export, inputs)
            .map(|(v, _)| v)
    }
    pub fn call_i32_timed(
        &mut self,
        instance: Instance,
        export: &str,
        inputs: &[u32],
    ) -> Result<(Vec<u32>, Option<Duration>), Error> {
        let (values, time) = self.call_timed(
            instance,
            export,
            &inputs
                .iter()
                .map(|&v| vec![Value::I32(v)])
                .collect::<Vec<_>>(),
            true,
        )?;
        let values = values
            .into_iter()
            .map(|row| match row.as_slice() {
                [Value::I32(v)] => Ok(*v),
                _ => Err(Error::Unsupported(
                    "export must have signature (i32) -> i32".into(),
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((values, time))
    }
    fn call_timed(
        &mut self,
        instance: Instance,
        export: &str,
        inputs: &[Vec<Value>],
        i32_only: bool,
    ) -> Result<(Vec<Vec<Value>>, Option<Duration>), Error> {
        let (index, _) = self.export(instance, export, ExternalKind::Func)?;
        if inputs.len() != self.lanes {
            return Err(Error::Unsupported(
                "input row count does not match store lane count".into(),
            ));
        }
        let key = (index, export.to_owned());
        let kernel = if let Some(kernel) = self.entries.remove(&key) {
            kernel
        } else {
            self.compile(index, Some(export))?.0
        };
        let result =
            if i32_only && (kernel.params != [ValType::I32] || kernel.results != [ValType::I32]) {
                Err(Error::Unsupported(
                    "export must have signature (i32) -> i32".into(),
                ))
            } else {
                self.execute(&kernel, inputs)
            };
        self.entries.insert(key, kernel);
        result
    }
    fn execute(
        &mut self,
        kernel: &Kernel,
        inputs: &[Vec<Value>],
    ) -> Result<(Vec<Vec<Value>>, Option<Duration>), Error> {
        self.check()?;
        for (lane, row) in inputs.iter().enumerate() {
            if row.len() != kernel.params.len()
                || row
                    .iter()
                    .zip(&kernel.params)
                    .any(|(&v, &t)| !self.accepts(v, t, lane))
            {
                return Err(Error::Unsupported(
                    "input values do not match the export signature or lane".into(),
                ));
            }
        }
        let fast = kernel.params == [ValType::I32] && kernel.results == [ValType::I32];
        let mut input = Vec::new();
        let mut metadata = self.metadata_words().to_vec();
        let retained = self.trace_references(false, &metadata, &self.host_exception_slots)?;
        let mut host_slots = self.host_exception_slots.clone();
        host_slots.retain(|&(_, id), _| retained.host_exceptions.contains(&(id as usize - 1)));
        for (lane, row) in inputs.iter().enumerate() {
            for &value in row {
                let words = self.encode(value, lane, &mut metadata, &mut host_slots)?;
                for word in words {
                    if fast {
                        input.extend_from_slice(&(word as u32).to_ne_bytes());
                    } else {
                        input.extend_from_slice(&word.to_ne_bytes());
                    }
                }
            }
        }
        let mut buffers = vec![vec![
            (4, self.resource_buffer.as_ref().unwrap().clone()),
            (6, self.metadata_buffer.as_ref().unwrap().clone()),
        ]];
        if let Some(gc) = kernel.gc {
            let host_capacity = gc.heap_words + self.limits.exception_records;
            let stride = gc.heap_words + gc.root_words + host_capacity + 1;
            let needs_allocation = self
                .heap
                .as_ref()
                .is_none_or(|(_, old_stride)| *old_stride != stride);
            if needs_allocation {
                let buffer = lane_buffer(
                    &self.device,
                    &[gc.heap_words as u64],
                    stride,
                    self.lanes as u32,
                )?;
                if let Some((old, old_stride)) = &self.heap {
                    for lane in 0..self.lanes {
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                old.contents().cast::<u64>().add(lane * old_stride),
                                buffer.contents().cast::<u64>().add(lane * stride),
                                gc.heap_words,
                            );
                        }
                    }
                }
                self.heap = Some((buffer, stride));
            }
            let (heap, _) = self.heap.as_ref().unwrap();
            let retained = self.trace_references(false, &metadata, &host_slots)?;
            for lane in 0..self.lanes {
                let roots =
                    retained
                        .gc
                        .iter()
                        .filter_map(|&(l, word)| (l == lane).then_some(word))
                        .chain(retained.exceptions.iter().filter_map(|&(l, slot)| {
                            (l == lane).then_some(0x400000000 | slot as u64)
                        }))
                        .collect::<Vec<_>>();
                if roots.len() > host_capacity {
                    return Err(Error::Unsupported(
                        "too many retained reference roots".into(),
                    ));
                }
                unsafe {
                    let ptr = heap
                        .contents()
                        .cast::<u64>()
                        .add(lane * stride + gc.heap_words + gc.root_words);
                    std::ptr::copy_nonoverlapping(roots.as_ptr(), ptr, roots.len());
                    *ptr.add(host_capacity) = roots.len() as u64;
                }
            }
            buffers.push(vec![(5, heap.clone())]);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                metadata.as_ptr(),
                self.metadata_buffer
                    .as_ref()
                    .unwrap()
                    .contents()
                    .cast::<u64>(),
                metadata.len(),
            );
        }
        self.host_exception_slots = host_slots;
        let result_words: usize = kernel.results.iter().map(|t| value_words(*t)).sum();
        let dispatched = kernel.dispatch(
            &input,
            self.lanes * result_words * if fast { 4 } else { 8 },
            self.lanes as u32,
            &buffers,
            None,
            0,
        );
        let (output, time) = match dispatched {
            Ok(v) => v,
            Err(e) => {
                if matches!(&e, Error::Metal(_)) {
                    self.poisoned = true;
                } else {
                    self.collect_garbage()?;
                }
                return Err(e);
            }
        };
        let mut rows = Vec::new();
        for lane in 0..self.lanes {
            let mut row = Vec::new();
            let mut offset = 0;
            for &ty in &kernel.results {
                let words = if fast {
                    vec![unsafe { *output.contents().cast::<u32>().add(lane) } as u64]
                } else {
                    unsafe {
                        std::slice::from_raw_parts(
                            output
                                .contents()
                                .cast::<u64>()
                                .add(lane * result_words + offset),
                            value_words(ty),
                        )
                        .to_vec()
                    }
                };
                row.push(self.decode(ty, &words, lane)?);
                offset += value_words(ty);
            }
            rows.push(row);
        }
        self.collect_garbage()?;
        Ok((rows, time))
    }
    pub fn get_global(&self, instance: Instance, export: &str) -> Result<Vec<Value>, Error> {
        let (m, g) = self.export(instance, export, ExternalKind::Global)?;
        let (owner, slot) = self.modules[m].globals[g];
        let ty = self.modules[m].global_types[g].content_type;
        let r = self.resource(owner, "globals");
        (0..self.lanes)
            .map(|lane| {
                let words = unsafe {
                    std::slice::from_raw_parts(
                        self.bytes(r)
                            .as_ptr()
                            .add(lane * r.size + slot * 8)
                            .cast::<u64>(),
                        value_words(ty),
                    )
                };
                self.decode(ty, words, lane)
            })
            .collect()
    }
    pub fn read_memory(
        &self,
        instance: Instance,
        export: &str,
        range: Range<usize>,
    ) -> Result<Vec<Vec<u8>>, Error> {
        let (m, i) = self.export(instance, export, ExternalKind::Memory)?;
        let (owner, index) = self.modules[m].memories[i].unwrap();
        if range.start > range.end
            || self
                .memory_pages(owner, index)
                .iter()
                .any(|&n| range.end > n as usize * 65536)
        {
            return Err(Error::Unsupported(
                "memory range exceeds current memory size".into(),
            ));
        }
        let r = self.resource(owner, &format!("memory{index}"));
        let bytes = self.bytes(r);
        Ok((0..self.lanes)
            .map(|lane| bytes[lane * r.size + range.start..lane * r.size + range.end].to_vec())
            .collect())
    }
    pub fn write_memory(
        &mut self,
        instance: Instance,
        export: &str,
        offset: usize,
        rows: &[Vec<u8>],
    ) -> Result<(), Error> {
        let (m, i) = self.export(instance, export, ExternalKind::Memory)?;
        let (owner, index) = self.modules[m].memories[i].unwrap();
        if rows.len() != self.lanes {
            return Err(Error::Unsupported(
                "memory row count does not match store lane count".into(),
            ));
        }
        let pages = self.memory_pages(owner, index);
        if rows.iter().enumerate().any(|(lane, row)| {
            offset
                .checked_add(row.len())
                .is_none_or(|end| end > pages[lane] as usize * 65536)
        }) {
            return Err(Error::Unsupported(
                "memory range exceeds current memory size".into(),
            ));
        }
        let r = self.resource(owner, &format!("memory{index}"));
        for (lane, row) in rows.iter().enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    self.resource_buffer
                        .as_ref()
                        .unwrap()
                        .contents()
                        .cast::<u8>()
                        .add(r.offset(self.lanes) + lane * r.size + offset),
                    row.len(),
                );
            }
        }
        Ok(())
    }
    pub fn function_reference(
        &self,
        instance: Instance,
        export: &str,
    ) -> Result<FunctionRef, Error> {
        let (m, local) = self.export(instance, export, ExternalKind::Func)?;
        let (modules, _) = self.parse()?;
        let mut function_ids = Vec::new();
        let mut next = 1;
        for module in &modules {
            let mut ids = vec![0; module.function_types.len()];
            for id in ids.iter_mut().skip(module.imports.len()) {
                *id = next;
                next += 1;
            }
            function_ids.push(ids);
        }
        for (m, module) in modules.iter().enumerate() {
            for (i, &(owner, local)) in module.resolved_imports.iter().enumerate() {
                function_ids[m][i] = function_ids[owner][local];
            }
        }
        let index = function_ids[m][local];
        Ok(FunctionRef {
            kernel: self.id,
            index,
            ty: self.function_types[index as usize - 1],
        })
    }
    pub fn external_reference<T: Any + Send + Sync>(&self, value: T) -> Result<ExternalRef, Error> {
        self.check()?;
        let mut values = self.externals.lock().unwrap();
        let index = lease_insert(&mut values, Arc::new(value) as Arc<dyn Any + Send + Sync>)?;
        Ok(ExternalRef {
            kernel: self.id,
            index: index as u64 + 1,
            generation: values[index].generation,
        })
    }
    pub fn external_reference_value<T: Any + Send + Sync>(
        &self,
        reference: ExternalRef,
    ) -> Result<Arc<T>, Error> {
        self.check()?;
        if reference.index >> 32 != 0 || !self.valid_external(reference, 0) {
            return Err(Error::Unsupported(
                "invalid external reference for this store".into(),
            ));
        }
        self.externals.lock().unwrap()[reference.index as usize - 1]
            .value
            .as_ref()
            .unwrap()
            .clone()
            .downcast::<T>()
            .map_err(|_| Error::Unsupported("external reference has the wrong host type".into()))
    }
    fn gc_record(&self, r: GcRef) -> Option<LiveGc> {
        if r.kernel != self.id || r.index >> 32 != 3 || r.index as u32 == 0 {
            return None;
        }
        self.gc_values
            .lock()
            .unwrap()
            .get(r.index as u32 as usize - 1)
            .filter(|v| v.valid(r.generation))
            .and_then(|v| v.value)
    }
    fn valid_external(&self, r: ExternalRef, lane: usize) -> bool {
        r.kernel == self.id
            && match r.index >> 32 {
                0 => self
                    .externals
                    .lock()
                    .unwrap()
                    .get((r.index as usize).wrapping_sub(1))
                    .is_some_and(|v| v.valid(r.generation)),
                1 => r.generation == 0 && r.index & 0x80000000 == 0,
                3 => self
                    .gc_values
                    .lock()
                    .unwrap()
                    .get((r.index as u32).wrapping_sub(1) as usize)
                    .is_some_and(|v| {
                        v.valid(r.generation) && v.value.as_ref().is_some_and(|r| r.lane == lane)
                    }),
                _ => false,
            }
    }
    fn exception_record(&self, r: ExceptionRef) -> Option<LiveException> {
        if r.kernel != self.id || r.index == 0 {
            return None;
        }
        self.exceptions
            .lock()
            .unwrap()
            .get(r.index as usize - 1)
            .filter(|v| v.valid(r.generation))
            .and_then(|v| v.value.clone())
    }
    // Internal graph edges remain valid after their public lease is released.
    // Extraction renews the current lease without changing guest identity.
    fn refresh_value(&self, mut value: Value, retain: bool) -> Result<Value, Error> {
        let invalid = || Error::Metal("missing retained reference graph node".into());
        match &mut value {
            Value::GcRef(Some(r)) if r.index >> 32 == 3 => {
                let mut values = self.gc_values.lock().unwrap();
                let v = values
                    .get_mut((r.index as u32).wrapping_sub(1) as usize)
                    .filter(|v| v.value.is_some())
                    .ok_or_else(invalid)?;
                v.retained |= retain;
                r.generation = v.generation;
            }
            Value::ExternRef(Some(r)) if r.index >> 32 == 3 => {
                let mut values = self.gc_values.lock().unwrap();
                let v = values
                    .get_mut((r.index as u32).wrapping_sub(1) as usize)
                    .filter(|v| v.value.is_some())
                    .ok_or_else(invalid)?;
                v.retained |= retain;
                r.generation = v.generation;
            }
            Value::GcRef(Some(r)) if r.index >> 32 == 2 => {
                let mut values = self.externals.lock().unwrap();
                let v = values
                    .get_mut((r.index as u32).wrapping_sub(1) as usize)
                    .filter(|v| v.value.is_some())
                    .ok_or_else(invalid)?;
                v.retained |= retain;
                r.generation = v.generation;
            }
            Value::ExternRef(Some(r)) if r.index >> 32 == 0 => {
                let mut values = self.externals.lock().unwrap();
                let v = values
                    .get_mut((r.index as usize).wrapping_sub(1))
                    .filter(|v| v.value.is_some())
                    .ok_or_else(invalid)?;
                v.retained |= retain;
                r.generation = v.generation;
            }
            Value::ExnRef(Some(r)) => {
                let mut values = self.exceptions.lock().unwrap();
                let v = values
                    .get_mut((r.index as usize).wrapping_sub(1))
                    .filter(|v| v.value.is_some())
                    .ok_or_else(invalid)?;
                v.retained |= retain;
                r.generation = v.generation;
            }
            _ => {}
        }
        Ok(value)
    }
    fn accepts(&self, value: Value, expected: ValType, lane: usize) -> bool {
        self.accepts_value(value, expected, lane, false)
    }
    fn accepts_value(&self, value: Value, expected: ValType, lane: usize, internal: bool) -> bool {
        // Public inputs must own an active lease. Parent payloads use their
        // internal identity, which is checked against the live registry instead.
        let value = if internal {
            match self.refresh_value(value, false) {
                Ok(v) => v,
                Err(_) => return false,
            }
        } else {
            value
        };
        match (value, expected) {
            (Value::FuncRef(None), ValType::Ref(t)) => {
                t.is_nullable() && is_function_ref_in(t, &self.reference_types)
            }
            (Value::ExternRef(None), ValType::Ref(t)) => t.is_nullable() && is_external_ref(t),
            (Value::GcRef(None), ValType::Ref(t)) => {
                t.is_nullable() && is_gc_ref(t, &self.reference_types)
            }
            (Value::ExnRef(None), ValType::Ref(t)) => t.is_nullable() && is_exception_ref(t),
            (Value::FuncRef(Some(r)), ValType::Ref(t)) => {
                r.kernel == self.id
                    && r.index > 0
                    && self.function_types.get(r.index as usize - 1) == Some(&r.ty)
                    && ref_subtype(r.ty, t, &self.reference_types)
            }
            (Value::ExternRef(Some(r)), ValType::Ref(t)) => {
                self.external_exists(r, lane, internal)
                    && ref_subtype(RefType::EXTERNREF.as_non_null(), t, &self.reference_types)
            }
            (Value::GcRef(Some(r)), ValType::Ref(t)) => {
                r.kernel == self.id
                    && ref_subtype(r.ty, t, &self.reference_types)
                    && if r.index >> 32 == 2 {
                        self.external_exists(
                            ExternalRef {
                                kernel: r.kernel,
                                index: r.index as u32 as u64,
                                generation: r.generation,
                            },
                            lane,
                            internal,
                        )
                    } else {
                        self.gc_values
                            .lock()
                            .unwrap()
                            .get((r.index as u32).wrapping_sub(1) as usize)
                            .is_some_and(|v| {
                                (internal || v.valid(r.generation))
                                    && v.value.is_some_and(|r| r.lane == lane)
                            })
                    }
            }
            (Value::ExnRef(Some(r)), ValType::Ref(t)) => {
                if r.kernel != self.id
                    || r.index == 0
                    || !ref_subtype(RefType::EXNREF.as_non_null(), t, &self.reference_types)
                {
                    return false;
                }
                let record = self
                    .exceptions
                    .lock()
                    .unwrap()
                    .get(r.index as usize - 1)
                    .filter(|v| internal || v.valid(r.generation))
                    .and_then(|v| v.value.clone());
                match record {
                    Some(LiveException::Lane { lane: owner, .. }) => lane == owner,
                    Some(LiveException::Host(record)) => record
                        .payload
                        .iter()
                        .zip(&self.tag_types[record.tag as usize])
                        .all(|(&v, &t)| self.accepts_value(v, t, lane, true)),
                    None => false,
                }
            }
            (Value::I31(_), ValType::Ref(t)) => {
                ref_subtype(RefType::I31REF.as_non_null(), t, &self.reference_types)
            }
            _ => value.ty() == expected,
        }
    }
    fn external_exists(&self, r: ExternalRef, lane: usize, internal: bool) -> bool {
        if !internal {
            return self.valid_external(r, lane);
        }
        r.kernel == self.id
            && match r.index >> 32 {
                0 => self
                    .externals
                    .lock()
                    .unwrap()
                    .get((r.index as usize).wrapping_sub(1))
                    .is_some_and(|v| v.value.is_some()),
                1 => r.index & 0x80000000 == 0,
                3 => self
                    .gc_values
                    .lock()
                    .unwrap()
                    .get((r.index as u32).wrapping_sub(1) as usize)
                    .is_some_and(|v| v.value.is_some_and(|r| r.lane == lane)),
                _ => false,
            }
    }
    fn encode(
        &self,
        value: Value,
        lane: usize,
        metadata: &mut [u64],
        host_slots: &mut HashMap<(usize, u32), usize>,
    ) -> Result<Vec<u64>, Error> {
        Ok(match value {
            Value::GcRef(Some(r)) if r.index >> 32 == 3 => vec![
                self.gc_values.lock().unwrap()[r.index as u32 as usize - 1]
                    .value
                    .unwrap()
                    .word,
            ],
            Value::ExternRef(Some(r)) if r.index >> 32 == 3 => vec![
                self.gc_values.lock().unwrap()[r.index as u32 as usize - 1]
                    .value
                    .unwrap()
                    .word,
            ],
            Value::ExnRef(Some(r)) => {
                let record = self.exceptions.lock().unwrap()[r.index as usize - 1]
                    .value
                    .clone()
                    .unwrap();
                let slot = match record {
                    LiveException::Lane { slot, .. } => slot,
                    LiveException::Host(record) => {
                        if let Some(&slot) = host_slots.get(&(lane, r.index)) {
                            slot
                        } else {
                            let (capacity, width) = self.metadata.exceptions.ok_or_else(|| {
                                Error::Unsupported(
                                    "module does not support exception references".into(),
                                )
                            })?;
                            let base = lane * self.metadata.stride() + self.metadata.fields.len();
                            let mut payload = Vec::new();
                            for value in record.payload {
                                payload.extend(self.encode(value, lane, metadata, host_slots)?);
                            }
                            let count = metadata[base] as usize;
                            let index = (0..count)
                                .find(|&i| metadata[base + 1 + i * (width + 1)] == 0)
                                .unwrap_or(count);
                            if index >= capacity {
                                return Err(Error::Trap {
                                    input_index: lane,
                                    reason: "exception record limit exceeded",
                                });
                            }
                            let start = base + 1 + index * (width + 1);
                            metadata[start..start + width + 1].fill(0);
                            metadata[start] = record.tag as u64;
                            metadata[start + 1..start + 1 + payload.len()]
                                .copy_from_slice(&payload);
                            metadata[base] = count.max(index + 1) as u64;
                            host_slots.insert((lane, r.index), index + 1);
                            index + 1
                        }
                    }
                };
                vec![slot as u64]
            }
            _ => value.words().collect(),
        })
    }
    fn retain_gc(&self, lane: usize, word: u64) -> Result<GcRef, Error> {
        let (heap, stride) = self
            .heap
            .as_ref()
            .ok_or_else(|| Error::Metal("missing persistent GC heap".into()))?;
        let offset = (word as u32)
            .checked_sub(1)
            .ok_or_else(|| Error::Metal("invalid GC reference".into()))?
            as usize;
        if offset + 3 > self.limits.gc_heap_words {
            return Err(Error::Metal("invalid GC reference".into()));
        }
        let ptr = unsafe { heap.contents().cast::<u64>().add(lane * stride + offset) };
        let type_id = unsafe { *ptr.add(1) } as usize;
        let ty = self
            .gc_types
            .get(type_id.wrapping_sub(1))
            .ok_or_else(|| Error::Metal("invalid GC object type".into()))?
            .reference;
        let mut objects = self.gc_values.lock().unwrap();
        let index = if let Some(i) = objects
            .iter()
            .position(|v| v.value.is_some_and(|r| r.lane == lane && r.word == word))
        {
            objects[i].retained = true;
            objects[i].value = Some(LiveGc { lane, word, ty });
            i
        } else {
            lease_insert(&mut objects, LiveGc { lane, word, ty })?
        };
        Ok(GcRef {
            kernel: self.id,
            index: 0x300000000 | (index as u64 + 1),
            ty,
            generation: objects[index].generation,
        })
    }
    fn decode(&self, ty: ValType, words: &[u64], lane: usize) -> Result<Value, Error> {
        let word = words[0];
        Ok(match ty {
            ValType::I32 => Value::I32(word as u32),
            ValType::I64 => Value::I64(word),
            ValType::F32 => Value::F32(word as u32),
            ValType::F64 => Value::F64(word),
            ValType::V128 => {
                let mut bytes = [0; 16];
                bytes[..8].copy_from_slice(&word.to_le_bytes());
                bytes[8..].copy_from_slice(&words[1].to_le_bytes());
                Value::V128(bytes)
            }
            ValType::Ref(t) if is_function_ref_in(t, &self.reference_types) => {
                Value::FuncRef(if word == 0 {
                    None
                } else {
                    Some(FunctionRef {
                        kernel: self.id,
                        index: word as u32,
                        ty: *self
                            .function_types
                            .get(word as usize - 1)
                            .ok_or_else(|| Error::Metal("invalid function reference".into()))?,
                    })
                })
            }
            ValType::Ref(t) if is_external_ref(t) => {
                if word == 0 {
                    Value::ExternRef(None)
                } else if word >> 32 == 3 {
                    let r = self.retain_gc(lane, word)?;
                    Value::ExternRef(Some(ExternalRef {
                        kernel: self.id,
                        index: r.index,
                        generation: r.generation,
                    }))
                } else {
                    self.refresh_value(
                        Value::ExternRef(Some(ExternalRef {
                            kernel: self.id,
                            index: word,
                            generation: 0,
                        })),
                        true,
                    )?
                }
            }
            ValType::Ref(t) if is_exception_ref(t) => {
                if word == 0 {
                    Value::ExnRef(None)
                } else {
                    let mapped =
                        self.host_exception_slots
                            .iter()
                            .find_map(|(&(l, index), &slot)| {
                                (l == lane && slot == word as usize).then_some(index as usize - 1)
                            });
                    let mut records = self.exceptions.lock().unwrap();
                    let index=if let Some(i)=mapped {i} else if let Some(i)=records.iter().position(|v|matches!(v.value,Some(LiveException::Lane{lane:l,slot}) if l==lane&&slot==word as usize)) {i} else {lease_insert(&mut records,LiveException::Lane{lane,slot:word as usize})?};
                    records[index].retained = true;
                    Value::ExnRef(Some(ExceptionRef {
                        kernel: self.id,
                        index: index as u32 + 1,
                        generation: records[index].generation,
                    }))
                }
            }
            ValType::Ref(_) => match word >> 32 {
                _ if word == 0 => Value::GcRef(None),
                1 => Value::I31(word as u32 & 0x7fffffff),
                2 => self.refresh_value(
                    Value::GcRef(Some(GcRef {
                        kernel: self.id,
                        index: word,
                        ty: RefType::ANYREF.as_non_null(),
                        generation: 0,
                    })),
                    true,
                )?,
                3 => Value::GcRef(Some(self.retain_gc(lane, word)?)),
                _ => return Err(Error::Metal("invalid GC reference tag".into())),
            },
        })
    }
    pub fn gc_reference_kind(&self, reference: GcRef) -> Result<AbstractHeapType, Error> {
        self.check()?;
        if reference.index >> 32 == 2
            && self.valid_external(
                ExternalRef {
                    kernel: reference.kernel,
                    index: reference.index as u32 as u64,
                    generation: reference.generation,
                },
                0,
            )
        {
            return Ok(AbstractHeapType::Any);
        }
        let record = self
            .gc_record(reference)
            .ok_or_else(|| Error::Unsupported("invalid GC reference for this store".into()))?;
        self.reference_types
            .get(&match record.ty.heap_type() {
                HeapType::Concrete(UnpackedIndex::Id(id)) => id,
                _ => return Err(Error::Metal("invalid GC reference type".into())),
            })
            .map(|(_, kind)| *kind)
            .ok_or_else(|| Error::Metal("unknown GC reference type".into()))
    }
    pub fn internalize_reference(&self, reference: Option<ExternalRef>) -> Result<Value, Error> {
        self.check()?;
        let Some(r) = reference else {
            return Ok(Value::GcRef(None));
        };
        if r.kernel != self.id {
            return Err(Error::Unsupported(
                "invalid external reference for this store".into(),
            ));
        }
        Ok(match r.index >> 32 {
            0 if self.valid_external(r, 0) => Value::GcRef(Some(GcRef {
                kernel: self.id,
                index: 0x200000000 | r.index,
                ty: RefType::ANYREF.as_non_null(),
                generation: r.generation,
            })),
            1 if self.valid_external(r, 0) => Value::I31(r.index as u32),
            3 => {
                let values = self.gc_values.lock().unwrap();
                let value = values
                    .get((r.index as u32).wrapping_sub(1) as usize)
                    .filter(|v| v.valid(r.generation))
                    .and_then(|v| v.value)
                    .ok_or_else(|| {
                        Error::Unsupported("invalid external reference for this store".into())
                    })?;
                Value::GcRef(Some(GcRef {
                    kernel: self.id,
                    index: r.index,
                    ty: value.ty,
                    generation: r.generation,
                }))
            }
            _ => {
                return Err(Error::Unsupported(
                    "invalid external reference for this store".into(),
                ));
            }
        })
    }
    pub fn externalize_reference(&self, value: Value) -> Result<Option<ExternalRef>, Error> {
        self.check()?;
        let (index, generation) = match value {
            Value::GcRef(None) => return Ok(None),
            Value::I31(v) => (0x100000000 | u64::from(v & 0x7fffffff), 0),
            Value::GcRef(Some(r)) if r.kernel == self.id => match r.index >> 32 {
                2 if self.valid_external(
                    ExternalRef {
                        kernel: r.kernel,
                        index: r.index as u32 as u64,
                        generation: r.generation,
                    },
                    0,
                ) =>
                {
                    (r.index as u32 as u64, r.generation)
                }
                3 if self.gc_record(r).is_some() => (r.index, r.generation),
                _ => {
                    return Err(Error::Unsupported(
                        "invalid GC reference for this store".into(),
                    ));
                }
            },
            _ => {
                return Err(Error::Unsupported(
                    "expected a GC reference from this store".into(),
                ));
            }
        };
        Ok(Some(ExternalRef {
            kernel: self.id,
            index,
            generation,
        }))
    }
    pub fn exception_reference(
        &self,
        instance: Instance,
        export: &str,
        payload: &[Value],
    ) -> Result<ExceptionRef, Error> {
        let (m, local) = self.export(instance, export, ExternalKind::Tag)?;
        let tag = self.modules[m].tags[local];
        let types = &self.tag_types[tag as usize];
        if payload.len() != types.len()
            || payload
                .iter()
                .zip(types)
                .any(|(&v, &t)| !(0..self.lanes).any(|lane| self.accepts(v, t, lane)))
        {
            return Err(Error::Unsupported(
                "exception payload does not match tag signature".into(),
            ));
        }
        let mut records = self.exceptions.lock().unwrap();
        let index = lease_insert(
            &mut records,
            LiveException::Host(ExceptionRecord {
                tag,
                payload: payload.to_vec(),
            }),
        )?;
        Ok(ExceptionRef {
            kernel: self.id,
            index: index as u32 + 1,
            generation: records[index].generation,
        })
    }
    pub fn exception_payload(&self, reference: ExceptionRef) -> Result<Vec<Value>, Error> {
        self.check()?;
        let record = self.exception_record(reference).ok_or_else(|| {
            Error::Unsupported("invalid exception reference for this store".into())
        })?;
        match record {
            LiveException::Host(record) => record
                .payload
                .into_iter()
                .map(|v| self.refresh_value(v, true))
                .collect(),
            LiveException::Lane { lane, slot } => {
                let (_, width) = self.metadata.exceptions.unwrap();
                let offset = lane * self.metadata.stride()
                    + self.metadata.fields.len()
                    + 1
                    + (slot - 1) * (width + 1);
                let words = self.metadata_words();
                let types = &self.tag_types[words[offset] as usize];
                let mut result = Vec::new();
                let mut word = offset + 1;
                for &ty in types {
                    result.push(self.decode(ty, &words[word..], lane)?);
                    word += value_words(ty);
                }
                Ok(result)
            }
        }
    }
    /// Release the shared lease for a reference and all its copied or converted
    /// aliases. Wasm state and other retained graphs can still keep it alive.
    /// Null references and i31 values require no release. Scalars and non-null
    /// functions are rejected. Releasing a stale handle is an error.
    pub fn release_reference(&mut self, value: Value) -> Result<(), Error> {
        self.check()?;
        let invalid = || Error::Unsupported("invalid or released reference for this store".into());
        match value {
            Value::GcRef(None)
            | Value::ExternRef(None)
            | Value::ExnRef(None)
            | Value::FuncRef(None)
            | Value::I31(_) => return Ok(()),
            Value::GcRef(Some(r)) if r.kernel == self.id && r.index >> 32 == 3 => self
                .gc_values
                .lock()
                .unwrap()
                .get_mut((r.index as u32).wrapping_sub(1) as usize)
                .ok_or_else(invalid)?
                .release(r.generation)?,
            Value::GcRef(Some(r)) if r.kernel == self.id && r.index >> 32 == 2 => self
                .externals
                .lock()
                .unwrap()
                .get_mut((r.index as u32).wrapping_sub(1) as usize)
                .ok_or_else(invalid)?
                .release(r.generation)?,
            Value::ExternRef(Some(r)) if r.kernel == self.id => match r.index >> 32 {
                0 => self
                    .externals
                    .lock()
                    .unwrap()
                    .get_mut((r.index as usize).wrapping_sub(1))
                    .ok_or_else(invalid)?
                    .release(r.generation)?,
                1 if self.valid_external(r, 0) => return Ok(()),
                3 => self
                    .gc_values
                    .lock()
                    .unwrap()
                    .get_mut((r.index as u32).wrapping_sub(1) as usize)
                    .ok_or_else(invalid)?
                    .release(r.generation)?,
                _ => return Err(invalid()),
            },
            Value::ExnRef(Some(r)) if r.kernel == self.id => self
                .exceptions
                .lock()
                .unwrap()
                .get_mut((r.index as usize).wrapping_sub(1))
                .ok_or_else(invalid)?
                .release(r.generation)?,
            _ => return Err(invalid()),
        }
        self.collect_garbage()
    }
    fn trace_references(
        &self,
        state: bool,
        metadata: &[u64],
        host_slots: &HashMap<(usize, u32), usize>,
    ) -> Result<Reachable, Error> {
        let mut reached = Reachable::default();
        let mut pending = Vec::new();
        for lease in self.gc_values.lock().unwrap().iter().filter(|v| v.retained) {
            if let Some(r) = lease.value {
                pending.push(TraceReference::Guest(r.lane, r.ty, r.word));
            }
        }
        for (i, lease) in self
            .externals
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, v)| v.retained && v.value.is_some())
        {
            let _ = lease;
            reached.externals.insert(i);
        }
        for (i, lease) in self
            .exceptions
            .lock()
            .unwrap()
            .iter()
            .enumerate()
            .filter(|(_, v)| v.retained && v.value.is_some())
        {
            pending.push(TraceReference::Host(Value::ExnRef(Some(ExceptionRef {
                kernel: self.id,
                index: i as u32 + 1,
                generation: lease.generation,
            }))));
        }
        if state {
            for &(m, slot, ty) in &self.metadata.global_roots {
                let r = self.resource(m, "globals");
                let bytes = self.bytes(r);
                for lane in 0..self.lanes {
                    let offset = lane * r.size + slot * 8;
                    pending.push(TraceReference::Guest(
                        lane,
                        ty,
                        u64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap()),
                    ));
                }
            }
            for &(m, table, ty) in &self.metadata.table_roots {
                let r = self.resource(m, &format!("table{table}"));
                let bytes = self.bytes(r);
                for lane in 0..self.lanes {
                    for i in 0..self.table_size(m, table, lane) {
                        let offset = lane * r.size + i * 8;
                        pending.push(TraceReference::Guest(
                            lane,
                            ty,
                            u64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap()),
                        ));
                    }
                }
            }
            for &(start, length, capacity, ty) in &self.metadata.element_roots {
                for lane in 0..self.lanes {
                    let base = lane * self.metadata.stride();
                    for i in 0..(metadata[base + length] as usize).min(capacity) {
                        pending.push(TraceReference::Guest(lane, ty, metadata[base + start + i]));
                    }
                }
            }
        }
        while let Some(reference) = pending.pop() {
            match reference {
                TraceReference::Host(value) => match value {
                    Value::ExnRef(Some(r)) => {
                        let index = r.index as usize - 1;
                        if !reached.host_exceptions.insert(index) {
                            continue;
                        }
                        let record = self
                            .exceptions
                            .lock()
                            .unwrap()
                            .get(index)
                            .and_then(|v| v.value.clone())
                            .ok_or_else(|| Error::Metal("missing exception graph node".into()))?;
                        match record {
                            LiveException::Lane { lane, slot } => pending
                                .push(TraceReference::Guest(lane, RefType::EXNREF, slot as u64)),
                            LiveException::Host(record) => {
                                pending
                                    .extend(record.payload.into_iter().map(TraceReference::Host));
                                for (&(lane, id), &slot) in host_slots {
                                    if id == r.index {
                                        pending.push(TraceReference::Guest(
                                            lane,
                                            RefType::EXNREF,
                                            slot as u64,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    Value::GcRef(Some(r)) if r.index >> 32 == 3 => {
                        let object = self
                            .gc_values
                            .lock()
                            .unwrap()
                            .get(r.index as u32 as usize - 1)
                            .and_then(|v| v.value)
                            .ok_or_else(|| Error::Metal("missing GC graph node".into()))?;
                        pending.push(TraceReference::Guest(object.lane, object.ty, object.word));
                    }
                    Value::ExternRef(Some(r)) if r.index >> 32 == 3 => {
                        let object = self
                            .gc_values
                            .lock()
                            .unwrap()
                            .get(r.index as u32 as usize - 1)
                            .and_then(|v| v.value)
                            .ok_or_else(|| Error::Metal("missing GC graph node".into()))?;
                        pending.push(TraceReference::Guest(object.lane, object.ty, object.word));
                    }
                    Value::GcRef(Some(r)) if r.index >> 32 == 2 => {
                        reached.externals.insert(r.index as u32 as usize - 1);
                    }
                    Value::ExternRef(Some(r)) if r.index >> 32 == 0 => {
                        reached.externals.insert(r.index as usize - 1);
                    }
                    _ => {}
                },
                TraceReference::Guest(lane, ty, word) => {
                    if word == 0 || is_function_ref_in(ty, &self.reference_types) {
                        continue;
                    }
                    if is_exception_ref(ty) {
                        let slot = word as usize;
                        if !reached.exceptions.insert((lane, slot)) {
                            continue;
                        }
                        let (capacity, width) = self
                            .metadata
                            .exceptions
                            .ok_or_else(|| Error::Metal("missing exception arena".into()))?;
                        let base = lane * self.metadata.stride() + self.metadata.fields.len();
                        if slot == 0 || slot > capacity || slot > metadata[base] as usize {
                            return Err(Error::Metal("invalid live exception slot".into()));
                        }
                        let offset = base + 1 + (slot - 1) * (width + 1);
                        let tag = metadata[offset] as usize;
                        let types = self
                            .tag_types
                            .get(tag)
                            .filter(|_| tag != 0)
                            .ok_or_else(|| Error::Metal("invalid live exception tag".into()))?;
                        let mut offset = offset + 1;
                        for &ty in types {
                            if let ValType::Ref(r) = ty {
                                pending.push(TraceReference::Guest(lane, r, metadata[offset]));
                            }
                            offset += value_words(ty);
                        }
                        for (&(l, id), &mapped) in host_slots {
                            if l == lane && mapped == slot {
                                pending.push(TraceReference::Host(Value::ExnRef(Some(
                                    ExceptionRef {
                                        kernel: self.id,
                                        index: id,
                                        generation: 0,
                                    },
                                ))));
                            }
                        }
                    } else if word >> 32 == 3 {
                        if !reached.gc.insert((lane, word)) {
                            continue;
                        }
                        let (heap, stride) = self
                            .heap
                            .as_ref()
                            .ok_or_else(|| Error::Metal("missing GC heap".into()))?;
                        let offset = (word as u32)
                            .checked_sub(1)
                            .ok_or_else(|| Error::Metal("invalid GC graph offset".into()))?
                            as usize;
                        if offset + 3 > self.limits.gc_heap_words {
                            return Err(Error::Metal("invalid GC graph offset".into()));
                        }
                        let words = unsafe {
                            std::slice::from_raw_parts(
                                heap.contents().cast::<u64>().add(lane * stride),
                                self.limits.gc_heap_words,
                            )
                        };
                        let size = words[offset] as u32 as usize;
                        if words[offset] & (1 << 63) == 0 || size < 3 || size > words.len() - offset
                        {
                            return Err(Error::Metal("invalid GC graph block".into()));
                        }
                        let ty = self
                            .gc_types
                            .get((words[offset + 1] as usize).wrapping_sub(1))
                            .ok_or_else(|| Error::Metal("invalid GC graph type".into()))?;
                        let length = words[offset + 2] as usize;
                        let fields = if ty.array { length } else { ty.fields.len() };
                        let mut at = offset + 3;
                        for index in 0..fields {
                            let field = ty.fields[if ty.array { 0 } else { index }];
                            if at + storage_words(&field) > offset + size {
                                return Err(Error::Metal("invalid GC graph size".into()));
                            }
                            if let StorageType::Val(ValType::Ref(r)) = field {
                                pending.push(TraceReference::Guest(lane, r, words[at]));
                            }
                            at += storage_words(&field);
                        }
                    } else if (is_external_ref(ty) && word >> 32 == 0) || word >> 32 == 2 {
                        reached.externals.insert(word as u32 as usize - 1);
                    }
                }
            }
        }
        Ok(reached)
    }
    /// Reclaim unreachable lane objects, exception slots, and host values.
    /// Collection runs between calls, never invokes Wasm, and preserves state.
    pub fn collect_garbage(&mut self) -> Result<(), Error> {
        self.check()?;
        let metadata = if self.metadata_buffer.is_some() {
            self.metadata_words()
        } else {
            &[]
        };
        let reached = self.trace_references(true, metadata, &self.host_exception_slots)?;
        if let Some((heap, stride)) = &self.heap {
            for lane in 0..self.lanes {
                let words = unsafe {
                    std::slice::from_raw_parts_mut(
                        heap.contents().cast::<u64>().add(lane * stride),
                        self.limits.gc_heap_words,
                    )
                };
                let mut offset = 0;
                let mut previous_free = None;
                while offset < words.len() {
                    let size = words[offset] as u32 as usize;
                    if size < 3 || size > words.len() - offset {
                        self.poisoned = true;
                        return Err(Error::Metal("invalid persistent GC heap".into()));
                    }
                    if reached
                        .gc
                        .contains(&(lane, 0x300000000 | (offset as u64 + 1)))
                    {
                        words[offset] = size as u64 | (1 << 63);
                        previous_free = None;
                    } else {
                        words[offset] = size as u64;
                        if let Some(prior) = previous_free {
                            words[prior] += size as u64;
                        } else {
                            previous_free = Some(offset);
                        }
                    }
                    offset += size;
                }
            }
        }
        if let Some((_, width)) = self.metadata.exceptions {
            let words = unsafe {
                std::slice::from_raw_parts_mut(
                    self.metadata_buffer
                        .as_ref()
                        .unwrap()
                        .contents()
                        .cast::<u64>(),
                    self.metadata.stride() * self.lanes,
                )
            };
            for lane in 0..self.lanes {
                let base = lane * self.metadata.stride() + self.metadata.fields.len();
                for slot in 1..=words[base] as usize {
                    if !reached.exceptions.contains(&(lane, slot)) {
                        let offset = base + 1 + (slot - 1) * (width + 1);
                        words[offset..offset + width + 1].fill(0);
                    }
                }
            }
        }
        for lease in self.gc_values.lock().unwrap().iter_mut() {
            if lease
                .value
                .is_some_and(|r| !reached.gc.contains(&(r.lane, r.word)))
            {
                debug_assert!(!lease.retained);
                lease.value = None;
            }
        }
        for (index, lease) in self.exceptions.lock().unwrap().iter_mut().enumerate() {
            let live = match &lease.value {
                Some(LiveException::Lane { lane, slot }) => {
                    reached.exceptions.contains(&(*lane, *slot))
                }
                Some(LiveException::Host(_)) => reached.host_exceptions.contains(&index),
                None => false,
            };
            if !live {
                debug_assert!(!lease.retained);
                lease.value = None;
            }
        }
        for (index, lease) in self.externals.lock().unwrap().iter_mut().enumerate() {
            if !reached.externals.contains(&index) {
                debug_assert!(!lease.retained);
                lease.value = None;
            }
        }
        self.host_exception_slots.retain(|&(lane, id), slot| {
            reached.exceptions.contains(&(lane, *slot))
                && reached.host_exceptions.contains(&(id as usize - 1))
        });
        Ok(())
    }
}

fn words_bytes(words: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(words.as_ptr().cast(), std::mem::size_of_val(words)) }
}
fn link_module(
    index: usize,
    modules: &mut [ParsedModule<'_>],
    imports: &HashMap<String, usize>,
) -> Result<(), Error> {
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
        let provider = *imports
            .get(&namespace)
            .ok_or_else(|| Error::Unsupported(format!("missing Wasm provider {namespace:?}")))?;
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
    Ok(())
}
