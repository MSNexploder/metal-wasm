use super::*;
use wasmparser::StorageType;

const I31_TAG: u64 = 0x100000000;
const FOREIGN_TAG: u64 = 0x200000000;
const OBJECT_TAG: u64 = 0x300000000;
const ALLOCATED: u64 = 1 << 63;

/// A GC reference owned by one kernel or store.
/// Kernel invocations copy the retained graph, preserving aliases and cycles.
/// Store references retain the live object and belong to the originating lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GcRef {
    pub(super) generation: u64,
    pub(super) kernel: u64,
    pub(super) index: u64,
    pub(super) ty: RefType,
}

pub(super) struct GcRecord {
    pub(super) type_index: usize,
    pub(super) values: Vec<Value>,
}

pub(super) struct EncodedGcRow {
    pub(super) arguments: Vec<u64>,
    pub(super) heap: Vec<u64>,
    pub(super) exceptions: Vec<u64>,
    pub(super) existing_exceptions: Vec<ExceptionRef>,
}

fn invalid(message: &str) -> Error {
    Error::Metal(message.into())
}

fn nonnull(ty: AbstractHeapType) -> RefType {
    RefType::new(false, HeapType::Abstract { shared: false, ty }).unwrap()
}

fn object_id(value: Value) -> Option<u32> {
    let word = match value {
        Value::GcRef(Some(reference)) => reference.index,
        Value::ExternRef(Some(reference)) => reference.index,
        _ => return None,
    };
    (word >> 32 == 3).then_some(word as u32)
}

impl Kernel {
    pub(super) fn validate_gc_reference(&self, reference: GcRef) -> bool {
        if reference.kernel != self.id || reference.index as u32 == 0 {
            return false;
        }
        let index = reference.index as u32 as usize - 1;
        match reference.index >> 32 {
            2 => self
                .external_values
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(index)
                .is_some_and(|v| v.valid(reference.generation)),
            3 => self
                .gc_values
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(index)
                .is_some_and(|v| v.valid(reference.generation)),
            _ => false,
        }
    }

    pub(super) fn validate_external_reference(&self, reference: ExternalRef) -> bool {
        if reference.kernel != self.id {
            return false;
        }
        match reference.index >> 32 {
            0 if reference.index != 0 => self
                .external_values
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(reference.index as usize - 1)
                .is_some_and(|v| v.valid(reference.generation)),
            1 => reference.index & 0x80000000 == 0,
            3 if reference.index as u32 != 0 => self
                .gc_values
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(reference.index as u32 as usize - 1)
                .is_some_and(|v| v.valid(reference.generation)),
            _ => false,
        }
    }

    /// Return the runtime kind of a retained object or internalized host value.
    pub fn gc_reference_kind(&self, reference: GcRef) -> Result<AbstractHeapType, Error> {
        if !self.validate_gc_reference(reference) {
            return Err(Error::Unsupported(
                "invalid GC reference for this kernel".into(),
            ));
        }
        if reference.index >> 32 == 2 {
            return Ok(AbstractHeapType::Any);
        }
        let records = self.gc_values.lock().unwrap_or_else(|e| e.into_inner());
        Ok(
            if self.gc_types[records[reference.index as u32 as usize - 1]
                .value
                .as_ref()
                .unwrap()
                .type_index]
                .array
            {
                AbstractHeapType::Array
            } else {
                AbstractHeapType::Struct
            },
        )
    }

    /// Apply `any.convert_extern` on the host, preserving reference identity.
    pub fn internalize_reference(&self, reference: Option<ExternalRef>) -> Result<Value, Error> {
        let Some(reference) = reference else {
            return Ok(Value::GcRef(None));
        };
        if !self.validate_external_reference(reference) {
            return Err(Error::Unsupported(
                "invalid external reference for this kernel".into(),
            ));
        }
        Ok(match reference.index >> 32 {
            0 => Value::GcRef(Some(GcRef {
                generation: reference.generation,
                kernel: self.id,
                index: FOREIGN_TAG | reference.index,
                ty: nonnull(AbstractHeapType::Any),
            })),
            1 => Value::I31(reference.index as u32),
            3 => {
                let records = self.gc_values.lock().unwrap_or_else(|e| e.into_inner());
                let record = records[reference.index as u32 as usize - 1]
                    .value
                    .as_ref()
                    .unwrap();
                Value::GcRef(Some(GcRef {
                    generation: reference.generation,
                    kernel: self.id,
                    index: reference.index,
                    ty: self.gc_types[record.type_index].reference,
                }))
            }
            _ => unreachable!(),
        })
    }

    /// Apply `extern.convert_any` on the host. Returned handles may be passed
    /// back into this kernel; only original host values support host downcasts.
    pub fn externalize_reference(&self, value: Value) -> Result<Option<ExternalRef>, Error> {
        let generation = match value {
            Value::GcRef(Some(r)) => r.generation,
            _ => 0,
        };
        let index = match value {
            Value::GcRef(None) => return Ok(None),
            Value::I31(bits) => I31_TAG | u64::from(bits & 0x7fffffff),
            Value::GcRef(Some(reference)) if self.validate_gc_reference(reference) => {
                if reference.index >> 32 == 2 {
                    u64::from(reference.index as u32)
                } else {
                    reference.index
                }
            }
            _ => {
                return Err(Error::Unsupported(
                    "expected a GC reference from this kernel".into(),
                ));
            }
        };
        Ok(Some(ExternalRef {
            generation,
            kernel: self.id,
            index,
        }))
    }

    pub(super) fn encode_gc_row(
        &self,
        row: &[Value],
        io: Option<ExceptionIo>,
    ) -> Result<EncodedGcRow, Error> {
        let layout = self
            .gc
            .as_ref()
            .ok_or_else(|| invalid("missing GC heap layout"))?;
        // Both codecs acquire these locks in this order. No retained object is
        // mutated; a row imports a private copy of the complete reachable graph.
        let objects = self.gc_values.lock().unwrap_or_else(|e| e.into_inner());
        let exceptions = self
            .exception_values
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut object_ids = Vec::new();
        let mut object_map = HashMap::new();
        let mut exception_ids = Vec::new();
        let mut exception_map = HashMap::new();
        let mut pending = row.to_vec();
        let mut used = 0usize;
        while let Some(value) = pending.pop() {
            if let Some(id) = object_id(value) {
                if object_map.contains_key(&id) {
                    continue;
                }
                let record = id
                    .checked_sub(1)
                    .and_then(|n| objects.get(n as usize))
                    .and_then(|v| v.value.as_ref())
                    .ok_or_else(|| invalid("unknown GC input reference"))?;
                let words = record
                    .values
                    .iter()
                    .map(|v| value_words(v.ty()))
                    .sum::<usize>()
                    + 3;
                let end = used
                    .checked_add(words)
                    .filter(|&end| end <= layout.heap_words)
                    .ok_or(Error::Trap {
                        input_index: 0,
                        reason: "GC heap limit exceeded",
                    })?;
                object_map.insert(id, OBJECT_TAG | (used as u64 + 1));
                object_ids.push(id);
                used = end;
                pending.extend_from_slice(&record.values);
            } else if let Value::ExnRef(Some(reference)) = value {
                if exception_map.contains_key(&reference.index) {
                    continue;
                }
                let record = reference
                    .index
                    .checked_sub(1)
                    .and_then(|n| exceptions.get(n as usize))
                    .and_then(|v| v.value.as_ref())
                    .ok_or_else(|| invalid("unknown exception input reference"))?;
                let io = io.ok_or_else(|| invalid("missing exception transfer layout"))?;
                if exception_ids.len() >= io.records {
                    return Err(Error::Trap {
                        input_index: 0,
                        reason: "exception record limit exceeded",
                    });
                }
                exception_ids.push(reference.index);
                exception_map.insert(reference.index, exception_ids.len() as u64);
                pending.extend_from_slice(&record.payload);
            }
        }
        let encode = |value: Value| -> Vec<u64> {
            if let Some(id) = object_id(value) {
                vec![object_map[&id]]
            } else if let Value::ExnRef(Some(reference)) = value {
                vec![exception_map[&reference.index]]
            } else {
                value.words().collect()
            }
        };
        let mut heap = vec![0; layout.heap_words];
        for &id in &object_ids {
            let record = objects[id as usize - 1].value.as_ref().unwrap();
            let offset = object_map[&id] as u32 as usize - 1;
            let values: Vec<_> = record.values.iter().copied().flat_map(&encode).collect();
            let words = values.len() + 3;
            heap[offset] = ALLOCATED | words as u64;
            heap[offset + 1] = record.type_index as u64 + 1;
            heap[offset + 2] = record.values.len() as u64;
            heap[offset + 3..offset + words].copy_from_slice(&values);
        }
        if used < layout.heap_words {
            let remaining = layout.heap_words - used;
            if remaining < 3 && !object_ids.is_empty() {
                let last = object_map[object_ids.last().unwrap()] as u32 as usize - 1;
                heap[last] += remaining as u64;
            } else {
                heap[used] = remaining as u64;
            }
        }
        let mut arena = if let Some(io) = io {
            vec![0; io.words()?]
        } else {
            Vec::new()
        };
        for (slot, &id) in exception_ids.iter().enumerate() {
            let io = io.unwrap();
            let record = exceptions[id as usize - 1].value.as_ref().unwrap();
            let offset = 1 + slot * (io.payload_words + 1);
            arena[offset] = u64::from(record.tag);
            let payload: Vec<_> = record.payload.iter().copied().flat_map(&encode).collect();
            arena[offset + 1..offset + 1 + payload.len()].copy_from_slice(&payload);
        }
        if !arena.is_empty() {
            arena[0] = exception_ids.len() as u64;
        }
        Ok(EncodedGcRow {
            arguments: row.iter().copied().flat_map(encode).collect(),
            heap,
            exceptions: arena,
            existing_exceptions: exception_ids
                .into_iter()
                .map(|index| ExceptionRef {
                    generation: exceptions[index as usize - 1].generation,
                    kernel: self.id,
                    index,
                })
                .collect(),
        })
    }

    fn gc_value_type(&self, ty: RefType) -> bool {
        match ty.heap_type() {
            HeapType::Abstract { shared: false, ty } => matches!(
                ty,
                AbstractHeapType::Any
                    | AbstractHeapType::Eq
                    | AbstractHeapType::I31
                    | AbstractHeapType::Struct
                    | AbstractHeapType::Array
                    | AbstractHeapType::None
            ),
            _ => self
                .gc_types
                .iter()
                .any(|gc| gc.reference.heap_type() == ty.heap_type()),
        }
    }

    pub(super) fn decode_gc_row(
        &self,
        results: &[u64],
        heap: &[u64],
        arena: &[u64],
        io: Option<ExceptionIo>,
        existing: &[ExceptionRef],
    ) -> Result<Vec<Value>, Error> {
        // First validate the reachable graph. Register every node before its
        // children so both object cycles and object/exception cycles terminate.
        let mut objects: HashMap<u32, (usize, Vec<(ValType, Vec<u64>)>)> = HashMap::new();
        let mut exceptions: HashMap<u32, (u32, Vec<(ValType, Vec<u64>)>)> = HashMap::new();
        let split = |types: &[ValType], words: &[u64]| -> Result<Vec<(ValType, Vec<u64>)>, Error> {
            let mut offset = 0;
            let mut values = Vec::new();
            for &ty in types {
                let end = offset + value_words(ty);
                values.push((
                    ty,
                    words
                        .get(offset..end)
                        .ok_or_else(|| invalid("truncated GC transfer value"))?
                        .to_vec(),
                ));
                offset = end;
            }
            Ok(values)
        };
        let output = split(&self.results, results)?;
        let mut pending = output.clone();
        let mut boundaries = HashSet::new();
        let mut offset = 0;
        while offset < heap.len() {
            let words = heap[offset] as u32 as usize;
            if words < 3 || words > heap.len() - offset {
                return Err(invalid("invalid GC heap block"));
            }
            if heap[offset] & ALLOCATED != 0 {
                boundaries.insert(offset);
            }
            offset += words;
        }
        while let Some((ty, words)) = pending.pop() {
            let ValType::Ref(ty) = ty else {
                continue;
            };
            let word = words[0];
            if word == 0 {
                continue;
            }
            if (self.gc_value_type(ty) || is_external_ref(ty)) && word >> 32 == 3 {
                let id = word as u32;
                if objects.contains_key(&id) {
                    continue;
                }
                let offset = id
                    .checked_sub(1)
                    .map(|n| n as usize)
                    .filter(|n| boundaries.contains(n))
                    .ok_or_else(|| invalid("invalid GC object reference"))?;
                let type_index = heap[offset + 1]
                    .checked_sub(1)
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n < self.gc_types.len())
                    .ok_or_else(|| invalid("invalid GC object type"))?;
                let gc = &self.gc_types[type_index];
                let count = usize::try_from(heap[offset + 2])
                    .map_err(|_| invalid("invalid GC object length"))?;
                let block_end = offset + heap[offset] as u32 as usize;
                let fields = if gc.array {
                    let field = gc
                        .fields
                        .first()
                        .ok_or_else(|| invalid("missing GC array element type"))?
                        .unpack();
                    if count > (block_end - offset - 3) / value_words(field) {
                        return Err(invalid("invalid GC array length"));
                    }
                    vec![field; count]
                } else {
                    if count != gc.fields.len() {
                        return Err(invalid("invalid GC struct field count"));
                    }
                    gc.fields.iter().map(StorageType::unpack).collect()
                };
                let values = split(&fields, &heap[offset + 3..block_end])?;
                pending.extend(values.clone());
                objects.insert(id, (type_index, values));
            } else if is_exception_ref(ty) {
                let id = u32::try_from(word).map_err(|_| invalid("invalid exception reference"))?;
                if exceptions.contains_key(&id) {
                    continue;
                }
                let io = io.ok_or_else(|| invalid("missing exception transfer layout"))?;
                if arena.len() != io.words()? || arena[0] > io.records as u64 || word > arena[0] {
                    return Err(invalid("invalid exception transfer count"));
                }
                let offset = 1 + (id as usize - 1) * (io.payload_words + 1);
                let tag = u32::try_from(arena[offset])
                    .ok()
                    .filter(|&tag| tag != 0)
                    .ok_or_else(|| invalid("invalid exception tag"))?;
                let types = self
                    .tag_payload_types
                    .get(tag as usize)
                    .ok_or_else(|| invalid("invalid exception tag"))?;
                let values = split(types, &arena[offset + 1..offset + 1 + io.payload_words])?;
                pending.extend(values.clone());
                exceptions.insert(id, (tag, values));
            }
        }
        // Imported immutable exception identities remain valid unless their
        // payload reaches a mutable object, whose returned state is a snapshot.
        let mut copy_exceptions = HashSet::new();
        loop {
            let before = copy_exceptions.len();
            for (&id, (_, values)) in &exceptions {
                if id as usize > existing.len()
                    || values.iter().any(|(ty, words)| {
                        matches!(ty, ValType::Ref(ty) if
                        ((self.gc_value_type(*ty) || is_external_ref(*ty)) && words[0] >> 32 == 3)
                        || (is_exception_ref(*ty) && copy_exceptions.contains(&(words[0] as u32))))
                    })
                {
                    copy_exceptions.insert(id);
                }
            }
            if copy_exceptions.len() == before {
                break;
            }
        }
        let mut retained_objects = self.gc_values.lock().unwrap_or_else(|e| e.into_inner());
        let mut retained_exceptions = self
            .exception_values
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut object_ids: Vec<_> = objects.keys().copied().collect();
        object_ids.sort_unstable();
        let mut exception_ids: Vec<_> = exceptions.keys().copied().collect();
        exception_ids.sort_unstable();
        let object_slots = lease_slots(&retained_objects, object_ids.len())?;
        let exception_slots = lease_slots(&retained_exceptions, copy_exceptions.len())?;
        let object_map: HashMap<_, _> = object_ids
            .iter()
            .zip(&object_slots)
            .map(|(&id, &slot)| {
                (
                    id,
                    GcRef {
                        generation: retained_objects.get(slot).map_or(1, |v| v.generation),
                        kernel: self.id,
                        index: OBJECT_TAG | (slot as u64 + 1),
                        ty: self.gc_types[objects[&id].0].reference,
                    },
                )
            })
            .collect();
        let mut slots = exception_slots.iter();
        let exception_map: HashMap<_, _> = exception_ids
            .iter()
            .map(|&id| {
                let reference = if copy_exceptions.contains(&id) {
                    let slot = *slots.next().unwrap();
                    ExceptionRef {
                        generation: retained_exceptions.get(slot).map_or(1, |v| v.generation),
                        kernel: self.id,
                        index: slot as u32 + 1,
                    }
                } else {
                    existing[id as usize - 1]
                };
                (id, reference)
            })
            .collect();
        let externals = self
            .external_values
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let external_generation = |id: u64| -> Result<u64, Error> {
            externals
                .get((id as u32 as usize).wrapping_sub(1))
                .filter(|v| v.value.is_some())
                .map(|v| v.generation)
                .ok_or_else(|| invalid("missing retained external value"))
        };
        let decode = |ty: ValType, words: &[u64]| -> Result<Value, Error> {
            let word = words[0];
            match ty {
                ValType::Ref(ty) if self.gc_value_type(ty) => Ok(match word >> 32 {
                    _ if word == 0 => Value::GcRef(None),
                    1 => Value::I31(word as u32 & 0x7fffffff),
                    2 if word as u32 != 0 => Value::GcRef(Some(GcRef {
                        generation: external_generation(word)?,
                        kernel: self.id,
                        index: word,
                        ty: nonnull(AbstractHeapType::Any),
                    })),
                    3 => Value::GcRef(Some(
                        *object_map
                            .get(&(word as u32))
                            .ok_or_else(|| invalid("missing GC snapshot"))?,
                    )),
                    _ => return Err(invalid("invalid GC reference tag")),
                }),
                ValType::Ref(ty) if is_external_ref(ty) => Ok(Value::ExternRef(if word == 0 {
                    None
                } else {
                    Some(ExternalRef {
                        generation: match word >> 32 {
                            3 => {
                                object_map
                                    .get(&(word as u32))
                                    .ok_or_else(|| invalid("missing external GC snapshot"))?
                                    .generation
                            }
                            1 => 0,
                            _ => external_generation(word)?,
                        },
                        kernel: self.id,
                        index: if word >> 32 == 3 {
                            object_map
                                .get(&(word as u32))
                                .ok_or_else(|| invalid("missing external GC snapshot"))?
                                .index
                        } else {
                            word
                        },
                    })
                })),
                ValType::Ref(ty) if is_exception_ref(ty) => Ok(Value::ExnRef(if word == 0 {
                    None
                } else {
                    Some(
                        *exception_map
                            .get(&(word as u32))
                            .ok_or_else(|| invalid("missing exception snapshot"))?,
                    )
                })),
                _ => self.decode_value(ty, words, &[]),
            }
        };
        // Decode into temporary records before publication. A malformed output
        // cannot leave partially retained graphs or handles visible to callers.
        let new_objects: Result<Vec<_>, Error> = object_ids
            .iter()
            .map(|id| {
                let (type_index, values) = &objects[id];
                Ok(GcRecord {
                    type_index: *type_index,
                    values: values
                        .iter()
                        .map(|(ty, words)| decode(*ty, words))
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect();
        let new_exceptions: Result<Vec<_>, Error> = exception_ids
            .iter()
            .filter(|id| copy_exceptions.contains(id))
            .map(|id| {
                let (tag, values) = &exceptions[id];
                Ok(ExceptionRecord {
                    tag: *tag,
                    payload: values
                        .iter()
                        .map(|(ty, words)| decode(*ty, words))
                        .collect::<Result<_, _>>()?,
                })
            })
            .collect();
        let output = output
            .iter()
            .map(|(ty, words)| decode(*ty, words))
            .collect::<Result<_, _>>()?;
        let new_objects = new_objects?;
        let new_exceptions = new_exceptions?;
        for (slot, record) in object_slots.into_iter().zip(new_objects) {
            if slot == retained_objects.len() {
                retained_objects.push(Lease {
                    generation: 1,
                    retained: false,
                    value: Some(record),
                });
            } else {
                retained_objects[slot].value = Some(record);
                retained_objects[slot].retained = false;
            }
        }
        for (slot, record) in exception_slots.into_iter().zip(new_exceptions) {
            if slot == retained_exceptions.len() {
                retained_exceptions.push(Lease {
                    generation: 1,
                    retained: false,
                    value: Some(record),
                });
            } else {
                retained_exceptions[slot].value = Some(record);
                retained_exceptions[slot].retained = false;
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_reference_conversions_preserve_identity_and_kernel_ownership() {
        let wasm = wat::parse_str(
            r#"(module (func (export "run") (param anyref) (result anyref) local.get 0))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "run").unwrap();
        let other = Kernel::new(&wasm, "run").unwrap();
        let external = kernel.external_reference(73u32).unwrap();
        let internal = kernel.internalize_reference(Some(external)).unwrap();
        assert_eq!(
            kernel.externalize_reference(internal).unwrap(),
            Some(external)
        );
        let rows = kernel
            .run_values(&[
                vec![internal],
                vec![Value::I31(0x7fffffff)],
                vec![Value::GcRef(None)],
            ])
            .unwrap();
        assert_eq!(
            rows,
            [
                vec![internal],
                vec![Value::I31(0x7fffffff)],
                vec![Value::GcRef(None)]
            ]
        );
        let boxed_i31 = kernel.externalize_reference(Value::I31(u32::MAX)).unwrap();
        assert_eq!(
            kernel.internalize_reference(boxed_i31).unwrap(),
            Value::I31(0x7fffffff)
        );
        assert!(other.internalize_reference(Some(external)).is_err());
        assert!(other.externalize_reference(internal).is_err());
        assert!(other.run_values(&[vec![internal]]).is_err());
        assert_eq!(
            *kernel.external_reference_value::<u32>(external).unwrap(),
            73
        );
    }

    #[test]
    fn external_array_snapshots_preserve_both_vector_words() {
        let wasm = wat::parse_str(
            r#"(module
          (type $array (array (mut v128)))
          (func (export "run") (param externref i32) (result externref i32)
            (local $array (ref null $array)) (local $old i32)
            local.get 0 ref.is_null
            if (result (ref $array))
              i32.const 2 array.new_default $array
            else
              local.get 0 any.convert_extern ref.cast (ref $array)
            end
            local.set $array
            local.get $array i32.const 1 array.get $array i32x4.extract_lane 3 local.set $old
            local.get $array i32.const 1 local.get 1 i32x4.splat array.set $array
            local.get $array extern.convert_any local.get $old))"#,
        )
        .unwrap();
        let kernel = Kernel::new(&wasm, "run").unwrap();
        let first = kernel
            .run_values(&[vec![Value::ExternRef(None), Value::I32(7)]])
            .unwrap();
        assert_eq!(first[0][1], Value::I32(0));
        let Value::ExternRef(reference) = first[0][0] else {
            panic!("expected external reference");
        };
        let Value::GcRef(Some(object)) = kernel.internalize_reference(reference).unwrap() else {
            panic!("expected object reference");
        };
        assert_eq!(
            kernel.gc_reference_kind(object).unwrap(),
            AbstractHeapType::Array
        );
        assert_eq!(
            kernel
                .externalize_reference(Value::GcRef(Some(object)))
                .unwrap(),
            reference
        );
        let second = kernel
            .run_values(&[vec![first[0][0], Value::I32(9)]])
            .unwrap();
        assert_eq!(second[0][1], Value::I32(7));
    }

    #[test]
    fn snapshots_preserve_object_exception_cycles_and_clone_each_lane() {
        let wasm = wat::parse_str(r#"(module
          (type $node (struct (field (mut exnref)) (field (mut i32))))
          (tag $tag (export "tag") (param (ref $node)))
          (func (export "run") (param $node (ref null $node)) (param $exception exnref) (param $value i32)
            (result (ref $node) exnref i32) (local $old i32)
            local.get $node ref.is_null
            if struct.new_default $node local.set $node end
            local.get $exception ref.is_null
            if
              (block (result exnref)
                (try_table (catch_all_ref 0)
                  local.get $node ref.as_non_null throw $tag)
                unreachable)
              local.set $exception
            end
            local.get $node struct.get $node 1 local.set $old
            local.get $node local.get $exception struct.set $node 0
            local.get $node local.get $value struct.set $node 1
            local.get $node ref.as_non_null local.get $exception local.get $old))"#).unwrap();
        let kernel = Kernel::new(&wasm, "run").unwrap();
        let original = kernel
            .run_values(&[vec![Value::GcRef(None), Value::ExnRef(None), Value::I32(7)]])
            .unwrap()
            .remove(0);
        assert_eq!(original[2], Value::I32(0));
        let Value::ExnRef(Some(exception)) = original[1] else {
            panic!("expected exception");
        };
        assert_eq!(kernel.exception_payload(exception).unwrap(), [original[0]]);
        let inputs = [10, 20].map(|n| vec![original[0], original[1], Value::I32(n)]);
        let rows = kernel.run_values(&inputs).unwrap();
        for row in &rows {
            assert_eq!(row[2], Value::I32(7));
            let Value::ExnRef(Some(exception)) = row[1] else {
                panic!("expected exception");
            };
            assert_eq!(kernel.exception_payload(exception).unwrap(), [row[0]]);
        }
        assert_ne!(rows[0][0], rows[1][0]);
        let reused = kernel
            .run_values(&[
                vec![original[0], original[1], Value::I32(30)],
                vec![rows[0][0], rows[0][1], Value::I32(40)],
            ])
            .unwrap();
        assert_eq!(reused[0][2], Value::I32(7));
        assert_eq!(reused[1][2], Value::I32(10));
    }
}
