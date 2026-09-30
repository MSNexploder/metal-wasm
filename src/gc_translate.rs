use super::*;

impl Compiler<'_> {
    pub(super) fn publish_gc_roots(&mut self, stack: &[String]) {
        let mut roots = self
            .local_types
            .iter()
            .enumerate()
            .filter(|(_, ty)| matches!(ty, ValType::Ref(_)))
            .map(|(i, _)| format!("l{i}"))
            .collect::<Vec<_>>();
        roots.extend(
            stack
                .iter()
                .filter(|v| !self.vector_names.contains(*v))
                .cloned(),
        );
        roots.extend(
            self.frames
                .iter()
                .flat_map(|f| f.params.iter())
                .filter(|v| !self.vector_names.contains(*v))
                .cloned(),
        );
        roots.sort();
        roots.dedup();
        self.max_gc_roots = self.max_gc_roots.max(roots.len());
        self.source
            .push_str("for(uint j=0u;j<GC_FRAME_SLOTSu;++j) gc.roots[gc_frame.base+j]=0ul;\n");
        for (i, v) in roots.iter().enumerate() {
            self.source
                .push_str(&format!("gc.roots[gc_frame.base+{i}u]={v};\n"));
        }
    }
    fn gc_type(&self, index: u32) -> Result<GcType, Error> {
        self.gc_types
            .get(index as usize)
            .and_then(Clone::clone)
            .ok_or_else(|| Error::Unsupported("GC type is out of range".into()))
    }
    fn gc_pointer(&mut self, value: &str) -> String {
        let name = format!("gc_p{}", self.next_value);
        self.next_value += 1;
        self.source.push_str(&format!(
            "device ulong* {name}=gc_ptr(gc,{value},trap); if(trap!=0u) return;\n"
        ));
        name
    }
    fn gc_bounds(&mut self, ptr: &str, index: &str, count: &str) {
        self.source.push_str(&format!(
            "if ({index} > {ptr}[-1] || {count} > {ptr}[-1] - {index}) {{trap=16u;return;}}\n"
        ));
    }
    fn gc_read(
        &mut self,
        stack: &mut Vec<String>,
        ptr: &str,
        index: &str,
        ty: &StorageType,
        signed: bool,
    ) {
        match ty {
            StorageType::Val(ValType::V128) => {
                self.emit_vector(stack, format!("ulong2({ptr}[{index}],{ptr}[({index})+1u])"))
            }
            StorageType::I8 if signed => self.emit_value(
                stack,
                format!("uint(int(as_type<char>(uchar({ptr}[{index}]))))"),
            ),
            StorageType::I16 if signed => self.emit_value(
                stack,
                format!("uint(int(as_type<short>(ushort({ptr}[{index}]))))"),
            ),
            _ => self.emit_value(stack, format!("{ptr}[{index}]")),
        }
    }
    fn gc_allocate(&mut self, index: u32, length: &str, words: &str) -> String {
        let value = self.fresh_value();
        let id = self.gc_ids[index as usize];
        self.source.push_str(&format!("if ({words} > ulong(gc.capacity)) {{trap=15u;return;}}\n{value}=gc_alloc(gc,{id}u,uint({length}),uint({words}),trap); if(trap!=0u) return;\n"));
        value
    }
    fn gc_segment(&self, index: u32) -> String {
        if self
            .dynamic_elements
            .iter()
            .any(|(s, _, _)| *s == index as usize)
        {
            format!("states[{}].element_values[{index}]", self.module_index)
        } else {
            format!("m{}_element_{index}", self.module_index)
        }
    }
    pub(super) fn emit_gc_operator(
        &mut self,
        op: &Operator<'_>,
        stack: &mut Vec<String>,
    ) -> Result<bool, Error> {
        match op {
            Operator::RefEq => {
                let b = pop(stack)?;
                let a = pop(stack)?;
                self.emit_value(stack, format!("uint({a}=={b})"));
            }
            Operator::RefI31 => {
                let v = pop(stack)?;
                self.emit_value(
                    stack,
                    format!("0x100000000ul | (ulong(uint({v})) & 0x7ffffffful)"),
                );
            }
            Operator::I31GetS | Operator::I31GetU => {
                let v = pop(stack)?;
                self.source
                    .push_str(&format!("if({v}==0ul) {{trap=9u;return;}}\n"));
                self.emit_value(
                    stack,
                    if matches!(op, Operator::I31GetS) {
                        format!("uint(as_type<int>(uint({v}) << 1u) >> 1u)")
                    } else {
                        format!("uint({v}) & 0x7fffffffu")
                    },
                );
            }
            Operator::AnyConvertExtern => {
                let v = pop(stack)?;
                self.emit_value(
                    stack,
                    format!("({v}==0ul ? 0ul : ({v}>>32u)==0ul ? 0x200000000ul | {v} : {v})"),
                );
            }
            Operator::ExternConvertAny => {
                let v = pop(stack)?;
                self.emit_value(stack, format!("(({v}>>32u)==2ul ? ulong(uint({v})) : {v})"));
            }
            Operator::StructNew {
                struct_type_index: index,
            }
            | Operator::StructNewDefault {
                struct_type_index: index,
            } => {
                let t = self.gc_type(*index)?;
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
                let id = self.gc_ids[*index as usize];
                let value = self.fresh_value();
                self.source.push_str(&format!(
                    "{value}=gc_struct_{id}({}gc,trap); if(trap!=0u) return;\n",
                    if args.is_empty() {
                        String::new()
                    } else {
                        format!("{},", args.join(","))
                    }
                ));
                stack.push(value);
            }
            Operator::StructGet {
                struct_type_index: index,
                field_index,
            }
            | Operator::StructGetS {
                struct_type_index: index,
                field_index,
            }
            | Operator::StructGetU {
                struct_type_index: index,
                field_index,
            } => {
                let t = self.gc_type(*index)?;
                let v = pop(stack)?;
                let ptr = self.gc_pointer(&v);
                let word: usize = t.fields[..*field_index as usize]
                    .iter()
                    .map(storage_words)
                    .sum();
                self.gc_read(
                    stack,
                    &ptr,
                    &format!("{word}u"),
                    &t.fields[*field_index as usize],
                    matches!(op, Operator::StructGetS { .. }),
                );
            }
            Operator::StructSet {
                struct_type_index: index,
                field_index,
            } => {
                let t = self.gc_type(*index)?;
                let v = pop(stack)?;
                let r = pop(stack)?;
                let ptr = self.gc_pointer(&r);
                let word: usize = t.fields[..*field_index as usize]
                    .iter()
                    .map(storage_words)
                    .sum();
                self.source.push_str(&storage_write(
                    &ptr,
                    &format!("{word}u"),
                    &t.fields[*field_index as usize],
                    &v,
                ));
            }
            Operator::ArrayNew {
                array_type_index: index,
            }
            | Operator::ArrayNewDefault {
                array_type_index: index,
            } => {
                let t = self.gc_type(*index)?;
                let len = pop(stack)?;
                let v = if matches!(op, Operator::ArrayNewDefault { .. }) {
                    if storage_words(&t.fields[0]) == 2 {
                        "ulong2(0ul)".into()
                    } else {
                        "0ul".into()
                    }
                } else {
                    pop(stack)?
                };
                let id = self.gc_ids[*index as usize];
                let value = self.fresh_value();
                self.source.push_str(&format!(
                    "{value}=gc_array_{id}({v},{len},gc,trap);if(trap!=0u)return;\n"
                ));
                stack.push(value);
            }
            Operator::ArrayNewFixed {
                array_type_index: index,
                array_size,
            } => {
                let t = self.gc_type(*index)?;
                let args = stack.split_off(stack.len() - *array_size as usize);
                let w = storage_words(&t.fields[0]);
                let r = self.gc_allocate(
                    *index,
                    &format!("{array_size}ul"),
                    &format!("{}ul", *array_size as usize * w),
                );
                let ptr = self.gc_pointer(&r);
                for (j, v) in args.iter().enumerate() {
                    self.source.push_str(&storage_write(
                        &ptr,
                        &format!("{}u", j * w),
                        &t.fields[0],
                        v,
                    ));
                }
                stack.push(r);
            }
            Operator::ArrayLen => {
                let r = pop(stack)?;
                let ptr = self.gc_pointer(&r);
                self.emit_value(stack, format!("{ptr}[-1]"));
            }
            Operator::ArrayGet {
                array_type_index: index,
            }
            | Operator::ArrayGetS {
                array_type_index: index,
            }
            | Operator::ArrayGetU {
                array_type_index: index,
            } => {
                let t = self.gc_type(*index)?;
                let i = pop(stack)?;
                let r = pop(stack)?;
                let ptr = self.gc_pointer(&r);
                self.gc_bounds(&ptr, &i, "1ul");
                self.gc_read(
                    stack,
                    &ptr,
                    &format!("({i})*{}ul", storage_words(&t.fields[0])),
                    &t.fields[0],
                    matches!(op, Operator::ArrayGetS { .. }),
                );
            }
            Operator::ArraySet {
                array_type_index: index,
            } => {
                let t = self.gc_type(*index)?;
                let v = pop(stack)?;
                let i = pop(stack)?;
                let r = pop(stack)?;
                let ptr = self.gc_pointer(&r);
                self.gc_bounds(&ptr, &i, "1ul");
                self.source.push_str(&storage_write(
                    &ptr,
                    &format!("({i})*{}ul", storage_words(&t.fields[0])),
                    &t.fields[0],
                    &v,
                ));
            }
            Operator::ArrayFill {
                array_type_index: index,
            } => {
                let t = self.gc_type(*index)?;
                let n = pop(stack)?;
                let v = pop(stack)?;
                let i = pop(stack)?;
                let r = pop(stack)?;
                let ptr = self.gc_pointer(&r);
                self.gc_bounds(&ptr, &i, &n);
                self.source
                    .push_str(&format!("for(ulong j=0ul;j<{n};++j) {{\n"));
                self.source.push_str(&storage_write(
                    &ptr,
                    &format!("({i}+j)*{}ul", storage_words(&t.fields[0])),
                    &t.fields[0],
                    &v,
                ));
                self.source.push_str("}\n");
            }
            Operator::ArrayCopy {
                array_type_index_dst: index,
                ..
            } => {
                let t = self.gc_type(*index)?;
                let n = pop(stack)?;
                let si = pop(stack)?;
                let sr = pop(stack)?;
                let di = pop(stack)?;
                let dr = pop(stack)?;
                let dst = self.gc_pointer(&dr);
                let src = self.gc_pointer(&sr);
                self.gc_bounds(&dst, &di, &n);
                self.gc_bounds(&src, &si, &n);
                let w = storage_words(&t.fields[0]);
                self.source.push_str(&format!("for(ulong j=0ul;j<{n}*{w}ul;++j) {{ ulong k=({dr}=={sr} && {di}>{si}) ? {n}*{w}ul-1ul-j:j; {dst}[{di}*{w}ul+k]={src}[{si}*{w}ul+k]; }}\n"));
            }
            Operator::ArrayNewData {
                array_type_index: index,
                array_data_index: segment,
            }
            | Operator::ArrayInitData {
                array_type_index: index,
                array_data_index: segment,
            } => {
                let t = self.gc_type(*index)?;
                let f = t.fields[0];
                let bytes = match f {
                    StorageType::I8 => 1,
                    StorageType::I16 => 2,
                    StorageType::Val(ValType::I32 | ValType::F32) => 4,
                    StorageType::Val(ValType::V128) => 16,
                    _ => 8,
                };
                let n = pop(stack)?;
                let si = pop(stack)?;
                let is_new = matches!(op, Operator::ArrayNewData { .. });
                let di = if is_new { "0ul".into() } else { pop(stack)? };
                let r = if is_new { String::new() } else { pop(stack)? };
                let mut ptr = if is_new {
                    String::new()
                } else {
                    self.gc_pointer(&r)
                };
                if !is_new {
                    self.gc_bounds(&ptr, &di, &n);
                }
                self.source.push_str(&format!("if({si}>ulong(data_lengths[{segment}]) || {n}> (ulong(data_lengths[{segment}])-{si})/{bytes}ul) {{trap=4u;return;}}\n"));
                let r = if is_new {
                    let r = self.gc_allocate(*index, &n, &format!("{n}*{}ul", storage_words(&f)));
                    ptr = self.gc_pointer(&r);
                    r
                } else {
                    r
                };
                let data_offset = self.data_offsets[*segment as usize];
                let w = storage_words(&f);
                self.source.push_str(&format!("for(ulong j=0ul;j<{n};++j) {{ for(uint k=0u;k<{w}u;++k) {{ ulong word=0ul; for(uint b=0u;b<{}u;++b) word |= ulong(passive_data[{data_offset}ul+{si}+j*{bytes}ul+ulong(k)*8ul+b]) << (b*8u); {ptr}[({di}+j)*{w}ul+k]=word; }} }}\n",bytes.min(8)));
                if is_new {
                    stack.push(r);
                }
            }
            Operator::ArrayNewElem {
                array_type_index: index,
                array_elem_index: segment,
            }
            | Operator::ArrayInitElem {
                array_type_index: index,
                array_elem_index: segment,
            } => {
                let n = pop(stack)?;
                let si = pop(stack)?;
                let is_new = matches!(op, Operator::ArrayNewElem { .. });
                let di = if is_new { "0ul".into() } else { pop(stack)? };
                let r = if is_new { String::new() } else { pop(stack)? };
                let mut ptr = if is_new {
                    String::new()
                } else {
                    self.gc_pointer(&r)
                };
                if !is_new {
                    self.gc_bounds(&ptr, &di, &n);
                }
                self.source.push_str(&format!("if({si}>ulong(element_lengths[{segment}]) || {n}>ulong(element_lengths[{segment}])-{si}) {{trap=14u;return;}}\n"));
                let r = if is_new {
                    let r = self.gc_allocate(*index, &n, &n);
                    ptr = self.gc_pointer(&r);
                    r
                } else {
                    r
                };
                let values = self.gc_segment(*segment);
                self.source.push_str(&format!(
                    "for(ulong j=0ul;j<{n};++j) {ptr}[{di}+j]={values}[{si}+j];\n"
                ));
                if is_new {
                    stack.push(r);
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

pub(super) fn gc_type_source(types: &[GcType], modules: &[ParsedModule<'_>]) -> String {
    let parents: ReferenceTypes = modules
        .iter()
        .flat_map(|m| m.supertypes.iter().map(|(&id, &p)| (id, p)))
        .collect();
    let mut s = String::from(
        "inline ulong gc_internalize(ulong v) { return v==0ul ? 0ul : (v>>32u)==0ul ? 0x200000000ul | v : v; }\ninline ulong gc_externalize(ulong v) { return (v>>32u)==2ul ? ulong(uint(v)) : v; }\ninline ulong gc_keep(thread GcState& gc, uint slot, ulong value) { gc.roots[slot] = value; return value; }\ninline uint gc_kind(uint type) { switch(type) {\n",
    );
    for (i, t) in types.iter().enumerate() {
        s.push_str(&format!(
            "case {}u: return {}u;\n",
            i + 1,
            if t.array { 2 } else { 1 }
        ));
    }
    s.push_str("default: return 0u; } }\ninline bool gc_type_match(uint actual, uint expected) { switch(expected) {\n");
    for (i, t) in types.iter().enumerate() {
        let accepted = types
            .iter()
            .enumerate()
            .filter(|(_, a)| ref_subtype(a.reference, t.reference, &parents))
            .map(|(j, _)| format!("actual == {}u", j + 1))
            .collect::<Vec<_>>()
            .join(" || ");
        s.push_str(&format!("case {}u: return {accepted};\n", i + 1));
    }
    s.push_str("default: return false; } }\ninline bool gc_ref_word(uint type, uint word) { switch(type) {\n");
    for (i, t) in types.iter().enumerate() {
        let mut words = Vec::new();
        let mut offset = 0;
        for f in &t.fields {
            if storage_is_ref(f) {
                words.push(offset);
            }
            offset += storage_words(f);
        }
        let condition = if words.is_empty() {
            "false".into()
        } else if t.array {
            format!("word % {}u == 0u", t.words())
        } else {
            words
                .iter()
                .map(|w| format!("word == {w}u"))
                .collect::<Vec<_>>()
                .join(" || ")
        };
        s.push_str(&format!("case {}u: return {condition};\n", i + 1));
    }
    s.push_str("default: return false; } }\ninline void gc_mark_state(thread GcState& gc) {\n");
    for (m, module) in modules.iter().enumerate() {
        let imported = module.global_types.len() - module.local_global_slots.len();
        for (i, &slot) in module.local_global_slots.iter().enumerate() {
            if matches!(module.global_types[imported+i].content_type,ValType::Ref(r) if is_gc_ref(r,&parents)||is_external_ref(r))
            {
                s.push_str(&format!("if (gc.states[{m}].globals != nullptr) gc_mark(gc, gc.states[{m}].globals[{slot}u]);\n"));
            }
        }
        let imported = module.table_types.len() - module.instance.tables.len();
        for (i, _) in module.instance.tables.iter().enumerate() {
            let r = module.table_types[imported + i].element_type;
            if is_gc_ref(r, &parents) || is_external_ref(r) {
                s.push_str(&format!("if (gc.states[{m}].table[{i}] != nullptr) for (uint j=0u;j<gc.states[{m}].table_size[{i}];++j) gc_mark(gc,gc.states[{m}].table[{i}][j]);\n"));
            }
        }
        for (i, values) in module.instance.element_info.elements.iter().enumerate() {
            if module.dynamic_elements.iter().any(|(seg, _, _)| *seg == i) {
                s.push_str(&format!("if (gc.states[{m}].element_values[{i}] != nullptr) for (uint j=0u;j<{}u;++j) gc_mark(gc,gc.states[{m}].element_values[{i}][j]);\n",values.len()));
            }
        }
    }
    if modules.iter().any(|m| m.uses_exceptions) {
        s.push_str("if (gc.states[0].exceptions != nullptr) { thread ExceptionState& ex = *gc.states[0].exceptions; for (uint i=0u;i<ex.count;++i) { switch(ex.tags[i]) {\n");
        let mut seen = HashSet::new();
        for m in modules {
            for (&tag, &ty) in m.tag_bindings.iter().zip(&m.tag_types) {
                if !seen.insert(tag) {
                    continue;
                }
                s.push_str(&format!("case {tag}u: {{\n"));
                let mut word = 0;
                for t in m.types[ty as usize].params() {
                    if matches!(t,ValType::Ref(r) if is_gc_ref(*r,&parents)||is_external_ref(*r)) {
                        s.push_str(&format!(
                            "gc_mark(gc,ex.values[i*EXCEPTION_WORDS+{word}u]);\n"
                        ));
                    }
                    word += value_words(*t);
                }
                s.push_str("break; }\n");
            }
        }
        s.push_str("default: break; } } }\n");
    }
    s.push_str("}\n");
    for (i, t) in types.iter().enumerate() {
        let id = i + 1;
        if t.array {
            let f = t.fields[0];
            let width = storage_words(&f);
            let argty = if width == 2 { "ulong2" } else { "ulong" };
            s.push_str(&format!("inline ulong gc_array_{id}({argty} value, ulong length, thread GcState& gc, thread uint& trap) {{\nif (length > ulong(gc.capacity) / {width}ul) {{ trap=15u; return 0ul; }}\nGcRootFrame roots(gc,1u,trap); if (trap!=0u) return 0ul;\n"));
            if storage_is_ref(&f) {
                s.push_str("gc.roots[roots.base]=value;\n");
            }
            s.push_str(&format!("ulong r=gc_alloc(gc,{id}u,uint(length),uint(length)*{width}u,trap); if(trap!=0u) return 0ul; device ulong* p=gc_ptr(gc,r,trap); for(uint j=0u;j<uint(length);++j) {{\n"));
            s.push_str(&storage_write("p", &format!("j*{width}u"), &f, "value"));
            s.push_str("} return r; }\n");
        } else {
            let args = t
                .fields
                .iter()
                .enumerate()
                .map(|(j, f)| {
                    format!(
                        "{} v{j}",
                        if storage_words(f) == 2 {
                            "ulong2"
                        } else {
                            "ulong"
                        }
                    )
                })
                .collect::<Vec<_>>();
            s.push_str(&format!("inline ulong gc_struct_{id}({}thread GcState& gc, thread uint& trap) {{ GcRootFrame roots(gc,{}u,trap); if(trap!=0u) return 0ul;\n",if args.is_empty(){String::new()}else{format!("{}, ",args.join(", "))},t.fields.len().max(1)));
            for (j, f) in t.fields.iter().enumerate() {
                if storage_is_ref(f) {
                    s.push_str(&format!("gc.roots[roots.base+{j}u]=v{j};\n"));
                }
            }
            s.push_str(&format!("ulong r=gc_alloc(gc,{id}u,{}u,{}u,trap); if(trap!=0u) return 0ul; device ulong* p=gc_ptr(gc,r,trap);\n",t.fields.len(),t.words()));
            let mut offset = 0;
            for (j, f) in t.fields.iter().enumerate() {
                s.push_str(&storage_write(
                    "p",
                    &offset.to_string(),
                    f,
                    &format!("v{j}"),
                ));
                offset += storage_words(f);
            }
            s.push_str("return r; }\n");
        }
    }
    let mut fixed = HashSet::new();
    for m in modules {
        for expr in m
            .dynamic_globals
            .iter()
            .map(|(_, _, e)| e)
            .chain(m.dynamic_elements.iter().map(|(_, _, e)| e))
            .chain(m.table_init_exprs.iter().map(|(_, e)| e))
        {
            for op in expr.get_operators_reader() {
                if let Ok(Operator::ArrayNewFixed {
                    array_type_index: index,
                    array_size,
                }) = op
                {
                    let id = m.gc_ids[index as usize];
                    if !fixed.insert((id, array_size)) {
                        continue;
                    }
                    let t = &types[id as usize - 1];
                    let f = t.fields[0];
                    let width = storage_words(&f);
                    let argty = if width == 2 { "ulong2" } else { "ulong" };
                    let args = (0..array_size)
                        .map(|j| format!("{argty} v{j}"))
                        .collect::<Vec<_>>();
                    s.push_str(&format!("inline ulong gc_fixed_{id}_{array_size}({}thread GcState& gc,thread uint& trap) {{\nulong r=gc_alloc(gc,{id}u,{array_size}u,{}u,trap);if(trap!=0u)return 0ul;device ulong* p=gc_ptr(gc,r,trap);\n",if args.is_empty(){String::new()}else{format!("{},",args.join(","))},array_size as usize*width));
                    for j in 0..array_size {
                        s.push_str(&storage_write(
                            "p",
                            &format!("{}u", j as usize * width),
                            &f,
                            &format!("v{j}"),
                        ));
                    }
                    s.push_str("return r;}\n");
                }
            }
        }
    }

    s
}
fn storage_write(ptr: &str, index: &str, ty: &StorageType, value: &str) -> String {
    match ty {
        StorageType::I8 => format!("{ptr}[{index}]=ulong(uchar({value}));\n"),
        StorageType::I16 => format!("{ptr}[{index}]=ulong(ushort({value}));\n"),
        StorageType::Val(ValType::V128) => {
            format!("{ptr}[{index}]=({value}).x; {ptr}[({index})+1u]=({value}).y;\n")
        }
        StorageType::Val(ValType::I32 | ValType::F32) => {
            format!("{ptr}[{index}]=ulong(uint({value}));\n")
        }
        _ => format!("{ptr}[{index}]={value};\n"),
    }
}
