use std::sync::Mutex;

use metal_wasm::{Error, Kernel, KernelLimits, Store, Value};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn wasm(source: &str) -> Vec<u8> {
    wat::parse_str(source).unwrap()
}

#[test]
fn compiled_rust_static_matches_independent_persistent_wasmtime_instances() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = include_bytes!("corpus/persistent.wasm");
    let mut metal = Store::new(3).unwrap();
    let instance = metal.instantiate(bytes, &[]).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, bytes).unwrap();
    let mut cpu = wasmtime::Store::new(&engine, ());
    let mut functions = Vec::new();
    for _ in 0..3 {
        let lane = wasmtime::Instance::new(&mut cpu, &module, &[]).unwrap();
        functions.push((
            lane.get_typed_func::<i32, i32>(&mut cpu, "step").unwrap(),
            lane.get_typed_func::<(), i32>(&mut cpu, "read").unwrap(),
        ));
    }
    for inputs in [[0, 1, 42], [1, 2, 99], [u32::MAX, 0x8000_0000, 10]] {
        let expected: Vec<_> = functions
            .iter()
            .zip(inputs)
            .map(|((step, _), input)| step.call(&mut cpu, input as i32).unwrap() as u32)
            .collect();
        assert_eq!(metal.call_i32(instance, "step", &inputs).unwrap(), expected);
        let expected: Vec<_> = functions
            .iter()
            .map(|(_, read)| vec![Value::I32(read.call(&mut cpu, ()).unwrap() as u32)])
            .collect();
        assert_eq!(
            metal
                .call(instance, "read", &[vec![], vec![], vec![]])
                .unwrap(),
            expected
        );
    }
}

#[test]
fn exports_share_lane_state_and_start_runs_once() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(2).unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (memory (export "memory") 1 3)
        (global $g (export "counter") (mut i32) (i32.const 10))
        (func $start global.get $g i32.const 1 i32.add global.set $g)
        (start $start)
        (func (export "step") (param i32) (result i32)
            global.get $g local.get 0 i32.add global.set $g
            i32.const 0 global.get $g i32.store global.get $g)
        (func (export "read") (result i32) i32.const 0 i32.load)
        (func (export "grow") (param i32) (result i32) local.get 0 memory.grow)
        (func (export "write_then_trap") (param i32)
            local.get 0 global.set $g
            i32.const 0 local.get 0 i32.store unreachable))"#,
            ),
            &[],
        )
        .unwrap();
    assert_eq!(
        store.get_global(instance, "counter").unwrap(),
        vec![Value::I32(11); 2]
    );
    assert_eq!(store.call_i32(instance, "step", &[1, 9]).unwrap(), [12, 20]);
    assert_eq!(
        store.call(instance, "read", &[vec![], vec![]]).unwrap(),
        [vec![Value::I32(12)], vec![Value::I32(20)]]
    );
    assert_eq!(store.call_i32(instance, "step", &[2, 3]).unwrap(), [14, 23]);
    assert_eq!(store.call_i32(instance, "grow", &[1, 2]).unwrap(), [1, 1]);
    assert_eq!(
        store.read_memory(instance, "memory", 65536..65540).unwrap(),
        vec![vec![0; 4]; 2]
    );
    store
        .write_memory(instance, "memory", 65536, &[vec![1, 2], vec![3, 4]])
        .unwrap();
    assert_eq!(
        store.read_memory(instance, "memory", 65536..65538).unwrap(),
        [vec![1, 2], vec![3, 4]]
    );
    let result = store.call(
        instance,
        "write_then_trap",
        &[vec![Value::I32(101)], vec![Value::I32(202)]],
    );
    assert!(matches!(
        result,
        Err(Error::Trap {
            input_index: 0,
            reason: "unreachable executed"
        })
    ));
    assert_eq!(
        store.get_global(instance, "counter").unwrap(),
        [Value::I32(101), Value::I32(202)]
    );
    assert_eq!(
        store.call(instance, "read", &[vec![], vec![]]).unwrap(),
        [vec![Value::I32(101)], vec![Value::I32(202)]]
    );
}

#[test]
fn import_limits_use_current_sizes_in_every_lane_and_alias_existing_state() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(2).unwrap();
    let provider = store
        .instantiate(
            &wasm(
                r#"(module
        (memory (export "memory") 0 3)
        (table (export "table") 0 3 funcref)
        (global $g (export "global") (mut i32) (i32.const 0))
        (func (export "grow_memory") (param i32) (result i32) local.get 0 memory.grow)
        (func (export "grow_table") (param i32) (result i32) ref.null func local.get 0 table.grow)
        (func (export "step") (param i32) (result i32)
            global.get $g local.get 0 i32.add global.set $g global.get $g))"#,
            ),
            &[],
        )
        .unwrap();
    assert_eq!(
        store.call_i32(provider, "grow_memory", &[1, 2]).unwrap(),
        [0, 0]
    );
    assert_eq!(
        store.call_i32(provider, "grow_table", &[1, 2]).unwrap(),
        [0, 0]
    );
    let consumer = wasm(
        r#"(module
        (import "a" "memory" (memory 2 3))
        (import "a" "table" (table 2 3 funcref))
        (import "a" "global" (global $a (mut i32)))
        (import "b" "global" (global $b (mut i32)))
        (import "b" "step" (func $step (param i32) (result i32)))
        (export "memory" (memory 0))
        (func $start i32.const 77 global.set $a)
        (start $start)
        (func (export "step") (param i32) (result i32)
            local.get 0 call $step drop global.get $b))"#,
    );
    assert!(matches!(
        store.instantiate(&consumer, &[("a", provider), ("b", provider)]),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        store.get_global(provider, "global").unwrap(),
        [Value::I32(0); 2]
    );
    assert_eq!(
        store.call_i32(provider, "grow_memory", &[1, 0]).unwrap(),
        [1, 2]
    );
    // Memory now satisfies the import, but lane zero's table still does not.
    assert!(matches!(
        store.instantiate(&consumer, &[("a", provider), ("b", provider)]),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        store.call_i32(provider, "grow_table", &[1, 0]).unwrap(),
        [1, 2]
    );
    let consumer = store
        .instantiate(&consumer, &[("a", provider), ("b", provider)])
        .unwrap();
    assert_eq!(store.call_i32(consumer, "step", &[1, 2]).unwrap(), [78, 79]);
    assert_eq!(
        store.call_i32(provider, "step", &[10, 20]).unwrap(),
        [88, 99]
    );
    assert_eq!(store.call_i32(consumer, "step", &[0, 0]).unwrap(), [88, 99]);
    store
        .write_memory(consumer, "memory", 0, &[vec![5], vec![6]])
        .unwrap();
    assert_eq!(
        store.read_memory(provider, "memory", 0..1).unwrap(),
        [vec![5], vec![6]]
    );
}

#[test]
fn reexported_resource_limits_use_the_underlying_definition() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for kind in ["memory", "table"] {
        let (declaration, size, grow) = if kind == "memory" {
            ("memory", "memory.size", "local.get 0 memory.grow")
        } else {
            (
                "table",
                "table.size",
                "ref.null func local.get 0 table.grow",
            )
        };
        let element = if kind == "table" { " funcref" } else { "" };
        let provider_bytes = wasm(&format!(
            "(module ({declaration} (export \"resource\") 1 2{element})
                (func (export \"grow\") (param i32) (result i32) {grow}))"
        ));
        let intermediary_bytes = wasm(&format!(
            "(module (import \"a\" \"resource\" ({declaration} 1 3{element}))
                (export \"resource\" ({declaration} 0)))"
        ));
        let consumer_bytes = wasm(&format!(
            "(module (import \"b\" \"resource\" ({declaration} 1 2{element}))
                (func (export \"size\") (param i32) (result i32) {size}))"
        ));
        let engine = wasmtime::Engine::default();
        let mut cpu = wasmtime::Store::new(&engine, ());
        let a = wasmtime::Instance::new(
            &mut cpu,
            &wasmtime::Module::new(&engine, &provider_bytes).unwrap(),
            &[],
        )
        .unwrap();
        let resource = a.get_export(&mut cpu, "resource").unwrap();
        let b = wasmtime::Instance::new(
            &mut cpu,
            &wasmtime::Module::new(&engine, &intermediary_bytes).unwrap(),
            &[resource],
        )
        .unwrap();
        let resource = b.get_export(&mut cpu, "resource").unwrap();
        let c = wasmtime::Instance::new(
            &mut cpu,
            &wasmtime::Module::new(&engine, &consumer_bytes).unwrap(),
            &[resource],
        )
        .unwrap();
        let expected = c
            .get_typed_func::<i32, i32>(&mut cpu, "size")
            .unwrap()
            .call(&mut cpu, 0)
            .unwrap() as u32;
        let mut store = Store::new(2).unwrap();
        let a = store.instantiate(&provider_bytes, &[]).unwrap();
        let b = store.instantiate(&intermediary_bytes, &[("a", a)]).unwrap();
        let c = store.instantiate(&consumer_bytes, &[("b", b)]).unwrap();
        assert_eq!(store.call_i32(c, "size", &[0, 0]).unwrap(), [expected; 2]);
        assert_eq!(store.call_i32(a, "grow", &[1, 0]).unwrap(), [1, 1]);
        assert_eq!(store.call_i32(c, "size", &[0, 0]).unwrap(), [2, 1]);
        let too_small = wasm(&format!(
            "(module (import \"b\" \"resource\" ({declaration} 1 1{element})))"
        ));
        assert!(matches!(
            store.instantiate(&too_small, &[("b", b)]),
            Err(Error::Unsupported(reason)) if reason.contains("type mismatch")
        ));
        let kernel = Kernel::new_with_imports(
            &consumer_bytes,
            "size",
            &[("a", &provider_bytes), ("b", &intermediary_bytes)],
        )
        .unwrap();
        assert_eq!(kernel.run(&[0, 0]).unwrap(), [expected; 2]);
    }
}

#[test]
fn dropped_segments_and_failed_bulk_operations_keep_their_state() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(1).unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (memory (export "memory") 1)
        (data $d "abc")
        (type $t (func (result i32)))
        (table 2 funcref)
        (func $f (type $t) i32.const 91)
        (elem $e func $f)
        (func (export "init")
            i32.const 0 i32.const 0 i32.const 3 memory.init $d
            i32.const 0 i32.const 0 i32.const 1 table.init $e)
        (func (export "drop") data.drop $d elem.drop $e)
        (func (export "call") (result i32) i32.const 0 call_indirect (type $t))
        (func (export "bad_copy") i32.const 0 i32.const 65535 i32.const 2 memory.copy))"#,
            ),
            &[],
        )
        .unwrap();
    store.call(instance, "init", &[vec![]]).unwrap();
    store.call(instance, "drop", &[vec![]]).unwrap();
    assert!(matches!(
        store.call(instance, "init", &[vec![]]),
        Err(Error::Trap {
            reason: "out of bounds memory access",
            ..
        })
    ));
    assert_eq!(
        store.call(instance, "call", &[vec![]]).unwrap(),
        [vec![Value::I32(91)]]
    );
    assert!(matches!(
        store.call(instance, "bad_copy", &[vec![]]),
        Err(Error::Trap {
            reason: "out of bounds memory access",
            ..
        })
    ));
    assert_eq!(
        store.read_memory(instance, "memory", 0..3).unwrap(),
        [b"abc".to_vec()]
    );
}

#[test]
fn failed_start_keeps_imported_writes_and_escaped_function_references() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(1).unwrap();
    let provider = store
        .instantiate(
            &wasm(
                r#"(module
        (type $t (func (result i32)))
        (table (export "table") 1 funcref)
        (global (export "global") (mut i32) (i32.const 0))
        (func (export "call") (result i32) i32.const 0 call_indirect (type $t)))"#,
            ),
            &[],
        )
        .unwrap();
    let failed = store.instantiate(
        &wasm(
            r#"(module
        (import "p" "table" (table 1 funcref))
        (import "p" "global" (global $g (mut i32)))
        (func $escaped (result i32) i32.const 123)
        (elem declare func $escaped)
        (func $start
            i32.const 0 ref.func $escaped table.set
            i32.const 7 global.set $g unreachable)
        (start $start))"#,
        ),
        &[("p", provider)],
    );
    assert!(matches!(
        failed,
        Err(Error::Trap {
            reason: "unreachable executed",
            ..
        })
    ));
    assert_eq!(
        store.get_global(provider, "global").unwrap(),
        [Value::I32(7)]
    );
    assert_eq!(
        store.call(provider, "call", &[vec![]]).unwrap(),
        [vec![Value::I32(123)]]
    );
    store
        .instantiate(
            &wasm("(module (func (export \"other\") (result i32) i32.const 9))"),
            &[],
        )
        .unwrap();
    assert_eq!(
        store.call(provider, "call", &[vec![]]).unwrap(),
        [vec![Value::I32(123)]]
    );
}

#[test]
fn live_gc_handles_preserve_aliases_and_survive_collection_in_other_instances() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new_with_limits(
        2,
        KernelLimits {
            gc_heap_words: 64,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (type $s (struct (field (mut i32))))
        (global $g (mut (ref null $s)) (ref.null $s))
        (func (export "create") (param i32) (result (ref $s))
            local.get 0 struct.new $s global.set $g global.get $g ref.as_non_null)
        (func (export "same") (param (ref $s)) (result i32) local.get 0 global.get $g ref.eq)
        (func (export "set") (param (ref $s) i32) local.get 0 local.get 1 struct.set $s 0)
        (func (export "get") (result i32) global.get $g struct.get $s 0))"#,
            ),
            &[],
        )
        .unwrap();
    let references = store
        .call(
            instance,
            "create",
            &[vec![Value::I32(10)], vec![Value::I32(20)]],
        )
        .unwrap();
    assert_eq!(
        store.call(instance, "same", &references).unwrap(),
        [vec![Value::I32(1)], vec![Value::I32(1)]]
    );
    store
        .call(
            instance,
            "set",
            &[
                vec![references[0][0], Value::I32(30)],
                vec![references[1][0], Value::I32(40)],
            ],
        )
        .unwrap();
    let allocator = store
        .instantiate(
            &wasm(
                r#"(module
        (type $a (array (mut i32)))
        (func (export "allocate") (param i32) (result i32)
            (local $i i32)
            (loop $again
                i32.const 0 i32.const 8 array.new $a drop
                local.get $i i32.const 1 i32.add local.tee $i
                i32.const 100 i32.lt_u br_if $again)
            local.get 0))"#,
            ),
            &[],
        )
        .unwrap();
    store.call_i32(allocator, "allocate", &[1, 2]).unwrap();
    assert_eq!(
        store.call(instance, "get", &[vec![], vec![]]).unwrap(),
        [vec![Value::I32(30)], vec![Value::I32(40)]]
    );
    assert_eq!(
        store.call(instance, "same", &references).unwrap(),
        [vec![Value::I32(1)], vec![Value::I32(1)]]
    );
    assert!(matches!(
        store.call(
            instance,
            "same",
            &[references[1].clone(), references[0].clone()]
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn exception_references_keep_gc_payloads_across_calls() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(1).unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (type $s (struct (field i32)))
        (tag $tag (param (ref $s)))
        (global $saved (mut exnref) (ref.null exn))
        (func (export "capture") (param i32) (result exnref)
            (local $object (ref $s))
            local.get 0 struct.new $s local.set $object
            (block $caught (result (ref $s) exnref)
                (try_table (catch_ref $tag $caught) local.get $object throw $tag)
                unreachable)
            global.set $saved drop global.get $saved)
        (func (export "read") (param exnref) (result i32)
            (block $caught (result (ref $s))
                (try_table (catch $tag $caught) local.get 0 throw_ref)
                unreachable)
            struct.get $s 0)
        (func (export "saved") (result i32)
            (block $caught (result (ref $s))
                (try_table (catch $tag $caught) global.get $saved throw_ref)
                unreachable)
            struct.get $s 0))"#,
            ),
            &[],
        )
        .unwrap();
    let reference = store
        .call(instance, "capture", &[vec![Value::I32(321)]])
        .unwrap();
    store
        .instantiate(&wasm("(module (global i32 (i32.const 8)))"), &[])
        .unwrap();
    assert_eq!(
        store.call(instance, "read", &reference).unwrap(),
        [vec![Value::I32(321)]]
    );
    assert_eq!(
        store.call(instance, "saved", &[vec![]]).unwrap(),
        [vec![Value::I32(321)]]
    );
}

#[test]
fn host_validation_rejects_foreign_handles_and_bad_rows_before_writes() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let module = wasm(
        r#"(module
        (global $g (export "global") (mut i32) (i32.const 0))
        (func (export "step") (param i32) (result i32)
            local.get 0 global.set $g global.get $g)
        (func (export "reference") (param externref) local.get 0 drop)
        (func (export "write_only") (param i32) local.get 0 global.set $g))"#,
    );
    let mut store = Store::new(2).unwrap();
    let instance = store.instantiate(&module, &[]).unwrap();
    let mut other = Store::new(2).unwrap();
    let foreign = other.instantiate(&module, &[]).unwrap();
    assert!(matches!(
        store.call_i32(foreign, "step", &[1, 2]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        store.call_i32(instance, "step", &[1]),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        store.call(
            instance,
            "step",
            &[vec![Value::I32(1)], vec![Value::I64(2)]]
        ),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        store.call_i32(instance, "write_only", &[9, 10]),
        Err(Error::Unsupported(_))
    ));
    let reference = other.external_reference(42_u32).unwrap();
    assert!(matches!(
        store.call(
            instance,
            "reference",
            &[
                vec![Value::ExternRef(Some(reference))],
                vec![Value::ExternRef(Some(reference))]
            ]
        ),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        store.instantiate(&module, &[("foreign", foreign)]),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        store.get_global(instance, "global").unwrap(),
        [Value::I32(0); 2]
    );
    assert_eq!(store.call_i32(instance, "step", &[3, 4]).unwrap(), [3, 4]);
}

#[test]
fn oversized_persistent_resources_are_rejected_before_allocation() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new_with_limits(
        u32::MAX as usize,
        KernelLimits {
            memory_bytes: 1_usize << 32,
            memory_growth_pages: 65536,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    // Initial memory is empty. The reserved capacity, not fixture allocation,
    // exercises lane multiplication near usize::MAX.
    let module = wasm(
        "(module (memory 0) (func (export \"grow\") (param i32) (result i32) local.get 0 memory.grow))",
    );
    assert!(matches!(
        store.instantiate(&module, &[]),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn private_table_accepts_host_functions_through_another_export() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(2).unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (type $t (func (result i32)))
        (table 1 funcref)
        (func $initial (result i32) i32.const 11)
        (elem (i32.const 0) func $initial)
        (func (export "set") (param funcref) i32.const 0 local.get 0 table.set)
        (func (export "call") (result i32) i32.const 0 call_indirect (type $t)))"#,
            ),
            &[],
        )
        .unwrap();
    assert_eq!(
        store.call(instance, "call", &[vec![], vec![]]).unwrap(),
        [vec![Value::I32(11)], vec![Value::I32(11)]]
    );
    // This target has no ref.func instruction and belongs to a later instance.
    let provider = store
        .instantiate(
            &wasm("(module (func (export \"target\") (result i32) i32.const 42))"),
            &[],
        )
        .unwrap();
    let reference = Value::FuncRef(Some(store.function_reference(provider, "target").unwrap()));
    store
        .call(
            instance,
            "set",
            &[vec![reference], vec![Value::FuncRef(None)]],
        )
        .unwrap();
    assert!(matches!(
        store.call(instance, "call", &[vec![], vec![]]),
        Err(Error::Trap { input_index: 1, .. })
    ));
    store
        .call(instance, "set", &[vec![reference], vec![reference]])
        .unwrap();
    assert_eq!(
        store.call(instance, "call", &[vec![], vec![]]).unwrap(),
        [vec![Value::I32(42)], vec![Value::I32(42)]]
    );
}

#[test]
fn bottom_reference_parameters_reject_non_null_host_values_before_writes() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(1).unwrap();
    let module = wasm(
        r#"(module
        (global $g (export "global") (mut i32) (i32.const 0))
        (tag (export "tag"))
        (func (export "extern") (param (ref noextern)) i32.const 1 global.set $g)
        (func (export "nullable_extern") (param (ref null noextern)) i32.const 2 global.set $g)
        (func (export "exception") (param (ref noexn)) i32.const 3 global.set $g)
        (func (export "nullable_exception") (param (ref null noexn)) i32.const 4 global.set $g))"#,
    );
    let instance = store.instantiate(&module, &[]).unwrap();
    let external = Value::ExternRef(Some(store.external_reference(42_u32).unwrap()));
    let exception = Value::ExnRef(Some(
        store.exception_reference(instance, "tag", &[]).unwrap(),
    ));
    for (export, value) in [
        ("extern", external),
        ("nullable_extern", external),
        ("exception", exception),
        ("nullable_exception", exception),
        ("extern", Value::ExternRef(None)),
        ("exception", Value::ExnRef(None)),
    ] {
        assert!(
            matches!(
                store.call(instance, export, &[vec![value]]),
                Err(Error::Unsupported(_))
            ),
            "{export}"
        );
        assert_eq!(
            store.get_global(instance, "global").unwrap(),
            [Value::I32(0)]
        );
    }
    store
        .call(instance, "nullable_extern", &[vec![Value::ExternRef(None)]])
        .unwrap();
    assert_eq!(
        store.get_global(instance, "global").unwrap(),
        [Value::I32(2)]
    );
    store
        .call(instance, "nullable_exception", &[vec![Value::ExnRef(None)]])
        .unwrap();
    assert_eq!(
        store.get_global(instance, "global").unwrap(),
        [Value::I32(4)]
    );
}
