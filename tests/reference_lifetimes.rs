use metal_wasm::{Error, Kernel, KernelLimits, Store, Value};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn wasm(source: &str) -> Vec<u8> {
    wat::parse_str(source).unwrap()
}

struct DropProbe(Arc<AtomicUsize>);
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn releasing_external_aliases_preserves_wasm_roots_in_every_lane() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(2).unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (global $g (mut externref) (ref.null extern))
        (table $t 1 externref)
        (func (export "hold") (param externref)
            local.get 0 global.set $g i32.const 0 local.get 0 table.set $t)
        (func (export "get") (result externref) global.get $g)
        (func (export "clear_global") ref.null extern global.set $g)
        (func (export "clear_table") (param i32)
            local.get 0 if i32.const 0 ref.null extern table.set $t end))"#,
            ),
            &[],
        )
        .unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let old = store.external_reference(DropProbe(drops.clone())).unwrap();
    let internal = store.internalize_reference(Some(old)).unwrap();
    let rows = vec![vec![Value::ExternRef(Some(old))]; 2];
    store.call(instance, "hold", &rows).unwrap();
    store.release_reference(internal).unwrap();
    assert!(store.external_reference_value::<DropProbe>(old).is_err());
    assert!(store.internalize_reference(Some(old)).is_err());
    assert!(store.externalize_reference(internal).is_err());
    assert!(
        store
            .release_reference(Value::ExternRef(Some(old)))
            .is_err()
    );
    assert!(matches!(
        store.call(instance, "hold", &rows),
        Err(Error::Unsupported(_))
    ));
    store.collect_garbage().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    let returned = store.call(instance, "get", &[vec![], vec![]]).unwrap();
    assert_eq!(returned[0], returned[1]);
    let Value::ExternRef(Some(fresh)) = returned[0][0] else {
        panic!("missing external")
    };
    assert_ne!(fresh, old);
    assert!(store.external_reference_value::<DropProbe>(fresh).is_ok());
    store
        .call(instance, "clear_global", &[vec![], vec![]])
        .unwrap();
    store
        .call(
            instance,
            "clear_table",
            &[vec![Value::I32(1)], vec![Value::I32(0)]],
        )
        .unwrap();
    store
        .release_reference(Value::ExternRef(Some(fresh)))
        .unwrap();
    assert_eq!(
        drops.load(Ordering::SeqCst),
        0,
        "lane 1 still holds the value"
    );
    store
        .call(
            instance,
            "clear_table",
            &[vec![Value::I32(0)], vec![Value::I32(1)]],
        )
        .unwrap();
    store.collect_garbage().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    for _ in 0..8 {
        let next = store.external_reference(DropProbe(drops.clone())).unwrap();
        assert_ne!(next, old);
        assert!(store.external_reference_value::<DropProbe>(old).is_err());
        store
            .release_reference(Value::ExternRef(Some(next)))
            .unwrap();
    }
    assert_eq!(drops.load(Ordering::SeqCst), 9);
}

#[test]
fn gc_leases_reopen_with_a_new_generation_and_do_not_retain_dead_objects() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new_with_limits(
        1,
        KernelLimits {
            gc_heap_words: 16,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (type $s (struct (field i32)))
        (global $g (export "saved") (mut (ref null $s)) (ref.null $s))
        (func (export "create") (param i32) (result (ref $s))
            local.get 0 struct.new $s global.set $g global.get $g ref.as_non_null)
        (func (export "read") (param (ref $s)) (result i32) local.get 0 struct.get $s 0)
        (func (export "clear") ref.null $s global.set $g))"#,
            ),
            &[],
        )
        .unwrap();
    let old = store
        .call(instance, "create", &[vec![Value::I32(42)]])
        .unwrap()[0][0];
    let external = store.externalize_reference(old).unwrap().unwrap();
    store
        .release_reference(Value::ExternRef(Some(external)))
        .unwrap();
    assert!(store.externalize_reference(old).is_err());
    assert!(store.internalize_reference(Some(external)).is_err());
    assert!(store.call(instance, "read", &[vec![old]]).is_err());
    store.collect_garbage().unwrap();
    let fresh = store.get_global(instance, "saved").unwrap()[0];
    assert_ne!(fresh, old);
    assert_eq!(
        store.call(instance, "read", &[vec![fresh]]).unwrap(),
        [vec![Value::I32(42)]]
    );
    store.call(instance, "clear", &[vec![]]).unwrap();
    store.release_reference(fresh).unwrap();
    for value in 0..64 {
        let next = store
            .call(instance, "create", &[vec![Value::I32(value)]])
            .unwrap()[0][0];
        assert_ne!(next, old);
        assert_eq!(
            store.call(instance, "read", &[vec![next]]).unwrap(),
            [vec![Value::I32(value)]]
        );
        store.call(instance, "clear", &[vec![]]).unwrap();
        store.release_reference(next).unwrap();
    }
    assert!(store.call(instance, "read", &[vec![old]]).is_err());
}

#[test]
fn retained_host_exception_owns_children_after_their_public_leases_end() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(1).unwrap();
    let instance = store.instantiate(&wasm(r#"(module
        (tag $leaf (export "leaf") (param externref))
        (tag $outer (export "outer") (param exnref))
        (func (export "read") (param exnref) (result externref)
            block $leaf_payload (result externref)
                try_table (catch $leaf $leaf_payload)
                    block $outer_payload (result exnref)
                        try_table (catch $outer $outer_payload) local.get 0 throw_ref end unreachable
                    end throw_ref
                end unreachable
            end))"#), &[]).unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let external = store.external_reference(DropProbe(drops.clone())).unwrap();
    let leaf = store
        .exception_reference(instance, "leaf", &[Value::ExternRef(Some(external))])
        .unwrap();
    let outer = store
        .exception_reference(instance, "outer", &[Value::ExnRef(Some(leaf))])
        .unwrap();
    store
        .release_reference(Value::ExternRef(Some(external)))
        .unwrap();
    store.release_reference(Value::ExnRef(Some(leaf))).unwrap();
    store.collect_garbage().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert!(store.exception_payload(leaf).is_err());
    let Value::ExnRef(Some(reopened_leaf)) = store.exception_payload(outer).unwrap()[0] else {
        panic!("missing child")
    };
    assert_ne!(reopened_leaf, leaf);
    let Value::ExternRef(Some(reopened_external)) =
        store.exception_payload(reopened_leaf).unwrap()[0]
    else {
        panic!("missing payload")
    };
    assert_ne!(reopened_external, external);
    assert!(
        store
            .external_reference_value::<DropProbe>(reopened_external)
            .is_ok()
    );
    assert_eq!(
        store
            .call(instance, "read", &[vec![Value::ExnRef(Some(outer))]])
            .unwrap(),
        [vec![Value::ExternRef(Some(reopened_external))]]
    );
    store
        .release_reference(Value::ExnRef(Some(reopened_leaf)))
        .unwrap();
    store
        .release_reference(Value::ExternRef(Some(reopened_external)))
        .unwrap();
    store.release_reference(Value::ExnRef(Some(outer))).unwrap();
    store.collect_garbage().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn runtime_reclaims_unreachable_mixed_cycles_inside_and_between_calls() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = wasm(include_str!("../examples/workloads/lifetime_cycles.wat"));
    let limits = KernelLimits {
        exception_records: 2,
        gc_heap_words: 24,
        ..KernelLimits::default()
    };
    let mut store = Store::new_with_limits(2, limits).unwrap();
    let instance = store.instantiate(&bytes, &[]).unwrap();
    store.call_i32(instance, "init", &[7, 11]).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &bytes).unwrap();
    let mut cpu = wasmtime::Store::new(&engine, ());
    let mut steps = Vec::new();
    for seed in [7, 11] {
        let instance = wasmtime::Instance::new(&mut cpu, &module, &[]).unwrap();
        instance
            .get_typed_func::<i32, i32>(&mut cpu, "init")
            .unwrap()
            .call(&mut cpu, seed)
            .unwrap();
        steps.push(
            instance
                .get_typed_func::<i32, i32>(&mut cpu, "step")
                .unwrap(),
        );
    }
    for round in 0..12 {
        let inputs = [round, round + 1];
        let expected: Vec<_> = steps
            .iter()
            .zip(inputs)
            .map(|(f, x)| f.call(&mut cpu, x as i32).unwrap() as u32)
            .collect();
        assert_eq!(store.call_i32(instance, "step", &inputs).unwrap(), expected);
        store.collect_garbage().unwrap();
    }
    let kernel = Kernel::new_with_limits(&bytes, "step", limits).unwrap();
    let mut expected = Vec::new();
    for input in [0, 1, 42] {
        let fresh = wasmtime::Instance::new(&mut cpu, &module, &[]).unwrap();
        expected.push(
            fresh
                .get_typed_func::<i32, i32>(&mut cpu, "step")
                .unwrap()
                .call(&mut cpu, input)
                .unwrap() as u32,
        );
    }
    assert_eq!(kernel.run(&[0, 1, 42]).unwrap(), expected);
}

#[test]
fn release_checks_ownership_and_accepts_only_reference_values() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new(1).unwrap();
    let other = Store::new(1).unwrap();
    let foreign = other.external_reference(42u32).unwrap();
    assert!(
        store
            .release_reference(Value::ExternRef(Some(foreign)))
            .is_err()
    );
    assert!(other.external_reference_value::<u32>(foreign).is_ok());
    assert!(store.release_reference(Value::I32(0)).is_err());
    for value in [
        Value::ExternRef(None),
        Value::ExnRef(None),
        Value::GcRef(None),
        Value::FuncRef(None),
        Value::I31(42),
    ] {
        store.release_reference(value).unwrap();
    }
    let instance = store
        .instantiate(&wasm("(module (func (export \"f\")))"), &[])
        .unwrap();
    let function = store.function_reference(instance, "f").unwrap();
    assert!(
        store
            .release_reference(Value::FuncRef(Some(function)))
            .is_err()
    );
    store.collect_garbage().unwrap();
    store.call(instance, "f", &[vec![]]).unwrap();
}

#[test]
fn exception_slots_reuse_holes_without_changing_live_payloads_or_reviving_handles() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new_with_limits(
        1,
        KernelLimits {
            exception_records: 2,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (tag $e (param i32)) (table $t 3 exnref)
        (func $capture (export "capture") (param i32 i32) (result exnref) (local exnref)
            block $caught (result exnref)
                try_table (catch_all_ref $caught) local.get 1 throw $e end unreachable
            end local.set 2
            local.get 0 local.get 2 table.set $t local.get 2)
        (func (export "capture_then_trap") (param i32 i32)
            local.get 0 local.get 1 call $capture drop unreachable)
        (func (export "clear") (param i32) local.get 0 ref.null exn table.set $t)
        (func (export "get") (param i32) (result exnref) local.get 0 table.get $t))"#,
            ),
            &[],
        )
        .unwrap();
    let first = store
        .call(instance, "capture", &[vec![Value::I32(0), Value::I32(11)]])
        .unwrap()[0][0];
    let second = store
        .call(instance, "capture", &[vec![Value::I32(1), Value::I32(22)]])
        .unwrap()[0][0];
    store.release_reference(first).unwrap();
    assert!(matches!(
        store.call(instance, "capture", &[vec![Value::I32(2), Value::I32(33)]]),
        Err(Error::Trap {
            input_index: 0,
            reason: "exception record limit exceeded"
        })
    ));
    store
        .call(instance, "clear", &[vec![Value::I32(0)]])
        .unwrap();
    assert!(matches!(
        store.call(
            instance,
            "capture_then_trap",
            &[vec![Value::I32(2), Value::I32(33)]]
        ),
        Err(Error::Trap {
            reason: "unreachable executed",
            ..
        })
    ));
    let third = store.call(instance, "get", &[vec![Value::I32(2)]]).unwrap()[0][0];
    assert_ne!(third, first);
    let Value::ExnRef(Some(old)) = first else {
        panic!("missing first exception")
    };
    assert!(store.exception_payload(old).is_err());
    store
        .instantiate(&wasm("(module (tag (param v128 v128)))"), &[])
        .unwrap();
    for (value, expected) in [(second, 22), (third, 33)] {
        let Value::ExnRef(Some(reference)) = value else {
            panic!("missing live exception")
        };
        assert_eq!(
            store.exception_payload(reference).unwrap(),
            [Value::I32(expected)]
        );
    }
    assert_eq!(
        store.call(instance, "get", &[vec![Value::I32(1)]]).unwrap(),
        [vec![second]]
    );
    assert_eq!(
        store.call(instance, "get", &[vec![Value::I32(2)]]).unwrap(),
        [vec![third]]
    );
}

#[test]
fn numeric_values_that_match_reference_words_are_not_collector_roots() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = wasm(
        r#"(module
        (tag $e)
        (type $s (struct (field i32)))
        (func (export "run") (param i32) (result i64)
            (local $slot i64) (local $i i32)
            i64.const 1 local.set $slot
            i64.const 0x300000001
            loop
                i32.const 0 struct.new $s drop
                block $caught try_table (catch $e $caught) throw $e end end
                local.get $i i32.const 1 i32.add local.tee $i local.get 0 i32.lt_u br_if 0
            end))"#,
    );
    let kernel = Kernel::new_with_limits(
        &bytes,
        "run",
        KernelLimits {
            exception_records: 1,
            gc_heap_words: 4,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    assert_eq!(
        kernel.run_values(&[vec![Value::I32(100)]]).unwrap(),
        [vec![Value::I64(0x300000001)]]
    );
}

#[test]
fn exception_operand_and_block_parameter_survive_collection_in_nested_calls() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = wasm(
        r#"(module
        (tag $e (param i32))
        (func $capture (param i32) (result exnref)
            block $caught (result exnref)
                try_table (catch_all_ref $caught) local.get 0 throw $e end unreachable
            end)
        (func $churn (local i32)
            loop
                block $caught try_table (catch_all $caught) i32.const 99 throw $e end end
                local.get 0 i32.const 1 i32.add local.tee 0 i32.const 20 i32.lt_u br_if 0
            end)
        (func (export "calculate") (param i32) (result i32)
            block $done (result i32)
                try_table (catch $e $done)
                    local.get 0 call $capture
                    call $churn
                    block (param exnref) (result exnref) call $churn end
                    throw_ref
                end unreachable
            end))"#,
    );
    let kernel = Kernel::new_with_limits(
        &bytes,
        "calculate",
        KernelLimits {
            exception_records: 2,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    assert_eq!(kernel.run(&[0, 42, u32::MAX]).unwrap(), [0, 42, u32::MAX]);
}

#[test]
fn active_and_declared_element_segments_release_initializer_roots() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut expressions = String::new();
    for i in 0..4 {
        expressions.push_str(&format!("(struct.new $s (i32.const {}))", 4 * i + 3));
    }
    let bytes = wasm(&format!(
        r#"(module
        (type $s (struct (field i32))) (type $a (array i32))
        (table $t 4 (ref null $s))
        (elem $active (i32.const 0) (ref $s) {expressions})
        (elem $decl declare (ref $s) (struct.new $s (i32.const 99)))
        (func (export "get") (param i32) (result i32) local.get 0 table.get $t struct.get $s 0)
        (func (export "clear_and_allocate") (result i32)
            i32.const 0 ref.null $s i32.const 4 table.fill $t
            i32.const 7 i32.const 14 array.new $a array.len)
        (func (export "calculate") (param i32) (result i32)
            i32.const 0 ref.null $s i32.const 4 table.fill $t
            i32.const 7 i32.const 14 array.new $a array.len)
        (func (export "init_active") i32.const 0 i32.const 0 i32.const 1 table.init $t $active)
        (func (export "init_declared") i32.const 0 i32.const 0 i32.const 1 table.init $t $decl))"#
    ));
    let mut store = Store::new_with_limits(
        1,
        KernelLimits {
            gc_heap_words: 24,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    let instance = store.instantiate(&bytes, &[]).unwrap();
    for (index, expected) in [3, 7, 11, 15].into_iter().enumerate() {
        assert_eq!(
            store.call_i32(instance, "get", &[index as u32]).unwrap(),
            [expected]
        );
    }
    for export in ["init_active", "init_declared"] {
        assert!(matches!(
            store.call(instance, export, &[vec![]]),
            Err(Error::Trap {
                reason: "out of bounds table access",
                ..
            })
        ));
    }
    assert_eq!(
        store
            .call(instance, "clear_and_allocate", &[vec![]])
            .unwrap(),
        [vec![Value::I32(14)]]
    );
    let kernel = Kernel::new_with_limits(
        &bytes,
        "calculate",
        KernelLimits {
            gc_heap_words: 24,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    assert_eq!(kernel.run(&[0]).unwrap(), [14]);
}

#[test]
fn consumed_exception_block_parameters_stop_retaining_records() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = wasm(
        r#"(module
        (tag $e)
        (func (export "calculate") (param i32) (result i32)
            block $capture (result exnref)
                try_table (catch_all_ref $capture) throw $e end unreachable
            end
            block (param exnref)
                drop
                block $caught try_table (catch $e $caught) throw $e end end
            end local.get 0))"#,
    );
    let kernel = Kernel::new_with_limits(
        &bytes,
        "calculate",
        KernelLimits {
            exception_records: 1,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    assert_eq!(kernel.run(&[42]).unwrap(), [42]);
}
