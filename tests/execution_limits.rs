use metal_wasm::{Error, Kernel, KernelLimits, Store, Value};
use std::sync::Mutex;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn wasm(source: &str) -> Vec<u8> {
    wat::parse_str(source).unwrap()
}

fn limits(fuel: Option<u64>) -> KernelLimits {
    KernelLimits {
        fuel,
        ..KernelLimits::default()
    }
}

fn exhausted<T: std::fmt::Debug>(result: Result<T, Error>, lane: usize) {
    assert!(
        matches!(
            result,
            Err(Error::Trap {
                input_index,
                reason: "fuel exhausted"
            }) if input_index == lane
        ),
        "{result:?}"
    );
}

#[test]
fn fuel_charges_entries_and_executed_loop_iterations() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = wasm(
        r#"(module
        (func (export "calculate") (param i32) (result i32) (local i32)
            local.get 0 if
                loop
                    local.get 1 i32.const 1 i32.add local.tee 1
                    local.get 0 i32.lt_u br_if 0
                end
            end local.get 1))"#,
    );
    let kernel = Kernel::new_with_limits(&bytes, "calculate", limits(Some(4))).unwrap();
    assert_eq!(kernel.run(&[0, 1, 3]).unwrap(), [0, 1, 3]);
    exhausted(kernel.run(&[0, 4]), 1);
    assert_eq!(kernel.run(&[3]).unwrap(), [3]);
    exhausted(
        Kernel::new_with_limits(&bytes, "calculate", limits(Some(0)))
            .unwrap()
            .run(&[0]),
        0,
    );
    assert_eq!(
        Kernel::new_with_limits(&bytes, "calculate", limits(None))
            .unwrap()
            .run(&[100])
            .unwrap(),
        [100]
    );
}

#[test]
fn infinite_loops_terminate_and_preserve_each_lanes_writes() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new_with_limits(3, limits(Some(5))).unwrap();
    let instance = store
        .instantiate(
            &wasm(
                r#"(module
        (global $g (export "counter") (mut i32) (i32.const 0))
        (func (export "run") (param i32) (result i32)
            loop
                global.get $g i32.const 1 i32.add global.set $g
                local.get 0 br_if 0
            end global.get $g))"#,
            ),
            &[],
        )
        .unwrap();
    exhausted(store.call_i32(instance, "run", &[0, 1, 1]), 1);
    assert_eq!(
        store.get_global(instance, "counter").unwrap(),
        [Value::I32(1), Value::I32(4), Value::I32(4)]
    );
    assert_eq!(
        store.call_i32(instance, "run", &[0, 0, 0]).unwrap(),
        [2, 5, 5]
    );
    let nested = wasm(
        r#"(module (func (export "calculate") (param i32) (result i32)
        loop loop br 1 end end i32.const 0))"#,
    );
    exhausted(
        Kernel::new_with_limits(&nested, "calculate", limits(Some(8)))
            .unwrap()
            .run(&[0]),
        0,
    );
}

#[test]
fn recursive_and_all_tail_call_forms_share_fuel() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    for call in [
        "call $again",
        "return_call $again",
        "i32.const 0 return_call_indirect (type $t)",
        "ref.func $again return_call_ref $t",
    ] {
        let bytes = wasm(&format!(
            r#"(module
            (type $t (func (param i32) (result i32)))
            (table 1 funcref) (elem (i32.const 0) $again)
            (func $again (export "calculate") (type $t) local.get 0 {call}))"#
        ));
        exhausted(
            Kernel::new_with_limits(&bytes, "calculate", limits(Some(7)))
                .unwrap()
                .run(&[0]),
            0,
        );
    }
}

#[test]
fn starts_and_linked_functions_share_the_dispatch_budget() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let provider = wasm(
        r#"(module
        (func $start) (start $start)
        (func (export "f") (result i32) i32.const 42))"#,
    );
    let root = wasm(
        r#"(module
        (func $f (import "p" "f") (result i32))
        (func $start) (start $start)
        (func (export "calculate") (param i32) (result i32) call $f))"#,
    );
    let kernel = Kernel::new_with_imports_and_limits(
        &root,
        "calculate",
        &[("p", &provider)],
        limits(Some(4)),
    )
    .unwrap();
    assert_eq!(kernel.run(&[0]).unwrap(), [42]);
    exhausted(
        Kernel::new_with_imports_and_limits(
            &root,
            "calculate",
            &[("p", &provider)],
            limits(Some(3)),
        )
        .unwrap()
        .run(&[0]),
        0,
    );
    let mut store = Store::new_with_limits(1, limits(Some(2))).unwrap();
    let p = store.instantiate(&provider, &[]).unwrap();
    let instance = store.instantiate(&root, &[("p", p)]).unwrap();
    assert_eq!(store.call_i32(instance, "calculate", &[0]).unwrap(), [42]);
    assert_eq!(store.call_i32(instance, "calculate", &[0]).unwrap(), [42]);
}

#[test]
fn exhausted_start_preserves_imported_state_and_is_not_catchable() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut store = Store::new_with_limits(1, limits(Some(4))).unwrap();
    let provider = store
        .instantiate(
            &wasm(
                r#"(module
        (global (export "counter") (mut i32) (i32.const 0)))"#,
            ),
            &[],
        )
        .unwrap();
    exhausted(
        store.instantiate(
            &wasm(
                r#"(module
        (global $g (import "p" "counter") (mut i32))
        (func $start loop global.get $g i32.const 1 i32.add global.set $g br 0 end)
        (start $start))"#,
            ),
            &[("p", provider)],
        ),
        0,
    );
    assert_eq!(
        store.get_global(provider, "counter").unwrap(),
        [Value::I32(3)]
    );
    let bytes = wasm(
        r#"(module
        (func (export "calculate") (param i32) (result i32)
            block $caught
                try_table (catch_all $caught) loop br 0 end end
            end i32.const 99))"#,
    );
    exhausted(
        Kernel::new_with_limits(&bytes, "calculate", limits(Some(4)))
            .unwrap()
            .run(&[0]),
        0,
    );
}
