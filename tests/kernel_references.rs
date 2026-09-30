use metal_wasm::{Error, Kernel, Value};
use std::sync::Arc;

#[test]
fn kernel_external_leases_release_values_and_keep_old_handles_invalid() {
    let wasm = wat::parse_str(
        r#"(module (func (export "f") (param externref) (result externref) local.get 0))"#,
    )
    .unwrap();
    let mut kernel = Kernel::new(&wasm, "f").unwrap();
    let payload = Arc::new(123u32);
    let weak = Arc::downgrade(&payload);
    let handle = kernel.external_reference(payload).unwrap();
    let alias = kernel.internalize_reference(Some(handle)).unwrap();
    assert_eq!(
        kernel
            .run_values(&[vec![Value::ExternRef(Some(handle))]])
            .unwrap(),
        [vec![Value::ExternRef(Some(handle))]]
    );
    kernel.release_reference(alias).unwrap();
    assert!(weak.upgrade().is_none());
    assert!(matches!(
        kernel.external_reference_value::<Arc<u32>>(handle),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        kernel.release_reference(Value::ExternRef(Some(handle))),
        Err(Error::Unsupported(_))
    ));
    for value in 0..256u32 {
        let next = kernel.external_reference(value).unwrap();
        assert!(matches!(
            kernel.run_values(&[vec![Value::ExternRef(Some(handle))]]),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(
            *kernel.external_reference_value::<u32>(next).unwrap(),
            value
        );
        kernel
            .release_reference(Value::ExternRef(Some(next)))
            .unwrap();
    }
    let held = kernel.external_reference(7u32).unwrap();
    let strong = kernel.external_reference_value::<u32>(held).unwrap();
    kernel
        .release_reference(Value::ExternRef(Some(held)))
        .unwrap();
    assert_eq!(*strong, 7);
}

#[test]
fn nested_exception_leases_survive_slot_reuse_and_renew_payload_handles() {
    let wasm = wat::parse_str(
        r#"(module
        (tag (export "child") (param i32))
        (tag (export "parent") (param exnref))
        (func (export "f") (param exnref) (result exnref) local.get 0))"#,
    )
    .unwrap();
    let mut kernel = Kernel::new(&wasm, "f").unwrap();
    let unused = kernel
        .exception_reference("child", &[Value::I32(0)])
        .unwrap();
    let child = kernel
        .exception_reference("child", &[Value::I32(42)])
        .unwrap();
    kernel
        .release_reference(Value::ExnRef(Some(unused)))
        .unwrap();
    let parent = kernel
        .exception_reference("parent", &[Value::ExnRef(Some(child))])
        .unwrap();
    kernel
        .release_reference(Value::ExnRef(Some(child)))
        .unwrap();
    assert!(matches!(
        kernel.exception_payload(unused),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        kernel.exception_payload(child),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        kernel
            .run_values(&[vec![Value::ExnRef(Some(parent))]])
            .unwrap(),
        [vec![Value::ExnRef(Some(parent))]]
    );
    let payload = kernel.exception_payload(parent).unwrap();
    let Value::ExnRef(Some(renewed)) = payload[0] else {
        panic!("expected child");
    };
    assert_ne!(renewed, child);
    assert_eq!(kernel.exception_payload(renewed).unwrap(), [Value::I32(42)]);
    kernel
        .release_reference(Value::ExnRef(Some(parent)))
        .unwrap();
    assert_eq!(kernel.exception_payload(renewed).unwrap(), [Value::I32(42)]);
    kernel
        .release_reference(Value::ExnRef(Some(renewed)))
        .unwrap();
    kernel.collect_garbage().unwrap();
}

#[test]
fn kernel_gc_exception_cycles_reclaim_host_payloads_and_conversion_aliases() {
    let wasm = wat::parse_str(
        r#"(module
        (type $node (struct (field (mut exnref)) (field externref)))
        (tag $tag (param (ref $node)))
        (func (export "f") (param externref) (result (ref $node))
            (local $node (ref null $node)) (local $exception exnref)
            ref.null exn local.get 0 struct.new $node local.set $node
            (block (result exnref)
                (try_table (catch_all_ref 0) local.get $node ref.as_non_null throw $tag)
                unreachable)
            local.set $exception
            local.get $node local.get $exception struct.set $node 0
            local.get $node ref.as_non_null))"#,
    )
    .unwrap();
    let mut kernel = Kernel::new(&wasm, "f").unwrap();
    let mut stale = None;
    for _ in 0..32 {
        let payload = Arc::new(());
        let weak = Arc::downgrade(&payload);
        let external = kernel.external_reference(payload).unwrap();
        let row = kernel
            .run_values(&[vec![Value::ExternRef(Some(external))]])
            .unwrap()
            .remove(0);
        let Value::GcRef(Some(object)) = row[0] else {
            panic!("expected object");
        };
        let converted = kernel.externalize_reference(row[0]).unwrap().unwrap();
        if let Some((old_object, old_external)) = stale {
            assert!(matches!(
                kernel.gc_reference_kind(old_object),
                Err(Error::Unsupported(_))
            ));
            assert!(matches!(
                kernel.run_values(&[vec![Value::ExternRef(Some(old_external))]]),
                Err(Error::Unsupported(_))
            ));
        }
        stale = Some((object, converted));
        kernel
            .release_reference(Value::ExternRef(Some(external)))
            .unwrap();
        assert!(
            weak.upgrade().is_some(),
            "snapshot must keep its payload alive"
        );
        kernel
            .release_reference(Value::ExternRef(Some(converted)))
            .unwrap();
        assert!(
            weak.upgrade().is_none(),
            "unleased mixed cycles must be reclaimed"
        );
        assert!(matches!(
            kernel.gc_reference_kind(object),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            kernel.internalize_reference(Some(converted)),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn discarded_batch_snapshots_do_not_leave_host_leases() {
    let wasm = wat::parse_str(
        r#"(module
        (type $s (struct (field externref)))
        (memory 1 1)
        (func (export "f") (param externref i32) (result (ref $s))
            local.get 1 if unreachable end local.get 0 struct.new $s))"#,
    )
    .unwrap();
    let mut kernel = Kernel::new(&wasm, "f").unwrap();
    let payload = Arc::new(());
    let weak = Arc::downgrade(&payload);
    let external = kernel.external_reference(payload).unwrap();
    let mut inputs = vec![vec![Value::ExternRef(Some(external)), Value::I32(0)]; 1000];
    inputs[999][1] = Value::I32(1);
    assert!(matches!(
        kernel.run_values(&inputs),
        Err(Error::Trap {
            input_index: 999,
            reason: "unreachable executed"
        })
    ));
    kernel
        .release_reference(Value::ExternRef(Some(external)))
        .unwrap();
    assert!(weak.upgrade().is_none());
}

#[test]
fn kernel_release_checks_owner_and_value_kind() {
    let wasm = wat::parse_str(
        r#"(module (func (export "f") (param externref) (result externref) local.get 0))"#,
    )
    .unwrap();
    let mut first = Kernel::new(&wasm, "f").unwrap();
    let other = Kernel::new(&wasm, "f").unwrap();
    let external = other.external_reference(7u32).unwrap();
    assert!(matches!(
        first.release_reference(Value::ExternRef(Some(external))),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(*other.external_reference_value::<u32>(external).unwrap(), 7);
    assert!(matches!(
        first.release_reference(Value::I32(7)),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        first.release_reference(Value::FuncRef(Some(first.function_reference("f").unwrap()))),
        Err(Error::Unsupported(_))
    ));
    for value in [
        Value::FuncRef(None),
        Value::ExternRef(None),
        Value::GcRef(None),
        Value::ExnRef(None),
        Value::I31(42),
    ] {
        first.release_reference(value).unwrap();
    }
}

#[test]
fn recycled_guest_exception_slots_preserve_forward_payload_references() {
    use metal_wasm::KernelLimits;
    let wasm = wat::parse_str(
        r#"(module
        (tag $int (param i32))
        (tag $parent (param exnref))
        (func $child (param i32) (result exnref)
            (block (result exnref)
                (try_table (catch_all_ref 0) local.get 0 throw $int)
                unreachable))
        (func $wrap (param exnref) (result exnref)
            (block (result exnref)
                (try_table (catch_all_ref 0) local.get 0 throw $parent)
                unreachable))
        (func (export "f") (result exnref exnref)
            (local $a exnref) (local $b exnref) (local $c exnref)
            i32.const 10 call $child local.set $a
            i32.const 20 call $child local.set $b
            i32.const 30 call $child local.set $c
            ref.null exn local.set $a
            local.get $c call $wrap local.set $a
            local.get $a local.get $b))"#,
    )
    .unwrap();
    let kernel = Kernel::new_with_limits(
        &wasm,
        "f",
        KernelLimits {
            exception_records: 3,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    let row = kernel.run_values(&[vec![]]).unwrap().remove(0);
    let Value::ExnRef(Some(parent)) = row[0] else {
        panic!("expected parent");
    };
    let payload = kernel.exception_payload(parent).unwrap();
    let Value::ExnRef(Some(child)) = payload[0] else {
        panic!("expected child");
    };
    assert_eq!(kernel.exception_payload(child).unwrap(), [Value::I32(30)]);
    let Value::ExnRef(Some(other)) = row[1] else {
        panic!("expected other child");
    };
    assert_eq!(kernel.exception_payload(other).unwrap(), [Value::I32(20)]);
}
