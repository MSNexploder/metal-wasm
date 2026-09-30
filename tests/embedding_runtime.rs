use metal_wasm::{Error, Store, Value};

#[test]
fn host_global_writes_preserve_bits_and_alias_provider_state() {
    let provider = wat::parse_str(
        r#"(module
        (global (export "integer") (mut i64) (i64.const 0))
        (global (export "vector") (mut v128) (v128.const i32x4 0 0 0 0))
        (global (export "float") (mut f64) (f64.const 0))
        (global (export "immutable") i32 (i32.const 7)))"#,
    )
    .unwrap();
    let consumer = wat::parse_str(
        r#"(module
        (global $g (import "p" "integer") (mut i64))
        (export "alias" (global $g))
        (func (export "get") (result i64) global.get $g))"#,
    )
    .unwrap();
    let mut store = Store::new(2).unwrap();
    let p = store.instantiate(&provider, &[]).unwrap();
    let c = store.instantiate(&consumer, &[("p", p)]).unwrap();
    let integers = [Value::I64(u64::MAX), Value::I64(1 << 63)];
    store.set_global(c, "alias", &integers).unwrap();
    assert_eq!(store.get_global(p, "integer").unwrap(), integers);
    assert_eq!(
        store.call(c, "get", &[vec![], vec![]]).unwrap(),
        [vec![integers[0]], vec![integers[1]]]
    );
    let vectors = [Value::V128([0x15; 16]), Value::V128([0xf1; 16])];
    store.set_global(p, "vector", &vectors).unwrap();
    assert_eq!(store.get_global(p, "vector").unwrap(), vectors);
    let floats = [
        Value::F64(0x8000_0000_0000_0000),
        Value::F64(0x7ff0_0000_0000_1234),
    ];
    store.set_global(p, "float", &floats).unwrap();
    assert_eq!(store.get_global(p, "float").unwrap(), floats);
    for invalid in [&[Value::I64(4)][..], &[Value::I64(4), Value::I32(5)][..]] {
        assert!(matches!(
            store.set_global(c, "alias", invalid),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(store.get_global(p, "integer").unwrap(), integers);
    }
    assert!(matches!(
        store.set_global(p, "immutable", &[Value::I32(1); 2]),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        store.get_global(p, "immutable").unwrap(),
        [Value::I32(7); 2]
    );
}

#[test]
fn host_table_writes_update_indirect_calls_and_check_each_lanes_current_size() {
    let wasm = wat::parse_str(
        r#"(module
        (type $sig (func (param i32) (result i32)))
        (func $a (export "a") (type $sig) local.get 0 i32.const 3 i32.add)
        (func $b (export "b") (type $sig) local.get 0 i32.const 7 i32.mul)
        (table $t (export "table") i64 2 4 funcref)
        (elem (table $t) (i64.const 0) func $a $b)
        (func (export "call") (param i32 i64) (result i32)
            local.get 0 local.get 1 call_indirect $t (type $sig))
        (func (export "grow") (param i32) (result i32)
            ref.null func local.get 0 i64.extend_i32_u table.grow $t i32.wrap_i64))"#,
    )
    .unwrap();
    let alias = wat::parse_str(
        r#"(module
        (table $t (import "p" "table") i64 2 4 funcref)
        (export "alias" (table $t)))"#,
    )
    .unwrap();
    let mut store = Store::new(2).unwrap();
    let p = store.instantiate(&wasm, &[]).unwrap();
    let c = store.instantiate(&alias, &[("p", p)]).unwrap();
    let a = Value::FuncRef(Some(store.function_reference(p, "a").unwrap()));
    let b = Value::FuncRef(Some(store.function_reference(p, "b").unwrap()));
    assert_eq!(
        store.read_table(c, "alias", 0..2).unwrap(),
        [vec![a, b], vec![a, b]]
    );
    store
        .write_table(c, "alias", 0, &[vec![b, a], vec![a, b]])
        .unwrap();
    assert_eq!(
        store
            .call(
                p,
                "call",
                &[
                    vec![Value::I32(4), Value::I64(0)],
                    vec![Value::I32(4), Value::I64(0)]
                ]
            )
            .unwrap(),
        [vec![Value::I32(28)], vec![Value::I32(7)]]
    );
    assert_eq!(store.call_i32(p, "grow", &[1, 2]).unwrap(), [2, 2]);
    assert_eq!(store.table_size(c, "alias").unwrap(), [3, 4]);
    let before = store.read_table(c, "alias", 0..3).unwrap();
    for (offset, rows) in [
        (3, vec![vec![a], vec![b]]),
        (u64::MAX, vec![vec![], vec![]]),
        (0, vec![vec![b], vec![Value::I32(0)]]),
    ] {
        assert!(matches!(
            store.write_table(c, "alias", offset, &rows),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(store.read_table(p, "table", 0..3).unwrap(), before);
    }
    assert!(matches!(
        store.read_table(c, "alias", 0..4),
        Err(Error::Unsupported(_))
    ));
    store
        .write_table(c, "alias", 3, &[vec![], vec![a]])
        .unwrap();
    assert_eq!(
        store.read_table(c, "alias", 3..3).unwrap(),
        [vec![], vec![]]
    );
}

#[test]
fn host_reference_writes_preserve_roots_and_reject_cross_lane_values() {
    let wasm = wat::parse_str(
        r#"(module
        (type $s (struct (field i32)))
        (tag (export "tag") (param externref))
        (global (export "g") (mut (ref null $s)) (ref.null $s))
        (global (export "e") (mut exnref) (ref.null exn))
        (table (export "objects") 1 (ref null $s))
        (table (export "externals") 1 externref)
        (table (export "exceptions") 1 exnref)
        (func (export "make") (param i32) (result (ref $s)) local.get 0 struct.new $s)
        (func (export "get") (result i32) global.get 0 struct.get $s 0))"#,
    )
    .unwrap();
    let mut store = Store::new(2).unwrap();
    let instance = store.instantiate(&wasm, &[]).unwrap();
    let objects = store
        .call(
            instance,
            "make",
            &[vec![Value::I32(11)], vec![Value::I32(22)]],
        )
        .unwrap();
    let values = [objects[0][0], objects[1][0]];
    assert!(matches!(
        store.set_global(instance, "g", &[values[1], values[0]]),
        Err(Error::Unsupported(_))
    ));
    store.set_global(instance, "g", &values).unwrap();
    store.write_table(instance, "objects", 0, &objects).unwrap();
    for value in values {
        store.release_reference(value).unwrap();
    }
    assert_eq!(
        store.call(instance, "get", &[vec![], vec![]]).unwrap(),
        [vec![Value::I32(11)], vec![Value::I32(22)]]
    );
    let renewed = store.read_table(instance, "objects", 0..1).unwrap();
    assert!(matches!(
        store.write_table(instance, "objects", 0, &objects),
        Err(Error::Unsupported(_))
    ));
    store.write_table(instance, "objects", 0, &renewed).unwrap();
    let external = store.external_reference(123u32).unwrap();
    let exception = store
        .exception_reference(instance, "tag", &[Value::ExternRef(Some(external))])
        .unwrap();
    store
        .set_global(instance, "e", &[Value::ExnRef(Some(exception)); 2])
        .unwrap();
    store
        .write_table(
            instance,
            "externals",
            0,
            &vec![vec![Value::ExternRef(Some(external))]; 2],
        )
        .unwrap();
    store
        .write_table(
            instance,
            "exceptions",
            0,
            &vec![vec![Value::ExnRef(Some(exception))]; 2],
        )
        .unwrap();
    store
        .release_reference(Value::ExternRef(Some(external)))
        .unwrap();
    store
        .release_reference(Value::ExnRef(Some(exception)))
        .unwrap();
    store.collect_garbage().unwrap();
    let ex = store.read_table(instance, "exceptions", 0..1).unwrap();
    let Value::ExnRef(Some(handle)) = ex[0][0] else {
        panic!("expected exception");
    };
    let payload = store.exception_payload(handle).unwrap();
    let Value::ExternRef(Some(handle)) = payload[0] else {
        panic!("expected external");
    };
    assert_eq!(*store.external_reference_value::<u32>(handle).unwrap(), 123);
}

#[test]
fn non_null_typed_tables_reject_null_and_wrong_signatures_atomically() {
    let wasm = wat::parse_str(
        r#"(module
        (type $t (func (result i32)))
        (func $a (export "a") (type $t) i32.const 7)
        (func (export "wrong") (param i32))
        (table (export "table") 1 (ref $t) (ref.func $a)))"#,
    )
    .unwrap();
    let mut store = Store::new(2).unwrap();
    let i = store.instantiate(&wasm, &[]).unwrap();
    let a = Value::FuncRef(Some(store.function_reference(i, "a").unwrap()));
    let wrong = Value::FuncRef(Some(store.function_reference(i, "wrong").unwrap()));
    for invalid in [Value::FuncRef(None), wrong] {
        assert!(matches!(
            store.write_table(i, "table", 0, &[vec![a], vec![invalid]]),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(
            store.read_table(i, "table", 0..1).unwrap(),
            [vec![a], vec![a]]
        );
    }
}

#[test]
fn per_lane_outcomes_keep_successes_and_persistent_writes() {
    use metal_wasm::{Kernel, KernelLimits, LaneError};
    let wasm = wat::parse_str(
        r#"(module
        (tag $tag (param i32))
        (global (export "calls") (mut i32) (i32.const 0))
        (func (export "f") (param i32) (result i32)
            global.get 0 i32.const 1 i32.add global.set 0
            local.get 0 i32.const 2 i32.eq if unreachable end
            local.get 0 i32.const 3 i32.eq if local.get 0 throw $tag end
            local.get 0 i32.const 4 i32.eq if i32.const 1 i32.const 0 i32.div_u return end
            local.get 0 i32.const 6 i32.eq if loop $forever br $forever end end
            local.get 0 i32.const 7 i32.mul i32.const 3 i32.add))"#,
    )
    .unwrap();
    let limits = KernelLimits {
        fuel: Some(8),
        ..KernelLimits::default()
    };
    let kernel = Kernel::new_with_limits(&wasm, "f", limits).unwrap();
    let inputs: Vec<_> = [1, 2, 3, 4, 5, 6].map(|v| vec![Value::I32(v)]).into();
    let expected = vec![
        Ok(vec![Value::I32(10)]),
        Err(LaneError::Trap {
            reason: "unreachable executed",
        }),
        Err(LaneError::UncaughtException),
        Err(LaneError::Trap {
            reason: "integer divide by zero",
        }),
        Ok(vec![Value::I32(38)]),
        Err(LaneError::Trap {
            reason: "fuel exhausted",
        }),
    ];
    assert_eq!(kernel.run_values_outcomes(&inputs).unwrap(), expected);
    assert!(matches!(
        kernel.run_values(&inputs),
        Err(Error::Trap {
            input_index: 1,
            reason: "unreachable executed"
        })
    ));
    let mut store = Store::new_with_limits(inputs.len(), limits).unwrap();
    let instance = store.instantiate(&wasm, &[]).unwrap();
    assert_eq!(
        store.call_outcomes(instance, "f", &inputs).unwrap(),
        expected
    );
    assert_eq!(
        store.get_global(instance, "calls").unwrap(),
        [Value::I32(1); 6]
    );
    assert_eq!(
        store.call_outcomes(instance, "f", &inputs).unwrap(),
        expected
    );
    assert_eq!(
        store.get_global(instance, "calls").unwrap(),
        [Value::I32(2); 6]
    );
    let mut invalid = inputs.clone();
    invalid[4] = vec![Value::I64(1)];
    assert!(matches!(
        store.call_outcomes(instance, "f", &invalid),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(
        store.get_global(instance, "calls").unwrap(),
        [Value::I32(2); 6]
    );
    assert!(matches!(
        kernel.run_values_outcomes(&invalid),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn kernel_outcomes_continue_after_failures_across_dispatches() {
    use metal_wasm::{Kernel, LaneError};
    let wasm = wat::parse_str(
        r#"(module
        (memory 2 2)
        (func (export "f") (param i32) (result i64)
            local.get 0 i32.load i64.extend_i32_u))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "f").unwrap();
    let mut inputs = vec![vec![Value::I32(0)]; 1100];
    for lane in [0, 550, 1099] {
        inputs[lane] = vec![Value::I32(u32::MAX)];
    }
    let results = kernel.run_values_outcomes(&inputs).unwrap();
    assert_eq!(results.len(), inputs.len());
    for (lane, result) in results.into_iter().enumerate() {
        if [0, 550, 1099].contains(&lane) {
            assert_eq!(
                result,
                Err(LaneError::Trap {
                    reason: "out of bounds memory access"
                })
            );
        } else {
            assert_eq!(result, Ok(vec![Value::I64(0)]));
        }
    }
    assert!(matches!(
        kernel.run_values(&inputs),
        Err(Error::Trap { input_index: 0, .. })
    ));
}

#[test]
fn mixed_outcomes_decode_only_successful_reference_rows() {
    use metal_wasm::{Kernel, LaneError};
    let wasm = wat::parse_str(
        r#"(module
        (type $s (struct (field i32)))
        (func (export "f") (param i32) (result (ref $s) exnref v128)
            local.get 0 if unreachable end
            i32.const 77 struct.new $s ref.null exn
            v128.const i32x4 1 2 3 4))"#,
    )
    .unwrap();
    let inputs = [
        vec![Value::I32(0)],
        vec![Value::I32(1)],
        vec![Value::I32(0)],
    ];
    let kernel = Kernel::new(&wasm, "f").unwrap();
    let results = kernel.run_values_outcomes(&inputs).unwrap();
    assert_eq!(
        results[1],
        Err(LaneError::Trap {
            reason: "unreachable executed"
        })
    );
    let mut store = Store::new(3).unwrap();
    let instance = store.instantiate(&wasm, &[]).unwrap();
    let persistent = store.call_outcomes(instance, "f", &inputs).unwrap();
    assert_eq!(persistent[1], results[1]);
    for lane in [0, 2] {
        for row in [&results[lane], &persistent[lane]] {
            let row = row.as_ref().unwrap();
            assert!(matches!(row[0], Value::GcRef(Some(_))));
            assert_eq!(row[1], Value::ExnRef(None));
            let mut bytes = [0; 16];
            for (i, v) in [1u32, 2, 3, 4].iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            assert_eq!(row[2], Value::V128(bytes));
        }
        store
            .release_reference(persistent[lane].as_ref().unwrap()[0])
            .unwrap();
    }
}
