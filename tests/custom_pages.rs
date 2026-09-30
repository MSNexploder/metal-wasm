use metal_wasm::{Error, Kernel, KernelLimits, Store, Value};

#[test]
fn kernel_byte_pages_use_exact_simd_and_host_io_bounds() {
    for address_type in ["i32", "i64"] {
        let memory_type = if address_type == "i64" { "i64" } else { "" };
        let wasm = wat::parse_str(format!(
            r#"(module
            (memory (export "memory") {memory_type} 17 21 (pagesize 1))
            (data ({address_type}.const 0) "X")
            (func $start {address_type}.const 0 i32.const 85 i32.store8)
            (start $start)
            (func (export "f") (param v128) (result v128 {address_type})
                {address_type}.const 1 local.get 0 v128.store
                {address_type}.const 1 v128.load memory.size)
            (func (export "oob") (param v128)
                {address_type}.const 2 local.get 0 v128.store))"#
        ))
        .unwrap();
        let limits = KernelLimits {
            memory_bytes: 21,
            ..KernelLimits::default()
        };
        let kernel = Kernel::new_with_imports_and_limits(&wasm, "f", &[], limits).unwrap();
        let vectors = [[0x12; 16], [0x98; 16]];
        let inputs: Vec<_> = vectors.iter().map(|&v| vec![Value::V128(v)]).collect();
        let memory_inputs = [vec![6], vec![7]];
        let (results, bytes) = kernel
            .run_values_with_memory_io(&inputs, "memory", 0, &memory_inputs, 0..17)
            .unwrap();
        for lane in 0..2 {
            let pages = if address_type == "i64" {
                Value::I64(17)
            } else {
                Value::I32(17)
            };
            assert_eq!(results[lane], [Value::V128(vectors[lane]), pages]);
            assert_eq!(bytes[lane][0], memory_inputs[lane][0]);
            assert_eq!(&bytes[lane][1..], &vectors[lane]);
        }
        assert!(matches!(
            kernel.run_values_with_memory(&inputs, "memory", 0..18),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            kernel.run_values_with_memory_io(&inputs, "memory", 17, &memory_inputs, 0..0),
            Err(Error::Unsupported(_))
        ));
        let oob = Kernel::new_with_imports_and_limits(&wasm, "oob", &[], limits).unwrap();
        assert!(matches!(
            oob.run_values(&inputs),
            Err(Error::Trap {
                input_index: 0,
                reason: "out of bounds memory access"
            })
        ));
    }
}

#[test]
fn imported_mixed_pages_share_a_byte_budget_and_current_sizes() {
    let provider = wat::parse_str(
        r#"(module
        (memory $small (export "small") i64 0 8 (pagesize 1))
        (memory $large (export "large") 0 1)
        (func (export "grow_small") (param i32) (result i64)
            local.get 0 i64.extend_i32_u memory.grow $small))"#,
    )
    .unwrap();
    let consumer = wat::parse_str(
        r#"(module
        (memory $small (import "p" "small") i64 0 8 (pagesize 1))
        (memory $large (import "p" "large") 0 1)
        (export "small" (memory $small))
        (export "large" (memory $large))
        (func (export "grow_large") (param i32) (result i32)
            local.get 0 memory.grow $large))"#,
    )
    .unwrap();
    let mut store = Store::new_with_limits(
        2,
        KernelLimits {
            memory_bytes: 65_538,
            memory_growth_pages: 65_536,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    let p = store.instantiate(&provider, &[]).unwrap();
    let c = store.instantiate(&consumer, &[("p", p)]).unwrap();
    assert_eq!(
        store
            .call(p, "grow_small", &[vec![Value::I32(2)], vec![Value::I32(3)]])
            .unwrap(),
        [vec![Value::I64(0)], vec![Value::I64(0)]]
    );
    assert_eq!(
        store.call_i32(c, "grow_large", &[1, 1]).unwrap(),
        [0, u32::MAX]
    );
    assert_eq!(
        store
            .call(p, "grow_small", &[vec![Value::I32(1)], vec![Value::I32(1)]])
            .unwrap(),
        [vec![Value::I64(u64::MAX)], vec![Value::I64(3)]]
    );
    assert_eq!(
        store.read_memory(c, "small", 0..2).unwrap(),
        [vec![0, 0], vec![0, 0]]
    );
    let bytes = [vec![11, 12], vec![21, 22]];
    store.write_memory(c, "small", 0, &bytes).unwrap();
    assert!(matches!(
        store.write_memory(c, "small", 2, &[vec![9], vec![9]]),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(store.read_memory(p, "small", 0..2).unwrap(), bytes);
    assert!(matches!(
        store.read_memory(c, "large", 0..1),
        Err(Error::Unsupported(_))
    ));
    let grown_import =
        wat::parse_str(r#"(module (memory (import "p" "small") i64 2 8 (pagesize 1)))"#).unwrap();
    store.instantiate(&grown_import, &[("p", p)]).unwrap();
}

#[test]
fn byte_page_bulk_operations_preserve_state_on_traps() {
    let wasm = wat::parse_str(
        r#"(module
        (memory $small (export "small") 20 24 (pagesize 1))
        (memory $large (export "large") 1 1)
        (data $data "ABCDEF")
        (func (export "init") (param i32 i32 i32)
            local.get 0 local.get 1 local.get 2 memory.init $small $data)
        (func (export "fill") (param i32 i32 i32)
            local.get 0 local.get 1 local.get 2 memory.fill $small)
        (func (export "copy") (param i32 i32 i32)
            local.get 0 local.get 1 local.get 2 memory.copy $large $small)
        (func (export "drop") data.drop $data))"#,
    )
    .unwrap();
    let mut store = Store::new(1).unwrap();
    let instance = store.instantiate(&wasm, &[]).unwrap();
    let mut call = |export, args: &[u32]| {
        store.call(
            instance,
            export,
            &[args.iter().copied().map(Value::I32).collect()],
        )
    };
    call("init", &[14, 0, 6]).unwrap();
    assert!(matches!(
        call("fill", &[19, 7, 2]),
        Err(Error::Trap {
            input_index: 0,
            reason: "out of bounds memory access"
        })
    ));
    assert!(matches!(
        call("copy", &[0, 19, 2]),
        Err(Error::Trap {
            input_index: 0,
            reason: "out of bounds memory access"
        })
    ));
    call("copy", &[65_530, 14, 6]).unwrap();
    call("drop", &[]).unwrap();
    call("init", &[20, 0, 0]).unwrap();
    assert!(matches!(
        call("init", &[20, 0, 1]),
        Err(Error::Trap {
            input_index: 0,
            reason: "out of bounds memory access"
        })
    ));
    assert_eq!(
        store.read_memory(instance, "small", 14..20).unwrap(),
        [b"ABCDEF".to_vec()]
    );
    assert_eq!(
        store
            .read_memory(instance, "large", 65_530..65_536)
            .unwrap(),
        [b"ABCDEF".to_vec()]
    );
    assert_eq!(
        store.read_memory(instance, "large", 0..2).unwrap(),
        [vec![0, 0]]
    );
}

#[test]
fn active_data_checks_byte_page_bounds_during_initialization() {
    for address_type in ["i32", "i64"] {
        let memory_type = if address_type == "i64" { "i64" } else { "" };
        let wasm = wat::parse_str(format!(
            r#"(module
            (memory {memory_type} 3 (pagesize 1))
            (data ({address_type}.const 3) "x")
            (func (export "f")))"#
        ))
        .unwrap();
        let kernel = Kernel::new(&wasm, "f").unwrap();
        assert!(matches!(
            kernel.run_values(&[vec![]]),
            Err(Error::Trap {
                input_index: 0,
                reason: "out of bounds memory access"
            })
        ));
        let mut store = Store::new(1).unwrap();
        assert!(matches!(
            store.instantiate(&wasm, &[]),
            Err(Error::Trap {
                input_index: 0,
                reason: "out of bounds memory access"
            })
        ));
    }
}
