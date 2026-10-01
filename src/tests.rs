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
            &format!("(module (func (export \"f\") (param i32) (result i32) local.get 0 {op}))"),
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
    // One no-argument invocation makes the packed dispatch input smaller
    // than the shader's ulong argument, including a zero-length input.
    for width in 0..=7 {
        let bytes = [10, 11, 12, 13, 14, 15, 16][..width].to_vec();
        let (values, memories) = kernel
            .run_values_with_memory_io(&[vec![]], "mem", 2, &[bytes], 2..9)
            .unwrap();
        assert_eq!(values, [vec![Value::I32(if width >= 2 { 11 } else { 0 })]]);
        let mut expected = vec![0; 7];
        expected[..width].copy_from_slice(&[10, 11, 12, 13, 14, 15, 16][..width]);
        assert_eq!(memories, [expected]);
    }
}

#[test]
fn shared_kernel_runs_independent_batches_on_host_threads() {
    let wasm = wat::parse_str(
        r#"(module
            (global $g (mut i32) (i32.const 1))
            (func (export "f") (param i32) (result i32)
                global.get $g local.get 0 i32.add
                global.set $g global.get $g))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "f").unwrap();
    std::thread::scope(|scope| {
        for thread in 0..4 {
            let kernel = &kernel;
            scope.spawn(move || {
                for run in 0..3 {
                    let inputs: Vec<_> =
                        (0..8).map(|lane| thread * 100 + run * 10 + lane).collect();
                    let expected: Vec<_> = inputs.iter().map(|input| input + 1).collect();
                    assert_eq!(kernel.run(&inputs).unwrap(), expected);
                }
            });
        }
    });
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
    let wasm = wat::parse_str(
        "(module (func (export \"f\") (param i64) (result i64) local.get 0 i64.const 0 i64.div_u))",
    )
    .unwrap();
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
                (Value::V128(a), wasmtime::Val::V128(b)) => u128::from_le_bytes(a) == b.as_u128(),
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
        "add", "sub", "mul", "div", "min", "max", "copysign", "eq", "ne", "lt", "gt", "le", "ge",
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
        "add", "sub", "mul", "div", "min", "max", "copysign", "eq", "ne", "lt", "gt", "le", "ge",
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
