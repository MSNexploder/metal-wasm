use metal_wasm::{Error, Kernel, KernelLimits, Value};
use std::sync::Mutex;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn compare(wat: &str, inputs: &[u32]) {
    let wasm = wat::parse_str(wat).unwrap();
    let kernel = Kernel::new(&wasm, "calculate").unwrap();
    let actual = kernel.run(inputs).unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    for (&input, &output) in inputs.iter().zip(&actual) {
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
        let function = instance
            .get_typed_func::<i32, i32>(&mut store, "calculate")
            .unwrap();
        assert_eq!(
            output,
            function.call(&mut store, input as i32).unwrap() as u32,
            "input {input}"
        );
    }
}

#[test]
fn declared_subtypes_work_in_typed_tables_and_calls() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    compare(
        r#"(module
      (type $base (sub (func (param i32) (result i32))))
      (type $child (sub $base (func (param i32) (result i32))))
      (func $child (type $child) local.get 0 i32.const 7 i32.add)
      (table $typed 1 (ref $base) (ref.func $child))
      (elem declare func $child)
      (func (export "calculate") (param i32) (result i32)
        local.get 0 i32.const 0 table.get $typed call_ref $base
        i32.const 0 call_indirect $typed (type $base)))"#,
        &[0, 1, 9, u32::MAX],
    );
}

#[test]
fn recursive_type_groups_preserve_nominal_identity() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wat = r#"(module
      (rec (type $a (func (result i32))) (type $b (func (result i32))))
      (func $a (type $a) i32.const 5)
      (func $b (type $b) i32.const 8)
      (table 2 funcref) (elem (i32.const 0) $a $b)
      (func (export "calculate") (param i32) (result i32)
        local.get 0 call_indirect (type $a)))"#;
    let wasm = wat::parse_str(wat).unwrap();
    let kernel = Kernel::new(&wasm, "calculate").unwrap();
    assert_eq!(kernel.run(&[0]).unwrap(), [5]);
    assert!(matches!(
        kernel.run(&[1]),
        Err(Error::Trap {
            reason: "indirect call type mismatch",
            ..
        })
    ));
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    assert!(
        instance
            .get_typed_func::<i32, i32>(&mut store, "calculate")
            .unwrap()
            .call(&mut store, 1)
            .is_err()
    );
    compare(
        r#"(module
      (rec (type $recursive (func (param (ref null $recursive) i32) (result i32))))
      (func $f (type $recursive) local.get 1 i32.const 3 i32.mul)
      (elem declare func $f)
      (func (export "calculate") (param i32) (result i32)
        ref.null $recursive local.get 0 ref.func $f call_ref $recursive))"#,
        &[0, 7, u32::MAX],
    );
}

#[test]
fn public_handles_enforce_kernel_and_function_types() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (type $base (sub (func (param i32) (result i32))))
      (type $child (sub $base (func (param i32) (result i32))))
      (func $child (export "child") (type $child) local.get 0 i32.const 1 i32.add)
      (func $wrong (export "wrong") (result i32) i32.const 0)
      (elem declare func $child $wrong)
      (func (export "apply") (param (ref null $base) i32) (result i32 (ref $base))
        local.get 1 local.get 0 call_ref $base
        local.get 0 ref.as_non_null))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "apply").unwrap();
    let reference = kernel.function_reference("child").unwrap();
    let output = kernel
        .run_values(&[vec![Value::FuncRef(Some(reference)), Value::I32(41)]])
        .unwrap();
    assert_eq!(
        output,
        [vec![Value::I32(42), Value::FuncRef(Some(reference))]]
    );
    assert!(matches!(
        kernel.run_values(&[vec![Value::FuncRef(None), Value::I32(1)]]),
        Err(Error::Trap {
            reason: "null function reference",
            ..
        })
    ));
    let wrong = kernel.function_reference("wrong").unwrap();
    assert!(matches!(
        kernel.run_values(&[vec![Value::FuncRef(Some(wrong)), Value::I32(1)]]),
        Err(Error::Unsupported(_))
    ));
    let other = Kernel::new(&wasm, "apply").unwrap();
    assert!(matches!(
        other.run_values(&[vec![Value::FuncRef(Some(reference)), Value::I32(1)]]),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn linked_function_types_and_immutable_globals_accept_subtypes() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let provider = wat::parse_str(
        r#"(module
      (type $base (sub (func (param i32) (result i32))))
      (type $child (sub $base (func (param i32) (result i32))))
      (func $child (export "child") (type $child) local.get 0 i32.const 3 i32.mul)
      (global (export "reference") (ref $child) (ref.func $child)))"#,
    )
    .unwrap();
    let root = wat::parse_str(
        r#"(module
      (type $base (sub (func (param i32) (result i32))))
      (func $child (import "provider" "child") (type $base))
      (global $reference (import "provider" "reference") (ref $base))
      (func (export "calculate") (param i32) (result i32)
        local.get 0 call $child global.get $reference call_ref $base))"#,
    )
    .unwrap();
    let kernel = Kernel::new_with_imports(&root, "calculate", &[("provider", &provider)]).unwrap();
    assert_eq!(kernel.run(&[0, 1, 7]).unwrap(), [0, 9, 63]);
    let engine = wasmtime::Engine::default();
    let mut store = wasmtime::Store::new(&engine, ());
    let mut linker = wasmtime::Linker::new(&engine);
    let instance = wasmtime::Instance::new(
        &mut store,
        &wasmtime::Module::new(&engine, &provider).unwrap(),
        &[],
    )
    .unwrap();
    linker.instance(&mut store, "provider", instance).unwrap();
    let instance = linker
        .instantiate(&mut store, &wasmtime::Module::new(&engine, &root).unwrap())
        .unwrap();
    assert_eq!(
        instance
            .get_typed_func::<i32, i32>(&mut store, "calculate")
            .unwrap()
            .call(&mut store, 7)
            .unwrap(),
        63
    );
}

#[test]
fn reexports_preserve_the_underlying_function_and_global_types() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let provider = wat::parse_str(
        r#"(module
      (type $base (sub (func (param i32) (result i32))))
      (type $child (sub $base (func (param i32) (result i32))))
      (func $child (export "child") (type $child) local.get 0 i32.const 3 i32.mul)
      (global (export "reference") (ref $child) (ref.func $child)))"#,
    )
    .unwrap();
    let middle = wat::parse_str(
        r#"(module
      (type $base (sub (func (param i32) (result i32))))
      (func $child (import "provider" "child") (type $base))
      (global $reference (import "provider" "reference") (ref $base))
      (export "child" (func $child)) (export "reference" (global $reference)))"#,
    )
    .unwrap();
    let root = wat::parse_str(
        r#"(module
      (type $base (sub (func (param i32) (result i32))))
      (type $child (sub $base (func (param i32) (result i32))))
      (func $child (import "middle" "child") (type $child))
      (global $reference (import "middle" "reference") (ref $child))
      (func (export "calculate") (param i32) (result i32)
        local.get 0 call $child global.get $reference call_ref $child))"#,
    )
    .unwrap();
    let engine = wasmtime::Engine::default();
    let mut store = wasmtime::Store::new(&engine, ());
    let mut linker = wasmtime::Linker::new(&engine);
    for (name, bytes) in [("provider", &provider), ("middle", &middle)] {
        let instance = linker
            .instantiate(&mut store, &wasmtime::Module::new(&engine, bytes).unwrap())
            .unwrap();
        linker.instance(&mut store, name, instance).unwrap();
    }
    let instance = linker
        .instantiate(&mut store, &wasmtime::Module::new(&engine, &root).unwrap())
        .unwrap();
    assert_eq!(
        instance
            .get_typed_func::<i32, i32>(&mut store, "calculate")
            .unwrap()
            .call(&mut store, 7)
            .unwrap(),
        63
    );
    let kernel = Kernel::new_with_imports(
        &root,
        "calculate",
        &[("provider", &provider), ("middle", &middle)],
    )
    .unwrap();
    assert_eq!(kernel.run(&[0, 1, 7]).unwrap(), [0, 9, 63]);
}

#[test]
fn exceptions_keep_vector_payloads_and_references_across_calls() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    compare(
        r#"(module
      (tag $vector (param v128 i32)) (tag $other)
      (func $raise (param i32)
        v128.const i32x4 7 11 13 17 local.get 0 throw $vector)
      (func (export "calculate") (param i32) (result i32)
        (local $saved exnref) (local $value i32)
        block $capture (result v128 i32 exnref)
          try_table (catch_ref $vector $capture) local.get 0 call $raise end
          unreachable
        end
        local.set $saved local.set $value drop
        block $discard (result exnref)
          try_table (catch_all_ref $discard) throw $other end unreachable
        end
        drop
        block $result (result v128 i32)
          try_table (catch $vector $result) local.get $saved throw_ref end unreachable
        end
        local.set $value i32x4.extract_lane 2 local.get $value i32.add))"#,
        &[0, 1, 99, u32::MAX],
    );
}

#[test]
fn exception_limits_and_uncaught_lane_indices_are_reported() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (tag $e)
      (func (export "calculate") (param i32) (result i32) (local i32)
        loop
          block $caught
            try_table (catch $e $caught) throw $e end
          end
          local.get 1 i32.const 1 i32.add local.tee 1 local.get 0 i32.lt_u br_if 0
        end local.get 1))"#,
    )
    .unwrap();
    let kernel = Kernel::new_with_limits(
        &wasm,
        "calculate",
        KernelLimits {
            exception_records: 2,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    assert_eq!(kernel.run(&[1, 2]).unwrap(), [1, 2]);
    assert_eq!(kernel.run(&[1, 100]).unwrap(), [1, 100]);
    assert_eq!(kernel.run(&[2]).unwrap(), [2]);
    let wasm = wat::parse_str(r#"(module (tag $e) (func (export "calculate") (param i32) (result i32) local.get 0 if throw $e end i32.const 42))"#).unwrap();
    assert!(matches!(
        Kernel::new(&wasm, "calculate").unwrap().run(&[0, 0, 1]),
        Err(Error::UncaughtException { input_index: 2 })
    ));
    assert!(matches!(
        Kernel::new_with_limits(
            &wasm,
            "calculate",
            KernelLimits {
                exception_records: 0,
                ..KernelLimits::default()
            }
        ),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn provider_start_exceptions_stop_initialization() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let provider = wat::parse_str(r#"(module (tag (export "error")) (func $start throw 0) (start $start) (func (export "f") (result i32) i32.const 9))"#).unwrap();
    let root = wat::parse_str(r#"(module (func $f (import "provider" "f") (result i32)) (func (export "calculate") (param i32) (result i32) call $f))"#).unwrap();
    let kernel = Kernel::new_with_imports(&root, "calculate", &[("provider", &provider)]).unwrap();
    assert!(matches!(
        kernel.run(&[0]),
        Err(Error::UncaughtException { input_index: 0 })
    ));
}

#[test]
fn exception_handlers_can_branch_to_loop_parameters() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    compare(
        r#"(module
      (tag $retry (param i32))
      (func (export "calculate") (param i32) (result i32) (local i32)
        local.get 0
        loop $again (param i32) (result i32)
          local.tee 1
          if (result i32)
            try_table (result i32) (catch $retry $again)
              local.get 1 i32.const 1 i32.sub throw $retry
            end
          else i32.const 42 end
        end))"#,
        &[0, 1, 7, 31],
    );
    let wasm = wat::parse_str(
        r#"(module (func (export "calculate") (param i32) (result i32)
      block $done try_table (catch_all $done) ref.null exn throw_ref end end i32.const 42))"#,
    )
    .unwrap();
    assert!(matches!(
        Kernel::new(&wasm, "calculate").unwrap().run(&[0]),
        Err(Error::Trap {
            reason: "null reference",
            ..
        })
    ));
}

#[test]
fn function_subtyping_preserves_parameter_and_result_variance() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    compare(
        r#"(module
      (type $base_value (sub (func (result i32))))
      (type $child_value (sub $base_value (func (result i32))))
      (type $base_step (sub (func (param (ref $child_value)) (result (ref $base_value)))))
      (type $child_step (sub $base_step (func (param (ref $base_value)) (result (ref $child_value)))))
      (func $value (type $child_value) i32.const 123)
      (func $step (type $child_step) ref.func $value)
      (elem declare func $step $value)
      (func (export "calculate") (param i32) (result i32)
        ref.func $value ref.func $step call_ref $base_step call_ref $base_value
        local.get 0 i32.add))"#,
        &[0, 7, u32::MAX],
    );
    let wasm = wat::parse_str(
        r#"(module
      (type $f (func (result i32))) (func $f (type $f) i32.const 7)
      (elem declare func $f)
      (func (export "calculate") (param i32) (result i32) (local (ref $f))
        ref.func $f local.set 1 local.get 1 call_ref $f))"#,
    )
    .unwrap();
    assert_eq!(
        Kernel::new(&wasm, "calculate")
            .unwrap()
            .run(&[0, 1])
            .unwrap(),
        [7, 7]
    );
    let invalid = wat::parse_str(r#"(module (type $f (func)) (func (export "calculate") (param i32) (result i32) (local (ref $f)) local.get 1 drop i32.const 0))"#).unwrap();
    assert!(matches!(
        Kernel::new(&invalid, "calculate"),
        Err(Error::InvalidWasm(_))
    ));
}

#[test]
fn host_function_handles_are_reachable_through_mutable_tables() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    for imported in [false, true] {
        let table = if imported {
            "(table (import \"provider\" \"table\") 1 funcref)"
        } else {
            "(table 1 funcref)"
        };
        // Exports make handles available to the host without an in-module ref.func.
        let wasm = wat::parse_str(format!(
            r#"(module
          (type $target (func (param i32) (result i32)))
          {table}
          (func (export "add") (type $target) local.get 0 i32.const 7 i32.add)
          (func (export "multiply") (type $target) local.get 0 i32.const 3 i32.mul)
          (func (export "apply") (param (ref $target) i32) (result i32)
            i32.const 0 local.get 0 table.set 0
            local.get 1 i32.const 0 call_indirect (type $target)))"#
        ))
        .unwrap();
        let provider = wat::parse_str("(module (table (export \"table\") 1 funcref))").unwrap();
        let imports: Vec<(&str, &[u8])> = if imported {
            vec![("provider", &provider)]
        } else {
            vec![]
        };
        let kernel = Kernel::new_with_imports(&wasm, "apply", &imports).unwrap();
        let add = kernel.function_reference("add").unwrap();
        let multiply = kernel.function_reference("multiply").unwrap();
        let output = kernel
            .run_values(&[
                vec![Value::FuncRef(Some(add)), Value::I32(10)],
                vec![Value::FuncRef(Some(multiply)), Value::I32(10)],
            ])
            .unwrap();
        assert_eq!(output, [vec![Value::I32(17)], vec![Value::I32(30)]]);
    }
}

#[test]
fn typed_select_accepts_nonnullable_and_nullable_function_references() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    compare(
        r#"(module
      (type $target (func (param i32) (result i32)))
      (func $add (type $target) local.get 0 i32.const 7 i32.add)
      (func $multiply (type $target) local.get 0 i32.const 3 i32.mul)
      (elem declare func $add $multiply)
      (func (export "calculate") (param i32) (result i32)
        local.get 0
        ref.func $add ref.func $multiply local.get 0 i32.const 1 i32.and
        select (result (ref $target)) call_ref $target
        ref.func $add ref.null $target i32.const 1
        select (result (ref null $target)) call_ref $target))"#,
        &[0, 1, 7, 8, u32::MAX],
    );
}

#[test]
fn function_reference_casts_and_branches_match_nominal_subtypes() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (type $base (sub (func (result i32))))
      (type $child (sub $base (func (result i32))))
      (type $other (func (result i32)))
      (func $child (type $child) i32.const 41)
      (func $other (type $other) i32.const 17)
      (table 3 funcref) (elem (i32.const 0) $child $other)
      (func (export "test") (param i32) (result i32)
        local.get 0 table.get 0 ref.test (ref $child))
      (func (export "nullable") (param i32) (result i32)
        local.get 0 table.get 0 ref.test (ref null $child))
      (func (export "cast") (param i32) (result i32)
        local.get 0 table.get 0 ref.cast (ref $child) call_ref $child)
      (func (export "branch") (param i32) (result i32)
        block $success (result (ref $child))
          local.get 0 table.get 0 br_on_cast $success (ref null func) (ref $child)
          drop i32.const 17 return
        end drop i32.const 41)
      (func (export "branch_fail") (param i32) (result i32)
        block $failure (result funcref)
          local.get 0 table.get 0 br_on_cast_fail $failure (ref null func) (ref $child)
          drop i32.const 41 return
        end drop i32.const 17))"#,
    )
    .unwrap();
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    for export in ["test", "nullable", "cast", "branch", "branch_fail"] {
        let kernel = Kernel::new(&wasm, export).unwrap();
        for input in 0..3 {
            let mut store = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let expected = instance
                .get_typed_func::<i32, i32>(&mut store, export)
                .unwrap()
                .call(&mut store, input);
            match (kernel.run(&[input as u32]), expected) {
                (Ok(actual), Ok(expected)) => assert_eq!(actual, [expected as u32]),
                (
                    Err(Error::Trap {
                        reason: "cast failure",
                        ..
                    }),
                    Err(error),
                ) => assert_eq!(
                    error.downcast_ref::<wasmtime::Trap>(),
                    Some(&wasmtime::Trap::CastFailure)
                ),
                (actual, expected) => panic!("{export}({input}): {actual:?}, {expected:?}"),
            }
        }
    }
}

#[test]
fn casts_recognize_host_handles_without_reachable_function_bodies() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (type $base (sub (func))) (type $child (sub $base (func)))
      (func (export "child") (type $child) unreachable)
      (func (export "test") (param funcref) (result i32)
        local.get 0 ref.test (ref $child)))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "test").unwrap();
    let child = kernel.function_reference("child").unwrap();
    assert_eq!(
        kernel
            .run_values(&[
                vec![Value::FuncRef(Some(child))],
                vec![Value::FuncRef(None)]
            ])
            .unwrap(),
        [vec![Value::I32(1)], vec![Value::I32(0)]]
    );
}

#[test]
fn external_handles_survive_tables_globals_bulk_operations_and_results() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (table $a 2 4 externref) (table $b 2 4 externref)
      (global $saved (mut externref) (ref.null extern))
      (elem $nulls externref (ref.null extern))
      (func (export "run") (param externref) (result externref externref i32)
        local.get 0 global.set $saved
        i32.const 0 local.get 0 table.set $a
        i32.const 1 local.get 0 i32.const 1 table.fill $a
        i32.const 0 i32.const 0 i32.const 2 table.copy $b $a
        local.get 0 i32.const 1 table.grow $b drop
        i32.const 1 i32.const 0 i32.const 1 table.init $b $nulls
        elem.drop $nulls
        i32.const 2 table.get $b global.get $saved
        i32.const 1 table.get $b ref.is_null))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "run").unwrap();
    let handle = kernel.external_reference(String::from("retained")).unwrap();
    assert_eq!(
        &*kernel.external_reference_value::<String>(handle).unwrap(),
        "retained"
    );
    assert!(kernel.external_reference_value::<u32>(handle).is_err());
    assert_eq!(
        kernel
            .run_values(&[
                vec![Value::ExternRef(Some(handle))],
                vec![Value::ExternRef(None)]
            ])
            .unwrap(),
        [
            vec![
                Value::ExternRef(Some(handle)),
                Value::ExternRef(Some(handle)),
                Value::I32(1)
            ],
            vec![
                Value::ExternRef(None),
                Value::ExternRef(None),
                Value::I32(1)
            ]
        ]
    );
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let external = wasmtime::ExternRef::new(&mut store, String::from("retained")).unwrap();
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let function = instance.get_func(&mut store, "run").unwrap();
    let mut results = [
        wasmtime::Val::ExternRef(None),
        wasmtime::Val::ExternRef(None),
        wasmtime::Val::I32(0),
    ];
    function
        .call(
            &mut store,
            &[wasmtime::Val::ExternRef(Some(external))],
            &mut results,
        )
        .unwrap();
    for result in &results[..2] {
        let reference = result.externref().unwrap().unwrap();
        assert_eq!(
            reference
                .data(&store)
                .unwrap()
                .unwrap()
                .downcast_ref::<String>()
                .unwrap(),
            "retained"
        );
    }
    assert_eq!(results[2].i32(), Some(1));
    let other = Kernel::new(&wasm, "run").unwrap();
    assert!(
        other
            .run_values(&[vec![Value::ExternRef(Some(handle))]])
            .is_err()
    );
    assert!(other.external_reference_value::<String>(handle).is_err());
}

#[test]
fn external_casts_enforce_nullability_and_bottom_types() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (func (export "run") (param externref) (result externref i32 i32)
        local.get 0 ref.cast (ref extern)
        local.get 0 ref.test (ref extern)
        local.get 0 ref.test (ref null noextern)))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "run").unwrap();
    let handle = kernel.external_reference(42u32).unwrap();
    assert_eq!(
        kernel
            .run_values(&[vec![Value::ExternRef(Some(handle))]])
            .unwrap(),
        [vec![
            Value::ExternRef(Some(handle)),
            Value::I32(1),
            Value::I32(0)
        ]]
    );
    assert!(matches!(
        kernel.run_values(&[vec![Value::ExternRef(None)]]),
        Err(Error::Trap {
            reason: "cast failure",
            ..
        })
    ));
    let wasm = wat::parse_str(
        r#"(module (func (export "run") (param (ref extern)) (result externref) local.get 0))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "run").unwrap();
    assert!(kernel.run_values(&[vec![Value::ExternRef(None)]]).is_err());
}

#[test]
fn exception_tables_retain_records_and_support_casts() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let source = r#"(module
      (tag $e (param i32))
      (global $saved (mut exnref) (ref.null exn))
      (table $table 2 4 exnref)
      (table $wide i64 2 exnref)
      (elem $nulls exnref (ref.null exn))
      (func (export "calculate") (param i32) (result i32)
        block $capture (result exnref)
          try_table (catch_all_ref $capture) local.get 0 throw $e end unreachable
        end
        global.set $saved
        i32.const 0 global.get $saved table.set $table
        i32.const 1 global.get $saved i32.const 1 table.fill $table
        i32.const 0 i32.const 1 i32.const 1 table.copy $table $table
        global.get $saved i32.const 1 table.grow $table
        i32.const 2 i32.ne if unreachable end
        i64.const 1 i32.const 2 i32.const 1 table.copy $wide $table
        i32.const 1 i32.const 0 i32.const 1 table.init $table $nulls
        elem.drop $nulls
        i32.const 1 i32.const 0 i32.const 0 table.init $table $nulls
        i32.const 1 table.get $table ref.is_null i32.eqz if unreachable end
        block $done (result i32)
          try_table (catch $e $done)
            i64.const 1 table.get $wide ref.cast (ref exn) throw_ref
          end unreachable
        end)
      (func (export "after_drop") (param i32) (result i32)
        elem.drop $nulls
        i32.const 0 i32.const 0 i32.const 1 table.init $table $nulls
        i32.const 0))"#;
    compare(source, &[0, 7, u32::MAX]);
    let wasm = wat::parse_str(source).unwrap();
    let kernel = Kernel::new(&wasm, "after_drop").unwrap();
    assert!(matches!(
        kernel.run(&[0]),
        Err(Error::Trap {
            reason: "out of bounds table access",
            ..
        })
    ));
}

#[test]
fn public_exceptions_preserve_vector_and_external_payloads_across_runs() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (tag $e (export "e") (param i32 v128 externref))
      (memory (export "memory") 1)
      (func (export "run") (param exnref i32 externref) (result exnref i32)
        (local $saved exnref)
        local.get 0 ref.is_null
        if
          block $capture (result i32 v128 externref exnref)
            try_table (catch_ref $e $capture)
              local.get 1 v128.const i32x4 7 11 13 17 local.get 2 throw $e
            end unreachable
          end
          local.set $saved drop drop drop
        else local.get 0 local.set $saved end
        local.get $saved
        block $payload (result i32 v128 externref)
          try_table (catch $e $payload) local.get $saved throw_ref end unreachable
        end drop drop))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "run").unwrap();
    let external = kernel.external_reference(String::from("payload")).unwrap();
    let result = kernel
        .run_values(&[vec![
            Value::ExnRef(None),
            Value::I32(98),
            Value::ExternRef(Some(external)),
        ]])
        .unwrap();
    let Value::ExnRef(Some(reference)) = result[0][0] else {
        panic!("missing exception");
    };
    let mut vector = [0u8; 16];
    for (lane, n) in [7u32, 11, 13, 17].iter().enumerate() {
        vector[lane * 4..lane * 4 + 4].copy_from_slice(&n.to_le_bytes());
    }
    let payload = vec![
        Value::I32(98),
        Value::V128(vector),
        Value::ExternRef(Some(external)),
    ];
    assert_eq!(kernel.exception_payload(reference).unwrap(), payload);
    assert_eq!(result[0][1], Value::I32(98));
    let input = vec![
        Value::ExnRef(Some(reference)),
        Value::I32(0),
        Value::ExternRef(None),
    ];
    assert_eq!(
        kernel.run_values(std::slice::from_ref(&input)).unwrap()[0],
        result[0]
    );
    // Odd memory input widths exercise alignment of the appended exception arena.
    // Enough lanes also force consecutive dispatches with the same retained handle.
    let (returned, memory) = kernel
        .run_values_with_memory_io(
            &vec![input; 1_100],
            "memory",
            0,
            &vec![vec![1, 2, 3]; 1_100],
            0..3,
        )
        .unwrap();
    assert!(returned.iter().all(|row| row == &result[0]));
    assert!(memory.iter().all(|row| row == &[1, 2, 3]));
    let created = kernel.exception_reference("e", &payload).unwrap();
    assert_eq!(kernel.exception_payload(created).unwrap(), payload);
    assert_eq!(
        kernel
            .run_values(&[vec![
                Value::ExnRef(Some(created)),
                Value::I32(0),
                Value::ExternRef(None)
            ]])
            .unwrap()[0][1],
        Value::I32(98)
    );
    assert!(kernel.exception_reference("e", &[Value::I32(1)]).is_err());
    let other = Kernel::new(&wasm, "run").unwrap();
    assert!(other.exception_payload(reference).is_err());
    assert!(
        other
            .run_values(&[vec![
                Value::ExnRef(Some(reference)),
                Value::I32(0),
                Value::ExternRef(None)
            ]])
            .is_err()
    );
    assert!(other.exception_reference("e", &payload).is_err());
}

#[test]
fn nested_public_exceptions_preserve_aliases_and_bound_imported_records() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(r#"(module
      (tag $leaf (export "leaf") (param i64))
      (tag $outer (export "outer") (param exnref exnref))
      (func (export "run") (param exnref i64) (result exnref i64)
        (local $saved exnref)
        local.get 0 ref.is_null
        if
          block $capture (result exnref)
            try_table (catch_all_ref $capture) local.get 1 throw $leaf end unreachable
          end local.set $saved
          block $capture (result exnref)
            try_table (catch_all_ref $capture) local.get $saved local.get $saved throw $outer end unreachable
          end local.set $saved
        else local.get 0 local.set $saved end
        local.get $saved
        block $unpack (result exnref exnref)
          try_table (catch $outer $unpack) local.get $saved throw_ref end unreachable
        end drop local.set $saved
        block $value (result i64)
          try_table (catch $leaf $value) local.get $saved throw_ref end unreachable
        end))"#).unwrap();
    let kernel = Kernel::new_with_limits(
        &wasm,
        "run",
        KernelLimits {
            exception_records: 2,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    let result = kernel
        .run_values(&[vec![Value::ExnRef(None), Value::I64(u64::MAX)]])
        .unwrap();
    let Value::ExnRef(Some(outer)) = result[0][0] else {
        panic!("missing outer");
    };
    let payload = kernel.exception_payload(outer).unwrap();
    assert_eq!(payload[0], payload[1]);
    let Value::ExnRef(Some(leaf)) = payload[0] else {
        panic!("missing leaf");
    };
    assert_eq!(
        kernel.exception_payload(leaf).unwrap(),
        [Value::I64(u64::MAX)]
    );
    assert_eq!(
        kernel
            .run_values(&[vec![Value::ExnRef(Some(outer)), Value::I64(0)]])
            .unwrap(),
        result
    );
    let deeper = kernel
        .exception_reference(
            "outer",
            &[Value::ExnRef(Some(outer)), Value::ExnRef(Some(outer))],
        )
        .unwrap();
    assert!(matches!(
        kernel.run_values(&[
            vec![Value::ExnRef(None), Value::I64(0)],
            vec![Value::ExnRef(Some(deeper)), Value::I64(0)]
        ]),
        Err(Error::Trap {
            input_index: 1,
            reason: "exception record limit exceeded"
        })
    ));
}

#[test]
fn exception_payload_function_handles_reach_indirect_calls() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (type $f (func (result i32)))
      (tag $e (export "e") (param funcref))
      (table 1 funcref)
      (func (export "target") (type $f) i32.const 42)
      (func (export "run") (param exnref) (result i32) (local funcref)
        block $unpack (result funcref)
          try_table (catch $e $unpack) local.get 0 throw_ref end unreachable
        end local.set 1
        i32.const 0 local.get 1 table.set 0
        i32.const 0 call_indirect (type $f)))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "run").unwrap();
    let target = kernel.function_reference("target").unwrap();
    let exception = kernel
        .exception_reference("e", &[Value::FuncRef(Some(target))])
        .unwrap();
    assert_eq!(
        kernel
            .run_values(&[vec![Value::ExnRef(Some(exception))]])
            .unwrap(),
        [vec![Value::I32(42)]]
    );
}

#[test]
fn linked_external_and_exception_tables_preserve_tag_aliases() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let provider = wat::parse_str(
        r#"(module
      (tag $e (export "e") (param f32 f64 externref))
      (table (export "external") 1 externref)
      (table (export "exceptions") 1 exnref)
      (func (export "stash") (param externref exnref)
        i32.const 0 local.get 0 table.set 0
        i32.const 0 local.get 1 table.set 1)
      (func (export "raise") i32.const 0 table.get 1 throw_ref))"#,
    )
    .unwrap();
    let root = wat::parse_str(
        r#"(module
      (import "p" "e" (tag $e (param f32 f64 externref)))
      (import "p" "external" (table $external 1 externref))
      (import "p" "exceptions" (table $exceptions 1 exnref))
      (import "p" "stash" (func $stash (param externref exnref)))
      (import "p" "raise" (func $raise))
      (export "alias" (tag $e))
      (func (export "run") (param externref exnref)
          (result f32 f64 externref exnref externref)
        local.get 0 local.get 1 call $stash
        block $caught (result f32 f64 externref exnref)
          try_table (catch_ref $e $caught) call $raise end unreachable
        end
        i32.const 0 table.get $external))"#,
    )
    .unwrap();
    let kernel = Kernel::new_with_imports(&root, "run", &[("p", &provider)]).unwrap();
    let external = kernel.external_reference(String::from("opaque")).unwrap();
    let payload = [
        Value::F32(0x7fc1_2345),
        Value::F64(0xfff8_0000_1234_5678),
        Value::ExternRef(Some(external)),
    ];
    let exception = kernel.exception_reference("alias", &payload).unwrap();
    let rows = vec![
        vec![
            Value::ExternRef(Some(external)),
            Value::ExnRef(Some(exception))
        ];
        33
    ];
    let expected = vec![
        payload[0],
        payload[1],
        payload[2],
        Value::ExnRef(Some(exception)),
        Value::ExternRef(Some(external)),
    ];
    assert_eq!(kernel.run_values(&rows).unwrap(), vec![expected; 33]);
    assert_eq!(kernel.exception_payload(exception).unwrap(), payload);
    assert_eq!(
        &*kernel.external_reference_value::<String>(external).unwrap(),
        "opaque"
    );
}

#[test]
fn external_and_exception_cast_branches_keep_the_correct_stack_values() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (tag (export "e"))
      (func (export "run") (param externref exnref) (result i32 i32)
        block $external (result i32 externref)
          i32.const 10 local.get 0
          br_on_cast $external externref (ref extern)
          drop drop i32.const 20 ref.null extern
        end drop
        block $exception (result i32 exnref)
          i32.const 30 local.get 1
          br_on_cast_fail $exception exnref (ref exn)
          drop drop i32.const 40 ref.null exn
        end drop))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "run").unwrap();
    let external = kernel.external_reference(7u32).unwrap();
    let exception = kernel.exception_reference("e", &[]).unwrap();
    assert_eq!(
        kernel
            .run_values(&[
                vec![Value::ExternRef(None), Value::ExnRef(None)],
                vec![
                    Value::ExternRef(Some(external)),
                    Value::ExnRef(Some(exception))
                ],
            ])
            .unwrap(),
        [
            vec![Value::I32(20), Value::I32(30)],
            vec![Value::I32(10), Value::I32(40)]
        ]
    );
}

#[test]
fn public_exception_input_and_output_match_wasmtime() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let wasm = wat::parse_str(
        r#"(module
      (tag $e (export "e") (param i64 externref))
      (func (export "run") (param exnref) (result i64 externref exnref)
        block $caught (result i64 externref exnref)
          try_table (catch_ref $e $caught)
            local.get 0 ref.cast (ref exn) throw_ref
          end unreachable
        end))"#,
    )
    .unwrap();
    let kernel = Kernel::new(&wasm, "run").unwrap();
    let external = kernel.external_reference(123u32).unwrap();
    let payload = [
        Value::I64(0xfedc_ba98_7654_3210),
        Value::ExternRef(Some(external)),
    ];
    let exception = kernel.exception_reference("e", &payload).unwrap();
    let actual = kernel
        .run_values(&[vec![Value::ExnRef(Some(exception))]])
        .unwrap();
    assert_eq!(
        actual,
        [vec![payload[0], payload[1], Value::ExnRef(Some(exception))]]
    );
    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let tag = instance.get_tag(&mut store, "e").unwrap();
    let exception_type = wasmtime::ExnType::from_tag_type(&tag.ty(&store)).unwrap();
    let allocator = wasmtime::ExnRefPre::new(&mut store, exception_type);
    let external = wasmtime::ExternRef::new(&mut store, 123u32).unwrap();
    let exception = wasmtime::ExnRef::new(
        &mut store,
        &allocator,
        &tag,
        &[
            wasmtime::Val::I64(0xfedc_ba98_7654_3210u64 as i64),
            wasmtime::Val::ExternRef(Some(external)),
        ],
    )
    .unwrap();
    let function = instance.get_func(&mut store, "run").unwrap();
    let mut results = [
        wasmtime::Val::I64(0),
        wasmtime::Val::ExternRef(None),
        wasmtime::Val::ExnRef(None),
    ];
    function
        .call(
            &mut store,
            &[wasmtime::Val::ExnRef(Some(exception))],
            &mut results,
        )
        .unwrap();
    assert_eq!(results[0].i64().unwrap() as u64, 0xfedc_ba98_7654_3210);
    let external = results[1].externref().unwrap().unwrap();
    assert_eq!(
        external
            .data(&store)
            .unwrap()
            .unwrap()
            .downcast_ref::<u32>(),
        Some(&123)
    );
    let returned = results[2].exnref().unwrap().unwrap();
    assert_eq!(
        returned.field(&mut store, 0).unwrap().i64(),
        results[0].i64()
    );
    let field = returned.field(&mut store, 1).unwrap();
    let field = field.externref().unwrap().unwrap();
    assert_eq!(
        field.data(&store).unwrap().unwrap().downcast_ref::<u32>(),
        Some(&123)
    );
}
