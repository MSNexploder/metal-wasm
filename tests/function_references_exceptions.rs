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
    assert!(matches!(
        kernel.run(&[1, 3]),
        Err(Error::Trap {
            input_index: 1,
            reason: "exception record limit exceeded"
        })
    ));
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
