use std::error::Error;
use std::fs;

use metal_wasm::{Error as WasmError, Kernel};

fn matching_trap(reason: &str, error: &wasmtime::Error) -> bool {
    let Some(trap) = error.downcast_ref::<wasmtime::Trap>() else {
        return false;
    };
    match reason {
        "integer divide by zero" => *trap == wasmtime::Trap::IntegerDivisionByZero,
        "integer overflow" => *trap == wasmtime::Trap::IntegerOverflow,
        "unreachable executed" => *trap == wasmtime::Trap::UnreachableCodeReached,
        "out of bounds memory access" => *trap == wasmtime::Trap::MemoryOutOfBounds,
        "invalid conversion to integer" => *trap == wasmtime::Trap::BadConversionToInteger,
        "undefined or uninitialized table element" => matches!(
            trap,
            wasmtime::Trap::TableOutOfBounds | wasmtime::Trap::IndirectCallToNull
        ),
        "indirect call type mismatch" => *trap == wasmtime::Trap::BadSignature,
        "null reference" | "null function reference" => *trap == wasmtime::Trap::NullReference,
        _ => false,
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: compat MODULE.wasm EXPORT I32_INPUT ...")?;
    let export = args
        .next()
        .ok_or("usage: compat MODULE.wasm EXPORT I32_INPUT ...")?;
    let inputs: Vec<u32> = args.map(|arg| arg.parse()).collect::<Result<_, _>>()?;
    if inputs.is_empty() {
        return Err("usage: compat MODULE.wasm EXPORT I32_INPUT ...".into());
    }
    let wasm = fs::read(path)?;
    check_invocations(&wasm, &export, &inputs)
}

fn check_invocations(wasm: &[u8], export: &str, inputs: &[u32]) -> Result<(), Box<dyn Error>> {
    let kernel = match Kernel::new(wasm, export) {
        Ok(kernel) => kernel,
        Err(WasmError::Unsupported(reason)) => {
            println!("unsupported: {reason}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let batch = match kernel.run(inputs) {
        Ok(actual) => Ok(actual),
        Err(error @ (WasmError::Trap { .. } | WasmError::UncaughtException { .. })) => Err(error),
        Err(WasmError::Unsupported(reason)) => {
            println!("unsupported: {reason}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, wasm)?;
    let mut first_failure = None;
    let mut trap_count = 0;
    let mut exception_count = 0;
    for (index, &input) in inputs.iter().enumerate() {
        let mut store = wasmtime::Store::new(&engine, ());
        let expected = (|| -> wasmtime::Result<i32> {
            let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
            let function = instance.get_typed_func::<i32, i32>(&mut store, export)?;
            function.call(&mut store, input as i32)
        })();
        let actual = match &batch {
            Ok(values) => Ok(values[index]),
            Err(_) => kernel.run(&[input]).map(|values| values[0]),
        };
        match (actual, expected) {
            (Ok(actual), Ok(expected)) if actual == expected as u32 => {}
            (Err(WasmError::Trap { reason, .. }), Err(error)) if matching_trap(reason, &error) => {
                first_failure.get_or_insert((index, reason));
                trap_count += 1;
            }
            (Err(WasmError::UncaughtException { .. }), Err(error))
                if error.is::<wasmtime::ThrownException>() =>
            {
                first_failure.get_or_insert((index, "uncaught exception"));
                exception_count += 1;
            }
            (actual, expected) => {
                return Err(format!(
                    "input {input:#x}: Metal returned {actual:?}, Wasmtime returned {expected:?}"
                )
                .into());
            }
        }
    }
    let batch_failure = match batch {
        Err(WasmError::Trap {
            input_index,
            reason,
        }) => Some((input_index, reason)),
        Err(WasmError::UncaughtException { input_index }) => {
            Some((input_index, "uncaught exception"))
        }
        _ => None,
    };
    if let Some((input_index, reason)) = batch_failure {
        if first_failure != Some((input_index, reason)) {
            return Err(format!(
                "Metal batch failed at input {input_index} with {reason:?}, but individual runs first failed at {first_failure:?}"
            ).into());
        }
    }
    println!(
        "supported: {} independent invocations matched Wasmtime ({trap_count} matching traps, {exception_count} matching exceptions)",
        inputs.len(),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_batch_failures_match_individual_wasmtime_invocations() {
        let wasm = wat::parse_str(
            r#"(module
          (tag $e)
          (func (export "calculate") (param i32) (result i32)
            local.get 0 i32.const 1 i32.eq if throw $e end
            local.get 0 i32.const 2 i32.eq if unreachable end
            local.get 0 i32.const 3 i32.eq if ref.null func ref.as_non_null drop end
            local.get 0 i32.const 7 i32.add))"#,
        )
        .unwrap();
        check_invocations(&wasm, "calculate", &[0, 1, 2, 3]).unwrap();
        check_invocations(&wasm, "calculate", &[0, 2, 1]).unwrap();
    }

    #[test]
    fn reference_traps_and_exceptions_match_wasmtime_errors() {
        let engine = wasmtime::Engine::default();
        for (source, reason) in [
            (
                "(module (func (export \"f\") (param i32) (result i32) ref.null func ref.as_non_null drop i32.const 0))",
                "null reference",
            ),
            (
                "(module (type $f (func (param i32) (result i32))) (func (export \"f\") (param i32) (result i32) local.get 0 ref.null $f call_ref $f))",
                "null function reference",
            ),
            (
                "(module (tag $e) (func (export \"f\") (param i32) (result i32) throw $e))",
                "uncaught exception",
            ),
        ] {
            let wasm = wat::parse_str(source).unwrap();
            let module = wasmtime::Module::new(&engine, &wasm).unwrap();
            let mut store = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
            let function = instance
                .get_typed_func::<i32, i32>(&mut store, "f")
                .unwrap();
            let error = function.call(&mut store, 0).unwrap_err();
            if reason == "uncaught exception" {
                assert!(error.is::<wasmtime::ThrownException>());
            } else {
                assert!(matching_trap(reason, &error), "{reason}: {error}");
            }
        }
    }
}
