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
    let kernel = match Kernel::new(&wasm, &export) {
        Ok(kernel) => kernel,
        Err(WasmError::Unsupported(reason)) => {
            println!("unsupported: {reason}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let batch = match kernel.run(&inputs) {
        Ok(actual) => Ok(actual),
        Err(error @ WasmError::Trap { .. }) => Err(error),
        Err(WasmError::Unsupported(reason)) => {
            println!("unsupported: {reason}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, &wasm)?;
    let mut first_trap = None;
    let mut trap_count = 0;
    for (index, &input) in inputs.iter().enumerate() {
        let mut store = wasmtime::Store::new(&engine, ());
        let expected = (|| -> wasmtime::Result<i32> {
            let instance = wasmtime::Instance::new(&mut store, &module, &[])?;
            let function = instance.get_typed_func::<i32, i32>(&mut store, &export)?;
            function.call(&mut store, input as i32)
        })();
        let actual = match &batch {
            Ok(values) => Ok(values[index]),
            Err(_) => kernel.run(&[input]).map(|values| values[0]),
        };
        match (actual, expected) {
            (Ok(actual), Ok(expected)) if actual == expected as u32 => {}
            (Err(WasmError::Trap { reason, .. }), Err(error)) if matching_trap(reason, &error) => {
                first_trap.get_or_insert((index, reason));
                trap_count += 1;
            }
            (actual, expected) => {
                return Err(format!(
                    "input {input:#x}: Metal returned {actual:?}, Wasmtime returned {expected:?}"
                )
                .into());
            }
        }
    }
    if let Err(WasmError::Trap {
        input_index,
        reason,
    }) = batch
    {
        if first_trap != Some((input_index, reason)) {
            return Err(format!(
                "Metal batch trapped at input {input_index} with {reason:?}, but individual runs first trapped at {first_trap:?}"
            )
            .into());
        }
    }
    println!(
        "supported: {} independent invocations matched Wasmtime ({trap_count} matching traps)",
        inputs.len(),
    );
    Ok(())
}
