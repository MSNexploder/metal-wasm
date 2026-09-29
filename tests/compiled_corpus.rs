use metal_wasm::{Error, Kernel};

fn check_compiled_module(wasm: &[u8]) {
    let kernel = Kernel::new(wasm, "calculate").unwrap();
    let inputs: Vec<u32> = (0..64).chain([u32::MAX, 0x8000_0000]).collect();
    let actual = kernel.run(&inputs).unwrap();

    let engine = wasmtime::Engine::default();
    let module = wasmtime::Module::new(&engine, wasm).unwrap();
    for (&input, &result) in inputs.iter().zip(&actual) {
        // A fresh instance matches the GPU's independent instance per lane.
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
        let function = instance
            .get_typed_func::<i32, i32>(&mut store, "calculate")
            .unwrap();
        let expected = function.call(&mut store, input as i32).unwrap() as u32;
        assert_eq!(result, expected, "input {input:#x}");
    }
}

#[test]
fn compiled_scalar_rust() {
    check_compiled_module(include_bytes!("corpus/scalar.wasm"));
}

#[test]
fn compiled_memory_rust() {
    check_compiled_module(include_bytes!("corpus/memory.wasm"));
}

#[test]
fn compiled_host_import_reports_unsupported_dependency() {
    let wasm = include_bytes!("corpus/host_import.wasm");
    assert!(matches!(
        Kernel::new(wasm, "calculate"),
        Err(Error::Unsupported(reason)) if reason == "missing Wasm provider \"env\""
    ));
}
