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

#[test]
fn compiled_c_cpp_numeric_rust_and_simd_rust() {
    for wasm in [
        include_bytes!("corpus/c_numeric_o0.wasm").as_slice(),
        include_bytes!("corpus/c_numeric_o3.wasm").as_slice(),
        include_bytes!("corpus/cpp_dispatch_o0.wasm").as_slice(),
        include_bytes!("corpus/cpp_dispatch_o3.wasm").as_slice(),
        include_bytes!("corpus/rust_numeric.wasm").as_slice(),
        include_bytes!("corpus/rust_simd.wasm").as_slice(),
    ] {
        check_compiled_module(wasm);
    }
}

#[test]
fn compiled_cpp_statics_and_indirect_calls_keep_per_lane_state() {
    use metal_wasm::Store;
    for wasm in [
        include_bytes!("corpus/cpp_dispatch_o0.wasm").as_slice(),
        include_bytes!("corpus/cpp_dispatch_o3.wasm").as_slice(),
    ] {
        let mut metal = Store::new(16).unwrap();
        let instance = metal.instantiate(wasm, &[]).unwrap();
        let engine = wasmtime::Engine::default();
        let module = wasmtime::Module::new(&engine, wasm).unwrap();
        let mut cpu = (0..16)
            .map(|_| {
                let mut store = wasmtime::Store::new(&engine, ());
                let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
                let function = instance
                    .get_typed_func::<i32, i32>(&mut store, "calculate")
                    .unwrap();
                (store, function)
            })
            .collect::<Vec<_>>();
        for round in 0..4u32 {
            let inputs = (0..16u32)
                .map(|i| i.wrapping_mul(0x9e3779b9) ^ round)
                .collect::<Vec<_>>();
            let actual = metal.call_i32(instance, "calculate", &inputs).unwrap();
            for (lane, ((store, function), &input)) in cpu.iter_mut().zip(&inputs).enumerate() {
                assert_eq!(
                    actual[lane],
                    function.call(store, input as i32).unwrap() as u32,
                    "round {round}, lane {lane}"
                );
            }
        }
    }
}
