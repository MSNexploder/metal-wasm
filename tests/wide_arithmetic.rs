use metal_wasm::{Kernel, Value};

#[test]
fn wide_results_match_rust_and_wasmtime_across_lanes() {
    let buffer = wast::parser::ParseBuffer::new(include_str!("spec/wide-arithmetic.wast")).unwrap();
    let mut script: wast::Wast = wast::parser::parse(&buffer).unwrap();
    let wast::WastDirective::Module(mut module) = script.directives.remove(0) else {
        panic!("expected operation module");
    };
    let wasm = module.encode().unwrap();
    let mut config = wasmtime::Config::new();
    config.wasm_wide_arithmetic(true);
    let engine = wasmtime::Engine::new(&config).unwrap();
    let module = wasmtime::Module::new(&engine, &wasm).unwrap();
    let mut store = wasmtime::Store::new(&engine, ());
    let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
    let edges = [
        0,
        1,
        2,
        u32::MAX as u64,
        1 << 32,
        i64::MAX as u64,
        1 << 63,
        u64::MAX,
    ];
    for operation in [
        "i64.add128",
        "i64.sub128",
        "i64.mul_wide_s",
        "i64.mul_wide_u",
    ] {
        let wide_pair = operation.ends_with("128");
        let mut inputs = Vec::new();
        for &a in &edges {
            for &b in &edges {
                if wide_pair {
                    for &c in &edges {
                        for &d in &edges {
                            inputs.push(vec![
                                Value::I64(a),
                                Value::I64(b),
                                Value::I64(c),
                                Value::I64(d),
                            ]);
                        }
                    }
                } else {
                    inputs.push(vec![Value::I64(a), Value::I64(b)]);
                }
            }
        }
        let mut random = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..512 {
            inputs.push(
                (0..if wide_pair { 4 } else { 2 })
                    .map(|_| {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        Value::I64(random)
                    })
                    .collect(),
            );
        }
        let kernel = Kernel::new(&wasm, operation).unwrap();
        let actual = kernel.run_values(&inputs).unwrap();
        let function = instance.get_func(&mut store, operation).unwrap();
        for (input, actual) in inputs.iter().zip(actual) {
            let bits: Vec<_> = input
                .iter()
                .map(|value| match value {
                    Value::I64(bits) => *bits,
                    _ => unreachable!(),
                })
                .collect();
            let expected = match operation {
                "i64.add128" | "i64.sub128" => {
                    let a = bits[0] as u128 | ((bits[1] as u128) << 64);
                    let b = bits[2] as u128 | ((bits[3] as u128) << 64);
                    if operation == "i64.add128" {
                        a.wrapping_add(b)
                    } else {
                        a.wrapping_sub(b)
                    }
                }
                "i64.mul_wide_s" => ((bits[0] as i64 as i128) * (bits[1] as i64 as i128)) as u128,
                "i64.mul_wide_u" => (bits[0] as u128) * (bits[1] as u128),
                _ => unreachable!(),
            };
            assert_eq!(
                actual,
                [
                    Value::I64(expected as u64),
                    Value::I64((expected >> 64) as u64)
                ],
                "{operation}: {bits:x?}"
            );
            let arguments: Vec<_> = bits.iter().map(|&v| wasmtime::Val::I64(v as i64)).collect();
            let mut results = [wasmtime::Val::I64(0), wasmtime::Val::I64(0)];
            function.call(&mut store, &arguments, &mut results).unwrap();
            assert_eq!(results[0].i64().unwrap() as u64, expected as u64);
            assert_eq!(results[1].i64().unwrap() as u64, (expected >> 64) as u64);
        }
    }
}
