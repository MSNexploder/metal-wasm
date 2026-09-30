use metal_wasm::{Kernel, LaneError, LaneResult, Store, Value};

fn next(random: &mut u64) -> u32 {
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    *random as u32
}

fn module_source(seed: u64) -> String {
    let mut random = seed ^ 0x9e3779b97f4a7c15;
    let constant = next(&mut random) | 1;
    let shift = next(&mut random) & 31;
    let rounds = 8 + next(&mut random) % 16;
    let mut operations = String::new();
    for _ in 0..6 {
        let operand = next(&mut random);
        let operation = match operand % 5 {
            0 => "add",
            1 => "xor",
            2 => "mul",
            3 => "rotl",
            _ => "shr_u",
        };
        operations.push_str(&format!("i32.const {operand} i32.{operation}\n"));
    }
    let call = match seed % 4 {
        0 => "local.get $x call $mix",
        1 => "local.get $x local.get $round i32.const 1 i32.and call_indirect (type $sig)",
        2 => "local.get $x ref.func $mix call_ref $sig",
        _ => "local.get $x call $tail",
    };
    format!(
        r#"(module
        (type $sig (func (param i32) (result i32)))
        (type $node (struct (field i32) (field (ref null $node))))
        (tag $tag (param i32))
        (memory $small (export "small") 64 68 (pagesize 1))
        (memory $large 1 1)
        (data $data "01234567")
        (global $calls (export "calls") (mut i32) (i32.const 0))
        (global $chain (mut (ref null $node)) (ref.null $node))
        (func $mix (type $sig) local.get 0 {operations})
        (func $alt (type $sig) local.get 0 i32.const {constant} i32.mul i32.const {shift} i32.rotl)
        (func $tail (type $sig) local.get 0 return_call $mix)
        (table 2 funcref)
        (elem (i32.const 0) func $mix $alt)
        (func (export "calculate") (param $input i32) (result i32 i64 v128 f64)
            (local $x i32) (local $round i32) (local $wide i64)
            (local $fraction f64) (local $vector v128)
            global.get $calls i32.const 1 i32.add global.set $calls
            local.get $input i32.const 15 i32.and i32.const {trap_input} i32.eq
            if i32.const 256 i32.const 0 i32.const 2 memory.fill $small end
            local.get $input i32.const 15 i32.and i32.const {throw_input} i32.eq
            if local.get $input throw $tag end
            i32.const 1 memory.grow $small drop
            i32.const 0 i32.const 0 i32.const 8 memory.init $small $data
            local.get $input global.get $calls i32.xor local.set $x
            local.get $input i64.extend_i32_s local.set $wide
            local.get $input f64.convert_i32_s f64.const 0.5 f64.mul local.set $fraction
            local.get $input i32x4.splat local.set $vector
            loop $again
                {call} local.set $x
                local.get $round local.get $x i32.store8 $small
                local.get $x i32.const 31 i32.and i32.load8_u $small
                local.get $x i32.add local.set $x
                local.get $x global.get $chain struct.new $node global.set $chain
                global.get $chain ref.cast (ref $node) struct.get $node 0 local.set $x
                local.get $wide i64.const {wide_constant} i64.mul
                local.get $x i64.extend_i32_u i64.xor local.set $wide
                local.get $fraction f64.const 0.5 f64.mul
                local.get $round f64.convert_i32_u f64.add local.set $fraction
                local.get $vector local.get $x i32x4.splat i32x4.add
                i32.const {shift} i32x4.shl local.set $vector
                local.get $round i32.const 1 i32.add local.tee $round
                i32.const {rounds} i32.lt_u br_if $again
            end
            i32.const 0 i32.const 0 i32.const 8 memory.copy $large $small
            local.get $input i32.const 128 i32.and if data.drop $data end
            local.get $x memory.size $small i32.xor
            local.get $wide local.get $vector local.get $fraction))"#,
        trap_input = seed & 15,
        throw_input = (seed + 1) & 15,
        wide_constant = ((constant as u64) << 32) | 0x85ebca6b
    )
}

fn cpu_call(store: &mut wasmtime::Store<()>, function: &wasmtime::Func, input: u32) -> LaneResult {
    let mut results = [
        wasmtime::Val::I32(0),
        wasmtime::Val::I64(0),
        wasmtime::Val::V128(0.into()),
        wasmtime::Val::F64(0),
    ];
    match function.call(
        &mut *store,
        &[wasmtime::Val::I32(input as i32)],
        &mut results,
    ) {
        Ok(()) => Ok(vec![
            Value::I32(results[0].i32().unwrap() as u32),
            Value::I64(results[1].i64().unwrap() as u64),
            Value::V128(results[2].v128().unwrap().as_u128().to_le_bytes()),
            Value::F64(results[3].f64().unwrap().to_bits()),
        ]),
        Err(error) => match error.downcast_ref::<wasmtime::Trap>() {
            Some(wasmtime::Trap::MemoryOutOfBounds) => Err(LaneError::Trap {
                reason: "out of bounds memory access",
            }),
            Some(trap) => panic!("unexpected CPU trap: {trap:?}: {error:#}"),
            None => {
                assert!(
                    store.has_pending_exception(),
                    "unexpected CPU error: {error:#}"
                );
                store.take_pending_exception().unwrap();
                Err(LaneError::UncaughtException)
            }
        },
    }
}

#[test]
fn generated_feature_combinations_match_fresh_and_persistent_wasmtime() {
    let mut config = wasmtime::Config::new();
    config.wasm_custom_page_sizes(true);
    let engine = wasmtime::Engine::new(&config).unwrap();
    for seed in 0..24u64 {
        let source = module_source(seed);
        let wasm = wat::parse_str(&source).unwrap_or_else(|e| panic!("seed {seed}: {e}\n{source}"));
        let module = wasmtime::Module::new(&engine, &wasm).unwrap();
        let kernel = Kernel::new(&wasm, "calculate")
            .unwrap_or_else(|e| panic!("seed {seed}: {e}\n{source}"));
        let mut random = seed ^ 0x85ebca6b;
        let inputs = (0..32u32)
            .chain([u32::MAX, 0x80000000])
            .chain((0..32).map(|_| next(&mut random)))
            .collect::<Vec<_>>();
        let rows = inputs
            .iter()
            .map(|&v| vec![Value::I32(v)])
            .collect::<Vec<_>>();
        let actual = kernel.run_values_outcomes(&rows).unwrap();
        for (lane, &input) in inputs.iter().enumerate() {
            let mut cpu = wasmtime::Store::new(&engine, ());
            let instance = wasmtime::Instance::new(&mut cpu, &module, &[]).unwrap();
            let function = instance.get_func(&mut cpu, "calculate").unwrap();
            assert_eq!(
                actual[lane],
                cpu_call(&mut cpu, &function, input),
                "fresh seed {seed}, input {input:#x}"
            );
        }
        let mut metal = Store::new(8).unwrap();
        let instance = metal.instantiate(&wasm, &[]).unwrap();
        let mut cpus = (0..8)
            .map(|_| {
                let mut cpu = wasmtime::Store::new(&engine, ());
                let instance = wasmtime::Instance::new(&mut cpu, &module, &[]).unwrap();
                let function = instance.get_func(&mut cpu, "calculate").unwrap();
                let global = instance.get_global(&mut cpu, "calls").unwrap();
                let memory = instance.get_memory(&mut cpu, "small").unwrap();
                (cpu, function, global, memory)
            })
            .collect::<Vec<_>>();
        for round in 0..4u32 {
            let inputs = (0..8u32)
                .map(|lane| lane * 97 + round * 43)
                .collect::<Vec<_>>();
            let rows = inputs
                .iter()
                .map(|&v| vec![Value::I32(v)])
                .collect::<Vec<_>>();
            let actual = metal.call_outcomes(instance, "calculate", &rows).unwrap();
            let globals = metal.get_global(instance, "calls").unwrap();
            let memories = metal.read_memory(instance, "small", 0..16).unwrap();
            for (lane, (cpu, function, global, memory)) in cpus.iter_mut().enumerate() {
                assert_eq!(
                    actual[lane],
                    cpu_call(cpu, function, inputs[lane]),
                    "persistent seed {seed}, round {round}, lane {lane}"
                );
                assert_eq!(
                    globals[lane],
                    Value::I32(global.get(&mut *cpu).i32().unwrap() as u32)
                );
                assert_eq!(memories[lane], memory.data(&*cpu)[..16]);
            }
        }
    }
}
