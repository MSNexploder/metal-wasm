use std::error::Error;
use std::hint::black_box;
use std::time::{Duration, Instant};

use metal_wasm::Kernel;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const WAT: &str = r#"
    (module
        (func (export "calculate") (param i32) (result i32)
            local.get 0
            i32.const 7
            i32.mul
            i32.const 3
            i32.add))
"#;
const COUNTS: [usize; 4] = [1, 1_000, 1_000_000, 10_000_000];
const MIX_ROUNDS: u32 = 32;
const MIX_MULTIPLIER: u32 = 0x9e37_79b1;
const OTHER_MULTIPLIER: u32 = 0x85eb_ca6b;
const MEMORY_DATA: [u8; 16] = [
    5, 17, 35, 55, 77, 91, 114, 138, 161, 184, 207, 229, 13, 47, 83, 121,
];

struct SuiteWorkload {
    name: &'static str,
    wat: &'static str,
    native: fn(u32) -> u32,
    counts: &'static [usize],
}

const SUITE_WORKLOADS: [SuiteWorkload; 3] = [
    SuiteWorkload {
        name: "dispatch_i64",
        wat: include_str!("workloads/dispatch_i64.wat"),
        native: dispatch_native,
        counts: &[1, 1_000, 100_000, 1_000_000],
    },
    SuiteWorkload {
        name: "float_f64",
        wat: include_str!("workloads/float_f64.wat"),
        native: float_native,
        counts: &[1, 1_000, 100_000, 1_000_000],
    },
    SuiteWorkload {
        name: "stateful_memory",
        wat: include_str!("workloads/stateful_memory.wat"),
        native: memory_native,
        counts: &[1, 1_000, 4_096],
    },
];

fn main() -> Result<()> {
    let large = match std::env::args().nth(1).as_deref() {
        Some("--mix") => return compare_compute(false),
        Some("--control") => return compare_compute(true),
        Some("--suite") => return compare_suite(),
        Some("--metal-native-100m") => true,
        Some(argument) => return Err(format!("unknown argument: {argument}").into()),
        None => false,
    };
    let wasm = wat::parse_str(WAT)?;
    let metal = Kernel::new(&wasm, "calculate")?;
    if large {
        return compare_large(&metal);
    }

    let wasmtime_engine = wasmtime::Engine::default();
    let wasmtime_module = wasmtime::Module::new(&wasmtime_engine, &wasm)?;
    let mut wasmtime_store = wasmtime::Store::new(&wasmtime_engine, ());
    let wasmtime_instance = wasmtime::Instance::new(&mut wasmtime_store, &wasmtime_module, &[])?;
    let wasmtime_func =
        wasmtime_instance.get_typed_func::<i32, i32>(&mut wasmtime_store, "calculate")?;

    let wasmi_engine = wasmi::Engine::default();
    let wasmi_module = wasmi::Module::new(&wasmi_engine, &wasm)?;
    let mut wasmi_store = wasmi::Store::new(&wasmi_engine, ());
    let wasmi_instance =
        wasmi::Linker::new(&wasmi_engine).instantiate_and_start(&mut wasmi_store, &wasmi_module)?;
    let wasmi_func = wasmi_instance.get_typed_func::<i32, i32>(&wasmi_store, "calculate")?;

    let mut run_wasmtime = |inputs: &[u32]| -> Result<Vec<u32>> {
        let mut output = Vec::with_capacity(inputs.len());
        for &input in inputs {
            output.push(wasmtime_func.call(&mut wasmtime_store, input as i32)? as u32);
        }
        Ok(output)
    };
    let mut run_wasmi = |inputs: &[u32]| -> Result<Vec<u32>> {
        let mut output = Vec::with_capacity(inputs.len());
        for &input in inputs {
            output.push(wasmi_func.call(&mut wasmi_store, input as i32)? as u32);
        }
        Ok(output)
    };
    // Warm each path before measuring it.
    let warmup = [42u32];
    metal.run(&warmup)?;
    run_wasmtime(&warmup)?;
    run_wasmi(&warmup)?;
    run_native(&warmup)?;

    println!("count,metal_us,wasmtime_us,wasmi_us,native_us,samples,repeats");
    for count in COUNTS {
        eprintln!("measuring {count} invocations...");
        let inputs: Vec<u32> = (0..count).map(|value| value as u32).collect();
        let samples = match count {
            1 => 31,
            1_000 => 11,
            _ => 3,
        };
        let repeats = match count {
            1 => 100,
            1_000 => 10,
            _ => 1,
        };
        let metal_time = measure(&inputs, samples, repeats, calculate_native, |inputs| {
            Ok(metal.run(inputs)?)
        })?;
        let wasmtime_time = measure(
            &inputs,
            samples,
            repeats,
            calculate_native,
            &mut run_wasmtime,
        )?;
        let wasmi_time = measure(&inputs, samples, repeats, calculate_native, &mut run_wasmi)?;
        let native_time = measure(&inputs, samples, repeats, calculate_native, run_native)?;
        println!(
            "{count},{:.2},{:.2},{:.2},{:.2},{samples},{repeats}",
            micros(metal_time),
            micros(wasmtime_time),
            micros(wasmi_time),
            micros(native_time),
        );
    }
    Ok(())
}

fn compare_large(metal: &Kernel) -> Result<()> {
    const COUNT: usize = 100_000_000;
    let inputs: Vec<u32> = (0..COUNT as u32).collect();
    metal.run(&[42])?;
    run_native(&[42])?;
    let metal_time = measure(&inputs, 3, 1, calculate_native, |inputs| {
        Ok(metal.run(inputs)?)
    })?;
    let native_time = measure(&inputs, 3, 1, calculate_native, run_native)?;
    println!("count,metal_us,native_us,samples");
    println!(
        "{COUNT},{:.2},{:.2},3",
        micros(metal_time),
        micros(native_time)
    );
    Ok(())
}

fn compare_compute(control: bool) -> Result<()> {
    let wasm = wat::parse_str(if control { control_wat() } else { mix_wat() })?;
    let expected: fn(u32) -> u32 = if control { control_native } else { mix_native };
    let metal = Kernel::new(&wasm, "calculate")?;
    let wasmtime_engine = wasmtime::Engine::default();
    let wasmtime_module = wasmtime::Module::new(&wasmtime_engine, &wasm)?;
    let mut wasmtime_store = wasmtime::Store::new(&wasmtime_engine, ());
    let wasmtime_instance = wasmtime::Instance::new(&mut wasmtime_store, &wasmtime_module, &[])?;
    let wasmtime_func =
        wasmtime_instance.get_typed_func::<i32, i32>(&mut wasmtime_store, "calculate")?;
    let wasmi_engine = wasmi::Engine::default();
    let wasmi_module = wasmi::Module::new(&wasmi_engine, &wasm)?;
    let mut wasmi_store = wasmi::Store::new(&wasmi_engine, ());
    let wasmi_instance =
        wasmi::Linker::new(&wasmi_engine).instantiate_and_start(&mut wasmi_store, &wasmi_module)?;
    let wasmi_func = wasmi_instance.get_typed_func::<i32, i32>(&wasmi_store, "calculate")?;
    let mut run_wasmtime = |inputs: &[u32]| -> Result<Vec<u32>> {
        let mut output = Vec::with_capacity(inputs.len());
        for &input in inputs {
            output.push(wasmtime_func.call(&mut wasmtime_store, input as i32)? as u32);
        }
        Ok(output)
    };
    let mut run_wasmi = |inputs: &[u32]| -> Result<Vec<u32>> {
        let mut output = Vec::with_capacity(inputs.len());
        for &input in inputs {
            output.push(wasmi_func.call(&mut wasmi_store, input as i32)? as u32);
        }
        Ok(output)
    };
    let warmup = [42u32, 43];
    metal.run_timed(&warmup)?;
    run_wasmtime(&warmup)?;
    run_wasmi(&warmup)?;
    std::hint::black_box(expected(warmup[0]));

    println!("pattern,count,metal_total_us,metal_gpu_us,wasmtime_us,wasmi_us,native_us,samples");
    let patterns: &[(&str, bool)] = if control {
        &[("grouped", true), ("mixed", false)]
    } else {
        &[("straight", false)]
    };
    for count in [1_000, 1_000_000, 10_000_000, 100_000_000] {
        for &(pattern, grouped) in patterns {
            eprintln!("measuring {pattern} at {count} invocations...");
            let inputs: Vec<u32> = if control {
                control_inputs(count, grouped)
            } else {
                (0..count).map(|value| value as u32).collect()
            };
            let (metal_total, metal_gpu) = measure_metal(&metal, &inputs, expected)?;
            let native = measure(&inputs, 3, 1, expected, |inputs| {
                Ok(inputs.iter().copied().map(expected).collect())
            })?;
            if count == 100_000_000 {
                println!(
                    "{pattern},{count},{:.2},{:.2},,,{:.2},3",
                    micros(metal_total),
                    micros(metal_gpu),
                    micros(native)
                );
            } else {
                let wasmtime = measure(&inputs, 3, 1, expected, &mut run_wasmtime)?;
                let wasmi = measure(&inputs, 3, 1, expected, &mut run_wasmi)?;
                println!(
                    "{pattern},{count},{:.2},{:.2},{:.2},{:.2},{:.2},3",
                    micros(metal_total),
                    micros(metal_gpu),
                    micros(wasmtime),
                    micros(wasmi),
                    micros(native)
                );
            }
        }
    }
    Ok(())
}

fn compare_suite() -> Result<()> {
    println!("workload,count,metal_total_us,metal_gpu_us,wasmtime_us,wasmi_us,native_us,samples");
    for workload in &SUITE_WORKLOADS {
        let wasm = wat::parse_str(workload.wat)?;
        let metal = Kernel::new(&wasm, "calculate")?;
        let wasmtime_engine = wasmtime::Engine::default();
        let wasmtime_module = wasmtime::Module::new(&wasmtime_engine, &wasm)?;
        let mut wasmtime_store = wasmtime::Store::new(&wasmtime_engine, ());
        let wasmtime_instance =
            wasmtime::Instance::new(&mut wasmtime_store, &wasmtime_module, &[])?;
        let wasmtime_func =
            wasmtime_instance.get_typed_func::<i32, i32>(&mut wasmtime_store, "calculate")?;
        let wasmi_engine = wasmi::Engine::default();
        let wasmi_module = wasmi::Module::new(&wasmi_engine, &wasm)?;
        let mut wasmi_store = wasmi::Store::new(&wasmi_engine, ());
        let wasmi_instance = wasmi::Linker::new(&wasmi_engine)
            .instantiate_and_start(&mut wasmi_store, &wasmi_module)?;
        let wasmi_func = wasmi_instance.get_typed_func::<i32, i32>(&wasmi_store, "calculate")?;
        let mut run_wasmtime = |inputs: &[u32]| -> Result<Vec<u32>> {
            let mut output = Vec::with_capacity(inputs.len());
            for &input in inputs {
                output.push(wasmtime_func.call(&mut wasmtime_store, input as i32)? as u32);
            }
            Ok(output)
        };
        let mut run_wasmi = |inputs: &[u32]| -> Result<Vec<u32>> {
            let mut output = Vec::with_capacity(inputs.len());
            for &input in inputs {
                output.push(wasmi_func.call(&mut wasmi_store, input as i32)? as u32);
            }
            Ok(output)
        };
        let warmup = [0u32, 1, 42, 43];
        verify_output(&warmup, &metal.run_timed(&warmup)?.0, workload.native)?;
        verify_output(&warmup, &run_wasmtime(&warmup)?, workload.native)?;
        verify_output(&warmup, &run_wasmi(&warmup)?, workload.native)?;

        for &count in workload.counts {
            eprintln!("measuring {} at {count} invocations...", workload.name);
            let inputs: Vec<u32> = (0..count).map(|value| value as u32).collect();
            let (metal_total, metal_gpu) = measure_metal(&metal, &inputs, workload.native)?;
            let wasmtime = measure(&inputs, 3, 1, workload.native, &mut run_wasmtime)?;
            let wasmi = measure(&inputs, 3, 1, workload.native, &mut run_wasmi)?;
            let native = measure(&inputs, 3, 1, workload.native, |inputs| {
                Ok(inputs.iter().copied().map(workload.native).collect())
            })?;
            println!(
                "{},{count},{:.2},{:.2},{:.2},{:.2},{:.2},3",
                workload.name,
                micros(metal_total),
                micros(metal_gpu),
                micros(wasmtime),
                micros(wasmi),
                micros(native)
            );
        }
    }
    Ok(())
}

fn mix_wat() -> String {
    let mut wat = String::from("(module (func (export \"calculate\") (param i32) (result i32) ");
    for round in 0..MIX_ROUNDS {
        wat.push_str(&format!(
            "local.get 0 i32.const 5 i32.rotl local.get 0 i32.xor i32.const {} i32.mul i32.const {round} i32.add local.set 0 ",
            MIX_MULTIPLIER as i32
        ));
    }
    wat.push_str("local.get 0))");
    wat
}

fn control_wat() -> String {
    format!(
        r#"(module
            (func (export "calculate") (param i32) (result i32) (local i32 i32)
                local.get 0
                local.set 1
                i32.const 0
                local.set 2
                loop
                    local.get 0
                    i32.const 1
                    i32.and
                    if
                        local.get 1
                        i32.const 5
                        i32.rotl
                        local.get 0
                        i32.xor
                        i32.const {odd_multiplier}
                        i32.mul
                        local.get 2
                        i32.add
                        local.set 1
                    else
                        local.get 1
                        i32.const 13
                        i32.rotl
                        local.get 0
                        i32.xor
                        i32.const {even_multiplier}
                        i32.mul
                        local.get 2
                        i32.add
                        local.set 1
                    end
                    local.get 2
                    i32.const 1
                    i32.add
                    local.tee 2
                    i32.const {rounds}
                    i32.ne
                    br_if 0
                end
                local.get 1))"#,
        odd_multiplier = MIX_MULTIPLIER as i32,
        even_multiplier = OTHER_MULTIPLIER as i32,
        rounds = MIX_ROUNDS
    )
}

fn control_inputs(count: usize, grouped: bool) -> Vec<u32> {
    let even_count = ((count / 2).div_ceil(32) * 32).min(count);
    let odd_count = count - even_count;
    let paired_count = even_count.min(odd_count);
    (0..count)
        .map(|index| {
            if grouped {
                if index < even_count {
                    (2 * index) as u32
                } else {
                    (2 * (index - even_count) + 1) as u32
                }
            } else if index < 2 * paired_count {
                index as u32
            } else if even_count > odd_count {
                (2 * (index - odd_count)) as u32
            } else {
                (2 * (index - even_count) + 1) as u32
            }
        })
        .collect()
}

fn run_native(inputs: &[u32]) -> Result<Vec<u32>> {
    Ok(inputs.iter().copied().map(calculate_native).collect())
}

#[inline(always)]
fn calculate_native(input: u32) -> u32 {
    input.wrapping_mul(7).wrapping_add(3)
}

#[inline(always)]
fn mix_native(mut value: u32) -> u32 {
    for round in 0..MIX_ROUNDS {
        value = (value.rotate_left(5) ^ value)
            .wrapping_mul(MIX_MULTIPLIER)
            .wrapping_add(round);
    }
    value
}

#[inline(always)]
fn control_native(input: u32) -> u32 {
    let mut value = input;
    for round in 0..MIX_ROUNDS {
        value = if input & 1 != 0 {
            (value.rotate_left(5) ^ input)
                .wrapping_mul(MIX_MULTIPLIER)
                .wrapping_add(round)
        } else {
            (value.rotate_left(13) ^ input)
                .wrapping_mul(OTHER_MULTIPLIER)
                .wrapping_add(round)
        };
    }
    value
}

#[inline(always)]
fn dispatch_native(input: u32) -> u32 {
    let mut value = input as u64;
    for round in 0..24u32 {
        let salt = round as u64;
        value = match (input ^ round) & 3 {
            0 => (value.rotate_left(7) ^ salt).wrapping_mul(6_364_136_223_846_793_005),
            1 => (value ^ salt.wrapping_mul(1_442_695_040_888_963_407))
                .rotate_right(11)
                .wrapping_add(2_862_933_555_777_941_757),
            2 => value.wrapping_add(salt).rotate_left(17) ^ 3_935_559_000_370_003_845,
            _ => {
                value
                    .wrapping_mul(2_862_933_555_777_941_757)
                    .rotate_right(23)
                    ^ salt
            }
        };
    }
    value as u32 ^ (value >> 32) as u32
}

#[inline(always)]
fn float_native(input: u32) -> u32 {
    let mut value = ((input & 2047) as f64 / 2048.0) + 0.25;
    for round in 0..12u32 {
        let factor = if (input ^ round) & 1 != 0 {
            1.015625
        } else {
            0.984375
        };
        value = (value * factor) / (value + 0.125) + 0.0625;
    }
    (value.sqrt() * 100_000.0) as u32
}

#[inline(always)]
fn memory_native(input: u32) -> u32 {
    const SEED: u32 = 0x9e37_79b1;
    let mut global = SEED;
    let mut value = input ^ global;
    let mut cells = [0u32; 16];
    for round in 0..24u32 {
        let slot = (round & 15) as usize;
        let data = MEMORY_DATA[(input.wrapping_add(round) & 15) as usize] as u32;
        value = (value.rotate_left(7) ^ data)
            .wrapping_add(cells[slot])
            .wrapping_mul(SEED)
            .wrapping_add(global);
        cells[slot] = value;
        global = value.rotate_left(13);
    }
    value ^ global
}

fn measure_metal(
    metal: &Kernel,
    inputs: &[u32],
    expected: fn(u32) -> u32,
) -> Result<(Duration, Duration)> {
    let mut total_times = Vec::with_capacity(3);
    let mut gpu_times = Vec::with_capacity(3);
    for _ in 0..3 {
        let start = Instant::now();
        let (output, gpu_time) = metal.run_timed(black_box(inputs))?;
        total_times.push(start.elapsed());
        gpu_times.push(gpu_time.ok_or("Metal GPU timestamps are unavailable")?);
        verify_output(inputs, &output, expected)?;
        black_box(output);
    }
    total_times.sort_unstable();
    gpu_times.sort_unstable();
    Ok((total_times[1], gpu_times[1]))
}

fn measure(
    inputs: &[u32],
    samples: usize,
    repeats: usize,
    expected: fn(u32) -> u32,
    mut run: impl FnMut(&[u32]) -> Result<Vec<u32>>,
) -> Result<Duration> {
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut outputs = Vec::with_capacity(repeats);
        let start = Instant::now();
        for _ in 0..repeats {
            outputs.push(run(black_box(inputs))?);
        }
        times.push(start.elapsed() / repeats as u32);
        for output in outputs {
            verify_output(inputs, &output, expected)?;
            black_box(output);
        }
    }
    times.sort_unstable();
    Ok(times[samples / 2])
}

fn verify_output(inputs: &[u32], output: &[u32], expected: fn(u32) -> u32) -> Result<()> {
    if output.len() != inputs.len() {
        return Err(format!(
            "benchmark output length: got {}, expected {}",
            output.len(),
            inputs.len()
        )
        .into());
    }
    for (index, (&actual, &input)) in output.iter().zip(inputs).enumerate() {
        let expected = expected(input);
        if actual != expected {
            return Err(format!("benchmark output mismatch at {index}: input {input}, got {actual}, expected {expected}").into());
        }
    }
    Ok(())
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::{SUITE_WORKLOADS, control_inputs};

    #[test]
    fn control_patterns_have_the_same_values_and_grouped_branches_agree() {
        for count in [1_000, 1_000_000] {
            let grouped = control_inputs(count, true);
            let mixed = control_inputs(count, false);
            assert!(
                grouped
                    .chunks(32)
                    .all(|chunk| chunk.iter().all(|value| value & 1 == chunk[0] & 1))
            );
            let mut grouped_sorted = grouped;
            let mut mixed_sorted = mixed;
            grouped_sorted.sort_unstable();
            mixed_sorted.sort_unstable();
            assert_eq!(grouped_sorted, mixed_sorted);
        }
    }

    #[test]
    fn suite_workloads_match_native_on_all_engines() {
        let inputs: Vec<u32> = (0..64).chain([u32::MAX, u32::MAX - 1]).collect();
        for workload in &SUITE_WORKLOADS {
            let wasm = wat::parse_str(workload.wat).unwrap();
            let expected: Vec<u32> = inputs.iter().copied().map(workload.native).collect();
            let metal = metal_wasm::Kernel::new(&wasm, "calculate").unwrap();
            let metal_output = metal.run(&inputs).unwrap();
            let wasmtime_engine = wasmtime::Engine::default();
            let wasmtime_module = wasmtime::Module::new(&wasmtime_engine, &wasm).unwrap();
            let mut wasmtime_store = wasmtime::Store::new(&wasmtime_engine, ());
            let wasmtime_instance =
                wasmtime::Instance::new(&mut wasmtime_store, &wasmtime_module, &[]).unwrap();
            let wasmtime_func = wasmtime_instance
                .get_typed_func::<i32, i32>(&mut wasmtime_store, "calculate")
                .unwrap();
            let wasmtime_output: Vec<u32> = inputs
                .iter()
                .map(|input| {
                    wasmtime_func
                        .call(&mut wasmtime_store, *input as i32)
                        .unwrap() as u32
                })
                .collect();
            let wasmi_engine = wasmi::Engine::default();
            let wasmi_module = wasmi::Module::new(&wasmi_engine, &wasm).unwrap();
            let mut wasmi_store = wasmi::Store::new(&wasmi_engine, ());
            let wasmi_instance = wasmi::Linker::new(&wasmi_engine)
                .instantiate_and_start(&mut wasmi_store, &wasmi_module)
                .unwrap();
            let wasmi_func = wasmi_instance
                .get_typed_func::<i32, i32>(&wasmi_store, "calculate")
                .unwrap();
            let wasmi_output: Vec<u32> = inputs
                .iter()
                .map(|input| wasmi_func.call(&mut wasmi_store, *input as i32).unwrap() as u32)
                .collect();
            assert_eq!(wasmtime_output, expected, "{}: Wasmtime", workload.name);
            assert_eq!(wasmi_output, expected, "{}: Wasmi", workload.name);
            assert_eq!(metal_output, expected, "{}: Metal", workload.name);
        }
    }
}
