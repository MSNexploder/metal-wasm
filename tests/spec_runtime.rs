use std::collections::HashMap;
use std::sync::Mutex;

use metal_wasm::{Error, Kernel, Value};
use wast::core::{NanPattern, V128Pattern, WastArgCore, WastRetCore};
use wast::parser::{self, ParseBuffer};
use wast::{Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet};

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn arg(value: WastArg<'_>) -> Value {
    match value {
        WastArg::Core(WastArgCore::I32(value)) => Value::I32(value as u32),
        WastArg::Core(WastArgCore::I64(value)) => Value::I64(value as u64),
        WastArg::Core(WastArgCore::F32(value)) => Value::F32(value.bits),
        WastArg::Core(WastArgCore::F64(value)) => Value::F64(value.bits),
        WastArg::Core(WastArgCore::V128(value)) => Value::V128(value.to_le_bytes()),
        other => panic!("unexpected spec argument: {other:?}"),
    }
}

fn matches_result(actual: Value, expected: &WastRet<'_>) -> bool {
    match expected {
        WastRet::Core(expected) => matches_core(actual, expected),
        other => panic!("unexpected spec result: {other:?}"),
    }
}

fn matches_core(actual: Value, expected: &WastRetCore<'_>) -> bool {
    match (actual, expected) {
        (actual, WastRetCore::Either(cases)) => cases.iter().any(|case| matches_core(actual, case)),
        (Value::I32(actual), WastRetCore::I32(expected)) => actual == *expected as u32,
        (Value::I64(actual), WastRetCore::I64(expected)) => actual == *expected as u64,
        (Value::F32(actual), WastRetCore::F32(pattern)) => match pattern {
            NanPattern::Value(expected) => actual == expected.bits,
            NanPattern::CanonicalNan => actual & 0x7fff_ffff == 0x7fc0_0000,
            NanPattern::ArithmeticNan => actual & 0x7fc0_0000 == 0x7fc0_0000,
        },
        (Value::F64(actual), WastRetCore::F64(pattern)) => match pattern {
            NanPattern::Value(expected) => actual == expected.bits,
            NanPattern::CanonicalNan => actual & 0x7fff_ffff_ffff_ffff == 0x7ff8_0000_0000_0000,
            NanPattern::ArithmeticNan => actual & 0x7ff8_0000_0000_0000 == 0x7ff8_0000_0000_0000,
        },
        (Value::V128(actual), WastRetCore::V128(pattern)) => match pattern {
            V128Pattern::I8x16(values) => actual.iter().zip(values).all(|(a, b)| *a == *b as u8),
            V128Pattern::I16x8(values) => actual
                .chunks_exact(2)
                .zip(values)
                .all(|(a, b)| a == b.to_le_bytes()),
            V128Pattern::I32x4(values) => actual
                .chunks_exact(4)
                .zip(values)
                .all(|(a, b)| a == b.to_le_bytes()),
            V128Pattern::I64x2(values) => actual
                .chunks_exact(8)
                .zip(values)
                .all(|(a, b)| a == b.to_le_bytes()),
            V128Pattern::F32x4(values) => actual.chunks_exact(4).zip(values).all(|(a, b)| {
                let bits = u32::from_le_bytes(a.try_into().unwrap());
                match b {
                    NanPattern::Value(v) => bits == v.bits,
                    NanPattern::CanonicalNan => bits & 0x7fff_ffff == 0x7fc0_0000,
                    NanPattern::ArithmeticNan => bits & 0x7fc0_0000 == 0x7fc0_0000,
                }
            }),
            V128Pattern::F64x2(values) => actual.chunks_exact(8).zip(values).all(|(a, b)| {
                let bits = u64::from_le_bytes(a.try_into().unwrap());
                match b {
                    NanPattern::Value(v) => bits == v.bits,
                    NanPattern::CanonicalNan => {
                        bits & 0x7fff_ffff_ffff_ffff == 0x7ff8_0000_0000_0000
                    }
                    NanPattern::ArithmeticNan => {
                        bits & 0x7ff8_0000_0000_0000 == 0x7ff8_0000_0000_0000
                    }
                }
            }),
        },
        other => panic!("unexpected spec result: {other:?}"),
    }
}

fn invoke(
    call: WastInvoke<'_>,
    module: &[u8],
    kernels: &mut HashMap<String, Kernel>,
) -> Result<Vec<Vec<Value>>, Error> {
    assert!(
        call.module.is_none(),
        "named modules are outside this harness"
    );
    let kernel = kernels
        .entry(call.name.to_owned())
        .or_insert_with(|| Kernel::new(module, call.name).unwrap());
    kernel.run_values(&[call.args.into_iter().map(arg).collect()])
}

fn check_script(source: &str, minimum_assertions: usize) {
    // Concurrent Metal compilation and execution can exhaust this device's resources.
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let buffer = ParseBuffer::new(source).unwrap();
    let script = parser::parse::<Wast<'_>>(&buffer).unwrap();
    let mut module = None;
    let mut kernels = HashMap::new();
    let mut checked = 0;
    let mut validation_cases = 0;
    for directive in script.directives {
        match directive {
            WastDirective::Module(mut wat) => {
                kernels.clear();
                module = Some(wat.encode().unwrap());
            }
            WastDirective::AssertReturn {
                exec: WastExecute::Invoke(call),
                results,
                ..
            } => {
                let invocation = format!("{call:?}");
                let actual = invoke(call, module.as_deref().unwrap(), &mut kernels)
                    .unwrap_or_else(|error| panic!("{invocation}: {error}"));
                assert_eq!(actual.len(), 1);
                assert_eq!(actual[0].len(), results.len());
                for (value, expected) in actual[0].iter().zip(&results) {
                    assert!(
                        matches_result(*value, expected),
                        "{invocation}: expected {expected:?}, got {value:?}"
                    );
                }
                checked += 1;
            }
            WastDirective::AssertTrap {
                exec: WastExecute::Invoke(call),
                message,
                ..
            } => {
                let actual = invoke(call, module.as_deref().unwrap(), &mut kernels);
                assert!(
                    matches!(actual, Err(Error::Trap { input_index: 0, reason })
                        if reason == message || (message == "undefined element"
                            && reason == "undefined or uninitialized table element")),
                    "expected trap {message:?}, got {actual:?}"
                );
                checked += 1;
            }
            WastDirective::AssertInvalid { mut module, .. }
            | WastDirective::AssertMalformed { mut module, .. } => {
                let rejected = match module.encode() {
                    Ok(bytes) => matches!(
                        Kernel::new(&bytes, "__invalid_spec_export"),
                        Err(Error::InvalidWasm(_))
                    ),
                    Err(_) => true,
                };
                assert!(rejected, "invalid module was accepted: {module:?}");
                validation_cases += 1;
            }
            other => panic!("unhandled spec directive: {other:?}"),
        }
    }
    assert!(checked >= minimum_assertions, "only {checked} cases ran");
    eprintln!("checked {checked} runtime cases and {validation_cases} validation cases");
}

#[test]
fn spec_i32_runtime() {
    check_script(include_str!("spec/i32.wast"), 300);
}

#[test]
fn spec_i64_runtime() {
    check_script(include_str!("spec/i64.wast"), 200);
}

#[test]
fn spec_f32_comparisons() {
    check_script(include_str!("spec/f32_cmp.wast"), 2000);
}

#[test]
fn spec_f64_comparisons() {
    check_script(include_str!("spec/f64_cmp.wast"), 2000);
}

#[test]
fn spec_conversions() {
    check_script(include_str!("spec/conversions.wast"), 500);
}

#[test]
fn spec_f32_arithmetic() {
    check_script(include_str!("spec/f32.wast"), 2500);
}

#[test]
fn spec_f64_arithmetic() {
    check_script(include_str!("spec/f64.wast"), 2500);
}

#[test]
fn spec_if_control() {
    check_script(include_str!("spec/if.wast"), 100);
}

#[test]
fn spec_simd_bitwise() {
    check_script(include_str!("spec/simd_bitwise.wast"), 100);
}

#[test]
fn spec_simd_i32x4_arithmetic() {
    check_script(include_str!("spec/simd_i32x4_arith.wast"), 100);
}

#[test]
fn spec_simd_i32x4_comparisons() {
    check_script(include_str!("spec/simd_i32x4_cmp.wast"), 100);
}

#[test]
fn spec_simd_select() {
    check_script(include_str!("spec/simd_select.wast"), 1);
}

#[test]
fn spec_simd_integer_families() {
    for source in [
        include_str!("spec/simd_bit_shift.wast"),
        include_str!("spec/simd_i8x16_arith.wast"),
        include_str!("spec/simd_i8x16_cmp.wast"),
        include_str!("spec/simd_i16x8_arith.wast"),
        include_str!("spec/simd_i16x8_cmp.wast"),
        include_str!("spec/simd_i64x2_arith.wast"),
        include_str!("spec/simd_i64x2_cmp.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_simd_values_and_saturation() {
    for source in [
        include_str!("spec/simd_boolean.wast"),
        include_str!("spec/simd_lane.wast"),
        include_str!("spec/simd_i8x16_sat_arith.wast"),
        include_str!("spec/simd_i16x8_sat_arith.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_simd_float_families() {
    for source in [
        include_str!("spec/simd_f32x4_arith.wast"),
        include_str!("spec/simd_f32x4_cmp.wast"),
        include_str!("spec/simd_f32x4_rounding.wast"),
        include_str!("spec/simd_f64x2_arith.wast"),
        include_str!("spec/simd_f64x2_cmp.wast"),
        include_str!("spec/simd_f64x2_rounding.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_simd_partial_memory() {
    for source in [
        include_str!("spec/simd_load_extend.wast"),
        include_str!("spec/simd_load_splat.wast"),
        include_str!("spec/simd_load_zero.wast"),
        include_str!("spec/simd_load8_lane.wast"),
        include_str!("spec/simd_load16_lane.wast"),
        include_str!("spec/simd_load32_lane.wast"),
        include_str!("spec/simd_load64_lane.wast"),
        include_str!("spec/simd_store8_lane.wast"),
        include_str!("spec/simd_store16_lane.wast"),
        include_str!("spec/simd_store32_lane.wast"),
        include_str!("spec/simd_store64_lane.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_simd_widening_integer() {
    for source in [
        include_str!("spec/simd_i16x8_extadd_pairwise_i8x16.wast"),
        include_str!("spec/simd_i16x8_extmul_i8x16.wast"),
        include_str!("spec/simd_i16x8_q15mulr_sat_s.wast"),
        include_str!("spec/simd_i32x4_dot_i16x8.wast"),
        include_str!("spec/simd_i32x4_extadd_pairwise_i16x8.wast"),
        include_str!("spec/simd_i32x4_extmul_i16x8.wast"),
        include_str!("spec/simd_i64x2_extmul_i32x4.wast"),
        include_str!("spec/simd_int_to_int_extend.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_simd_float_conversion_and_pseudo_minmax() {
    for source in [
        include_str!("spec/simd_i32x4_trunc_sat_f32x4.wast"),
        include_str!("spec/simd_i32x4_trunc_sat_f64x2.wast"),
        include_str!("spec/simd_conversions.wast"),
        include_str!("spec/simd_f32x4_pmin_pmax.wast"),
        include_str!("spec/simd_f64x2_pmin_pmax.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_relaxed_simd() {
    for source in [
        include_str!("spec/i16x8_relaxed_q15mulr_s.wast"),
        include_str!("spec/i8x16_relaxed_swizzle.wast"),
        include_str!("spec/relaxed_dot_product.wast"),
        include_str!("spec/relaxed_laneselect.wast"),
        include_str!("spec/relaxed_madd_nmadd.wast"),
        include_str!("spec/relaxed_min_max.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_simd_remaining_arithmetic_and_splat() {
    for source in [
        include_str!("spec/simd_i8x16_arith2.wast"),
        include_str!("spec/simd_i16x8_arith2.wast"),
        include_str!("spec/simd_i32x4_arith2.wast"),
        include_str!("spec/simd_i64x2_arith2.wast"),
        include_str!("spec/simd_splat.wast"),
        include_str!("spec/simd_f32x4.wast"),
        include_str!("spec/simd_f64x2.wast"),
    ] {
        check_script(source, 1);
    }
}

#[test]
fn spec_simd_memory() {
    for source in [
        include_str!("spec/simd_address.wast"),
        include_str!("spec/simd_load.wast"),
        include_str!("spec/simd_store.wast"),
    ] {
        check_script(source, 1);
    }
}
