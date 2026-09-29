use std::collections::HashMap;

use metal_wasm::{Error, Kernel, Value};
use wast::core::{NanPattern, WastArgCore, WastRetCore};
use wast::parser::{self, ParseBuffer};
use wast::{Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet};

fn arg(value: WastArg<'_>) -> Value {
    match value {
        WastArg::Core(WastArgCore::I32(value)) => Value::I32(value as u32),
        WastArg::Core(WastArgCore::I64(value)) => Value::I64(value as u64),
        WastArg::Core(WastArgCore::F32(value)) => Value::F32(value.bits),
        WastArg::Core(WastArgCore::F64(value)) => Value::F64(value.bits),
        other => panic!("unexpected spec argument: {other:?}"),
    }
}

fn matches_result(actual: Value, expected: &WastRet<'_>) -> bool {
    match (actual, expected) {
        (Value::I32(actual), WastRet::Core(WastRetCore::I32(expected))) => {
            actual == *expected as u32
        }
        (Value::I64(actual), WastRet::Core(WastRetCore::I64(expected))) => {
            actual == *expected as u64
        }
        (Value::F32(actual), WastRet::Core(WastRetCore::F32(pattern))) => match pattern {
            NanPattern::Value(expected) => actual == expected.bits,
            NanPattern::CanonicalNan => actual & 0x7fff_ffff == 0x7fc0_0000,
            NanPattern::ArithmeticNan => actual & 0x7fc0_0000 == 0x7fc0_0000,
        },
        (Value::F64(actual), WastRet::Core(WastRetCore::F64(pattern))) => match pattern {
            NanPattern::Value(expected) => actual == expected.bits,
            NanPattern::CanonicalNan => actual & 0x7fff_ffff_ffff_ffff == 0x7ff8_0000_0000_0000,
            NanPattern::ArithmeticNan => actual & 0x7ff8_0000_0000_0000 == 0x7ff8_0000_0000_0000,
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
    let buffer = ParseBuffer::new(source).unwrap();
    let script = parser::parse::<Wast<'_>>(&buffer).unwrap();
    let mut module = None;
    let mut kernels = HashMap::new();
    let mut checked = 0;
    let mut validation_cases = 0;
    for directive in script.directives {
        match directive {
            WastDirective::Module(mut wat) => {
                assert!(module.is_none(), "expected one module per script");
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
            WastDirective::AssertInvalid { .. } | WastDirective::AssertMalformed { .. } => {
                // The GPU test targets execution; wasmparser already validates modules.
                validation_cases += 1;
            }
            other => panic!("unhandled spec directive: {other:?}"),
        }
    }
    assert!(checked >= minimum_assertions, "only {checked} cases ran");
    eprintln!("checked {checked} runtime cases; excluded {validation_cases} validation cases");
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
