use std::collections::HashMap;

use metal_wasm::{Error, Kernel, Value};
use wast::core::{WastArgCore, WastRetCore};
use wast::parser::{self, ParseBuffer};
use wast::{Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet};

fn arg(value: WastArg<'_>) -> Value {
    match value {
        WastArg::Core(WastArgCore::I32(value)) => Value::I32(value as u32),
        WastArg::Core(WastArgCore::I64(value)) => Value::I64(value as u64),
        other => panic!("unexpected spec argument: {other:?}"),
    }
}

fn result(value: WastRet<'_>) -> Value {
    match value {
        WastRet::Core(WastRetCore::I32(value)) => Value::I32(value as u32),
        WastRet::Core(WastRetCore::I64(value)) => Value::I64(value as u64),
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
                let expected: Vec<_> = results.into_iter().map(result).collect();
                let actual = invoke(call, module.as_deref().unwrap(), &mut kernels).unwrap();
                assert_eq!(actual, [expected]);
                checked += 1;
            }
            WastDirective::AssertTrap {
                exec: WastExecute::Invoke(call),
                message,
                ..
            } => {
                let actual = invoke(call, module.as_deref().unwrap(), &mut kernels);
                assert!(
                    matches!(actual, Err(Error::Trap { input_index: 0, reason }) if reason == message),
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
