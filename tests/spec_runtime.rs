use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use metal_wasm::{Error, Kernel, KernelLimits, Value};
use wast::core::{NanPattern, V128Pattern, WastArgCore, WastRetCore};
use wast::lexer::{Lexer, TokenKind};
use wast::parser::{self, ParseBuffer};
use wast::{Wast, WastArg, WastDirective, WastExecute, WastInvoke, WastRet};

static GPU_LOCK: Mutex<()> = Mutex::new(());
static SPECTEST: OnceLock<Vec<u8>> = OnceLock::new();

#[derive(Clone, Copy)]
enum SpecArg {
    Value(Value),
    External(u32),
    Host(u32),
}

fn arg(value: &WastArg<'_>) -> SpecArg {
    let value = match value {
        WastArg::Core(WastArgCore::I32(value)) => Value::I32(*value as u32),
        WastArg::Core(WastArgCore::I64(value)) => Value::I64(*value as u64),
        WastArg::Core(WastArgCore::F32(value)) => Value::F32(value.bits),
        WastArg::Core(WastArgCore::F64(value)) => Value::F64(value.bits),
        WastArg::Core(WastArgCore::V128(value)) => Value::V128(value.to_le_bytes()),
        WastArg::Core(WastArgCore::RefExtern(id)) => return SpecArg::External(*id),
        WastArg::Core(WastArgCore::RefHost(id)) => return SpecArg::Host(*id),
        WastArg::Core(WastArgCore::RefNull(wast::core::HeapType::Abstract { ty, .. })) => {
            match ty {
                wast::core::AbstractHeapType::Extern | wast::core::AbstractHeapType::NoExtern => {
                    Value::ExternRef(None)
                }
                wast::core::AbstractHeapType::Exn | wast::core::AbstractHeapType::NoExn => {
                    Value::ExnRef(None)
                }
                wast::core::AbstractHeapType::Func | wast::core::AbstractHeapType::NoFunc => {
                    Value::FuncRef(None)
                }
                wast::core::AbstractHeapType::Any
                | wast::core::AbstractHeapType::Eq
                | wast::core::AbstractHeapType::I31
                | wast::core::AbstractHeapType::Struct
                | wast::core::AbstractHeapType::Array
                | wast::core::AbstractHeapType::None => Value::GcRef(None),
                other => panic!("unsupported null argument: {other:?}"),
            }
        }
        other => panic!("unexpected spec argument: {other:?}"),
    };
    SpecArg::Value(value)
}

fn matches_result(kernel: &Kernel, actual: Value, expected: &WastRet<'_>) -> bool {
    match expected {
        WastRet::Core(expected) => matches_core(kernel, actual, expected),
        other => panic!("unexpected spec result: {other:?}"),
    }
}

fn matches_core(kernel: &Kernel, actual: Value, expected: &WastRetCore<'_>) -> bool {
    match (actual, expected) {
        (
            Value::FuncRef(None)
            | Value::ExternRef(None)
            | Value::ExnRef(None)
            | Value::GcRef(None),
            WastRetCore::RefNull(_),
        ) => true,
        (Value::ExternRef(Some(handle)), WastRetCore::RefExtern(id)) => {
            id.is_none_or(|id| *kernel.external_reference_value::<u32>(handle).unwrap() == id)
        }
        (Value::FuncRef(Some(_)), WastRetCore::RefFunc(None)) => true,
        (Value::I31(_), WastRetCore::RefI31 | WastRetCore::RefEq | WastRetCore::RefAny) => true,
        (Value::GcRef(Some(_)), WastRetCore::RefAny) => true,
        (Value::GcRef(Some(handle)), WastRetCore::RefStruct) => {
            kernel.gc_reference_kind(handle).unwrap() == wasmparser::AbstractHeapType::Struct
        }
        (Value::GcRef(Some(handle)), WastRetCore::RefArray) => {
            kernel.gc_reference_kind(handle).unwrap() == wasmparser::AbstractHeapType::Array
        }
        (Value::GcRef(Some(handle)), WastRetCore::RefEq) => matches!(
            kernel.gc_reference_kind(handle).unwrap(),
            wasmparser::AbstractHeapType::Struct | wasmparser::AbstractHeapType::Array
        ),
        (Value::GcRef(Some(handle)), WastRetCore::RefHost(id)) => {
            let external = kernel
                .externalize_reference(Value::GcRef(Some(handle)))
                .unwrap()
                .unwrap();
            *kernel.external_reference_value::<u32>(external).unwrap() == *id
        }
        (actual, WastRetCore::Either(cases)) => {
            cases.iter().any(|case| matches_core(kernel, actual, case))
        }
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
    providers: &HashMap<String, Vec<u8>>,
    sequence: Option<(&str, &[(String, Vec<SpecArg>)])>,
) -> Result<Vec<Vec<Value>>, Error> {
    assert!(
        call.module.is_none(),
        "named invocations require a module cache key"
    );
    let mut wrapped = None;
    let (export, inputs) = if let Some((text, history)) = sequence {
        let (bytes, inputs) = sequence_module(text, module, history, &call);
        wrapped = Some(bytes);
        kernels.clear();
        ("__spec_sequence", inputs)
    } else {
        (call.name, call.args.iter().map(arg).collect())
    };
    let module = wrapped.as_deref().unwrap_or(module);
    if !kernels.contains_key(export) {
        let imports = spec_imports(providers);
        let kernel = Kernel::new_with_imports_and_limits(
            module,
            export,
            &imports,
            KernelLimits {
                call_depth: 256,
                table_growth_elements: 65_536,
                ..KernelLimits::default()
            },
        )?;
        kernels.insert(export.to_owned(), kernel);
    }
    let kernel = &kernels[export];
    let mut external = HashMap::new();
    let inputs: Result<Vec<_>, Error> = inputs
        .into_iter()
        .map(|arg| match arg {
            SpecArg::Value(value) => Ok(value),
            SpecArg::External(id) | SpecArg::Host(id) => {
                let handle = if let Some(handle) = external.get(&id) {
                    *handle
                } else {
                    let handle = kernel.external_reference(id)?;
                    external.insert(id, handle);
                    handle
                };
                if matches!(arg, SpecArg::Host(_)) {
                    kernel.internalize_reference(Some(handle))
                } else {
                    Ok(Value::ExternRef(Some(handle)))
                }
            }
        })
        .collect();
    kernel.run_values(&[inputs?])
}

// Replay successful actions in one lane to test scripts with persistent state.
// The production API continues to create independent instances per invocation.
fn module_text(source: &str, offset: usize) -> &str {
    let start = source[..=offset].rfind('(').unwrap();
    let mut depth = 0;
    for token in Lexer::new(source).iter(start) {
        let token = token.unwrap();
        match token.kind {
            TokenKind::LParen => depth += 1,
            TokenKind::RParen => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..token.offset + token.len as usize];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated module");
}

fn sequence_module(
    text: &str,
    bytes: &[u8],
    history: &[(String, Vec<SpecArg>)],
    call: &WastInvoke<'_>,
) -> (Vec<u8>, Vec<SpecArg>) {
    use wasmparser::{
        CompositeInnerType, ExternalKind, HeapType, Parser, Payload, ValType, Validator,
    };
    let validated = Validator::new().validate_all(bytes).unwrap();
    let validated = validated.as_ref();
    let mut exports = HashMap::new();
    for payload in Parser::new(0).parse_all(bytes) {
        if let Payload::ExportSection(section) = payload.unwrap() {
            for export in section {
                let export = export.unwrap();
                if export.kind == ExternalKind::Func {
                    exports.insert(export.name, export.index);
                }
            }
        }
    }
    let result_types = |name: &str| match &validated[validated.core_function_at(exports[name])]
        .composite_type
        .inner
    {
        CompositeInnerType::Func(ty) => ty.results().to_vec(),
        _ => unreachable!(),
    };
    let results = result_types(call.name);
    let mut source = text[..text.len() - 1].to_owned();
    source.push_str(" (func (export \"__spec_sequence\")");
    let actions: Vec<_> = history
        .iter()
        .map(|(name, args)| (name.as_str(), args.clone(), true))
        .chain(std::iter::once((
            call.name,
            call.args.iter().map(arg).collect(),
            false,
        )))
        .collect();
    let mut external = HashMap::new();
    let mut inputs = Vec::new();
    for (_, args, _) in &actions {
        for arg in args {
            if let SpecArg::External(id) | SpecArg::Host(id) = arg {
                external.entry(*id).or_insert_with(|| {
                    let index = inputs.len();
                    inputs.push(SpecArg::External(*id));
                    source.push_str(" (param (ref extern))");
                    index
                });
            }
        }
    }

    if !results.is_empty() {
        source.push_str(&format!(
            " (result {})",
            results
                .iter()
                .map(|ty| match ty {
                    ValType::Ref(reference) => match reference.heap_type() {
                        HeapType::Concrete(id) => {
                            let module_index = (0..validated.core_type_count_in_module())
                                .find(|index| {
                                    validated.core_type_at_in_module(*index)
                                        == id.as_core_type_id().unwrap()
                                })
                                .unwrap();
                            format!(
                                "(ref {}{module_index})",
                                if reference.is_nullable() { "null " } else { "" }
                            )
                        }
                        _ => ty.to_string(),
                    },
                    _ => ty.to_string(),
                })
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    for (name, args, discard) in actions {
        for value in args {
            source.push_str(&match value {
                SpecArg::External(id) => format!(" local.get {}", external[&id]),
                SpecArg::Host(id) => {
                    format!(" local.get {} any.convert_extern", external[&id])
                }
                SpecArg::Value(Value::I32(bits)) => format!(" i32.const {}", bits as i32),
                SpecArg::Value(Value::I64(bits)) => format!(" i64.const {}", bits as i64),
                SpecArg::Value(Value::F32(bits)) => {
                    format!(" i32.const {} f32.reinterpret_i32", bits as i32)
                }
                SpecArg::Value(Value::F64(bits)) => {
                    format!(" i64.const {} f64.reinterpret_i64", bits as i64)
                }
                SpecArg::Value(Value::FuncRef(None)) => " ref.null nofunc".into(),
                SpecArg::Value(Value::ExternRef(None)) => " ref.null noextern".into(),
                SpecArg::Value(Value::ExnRef(None)) => " ref.null noexn".into(),
                SpecArg::Value(Value::GcRef(None)) => " ref.null none".into(),
                _ => {
                    panic!("sequence arguments must be scalar numbers or null function references")
                }
            });
        }
        source.push_str(&format!(" call {}", exports[name]));
        if discard {
            for _ in result_types(name) {
                source.push_str(" drop");
            }
        }
    }
    source.push_str("))");
    (wat::parse_str(source).unwrap(), inputs)
}

fn spec_imports(providers: &HashMap<String, Vec<u8>>) -> Vec<(&str, &[u8])> {
    let mut imports: Vec<_> = providers
        .iter()
        .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
        .collect();
    if !providers.contains_key("spectest") {
        let spectest = SPECTEST.get_or_init(|| wat::parse_str("(module (func (export \"print_i32_f32\") (param i32 f32)) (global (export \"global_i32\") i32 (i32.const 666)) (table (export \"table\") 10 20 funcref) (table (export \"table64\") i64 10 20 funcref))").unwrap());
        imports.push(("spectest", spectest));
    }
    imports
}

// Translation resolves imports before looking up an export. A missing sentinel
// export lets module-only directives check validation and linking without a GPU dispatch.
fn assert_links(bytes: &[u8], providers: &HashMap<String, Vec<u8>>) {
    let imports = spec_imports(providers);
    let actual = Kernel::new_with_imports(bytes, "__invalid_spec_export", &imports);
    assert!(
        matches!(actual, Err(Error::Unsupported(ref reason)) if reason == "export \"__invalid_spec_export\" is not a function"),
        "valid module failed validation or linking: {:?}",
        actual.err()
    );
}

fn check_script(source: &str, minimum_assertions: usize) {
    check_script_mode(source, minimum_assertions, false);
}

// Parse directives separately because some upstream invalid modules use syntax
// rejected by the text parser itself (for example, multiple supertypes).
fn directive_sources(source: &str) -> Vec<(usize, &str)> {
    let mut directives = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    for token in Lexer::new(source).iter(0) {
        let token = token.unwrap();
        match token.kind {
            TokenKind::LParen => {
                if depth == 0 {
                    start = token.offset;
                }
                depth += 1;
            }
            TokenKind::RParen => {
                depth = depth.checked_sub(1).expect("unmatched closing parenthesis");
                if depth == 0 {
                    directives.push((start, &source[start..token.offset + token.len as usize]));
                }
            }
            TokenKind::Whitespace | TokenKind::LineComment | TokenKind::BlockComment => {}
            _ => assert!(depth != 0, "unexpected token outside a directive"),
        }
    }
    assert_eq!(depth, 0, "unterminated directive");
    directives
}

fn check_script_mode(source: &str, minimum_assertions: usize, replay: bool) {
    // Concurrent Metal compilation and execution can exhaust this device's resources.
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let mut module = None;
    let mut kernels = HashMap::new();
    let mut providers = HashMap::new();
    let mut named_modules = HashMap::new();
    let mut checked = 0;
    let mut validation_cases = 0;
    let mut text = None;
    let mut history: Vec<(String, Vec<SpecArg>)> = Vec::new();
    for (offset, directive_source) in directive_sources(source) {
        let buffer = ParseBuffer::new(directive_source).unwrap();
        let mut script = match parser::parse::<Wast<'_>>(&buffer) {
            Ok(script) => script,
            Err(error)
                if directive_source.starts_with("(assert_invalid")
                    || directive_source.starts_with("(assert_malformed") =>
            {
                eprintln!("text parser rejected invalid module: {error}");
                validation_cases += 1;
                continue;
            }
            Err(error) => panic!("{error}"),
        };
        assert_eq!(script.directives.len(), 1);
        let directive = script.directives.pop().unwrap();
        match directive {
            WastDirective::ModuleDefinition(mut wat) => {
                let bytes = wat.encode().unwrap();
                wasmparser::Validator::new().validate_all(&bytes).unwrap();
                validation_cases += 1;
            }
            WastDirective::Module(mut wat) => {
                kernels.clear();
                history.clear();
                text = replay.then(|| module_text(source, offset + wat.span().offset()));
                let name = wat.name().map(|id| id.name().to_owned());
                let bytes = wat.encode().unwrap();
                assert_links(&bytes, &providers);
                if let Some(name) = name {
                    named_modules.insert(name, bytes.clone());
                }
                module = Some(bytes);
            }
            WastDirective::AssertReturn {
                exec: WastExecute::Invoke(call),
                results,
                ..
            } => {
                let action = (call.name.to_owned(), call.args.iter().map(arg).collect());
                let export = if replay { "__spec_sequence" } else { call.name };
                let line = source[..offset].lines().count() + 1;
                let invocation = format!("line {line}: {call:?}");
                let actual = invoke(
                    call,
                    module.as_deref().unwrap(),
                    &mut kernels,
                    &providers,
                    text.map(|text| (text, history.as_slice())),
                );

                let actual = actual.unwrap_or_else(|error| panic!("{invocation}: {error}"));
                assert_eq!(actual.len(), 1);
                assert_eq!(actual[0].len(), results.len());
                for (value, expected) in actual[0].iter().zip(&results) {
                    assert!(
                        matches_result(&kernels[export], *value, expected),
                        "{invocation}: expected {expected:?}, got {value:?}"
                    );
                }
                if replay {
                    history.push(action);
                }
                checked += 1;
            }
            WastDirective::AssertTrap {
                exec: WastExecute::Invoke(call),
                message,
                ..
            } => {
                let actual = invoke(
                    call,
                    module.as_deref().unwrap(),
                    &mut kernels,
                    &providers,
                    text.map(|text| (text, history.as_slice())),
                );

                assert!(
                    matches!(actual, Err(Error::Trap { input_index: 0, reason })
                        if reason == message || (matches!(message, "null i31 reference" | "null struct reference" | "null structure reference" | "null array reference") && reason == "null reference") || ((message == "unreachable" || message == "unreachable executed") && reason == "unreachable executed") || ((message == "undefined element" || message == "uninitialized element")
                            && reason == "undefined or uninitialized table element")),
                    "expected trap {message:?}, got {actual:?}"
                );
                checked += 1;
            }
            WastDirective::Invoke(call) => {
                assert!(replay, "stateful invoke requires replay mode");
                let action = (call.name.to_owned(), call.args.iter().map(arg).collect());
                let line = source[..offset].lines().count() + 1;
                let invocation = format!("line {line}: {call:?}");
                invoke(
                    call,
                    module.as_deref().unwrap(),
                    &mut kernels,
                    &providers,
                    text.map(|text| (text, history.as_slice())),
                )
                .unwrap_or_else(|error| panic!("{invocation}: {error}"));
                history.push(action);
            }
            WastDirective::Register {
                name,
                module: selected,
                ..
            } => {
                let bytes = selected
                    .map(|id| named_modules.get(id.name()).unwrap())
                    .unwrap_or_else(|| module.as_ref().unwrap());
                providers.insert(name.to_owned(), bytes.clone());
                kernels.clear();
            }
            WastDirective::AssertException {
                exec: WastExecute::Invoke(call),
                ..
            } => {
                let actual = invoke(
                    call,
                    module.as_deref().unwrap(),
                    &mut kernels,
                    &providers,
                    text.map(|text| (text, history.as_slice())),
                );
                assert!(
                    matches!(actual, Err(Error::UncaughtException { input_index: 0 })),
                    "expected exception, got {actual:?}"
                );
                checked += 1;
            }
            WastDirective::AssertUnlinkable { mut module, .. } => {
                let bytes = module.encode().unwrap();
                let imports = spec_imports(&providers);
                let actual = Kernel::new_with_imports(&bytes, "__invalid_spec_export", &imports);
                assert!(
                    matches!(actual, Err(Error::Unsupported(ref reason)) if reason.contains("mismatch")),
                    "unlinkable module accepted or unexpected error"
                );
                validation_cases += 1;
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
fn spec_constant_expressions() {
    check_script(include_str!("spec/const.wast"), 300);
}

#[test]
fn spec_extended_constant_elements() {
    check_script(include_str!("spec/extended_const_elements.wast"), 8);
}

#[test]
fn spec_tail_calls() {
    check_script(include_str!("spec/return_call.wast"), 20);
}

#[test]
fn spec_tail_calls_indirect() {
    check_script(include_str!("spec/return_call_indirect.wast"), 30);
}

#[test]
fn spec_multiple_memories() {
    check_script(include_str!("spec/multi_memory_load0.wast"), 2);
}

#[test]
fn spec_memory64_loads() {
    check_script(include_str!("spec/memory64_load64.wast"), 37);
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

macro_rules! new_feature_spec {
    ($name:ident, $file:literal, $minimum:expr) => {
        #[test]
        fn $name() {
            check_script(include_str!($file), $minimum);
        }
    };
}

new_feature_spec!(spec_memory64_address, "spec/memory64_address64.wast", 238);
new_feature_spec!(spec_memory64_align, "spec/memory64_align64.wast", 48);
new_feature_spec!(
    spec_table64_indirect,
    "spec/memory64_call_indirect64.wast",
    1
);
new_feature_spec!(
    spec_memory64_endianness,
    "spec/memory64_endianness64.wast",
    68
);
new_feature_spec!(spec_multi_memory_load2, "spec/multi_memory_load2.wast", 37);
new_feature_spec!(
    spec_multi_memory_address0,
    "spec/multi_memory_address0.wast",
    91
);
new_feature_spec!(
    spec_multi_memory_address1,
    "spec/multi_memory_address1.wast",
    126
);
new_feature_spec!(spec_multi_memory_traps, "spec/multi_memory_traps0.wast", 14);

#[test]
fn spec_br_on_null() {
    check_script(include_str!("spec/br_on_null.wast"), 4);
}

#[test]
fn spec_br_on_non_null() {
    check_script(include_str!("spec/br_on_non_null.wast"), 4);
}

#[test]
fn spec_call_ref() {
    check_script(include_str!("spec/call_ref.wast"), 20);
}

#[test]
fn spec_ref_as_non_null() {
    check_script(include_str!("spec/ref_as_non_null.wast"), 4);
}

#[test]
fn spec_return_call_ref() {
    check_script(include_str!("spec/return_call_ref.wast"), 30);
}

#[test]
fn spec_exceptions_tag() {
    check_script(include_str!("spec/exceptions_tag.wast"), 0);
}

#[test]
fn spec_exceptions_throw() {
    check_script(include_str!("spec/exceptions_throw.wast"), 9);
}

#[test]
fn spec_exceptions_throw_ref() {
    check_script(include_str!("spec/exceptions_throw_ref.wast"), 12);
}

#[test]
fn spec_exceptions_try_table() {
    check_script(include_str!("spec/exceptions_try_table.wast"), 35);
}

#[test]
fn spec_ref_func() {
    check_script_mode(include_str!("spec/ref_func.wast"), 8, true);
}

#[test]
fn spec_function_type_equivalence() {
    check_script(include_str!("spec/type-equivalence.wast"), 4);
}

#[test]
fn spec_type_recursion() {
    check_script(include_str!("spec/type-rec.wast"), 3);
}

#[test]
fn spec_function_select() {
    check_script_mode(include_str!("spec/select_functions.wast"), 90, true);
}

#[test]
fn spec_external_locals() {
    check_script(include_str!("spec/local_init.wast"), 4);
}

#[test]
fn spec_reference_tests() {
    check_script(include_str!("spec/ref_null.wast"), 32);
}

macro_rules! gc_script {
    ($name:ident, $file:literal, $assertions:literal) => {
        #[test]
        fn $name() {
            check_script_mode(include_str!($file), $assertions, true);
        }
    };
}

gc_script!(spec_gc_i31, "spec/gc_i31.wast", 57);
gc_script!(spec_gc_ref_eq, "spec/gc_ref_eq.wast", 81);
gc_script!(spec_gc_struct, "spec/gc_struct.wast", 19);
gc_script!(spec_gc_array, "spec/gc_array.wast", 41);
gc_script!(spec_gc_array_copy, "spec/gc_array_copy.wast", 30);
gc_script!(spec_gc_array_fill, "spec/gc_array_fill.wast", 26);
gc_script!(spec_gc_array_init_data, "spec/gc_array_init_data.wast", 42);
gc_script!(spec_gc_array_init_elem, "spec/gc_array_init_elem.wast", 30);
gc_script!(spec_gc_array_new_data, "spec/gc_array_new_data.wast", 23);
gc_script!(spec_gc_array_new_elem, "spec/gc_array_new_elem.wast", 19);
gc_script!(spec_gc_ref_test, "spec/gc_ref_test.wast", 68);
gc_script!(spec_gc_ref_cast, "spec/gc_ref_cast.wast", 40);
gc_script!(spec_gc_br_on_cast, "spec/gc_br_on_cast.wast", 25);
gc_script!(spec_gc_br_on_cast_fail, "spec/gc_br_on_cast_fail.wast", 25);
gc_script!(spec_gc_extern, "spec/gc_extern.wast", 16);
gc_script!(spec_gc_subtyping, "spec/gc_type-subtyping.wast", 29);
gc_script!(spec_gc_binary, "spec/gc_binary-gc.wast", 0);

#[test]
fn spec_reference_tables() {
    check_script_mode(include_str!("spec/memory64_table64.wast"), 0, true);
    for source in [
        include_str!("spec/table.wast"),
        include_str!("spec/table_get.wast"),
        include_str!("spec/table_set.wast"),
        include_str!("spec/table_size.wast"),
        include_str!("spec/table_grow_instances.wast"),
        include_str!("spec/bulk_memory_table_fill.wast"),
        include_str!("spec/memory64_table_get64.wast"),
        include_str!("spec/memory64_table_set64.wast"),
        include_str!("spec/memory64_table_size64.wast"),
        include_str!("spec/memory64_table_grow64.wast"),
        include_str!("spec/memory64_table_fill64.wast"),
    ] {
        check_script_mode(source, 1, true);
    }
}
