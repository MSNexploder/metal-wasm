use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use metal_wasm::{Error, ExternalRef, Instance, KernelLimits, Store, Value};
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

fn matches_result(kernel: &Store, actual: Value, expected: &WastRet<'_>) -> bool {
    match expected {
        WastRet::Core(expected) => matches_core(kernel, actual, expected),
        other => panic!("unexpected spec result: {other:?}"),
    }
}

fn matches_core(kernel: &Store, actual: Value, expected: &WastRetCore<'_>) -> bool {
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

// Script identities and resources belong to one persistent Store. Registrations
// retain the actual instance, including its state after trapping actions.
struct ScriptState {
    store: Store,
    current: Option<Instance>,
    named: HashMap<String, Instance>,
    definitions: HashMap<String, Vec<u8>>,
    last_definition: Option<Vec<u8>>,
    providers: HashMap<String, Instance>,
    external: HashMap<u32, ExternalRef>,
}

impl ScriptState {
    fn new(limits: KernelLimits) -> Self {
        let mut store = Store::new_with_limits(1, limits).unwrap();
        let spectest = SPECTEST.get_or_init(|| {
            wat::parse_str(
                r#"(module
            (func (export "print"))
            (func (export "print_i32") (param i32))
            (func (export "print_i64") (param i64))
            (func (export "print_f32") (param f32))
            (func (export "print_f64") (param f64))
            (func (export "print_i32_f32") (param i32 f32))
            (func (export "print_f64_f64") (param f64 f64))
            (global (export "global_i32") i32 (i32.const 666))
            (global (export "global_i64") i64 (i64.const 666))
            (global (export "global_f32") f32 (f32.const 666.6))
            (global (export "global_f64") f64 (f64.const 666.6))
            (table (export "table") 10 20 funcref)
            (table (export "table64") i64 10 20 funcref)
            (memory (export "memory") 1 2))"#,
            )
            .unwrap()
        });
        let provider = store.instantiate(spectest, &[]).unwrap();
        Self {
            store,
            current: None,
            named: HashMap::new(),
            definitions: HashMap::new(),
            last_definition: None,
            providers: HashMap::from([("spectest".into(), provider)]),
            external: HashMap::new(),
        }
    }

    fn selected(&self, name: Option<wast::token::Id<'_>>) -> Instance {
        name.map(|id| *self.named.get(id.name()).expect("unknown instance"))
            .unwrap_or_else(|| self.current.expect("no current instance"))
    }

    fn instantiate(&mut self, bytes: &[u8]) -> Result<Instance, Error> {
        let imports: Vec<_> = self
            .providers
            .iter()
            .map(|(name, &instance)| (name.as_str(), instance))
            .collect();
        self.store.instantiate(bytes, &imports)
    }

    fn invoke(&mut self, call: WastInvoke<'_>) -> Result<Vec<Vec<Value>>, Error> {
        let instance = self.selected(call.module);
        let mut inputs = Vec::new();
        for argument in &call.args {
            inputs.push(match arg(argument) {
                SpecArg::Value(value) => value,
                SpecArg::External(id) | SpecArg::Host(id) => {
                    let handle = if let Some(&handle) = self.external.get(&id) {
                        handle
                    } else {
                        let handle = self.store.external_reference(id)?;
                        self.external.insert(id, handle);
                        handle
                    };
                    if matches!(arg(argument), SpecArg::Host(_)) {
                        self.store.internalize_reference(Some(handle))?
                    } else {
                        Value::ExternRef(Some(handle))
                    }
                }
            });
        }
        self.store.call(instance, call.name, &[inputs])
    }

    fn execute(&mut self, action: WastExecute<'_>) -> Result<Vec<Vec<Value>>, Error> {
        match action {
            WastExecute::Invoke(call) => self.invoke(call),
            WastExecute::Get { module, global, .. } => {
                let instance = self.selected(module);
                Ok(self
                    .store
                    .get_global(instance, global)?
                    .into_iter()
                    .map(|v| vec![v])
                    .collect())
            }
            WastExecute::Wat(mut module) => {
                self.instantiate(&module.encode().unwrap())?;
                Ok(vec![vec![]])
            }
        }
    }
}

fn script_limits() -> KernelLimits {
    KernelLimits {
        // Official tail-call scripts include one million calls plus entry/exit.
        fuel: Some(2_000_000),
        memory_growth_pages: 8,
        call_depth: 32,
        table_growth_elements: 65_536,
        exception_records: 512,
        gc_heap_words: 4096,
        ..KernelLimits::default()
    }
}

fn check_script(source: &str, minimum_assertions: usize) {
    check_script_with_limits(source, minimum_assertions, script_limits());
}

// Parse directives separately because some upstream invalid modules use syntax
// rejected by the text parser itself (for example, multiple supertypes).
fn directive_sources(source: &str) -> Vec<(usize, &str)> {
    let mut directives = Vec::new();
    let mut start = 0;
    let mut depth = 0usize;
    let mut lexer = Lexer::new(source);
    // The official names script deliberately uses valid bidi/control characters.
    lexer.allow_confusing_unicode(true);
    for token in lexer.iter(0) {
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

fn check_script_with_limits(source: &str, minimum_assertions: usize, limits: KernelLimits) {
    // Concurrent Metal compilation and execution can exhaust this device's resources.
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let mut state = ScriptState::new(limits);
    let mut checked = 0;
    let mut validation_cases = 0;
    for (offset, directive_source) in directive_sources(source) {
        let mut lexer = Lexer::new(directive_source);
        lexer.allow_confusing_unicode(true);
        let buffer = ParseBuffer::new_with_lexer(lexer).unwrap();
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
        let line = source[..offset].lines().count() + 1;
        let context = format!("line {line}: {directive:?}");
        match directive {
            WastDirective::ModuleDefinition(mut wat) => {
                let name = wat.name().map(|id| id.name().to_owned());
                let bytes = wat.encode().unwrap();
                wasmparser::Validator::new_with_features(
                    wasmparser::WasmFeatures::default()
                        | wasmparser::WasmFeatures::CUSTOM_PAGE_SIZES,
                )
                .validate_all(&bytes)
                .unwrap();
                if let Some(name) = name {
                    state.definitions.insert(name, bytes.clone());
                }
                state.last_definition = Some(bytes);
                validation_cases += 1;
            }
            WastDirective::Module(mut wat) => {
                let name = wat.name().map(|id| id.name().to_owned());
                let bytes = wat.encode().unwrap();
                let instance = state
                    .instantiate(&bytes)
                    .unwrap_or_else(|error| panic!("{context}: {error}"));
                if let Some(name) = name {
                    state.named.insert(name, instance);
                }
                state.current = Some(instance);
                validation_cases += 1;
            }
            WastDirective::ModuleInstance {
                instance, module, ..
            } => {
                let bytes = module
                    .map(|id| state.definitions.get(id.name()).unwrap())
                    .unwrap_or_else(|| state.last_definition.as_ref().unwrap())
                    .clone();
                let handle = state
                    .instantiate(&bytes)
                    .unwrap_or_else(|error| panic!("{context}: {error}"));
                if let Some(instance) = instance {
                    state.named.insert(instance.name().into(), handle);
                }
                state.current = Some(handle);
                validation_cases += 1;
            }
            WastDirective::AssertReturn { exec, results, .. } => {
                let actual = state
                    .execute(exec)
                    .unwrap_or_else(|error| panic!("{context}: {error}"));
                assert_eq!(actual.len(), 1, "{context}");
                assert_eq!(actual[0].len(), results.len(), "{context}");
                for (value, expected) in actual[0].iter().zip(&results) {
                    assert!(
                        matches_result(&state.store, *value, expected),
                        "{context}: expected {expected:?}, got {value:?}"
                    );
                }
                checked += 1;
            }
            WastDirective::AssertTrap { exec, message, .. } => {
                let actual = state.execute(exec);
                assert!(
                    matches!(actual, Err(Error::Trap { input_index: 0, reason })
                    if reason == message
                        || (matches!(message, "null i31 reference" | "null struct reference" | "null structure reference" | "null array reference") && reason == "null reference")
                        || ((message == "unreachable" || message == "unreachable executed") && reason == "unreachable executed")
                        || ((message.starts_with("undefined element") || message.starts_with("uninitialized element")) && reason == "undefined or uninitialized table element")),
                    "{context}: expected trap {message:?}, got {actual:?}"
                );
                checked += 1;
            }
            WastDirective::AssertExhaustion { call, .. } => {
                let actual = state.invoke(call);
                assert!(
                    matches!(
                        actual,
                        Err(Error::Trap {
                            input_index: 0,
                            reason: "call depth limit exceeded"
                        })
                    ),
                    "{context}: expected stack exhaustion, got {actual:?}"
                );
                checked += 1;
            }
            WastDirective::Invoke(call) => {
                state
                    .invoke(call)
                    .unwrap_or_else(|error| panic!("{context}: {error}"));
            }
            WastDirective::Register { name, module, .. } => {
                state
                    .providers
                    .insert(name.to_owned(), state.selected(module));
            }
            WastDirective::AssertException { exec, .. } => {
                let actual = state.execute(exec);
                assert!(
                    matches!(actual, Err(Error::UncaughtException { input_index: 0 })),
                    "{context}: expected exception, got {actual:?}"
                );
                checked += 1;
            }
            WastDirective::AssertUnlinkable { mut module, .. } => {
                let actual = state.instantiate(&module.encode().unwrap());
                assert!(
                    matches!(actual, Err(Error::Unsupported(ref reason))
                    if reason.contains("mismatch") || reason.starts_with("missing Wasm provider")
                        || reason.contains("has no export") || reason.contains("wrong kind")
                        || reason.contains("is not a function")),
                    "{context}: expected linking rejection, got {actual:?}"
                );
                validation_cases += 1;
            }
            WastDirective::AssertInvalid { mut module, .. }
            | WastDirective::AssertMalformed { mut module, .. } => {
                let rejected = match module.encode() {
                    Ok(bytes) => matches!(state.instantiate(&bytes), Err(Error::InvalidWasm(_))),
                    Err(_) => true,
                };
                assert!(rejected, "{context}: invalid module was accepted");
                validation_cases += 1;
            }
            other => panic!("{context}: unhandled spec directive: {other:?}"),
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
    check_script_with_limits(
        include_str!("spec/call_ref.wast"),
        20,
        KernelLimits {
            call_depth: 256,
            memory_growth_pages: 8,
            table_growth_elements: 65_536,
            exception_records: 512,
            gc_heap_words: 4096,
            ..KernelLimits::default()
        },
    );
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
    check_script(include_str!("spec/ref_func.wast"), 8);
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
    check_script(include_str!("spec/select_functions.wast"), 90);
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
            check_script(include_str!($file), $assertions);
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
    check_script(include_str!("spec/memory64_table64.wast"), 0);
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
        check_script(source, 1);
    }
}

// Whole official scripts for persistent state, linking, and initialization.

#[test]
fn spec_persistent_data() {
    check_script(include_str!("spec/data.wast"), 15);
}

#[test]
fn spec_persistent_elem() {
    check_script(include_str!("spec/elem.wast"), 47);
}

#[test]
fn spec_persistent_global() {
    check_script(include_str!("spec/global.wast"), 67);
}

#[test]
fn spec_persistent_imports() {
    check_script(include_str!("spec/imports.wast"), 34);
}

#[test]
fn spec_persistent_linking() {
    check_script(include_str!("spec/linking.wast"), 90);
}

#[test]
fn spec_persistent_start() {
    check_script(include_str!("spec/start.wast"), 7);
}

#[test]
fn spec_persistent_instance() {
    check_script(include_str!("spec/instance.wast"), 12);
}

#[test]
fn spec_persistent_memory_size() {
    check_script(include_str!("spec/memory_size.wast"), 36);
}

#[test]
fn spec_persistent_memory_redundancy() {
    check_script(include_str!("spec/memory_redundancy.wast"), 4);
}

#[test]
fn spec_persistent_memory_trap() {
    check_script(include_str!("spec/memory_trap.wast"), 180);
}

#[test]
fn spec_persistent_bulk_memory_table_copy() {
    check_script(include_str!("spec/bulk_memory_table_copy.wast"), 1649);
}

#[test]
fn spec_persistent_bulk_memory_table_sub() {
    check_script(include_str!("spec/bulk_memory_table-sub.wast"), 0);
}

#[test]
fn spec_persistent_bulk_memory_memory_copy() {
    check_script(include_str!("spec/bulk_memory_memory_copy.wast"), 4338);
}

#[test]
fn spec_persistent_bulk_memory_bulk() {
    check_script(include_str!("spec/bulk_memory_bulk.wast"), 66);
}

#[test]
fn spec_persistent_bulk_memory_memory_fill() {
    check_script(include_str!("spec/bulk_memory_memory_fill.wast"), 20);
}

#[test]
fn spec_persistent_bulk_memory_memory_init() {
    check_script(include_str!("spec/bulk_memory_memory_init.wast"), 142);
}

#[test]
fn spec_persistent_bulk_memory_table_init() {
    check_script(include_str!("spec/bulk_memory_table_init.wast"), 665);
}

#[test]
fn spec_persistent_simd_align() {
    check_script(include_str!("spec/simd_align.wast"), 8);
}

#[test]
fn spec_persistent_simd_const() {
    check_script(include_str!("spec/simd_const.wast"), 265);
}

#[test]
fn spec_persistent_simd_linking() {
    check_script(include_str!("spec/simd_linking.wast"), 0);
}

#[test]
fn spec_persistent_simd_memory_multi() {
    check_script(include_str!("spec/simd_memory-multi.wast"), 0);
}

#[test]
fn spec_persistent_memory_grow() {
    // This official script successfully grows an unbounded memory to 804 pages.
    check_script_with_limits(
        include_str!("spec/memory_grow.wast"),
        87,
        KernelLimits {
            memory_bytes: 64 * 1024 * 1024,
            memory_growth_pages: 1024,
            table_growth_elements: 65_536,
            call_depth: 256,
            fuel: Some(1_000_000),
            exception_records: 512,
            gc_heap_words: 4096,
        },
    );
}

// Remaining whole scripts from the pinned official core tree.
new_feature_spec!(spec_complete_address, "spec/core/address.wast", 0);
new_feature_spec!(spec_complete_align, "spec/core/align.wast", 0);
new_feature_spec!(spec_complete_annotations, "spec/core/annotations.wast", 0);
new_feature_spec!(
    spec_complete_binary_leb128,
    "spec/core/binary-leb128.wast",
    0
);
new_feature_spec!(spec_complete_binary, "spec/core/binary.wast", 0);
new_feature_spec!(spec_complete_block, "spec/core/block.wast", 0);
new_feature_spec!(spec_complete_br, "spec/core/br.wast", 0);
new_feature_spec!(spec_complete_br_if, "spec/core/br_if.wast", 0);
new_feature_spec!(spec_complete_br_table, "spec/core/br_table.wast", 0);
#[test]
fn spec_complete_call() {
    check_script_with_limits(
        include_str!("spec/core/call.wast"),
        0,
        KernelLimits {
            call_depth: 256,
            memory_bytes: 32 * 1024 * 1024,
            memory_growth_pages: 512,
            ..script_limits()
        },
    );
}
#[test]
fn spec_complete_call_indirect() {
    check_script_with_limits(
        include_str!("spec/core/call_indirect.wast"),
        0,
        KernelLimits {
            call_depth: 256,
            memory_bytes: 32 * 1024 * 1024,
            memory_growth_pages: 512,
            ..script_limits()
        },
    );
}
new_feature_spec!(spec_complete_comments, "spec/core/comments.wast", 0);
new_feature_spec!(spec_complete_custom, "spec/core/custom.wast", 0);
new_feature_spec!(spec_complete_endianness, "spec/core/endianness.wast", 0);
new_feature_spec!(spec_complete_exports, "spec/core/exports.wast", 0);
new_feature_spec!(spec_complete_f32_bitwise, "spec/core/f32_bitwise.wast", 0);
new_feature_spec!(spec_complete_f64_bitwise, "spec/core/f64_bitwise.wast", 0);
new_feature_spec!(spec_complete_fac, "spec/core/fac.wast", 0);
new_feature_spec!(spec_complete_float_exprs, "spec/core/float_exprs.wast", 0);
new_feature_spec!(
    spec_complete_float_literals,
    "spec/core/float_literals.wast",
    0
);
new_feature_spec!(spec_complete_float_memory, "spec/core/float_memory.wast", 0);
new_feature_spec!(spec_complete_float_misc, "spec/core/float_misc.wast", 0);
new_feature_spec!(spec_complete_forward, "spec/core/forward.wast", 0);
new_feature_spec!(spec_complete_func, "spec/core/func.wast", 0);
new_feature_spec!(spec_complete_func_ptrs, "spec/core/func_ptrs.wast", 0);
new_feature_spec!(spec_complete_id, "spec/core/id.wast", 0);
new_feature_spec!(
    spec_complete_inline_module,
    "spec/core/inline-module.wast",
    0
);
new_feature_spec!(spec_complete_int_exprs, "spec/core/int_exprs.wast", 0);
new_feature_spec!(spec_complete_int_literals, "spec/core/int_literals.wast", 0);
new_feature_spec!(spec_complete_labels, "spec/core/labels.wast", 0);
new_feature_spec!(
    spec_complete_left_to_right,
    "spec/core/left-to-right.wast",
    0
);
new_feature_spec!(spec_complete_load, "spec/core/load.wast", 0);
new_feature_spec!(spec_complete_local_get, "spec/core/local_get.wast", 0);
new_feature_spec!(spec_complete_local_set, "spec/core/local_set.wast", 0);
#[test]
fn spec_complete_local_tee() {
    check_script_with_limits(
        include_str!("spec/core/local_tee.wast"),
        0,
        KernelLimits {
            memory_growth_pages: 64,
            ..script_limits()
        },
    );
}
new_feature_spec!(spec_complete_loop, "spec/core/loop.wast", 0);
new_feature_spec!(spec_complete_memory, "spec/core/memory.wast", 0);
new_feature_spec!(
    spec_complete_memory64_binary_leb128_64,
    "spec/core/memory64/binary_leb128_64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_bulk64,
    "spec/core/memory64/bulk64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_float_memory64,
    "spec/core/memory64/float_memory64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_memory64_imports,
    "spec/core/memory64/memory64-imports.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_memory64,
    "spec/core/memory64/memory64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_memory_copy64,
    "spec/core/memory64/memory_copy64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_memory_fill64,
    "spec/core/memory64/memory_fill64.wast",
    0
);

#[test]
fn spec_complete_memory64_memory_grow64() {
    check_script_with_limits(
        include_str!("spec/core/memory64/memory_grow64.wast"),
        0,
        KernelLimits {
            memory_bytes: 64 * 1024 * 1024,
            memory_growth_pages: 1024,
            fuel: Some(2_000_000),
            ..KernelLimits::default()
        },
    );
}
new_feature_spec!(
    spec_complete_memory64_memory_init64,
    "spec/core/memory64/memory_init64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_memory_redundancy64,
    "spec/core/memory64/memory_redundancy64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_memory_trap64,
    "spec/core/memory64/memory_trap64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_table_copy64,
    "spec/core/memory64/table_copy64.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_table_copy_mixed,
    "spec/core/memory64/table_copy_mixed.wast",
    0
);
new_feature_spec!(
    spec_complete_memory64_table_init64,
    "spec/core/memory64/table_init64.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_address1,
    "spec/core/multi-memory/address1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_align0,
    "spec/core/multi-memory/align0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_binary0,
    "spec/core/multi-memory/binary0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_data0,
    "spec/core/multi-memory/data0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_data1,
    "spec/core/multi-memory/data1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_data_drop0,
    "spec/core/multi-memory/data_drop0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_exports0,
    "spec/core/multi-memory/exports0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_float_exprs0,
    "spec/core/multi-memory/float_exprs0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_float_exprs1,
    "spec/core/multi-memory/float_exprs1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_float_memory0,
    "spec/core/multi-memory/float_memory0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_imports0,
    "spec/core/multi-memory/imports0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_imports1,
    "spec/core/multi-memory/imports1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_imports2,
    "spec/core/multi-memory/imports2.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_imports3,
    "spec/core/multi-memory/imports3.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_imports4,
    "spec/core/multi-memory/imports4.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_linking0,
    "spec/core/multi-memory/linking0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_linking1,
    "spec/core/multi-memory/linking1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_linking2,
    "spec/core/multi-memory/linking2.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_linking3,
    "spec/core/multi-memory/linking3.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_load1,
    "spec/core/multi-memory/load1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_multi,
    "spec/core/multi-memory/memory-multi.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_copy0,
    "spec/core/multi-memory/memory_copy0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_copy1,
    "spec/core/multi-memory/memory_copy1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_fill0,
    "spec/core/multi-memory/memory_fill0.wast",
    0
);
#[test]
fn spec_complete_multi_memory_memory_grow() {
    check_script_with_limits(
        include_str!("spec/core/multi-memory/memory_grow.wast"),
        0,
        KernelLimits {
            memory_growth_pages: 16,
            ..script_limits()
        },
    );
}
new_feature_spec!(
    spec_complete_multi_memory_memory_init0,
    "spec/core/multi-memory/memory_init0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_size0,
    "spec/core/multi-memory/memory_size0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_size1,
    "spec/core/multi-memory/memory_size1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_size2,
    "spec/core/multi-memory/memory_size2.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_size3,
    "spec/core/multi-memory/memory_size3.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_size_import,
    "spec/core/multi-memory/memory_size_import.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_trap0,
    "spec/core/multi-memory/memory_trap0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_memory_trap1,
    "spec/core/multi-memory/memory_trap1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_start0,
    "spec/core/multi-memory/start0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_store0,
    "spec/core/multi-memory/store0.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_store1,
    "spec/core/multi-memory/store1.wast",
    0
);
new_feature_spec!(
    spec_complete_multi_memory_store2,
    "spec/core/multi-memory/store2.wast",
    0
);
new_feature_spec!(spec_complete_names, "spec/core/names.wast", 0);
#[test]
fn spec_complete_nop() {
    check_script_with_limits(
        include_str!("spec/core/nop.wast"),
        0,
        KernelLimits {
            memory_growth_pages: 64,
            ..script_limits()
        },
    );
}
new_feature_spec!(
    spec_complete_obsolete_keywords,
    "spec/core/obsolete-keywords.wast",
    0
);
new_feature_spec!(spec_complete_ref, "spec/core/ref.wast", 0);
new_feature_spec!(spec_complete_ref_is_null, "spec/core/ref_is_null.wast", 0);
new_feature_spec!(
    spec_complete_relaxed_simd_i32x4_relaxed_trunc,
    "spec/core/relaxed-simd/i32x4_relaxed_trunc.wast",
    0
);
new_feature_spec!(spec_complete_return, "spec/core/return.wast", 0);
new_feature_spec!(
    spec_complete_skip_stack_guard_page,
    "spec/core/skip-stack-guard-page.wast",
    0
);
new_feature_spec!(spec_complete_stack, "spec/core/stack.wast", 0);
new_feature_spec!(spec_complete_store, "spec/core/store.wast", 0);
new_feature_spec!(spec_complete_switch, "spec/core/switch.wast", 0);
new_feature_spec!(spec_complete_token, "spec/core/token.wast", 0);
new_feature_spec!(spec_complete_traps, "spec/core/traps.wast", 0);
new_feature_spec!(spec_complete_type_canon, "spec/core/type-canon.wast", 0);
new_feature_spec!(spec_complete_type, "spec/core/type.wast", 0);
new_feature_spec!(spec_complete_unreachable, "spec/core/unreachable.wast", 0);
new_feature_spec!(
    spec_complete_unreached_invalid,
    "spec/core/unreached-invalid.wast",
    0
);
new_feature_spec!(
    spec_complete_unreached_valid,
    "spec/core/unreached-valid.wast",
    0
);
new_feature_spec!(spec_complete_unwind, "spec/core/unwind.wast", 0);
new_feature_spec!(
    spec_complete_utf8_custom_section_id,
    "spec/core/utf8-custom-section-id.wast",
    0
);
new_feature_spec!(
    spec_complete_utf8_import_field,
    "spec/core/utf8-import-field.wast",
    0
);
new_feature_spec!(
    spec_complete_utf8_import_module,
    "spec/core/utf8-import-module.wast",
    0
);
new_feature_spec!(
    spec_complete_utf8_invalid_encoding,
    "spec/core/utf8-invalid-encoding.wast",
    0
);

new_feature_spec!(spec_wide_arithmetic, "spec/wide-arithmetic.wast", 99);

#[test]
fn spec_custom_pages_custom_page_sizes_invalid() {
    check_script_with_limits(
        include_str!("spec/custom-page-sizes/custom-page-sizes-invalid.wast"),
        0,
        KernelLimits {
            memory_growth_pages: 65_536,
            ..script_limits()
        },
    );
}

#[test]
fn spec_custom_pages_custom_page_sizes() {
    check_script_with_limits(
        include_str!("spec/custom-page-sizes/custom-page-sizes.wast"),
        31,
        KernelLimits {
            memory_growth_pages: 131_072,
            ..script_limits()
        },
    );
}

#[test]
fn spec_custom_pages_memory_max() {
    check_script_with_limits(
        include_str!("spec/custom-page-sizes/memory_max.wast"),
        0,
        KernelLimits {
            memory_growth_pages: 65_536,
            ..script_limits()
        },
    );
}

#[test]
fn spec_custom_pages_memory_max_i64() {
    check_script_with_limits(
        include_str!("spec/custom-page-sizes/memory_max_i64.wast"),
        0,
        KernelLimits {
            memory_growth_pages: 65_536,
            ..script_limits()
        },
    );
}
