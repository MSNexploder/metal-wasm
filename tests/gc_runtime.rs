use metal_wasm::{Error, Kernel, KernelLimits, Value};
use std::sync::Mutex;

static GPU_LOCK: Mutex<()> = Mutex::new(());

fn kernel(source: &str, heap_words: usize) -> Kernel {
    Kernel::new_with_limits(
        &wat::parse_str(source).unwrap(),
        "run",
        KernelLimits {
            gc_heap_words: heap_words,
            ..KernelLimits::default()
        },
    )
    .unwrap()
}

#[test]
fn bounded_heap_reuses_dead_objects_and_traps_when_all_objects_are_live() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let source = r#"(module
      (type $box (struct (field i32)))
      (func (export "run") (param $n i32) (result i32) (local $i i32) (local $sum i32)
        (loop $next
          local.get $sum local.get $i struct.new $box struct.get $box 0 i32.add local.set $sum
          local.get $i i32.const 1 i32.add local.tee $i local.get $n i32.lt_u br_if $next)
        local.get $sum))"#;
    let compiled = kernel(source, 24);
    let inputs = [1, 37, 1000, 10000];
    assert_eq!(
        compiled.run(&inputs).unwrap(),
        inputs.map(|n| (0..n).sum::<u32>())
    );
    let live = kernel(
        r#"(module
      (type $node (struct (field (ref null $node))))
      (func (export "run") (param $n i32) (result i32) (local $head (ref null $node))
        (loop $next
          local.get $head struct.new $node local.set $head
          local.get $n i32.const 1 i32.sub local.tee $n br_if $next)
        i32.const 1))"#,
        24,
    );
    assert_eq!(live.run(&[5]).unwrap(), [1]);
    assert!(matches!(
        live.run(&[7]),
        Err(Error::Trap {
            input_index: 0,
            reason: "GC resource limit exceeded"
        })
    ));
    let wasm = wat::parse_str(source).unwrap();
    for words in [0, 3] {
        assert!(matches!(
            Kernel::new_with_limits(
                &wasm,
                "run",
                KernelLimits {
                    gc_heap_words: words,
                    ..KernelLimits::default()
                }
            ),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn caller_operand_and_exception_payload_survive_callee_collection() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let compiled = kernel(
        r#"(module
      (type $box (struct (field i32)))
      (tag $tag (param (ref $box)))
      (func $churn (param $n i32)
        (loop $next
          local.get $n struct.new $box drop
          local.get $n i32.const 1 i32.sub local.tee $n br_if $next))
      (func (export "run") (param $n i32) (result i32) (local $ex exnref) (local $first i32)
        i32.const 47 struct.new $box
        local.get $n call $churn
        struct.get $box 0 local.set $first
        (block (result exnref)
          (try_table (catch_all_ref 0)
            i32.const 83 struct.new $box throw $tag)
          unreachable)
        local.set $ex
        local.get $n call $churn
        (block (result (ref $box))
          (try_table (catch $tag 0) local.get $ex throw_ref)
          unreachable)
        struct.get $box 0 local.get $first i32.add))"#,
        24,
    );
    assert_eq!(compiled.run(&[100, 1000, 10000]).unwrap(), [130; 3]);
}

#[test]
fn collection_preserves_live_cycles_and_reclaims_unreachable_cycles() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let compiled = kernel(
        r#"(module
      (type $node (struct (field (mut (ref null $node))) (field i32)))
      (func $cycle (param $value i32) (result (ref $node)) (local $node (ref null $node))
        ref.null $node local.get $value struct.new $node local.set $node
        local.get $node local.get $node struct.set $node 0
        local.get $node ref.as_non_null)
      (func (export "run") (param $n i32) (result i32) (local $live (ref null $node))
        i32.const 91 call $cycle local.set $live
        (loop $next
          local.get $n call $cycle drop
          local.get $n i32.const 1 i32.sub local.tee $n br_if $next)
        local.get $live struct.get $node 0 local.get $live ref.eq
        if (result i32) local.get $live struct.get $node 1 else i32.const -1 end))"#,
        24,
    );
    assert_eq!(compiled.run(&[100, 1000, 10000]).unwrap(), [91; 3]);
}

#[test]
fn provider_globals_and_tables_keep_start_objects_alive() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let provider = wat::parse_str(
        r#"(module
      (type $node (struct (field (mut (ref null $node))) (field i32)))
      (global $root (export "root") (mut (ref null $node)) (ref.null $node))
      (table $table 1 (ref null $node))
      (func $start (local $node (ref null $node))
        ref.null $node i32.const 17 struct.new $node global.set $root
        ref.null $node i32.const 29 struct.new $node local.set $node
        local.get $node local.get $node struct.set $node 0
        i32.const 0 local.get $node table.set $table)
      (start $start)
      (func (export "read") (result i32)
        global.get $root struct.get $node 1
        i32.const 0 table.get $table struct.get $node 0 struct.get $node 1 i32.add))"#,
    )
    .unwrap();
    let root = wat::parse_str(
        r#"(module
      (type $node (struct (field (mut (ref null $node))) (field i32)))
      (import "provider" "root" (global $root (mut (ref null $node))))
      (import "provider" "read" (func $read (result i32)))
      (type $box (struct (field i32)))
      (func (export "run") (param $n i32) (result i32)
        (loop $next
          local.get $n struct.new $box drop
          local.get $n i32.const 1 i32.sub local.tee $n br_if $next)
        call $read global.get $root struct.get $node 1 i32.add))"#,
    )
    .unwrap();
    let compiled = Kernel::new_with_imports_and_limits(
        &root,
        "run",
        &[("provider", &provider)],
        KernelLimits {
            gc_heap_words: 32,
            ..KernelLimits::default()
        },
    )
    .unwrap();
    assert_eq!(compiled.run(&[100, 1000, 10000]).unwrap(), [63; 3]);
}

#[test]
fn passive_elements_remain_roots_until_drop() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let compiled = kernel(
        r#"(module
      (type $box (struct (field i32)))
      (type $refs (array (ref $box)))
      (type $bytes (array i8))
      (elem $segment (ref $box) (struct.new $box (i32.const 77)))
      (func (export "run") (param $n i32) (result i32) (local $saved i32)
        (loop $next
          local.get $n struct.new $box drop
          local.get $n i32.const 1 i32.sub local.tee $n br_if $next)
        i32.const 0 i32.const 1 array.new_elem $refs $segment
        i32.const 0 array.get $refs struct.get $box 0 local.set $saved
        elem.drop $segment
        i32.const 21 array.new_default $bytes array.len local.get $saved i32.add))"#,
        24,
    );
    assert_eq!(compiled.run(&[100, 1000]).unwrap(), [98; 2]);
}

#[test]
fn failed_array_mutation_does_not_change_imported_host_snapshot() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let compiled = kernel(
        r#"(module
      (type $array (array (mut i32)))
      (data $data "abcdefghijkl")
      (func (export "run") (param $input (ref null $array)) (param $fail i32)
        (result (ref $array) i32) (local $old i32)
        local.get $input ref.is_null
        if i32.const 7 i32.const 3 array.new $array local.set $input end
        local.get $input i32.const 0 array.get $array local.set $old
        local.get $fail
        if
          local.get $input i32.const 0 i32.const 99 array.set $array
          local.get $fail i32.const 1 i32.eq
          if local.get $input i32.const 1 i32.const 42 i32.const 3 array.fill $array end
          local.get $fail i32.const 2 i32.eq
          if local.get $input i32.const 1 local.get $input i32.const 0 i32.const 3 array.copy $array $array end
          local.get $fail i32.const 3 i32.eq
          if local.get $input i32.const 1 i32.const 0 i32.const 3 array.init_data $array $data end
        end
        local.get $input ref.as_non_null local.get $old))"#,
        24,
    );
    let first = compiled
        .run_values(&[vec![Value::GcRef(None), Value::I32(0)]])
        .unwrap();
    assert_eq!(first[0][1], Value::I32(7));
    for operation in 1..=3 {
        assert!(matches!(
            compiled.run_values(&[vec![first[0][0], Value::I32(operation)]]),
            Err(Error::Trap {
                input_index: 0,
                reason: "out of bounds array access"
            })
        ));
        let retry = compiled
            .run_values(&[vec![first[0][0], Value::I32(0)]])
            .unwrap();
        assert_eq!(retry[0][1], Value::I32(7));
    }
}

#[test]
fn abstract_gc_types_work_without_declared_object_types() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let compiled = kernel(
        r#"(module
      (global $null anyref (ref.null any))
      (table 1 eqref)
      (func (export "run") (param i32) (result i32) (local anyref)
        global.get $null local.set 1
        local.get 1 ref.test (ref null struct)
        i32.const 0 table.get ref.test (ref array) i32.add))"#,
        16,
    );
    assert_eq!(compiled.run(&[0, 1]).unwrap(), [1, 1]);
}

#[test]
fn elements_read_per_lane_gc_globals_during_initialization() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let compiled = kernel(
        r#"(module
      (type $box (struct (field i32)))
      (type $refs (array (ref $box)))
      (global $root (ref $box) (struct.new $box (i32.const 11)))
      (table 1 (ref null $box))
      (elem (i32.const 0) (ref $box) (global.get $root))
      (elem $segment (ref $box) (global.get $root))
      (func (export "run") (param $n i32) (result i32)
        (loop $next
          local.get $n struct.new $box drop
          local.get $n i32.const 1 i32.sub local.tee $n br_if $next)
        global.get $root struct.get $box 0
        i32.const 0 table.get global.get $root ref.eq i32.add
        i32.const 0 i32.const 1 array.new_elem $refs $segment
        i32.const 0 array.get $refs global.get $root ref.eq i32.add))"#,
        24,
    );
    assert_eq!(compiled.run(&[100, 1000]).unwrap(), [13; 2]);
}

#[test]
fn active_segment_offsets_use_globals_initialized_after_gc_constructors() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let compiled = kernel(
        r#"(module
      (type $box (struct (field i32)))
      (global (ref $box) (struct.new $box (i32.const 11)))
      (global $table_offset i32 (i32.const 1))
      (global $data_offset i32 (i32.const 5))
      (table 2 funcref)
      (memory (export "memory") 1)
      (elem (global.get $table_offset) func $target)
      (data (global.get $data_offset) "\2a")
      (func $target)
      (func (export "run") (param i32) (result i32)
        i32.const 0 table.get ref.is_null))"#,
        16,
    );
    let (values, memories) = compiled
        .run_values_with_memory(&[vec![Value::I32(0)]], "memory", 0..6)
        .unwrap();
    assert_eq!(values, [vec![Value::I32(1)]]);
    assert_eq!(memories, [vec![0, 0, 0, 0, 0, 42]]);
}
