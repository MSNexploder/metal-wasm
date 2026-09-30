(module
  (type $base (sub (func (param i32) (result i32))))
  (type $multiply (sub $base (func (param i32) (result i32))))
  (type $rotate (sub $base (func (param i32) (result i32))))
  (tag $recover (param i32 i32))
  (table $functions 2 funcref)
  (table $exceptions 1 exnref)
  (table $external 1 externref)
  (func $multiply (type $multiply)
    local.get 0 i32.const 0x9e3779b1 i32.mul i32.const 17 i32.add)
  (func $rotate (type $rotate)
    local.get 0 i32.const 5 i32.rotl i32.const 0x85ebca6b i32.xor)
  (elem (table $functions) (i32.const 0) func $multiply $rotate)
  (func (export "calculate") (param i32) (result i32)
    (local $value i32) (local $round i32) (local $function funcref)
    (local $exception exnref)
    local.get 0 local.set $value
    loop $again
      local.get 0 local.get $round i32.xor i32.const 1 i32.and
      table.get $functions local.set $function
      block $dispatched
        block $multiply (result (ref $multiply))
          block $rotate (result funcref)
            local.get $function
            br_on_cast_fail $rotate funcref (ref $multiply)
            br $multiply
          end
          local.set $function
          local.get $value local.get $round i32.xor
          local.get $function ref.cast (ref $rotate) call_ref $rotate
          local.set $value
          br $dispatched
        end
        local.set $function
        local.get $value local.get $round i32.xor
        local.get $function ref.cast (ref $multiply) call_ref $multiply
        local.set $value
      end
      block $captured (result i32 i32 exnref)
        try_table (catch_ref $recover $captured)
          local.get $value local.get $round throw $recover
        end
        unreachable
      end
      local.set $exception drop drop
      i32.const 0 local.get $exception table.set $exceptions
      block $recovered (result i32 i32)
        try_table (catch $recover $recovered)
          i32.const 0 table.get $exceptions
          ref.cast (ref exn) throw_ref
        end
        unreachable
      end
      i32.xor
      i32.const 0 table.get $external ref.test nullexternref
      i32.add local.set $value
      local.get $round i32.const 1 i32.add local.tee $round
      i32.const 32 i32.ne br_if $again
    end
    local.get $value))
