(module
  (type $step (func (param i32) (result i32)))
  (tag $recover (param i32))
  (func $multiply (type $step) (local i32)
    local.get 0 i32.const 0x9e3779b1 i32.mul i32.const 17 i32.add local.set 1
    local.get 1 i32.const 7 i32.and i32.eqz
    if
      local.get 1 i32.const 0x85ebca6b i32.xor throw $recover
    end
    local.get 1)
  (func $rotate (type $step) (local i32)
    local.get 0 i32.const 5 i32.rotl i32.const 0x85ebca6b i32.xor local.set 1
    local.get 1 i32.const 7 i32.and i32.eqz
    if
      local.get 1 i32.const 0x9e3779b1 i32.add throw $recover
    end
    local.get 1)
  (elem declare func $multiply $rotate)
  (func (export "calculate") (param i32) (result i32)
    (local $value i32) (local $round i32) (local $function (ref $step))
    local.get 0 local.set $value
    loop $again
      local.get 0 local.get $round i32.xor i32.const 1 i32.and
      if (result (ref $step))
        ref.func $rotate
      else
        ref.func $multiply
      end
      local.set $function
      block $recover_result (result i32)
        try_table (result i32) (catch $recover $recover_result)
          local.get $value local.get $round i32.xor
          local.get $function call_ref $step
        end
      end
      local.set $value
      local.get $round i32.const 1 i32.add local.tee $round
      i32.const 32 i32.ne br_if $again
    end
    local.get $value))
