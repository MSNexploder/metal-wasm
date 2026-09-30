(module
  (memory 261 261 (pagesize 1))
  (func (export "calculate") (param $input i32) (result i32)
    (local $value i32) (local $round i32) (local $index i32)
    i32.const 0 local.get $input memory.size memory.fill
    local.get $input i32.const 0x9e3779b9 i32.xor local.set $value
    loop $again
      local.get $value local.get $round i32.xor
      i32.const 255 i32.and local.set $index
      local.get $value i32.const 5 i32.rotl
      local.get $index i32.load8_u i32.add
      i32.const 0x85ebca6b i32.mul local.set $value
      local.get $index local.get $value i32.const 24 i32.shr_u i32.store8
      local.get $round i32.const 1 i32.add local.tee $round
      i32.const 128 i32.lt_u br_if $again
    end
    local.get $value memory.size i32.xor))
