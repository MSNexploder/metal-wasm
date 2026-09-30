(module
  (func (export "calculate") (param $input i32) (result i32)
    (local $low i64) (local $high i64) (local $round i32)
    (local $a_low i64) (local $a_high i64)
    (local $b_low i64) (local $b_high i64)
    local.get $input i64.extend_i32_u i64.const 32 i64.shl
    i64.const 0x9e3779b97f4a7c15 i64.xor local.set $low
    i64.const 0x85ebca6bc2b2ae35 local.set $high
    loop $again
      local.get $low i64.const 0xd6e8feb86659fd93 i64.mul_wide_u
      local.set $a_high local.set $a_low
      local.get $high i64.const 0x9e3779b97f4a7c15 i64.mul_wide_s
      local.set $b_high local.set $b_low
      local.get $a_low local.get $a_high
      local.get $b_low local.get $b_high i64.add128
      local.set $high local.set $low
      local.get $low local.get $high
      local.get $round i64.extend_i32_u i64.const 0 i64.sub128
      local.set $high local.set $low
      local.get $round i32.const 1 i32.add local.tee $round
      i32.const 128 i32.lt_u br_if $again
    end
    local.get $low local.get $high i64.xor
    local.get $low i64.const 32 i64.shr_u i64.xor i32.wrap_i64))
