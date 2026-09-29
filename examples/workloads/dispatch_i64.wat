(module
    (type $step (func (param i64 i64) (result i64)))
    (table 4 funcref)
    (elem (i32.const 0) $rotate_mix $xor_rotate $add_rotate $multiply_rotate)

    (func $rotate_mix (type $step)
        local.get 0
        i64.const 7
        i64.rotl
        local.get 1
        i64.xor
        i64.const 6364136223846793005
        i64.mul)

    (func $xor_rotate (type $step)
        local.get 0
        local.get 1
        i64.const 1442695040888963407
        i64.mul
        i64.xor
        i64.const 11
        i64.rotr
        i64.const 2862933555777941757
        i64.add)

    (func $add_rotate (type $step)
        local.get 0
        local.get 1
        i64.add
        i64.const 17
        i64.rotl
        i64.const 3935559000370003845
        i64.xor)

    (func $multiply_rotate (type $step)
        local.get 0
        i64.const 2862933555777941757
        i64.mul
        i64.const 23
        i64.rotr
        local.get 1
        i64.xor)

    (func (export "calculate") (param i32) (result i32) (local i64 i32)
        local.get 0
        i64.extend_i32_u
        local.set 1
        i32.const 0
        local.set 2
        loop
            local.get 1
            local.get 2
            i64.extend_i32_u
            local.get 0
            local.get 2
            i32.xor
            i32.const 3
            i32.and
            call_indirect (type $step)
            local.set 1
            local.get 2
            i32.const 1
            i32.add
            local.tee 2
            i32.const 24
            i32.ne
            br_if 0
        end
        local.get 1
        i32.wrap_i64
        local.get 1
        i64.const 32
        i64.shr_u
        i32.wrap_i64
        i32.xor))
