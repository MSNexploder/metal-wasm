(module
    (func (export "calculate") (param i32) (result i32) (local v128 i32)
        local.get 0
        i32x4.splat
        v128.const i32x4 1 2 3 4
        i32x4.add
        local.set 1
        i32.const 0
        local.set 2
        loop
            local.get 1
            v128.const i32x4 7 11 13 17
            i32x4.mul
            local.get 1
            i32.const 5
            i32x4.shl
            v128.xor
            v128.const i32x4 0x9e3779b9 0x9e3779b9 0x9e3779b9 0x9e3779b9
            i32x4.add
            local.set 1
            local.get 2
            i32.const 1
            i32.add
            local.tee 2
            i32.const 32
            i32.ne
            br_if 0
        end
        local.get 1
        i32x4.extract_lane 0
        local.get 1
        i32x4.extract_lane 1
        i32.xor
        local.get 1
        i32x4.extract_lane 2
        i32.xor
        local.get 1
        i32x4.extract_lane 3
        i32.xor))
