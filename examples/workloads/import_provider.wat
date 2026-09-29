(module
    (func (export "mix") (param i32 i32) (result i32) (local i32)
        i32.const 0
        local.set 2
        loop
            local.get 0
            i32.const 5
            i32.rotl
            local.get 1
            i32.xor
            i32.const -1640531535
            i32.mul
            local.get 2
            i32.add
            local.set 0
            local.get 2
            i32.const 1
            i32.add
            local.tee 2
            i32.const 4
            i32.ne
            br_if 0
        end
        local.get 0))
