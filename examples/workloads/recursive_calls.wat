(module
    (func $step (param i32 i32) (result i32) (local i32)
        local.get 1
        i32.const 5
        i32.rotl
        local.get 0
        i32.xor
        i32.const -1640531535
        i32.mul
        local.set 2
        local.get 0
        if (result i32)
            local.get 0
            i32.const 1
            i32.sub
            local.get 2
            call $step
            local.get 0
            i32.const -2048144789
            i32.mul
            i32.xor
        else
            local.get 2
        end)

    (func (export "calculate") (param i32) (result i32)
        local.get 0
        i32.const 15
        i32.and
        i32.const 8
        i32.add
        local.get 0
        call $step))
