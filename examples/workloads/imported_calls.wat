(module
    (import "math" "mix" (func $mix (param i32 i32) (result i32)))
    (func (export "calculate") (param i32) (result i32) (local i32 i32)
        local.get 0
        local.set 1
        i32.const 0
        local.set 2
        loop
            local.get 1
            local.get 0
            local.get 2
            i32.xor
            call $mix
            local.set 1
            local.get 0
            i32.const 1
            i32.and
            if
                local.get 1
                local.get 2
                i32.xor
                local.set 1
            else
                local.get 1
                i32.const 3
                i32.rotr
                local.set 1
            end
            local.get 2
            i32.const 1
            i32.add
            local.tee 2
            i32.const 16
            i32.ne
            br_if 0
        end
        local.get 1))
