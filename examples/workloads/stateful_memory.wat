(module
    (memory 1)
    (data (i32.const 0) "\05\11\23\37\4d\5b\72\8a\a1\b8\cf\e5\0d\2f\53\79")
    (global $seed (mut i32) (i32.const -1640531535))

    (func $step (param i32 i32 i32) (result i32)
        local.get 0
        i32.const 7
        i32.rotl
        local.get 1
        i32.const 15
        i32.and
        i32.load8_u
        i32.xor
        local.get 2
        i32.add
        i32.const -1640531535
        i32.mul)

    (func (export "calculate") (param i32) (result i32) (local i32 i32 i32)
        global.get $seed
        local.set 1
        local.get 0
        global.get $seed
        i32.xor
        local.set 2
        i32.const 0
        local.set 3
        loop
            i32.const 64
            local.get 3
            i32.const 2
            i32.shl
            i32.add
            i32.const 0
            i32.store
            local.get 3
            i32.const 1
            i32.add
            local.tee 3
            i32.const 16
            i32.ne
            br_if 0
        end
        i32.const 0
        local.set 3
        loop
            local.get 2
            local.get 0
            local.get 3
            i32.add
            i32.const 64
            local.get 3
            i32.const 15
            i32.and
            i32.const 2
            i32.shl
            i32.add
            i32.load
            call $step
            global.get $seed
            i32.add
            local.set 2
            i32.const 64
            local.get 3
            i32.const 15
            i32.and
            i32.const 2
            i32.shl
            i32.add
            local.get 2
            i32.store
            local.get 2
            i32.const 13
            i32.rotl
            global.set $seed
            local.get 3
            i32.const 1
            i32.add
            local.tee 3
            i32.const 24
            i32.ne
            br_if 0
        end
        local.get 2
        global.get $seed
        i32.xor
        local.set 2
        local.get 1
        global.set $seed
        local.get 2))
