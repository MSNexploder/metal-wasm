(module
    (memory 1)
    (data $seed "\05\11\23\37\4d\5b\72\8a\a1\b8\cf\e5\0d\2f\53\79")
    (global $key (mut i32) (i32.const 0))

    (func $start
        i32.const 0
        i32.const 0
        i32.const 16
        memory.init $seed
        data.drop $seed
        i32.const 17
        global.set $key)
    (start $start)

    (func (export "calculate") (param i32) (result i32) (local i32 i32)
        local.get 0
        global.get $key
        i32.xor
        local.set 1
        i32.const 64
        local.get 0
        i32.const 32
        memory.fill
        i32.const 96
        i32.const 64
        i32.const 32
        memory.copy
        i32.const 0
        local.set 2
        loop
            local.get 1
            i32.const 7
            i32.rotl
            local.get 0
            local.get 2
            i32.add
            i32.const 15
            i32.and
            i32.load8_u
            i32.xor
            i32.const 96
            local.get 2
            i32.const 31
            i32.and
            i32.add
            i32.load8_u
            i32.add
            i32.const -1640531535
            i32.mul
            local.get 2
            i32.add
            local.set 1
            local.get 2
            i32.const 1
            i32.add
            local.tee 2
            i32.const 24
            i32.ne
            br_if 0
        end
        local.get 1))
