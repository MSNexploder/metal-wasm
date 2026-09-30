(module
    (memory 1 1)
    (data $seed "\05\11\23\37\4d\5b\72\8a\a1\b8\cf\e5\0d\2f\53\79")
    (global $state (mut i32) (i32.const 0))

    (func $start
        i32.const 0
        i32.const 0
        i32.const 16
        memory.init $seed
        data.drop $seed
        i32.const 32
        i32.const 0
        i32.store
        i32.const 36
        i32.const 0
        i32.store
        i32.const 40
        i32.const 0
        i32.store
        i32.const 44
        i32.const 0
        i32.store
        i32.const 0
        global.set $state)
    (start $start)

    ;; Reset the mutable global and four working memory cells for one sample.
    (func (export "init") (param i32) (result i32)
        local.get 0
        global.set $state
        i32.const 32
        local.get 0
        i32.const -1515870811
        i32.xor
        i32.store
        i32.const 36
        local.get 0
        i32.const 1013904242
        i32.xor
        i32.store
        i32.const 40
        local.get 0
        i32.const -626627285
        i32.xor
        i32.store
        i32.const 44
        local.get 0
        i32.const 2027874020
        i32.xor
        i32.store
        local.get 0)

    ;; Preserve global and memory state across all calls for this lane.
    (func (export "step") (param i32) (result i32) (local i32 i32)
        global.get $state
        local.get 0
        i32.xor
        local.set 1
        i32.const 0
        local.set 2
        loop
            local.get 1
            global.get $state
            i32.xor
            i32.const 5
            i32.rotl
            i32.const 32
            local.get 2
            i32.const 3
            i32.and
            i32.const 2
            i32.shl
            i32.add
            i32.load
            i32.add
            local.get 2
            i32.const 15
            i32.and
            i32.load8_u
            i32.xor
            i32.const -1640531535
            i32.mul
            local.get 2
            i32.add
            local.set 1

            i32.const 32
            local.get 2
            i32.const 3
            i32.and
            i32.const 2
            i32.shl
            i32.add
            local.get 1
            i32.store
            local.get 1
            i32.const 13
            i32.rotl
            global.set $state

            local.get 2
            i32.const 1
            i32.add
            local.tee 2
            i32.const 24
            i32.ne
            br_if 0
        end
        local.get 1
        global.get $state
        i32.xor)

    (func (export "digest") (param i32) (result i32)
        local.get 0
        global.get $state
        i32.xor
        i32.const 32
        i32.load
        i32.xor
        i32.const 36
        i32.load
        i32.const 3
        i32.rotl
        i32.xor
        i32.const 40
        i32.load
        i32.const 7
        i32.rotl
        i32.xor
        i32.const 44
        i32.load
        i32.const 11
        i32.rotl
        i32.xor)
)
