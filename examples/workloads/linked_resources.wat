(module
    (type $step (func (param i32 i32) (result i32)))
    (import "state" "mem" (memory 1))
    (import "state" "tab" (table 2 funcref))
    (import "state" "g" (global $g (mut i32)))
    (func $multiply (type $step)
        local.get 0 i32.const -2048144789 i32.mul
        local.get 1 i32.add)
    (elem declare func $multiply)

    (func (export "calculate") (param i32) (result i32) (local i32 i32 i32)
        local.get 0
        i32.const 0 i32.load
        i32.xor
        global.set $g
        i32.const 64 i32.const 0 i32.const 16 memory.fill
        i32.const 1 ref.func $multiply table.set 0
        local.get 0 local.set 1
        i32.const 0 local.set 2
        loop
            i32.const 64
            local.get 2 i32.const 3 i32.and i32.const 2 i32.shl
            i32.add local.set 3

            local.get 1 global.get $g i32.xor
            local.get 2
            local.get 2 i32.const 1 i32.and
            call_indirect (type $step)
            local.get 3 i32.load
            i32.add
            i32.const -1640531535 i32.mul
            local.set 1

            local.get 3 local.get 1 i32.store
            local.get 1 i32.const 13 i32.rotl global.set $g
            local.get 2 i32.const 1 i32.add local.tee 2
            i32.const 24 i32.ne br_if 0
        end
        local.get 1 global.get $g i32.xor))
