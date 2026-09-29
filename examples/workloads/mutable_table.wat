(module
    (type $step (func (param i32 i32) (result i32)))
    (table 4 4 funcref)
    (func $rotate (type $step)
        local.get 0 i32.const 7 i32.rotl
        local.get 1 i32.xor)
    (func $multiply (type $step)
        local.get 0 i32.const -1640531535 i32.mul
        local.get 1 i32.add)
    (func $shift (type $step)
        local.get 0 i32.const 3 i32.shr_u
        local.get 1 i32.xor)
    (func $scramble (type $step)
        local.get 0 i32.const 13 i32.rotl
        local.get 1 i32.const -2048144789 i32.mul
        i32.xor)
    (elem $choices funcref
        (ref.func $rotate)
        (ref.func $multiply)
        (ref.func $shift)
        (ref.func $scramble))

    (func (export "calculate") (param i32) (result i32) (local i32 i32)
        i32.const 0 i32.const 0 i32.const 4 table.init $choices
        local.get 0 i32.const 3 i32.and
        ref.func $scramble
        table.set 0
        local.get 0 local.set 1
        i32.const 0 local.set 2
        loop
            local.get 1
            local.get 2
            local.get 0 local.get 2 i32.add i32.const 3 i32.and
            call_indirect (type $step)
            local.set 1
            local.get 2 i32.const 1 i32.add local.tee 2
            i32.const 24 i32.ne br_if 0
        end
        local.get 1
        i32.const 0 table.get 0 ref.is_null
        i32.xor))
