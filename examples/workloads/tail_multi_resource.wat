(module
    (type $identity_type (func (param i32) (result i32)))
    (global $seed i32 (i32.add (i32.const 17) (i32.const 23)))
    (memory $small 1)
    (memory $wide i64 1)
    (table $functions i64 1 funcref)
    (func $identity (type $identity_type) local.get 0)
    (elem (table $functions) (i64.const 0) func $identity)
    (func $step (param $remaining i32) (param $value i32) (result i32)
        local.get $remaining
        i32.eqz
        if (result i32)
            local.get $value
            i64.const 0
            call_indirect $functions (type $identity_type)
        else
            i32.const 0
            local.get $value
            i32.store8 $small
            i64.const 0
            local.get $value
            i32.store8 $wide
            local.get $remaining
            i32.const 1
            i32.sub
            local.get $value
            i32.const 5
            i32.rotl
            i32.const 0
            i32.load8_u $small
            i32.xor
            i64.const 0
            i32.load8_u $wide
            i32.add
            global.get $seed
            i32.mul
            return_call $step
        end)
    (func (export "calculate") (param i32) (result i32)
        i32.const 16
        local.get 0
        return_call $step))
