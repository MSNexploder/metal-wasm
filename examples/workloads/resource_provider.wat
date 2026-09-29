(module
    (type $step (func (param i32 i32) (result i32)))
    (memory (export "mem") 1)
    (data (i32.const 0) "\05\11\23\37")
    (table (export "tab") 2 funcref)
    (global (export "g") (mut i32) (i32.const 0))
    (func $rotate (type $step)
        local.get 0 i32.const 5 i32.rotl
        local.get 1 i32.xor)
    (elem (i32.const 0) func $rotate))
