(module
    (func $step (param f64 f64) (result f64)
        local.get 0
        local.get 1
        f64.mul
        local.get 0
        f64.const 0.125
        f64.add
        f64.div
        f64.const 0.0625
        f64.add)

    (func (export "calculate") (param i32) (result i32) (local f64 i32)
        local.get 0
        i32.const 2047
        i32.and
        f64.convert_i32_u
        f64.const 2048
        f64.div
        f64.const 0.25
        f64.add
        local.set 1
        i32.const 0
        local.set 2
        loop
            local.get 1
            local.get 0
            local.get 2
            i32.xor
            i32.const 1
            i32.and
            if (result f64)
                f64.const 1.015625
            else
                f64.const 0.984375
            end
            call $step
            local.set 1
            local.get 2
            i32.const 1
            i32.add
            local.tee 2
            i32.const 12
            i32.ne
            br_if 0
        end
        local.get 1
        f64.sqrt
        f64.const 100000
        f64.mul
        i32.trunc_sat_f64_u))
