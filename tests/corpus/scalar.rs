#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[inline(never)]
fn mix(value: u32, salt: u32) -> u32 {
    value.rotate_left(5).wrapping_mul(0x9e37_79b1) ^ salt
}

#[unsafe(no_mangle)]
pub extern "C" fn calculate(input: u32) -> u32 {
    let mut value = input ^ 0x85eb_ca6b;
    for round in 0..16u32 {
        value = mix(value, input.wrapping_add(round));
        if input & round & 1 != 0 {
            value = value.rotate_right(3);
        } else {
            value = value.wrapping_add(round);
        }
    }
    value
}
