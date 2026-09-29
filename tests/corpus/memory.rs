#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn calculate(input: u32) -> u32 {
    let mut bytes = [0u8; 64];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = input.wrapping_add(index as u32).rotate_left(3) as u8;
    }
    let mut value = input;
    for round in 0..32u32 {
        let index = ((input ^ round) & 63) as usize;
        value = value
            .rotate_left(7)
            .wrapping_mul(0x9e37_79b1)
            .wrapping_add(bytes[index] as u32);
    }
    value
}
