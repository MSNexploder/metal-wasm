#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { loop {} }

#[unsafe(no_mangle)]
pub extern "C" fn calculate(input: u32) -> u32 {
    let mut value = ((input as u128) << 64) | 0x9e37_79b9_7f4a_7c15;
    let mut fraction = input as i32 as f64 / 17.0;
    for round in 0..32u32 {
        value = value.wrapping_mul(0xd6e8_feb8_6659_fd93).rotate_left(13);
        value ^= (fraction as i64 as u64 as u128) << (round & 63);
        fraction = fraction * 0.5 + round as f64;
    }
    (value as u32) ^ ((value >> 64) as u32) ^ (fraction.to_bits() as u32)
}
