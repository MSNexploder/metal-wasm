#![no_std]
use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! { loop {} }

#[unsafe(no_mangle)]
pub extern "C" fn calculate(input: u32) -> u32 {
    let mut value = i32x4(input as i32, input.wrapping_add(1) as i32, !input as i32, 42);
    let multiplier = i32x4(7, 11, 13, 17);
    for round in 0..32 {
        value = i32x4_add(i32x4_mul(value, multiplier), i32x4_splat(round));
        value = v128_xor(value, i32x4_shuffle::<1, 2, 3, 0>(value, value));
        value = v128_bitselect(value, i32x4_shr(value, 3), i32x4_gt(value, i32x4_splat(0)));
    }
    (i32x4_extract_lane::<0>(value) ^ i32x4_extract_lane::<1>(value)
        ^ i32x4_extract_lane::<2>(value) ^ i32x4_extract_lane::<3>(value)) as u32
}
