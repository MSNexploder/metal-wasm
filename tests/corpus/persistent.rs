#![no_std]

static mut STATE: u32 = 0x1234_5678;

#[unsafe(no_mangle)]
pub extern "C" fn step(input: u32) -> u32 {
    unsafe {
        STATE = (STATE ^ input).rotate_left(7).wrapping_mul(0x9e37_79b1);
        STATE
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn read() -> u32 {
    unsafe { STATE }
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}
