#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

unsafe extern "C" {
    fn host_mix(value: u32) -> u32;
}

#[unsafe(no_mangle)]
pub extern "C" fn calculate(input: u32) -> u32 {
    unsafe { host_mix(input) }
}
