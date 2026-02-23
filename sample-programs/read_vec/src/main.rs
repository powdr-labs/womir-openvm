use womir_guest_io::{debug_print, read_bytes};

pub fn main() {
    debug_print("Hello, world!");

    let v = read_bytes();
    assert_eq!(v.len(), 4);
    assert_eq!(v[0], 0xcc);
    assert_eq!(v[1], 0xbb);
    assert_eq!(v[2], 0xaa);
    assert_eq!(v[3], 0xff);

    let v = read_bytes();
    assert_eq!(v.len(), 4);
    assert_eq!(v[0], 0x66);
    assert_eq!(v[1], 0x00);
    assert_eq!(v[2], 0xdd);
    assert_eq!(v[3], 0xee);
}
