use serde::Deserialize;
use womir_guest_io::{debug_print, read_vec};

/// A simple struct to demonstrate host-serialized, guest-deserialized data.
/// Postcard serialization of 4 u8 fields is exactly 4 bytes (1 byte each),
/// which fits in a single u32 hint item.
#[derive(Deserialize, PartialEq)]
struct Quad {
    a: u8,
    b: u8,
    c: u8,
    d: u8,
}

pub fn main() {
    debug_print("read_serde: reading postcard-serialized struct");

    // read_vec returns the raw bytes from the hint stream item.
    // The host wrote postcard::to_allocvec(&Quad { a: 1, b: 2, c: 3, d: 4 })
    // which is [1, 2, 3, 4], packed as u32 0x04030201.
    let bytes = read_vec();
    let q: Quad = postcard::from_bytes(&bytes).expect("postcard deserialization failed");
    assert_eq!(q.a, 1);
    assert_eq!(q.b, 2);
    assert_eq!(q.c, 3);
    assert_eq!(q.d, 4);

    debug_print("read_serde: success");
}
