extern crate alloc;

use alloc::string::String;
use serde::Deserialize;
use womir_guest_io::{debug_print, read};

/// A struct whose postcard encoding differs from raw field layout:
/// - `label` is length-prefixed (varint length + UTF-8 bytes)
/// - `value` uses varint encoding (1–10 bytes, not fixed 8)
#[derive(Deserialize, PartialEq)]
struct SampleData {
    label: String,
    value: u64,
}

pub fn main() {
    debug_print("read_serde: reading postcard-serialized struct");

    let data: SampleData = read();
    assert_eq!(data.label.as_str(), "hello");
    assert_eq!(data.value, 1_000_000);

    debug_print("read_serde: success");
}
