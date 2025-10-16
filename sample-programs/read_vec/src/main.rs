#[link(wasm_import_module = "env")]
unsafe extern "C" {
    pub safe fn __hint_input();
    pub unsafe fn __hint_store_vec(ptr: *mut u8, num_words: u32);
    pub unsafe fn __debug_print(ptr: *const u8, num_bytes: u32);
}

pub fn read_vec() -> Vec<u8> {
    __hint_input();
    let len = read_word();
    assert_eq!(len, 4);
    read_vec_by_len(len as usize)
}

// len in bytes
pub fn read_vec_by_len(len: usize) -> Vec<u8> {
    let num_words = len.div_ceil(4);
    let capacity = num_words * 4;

    let mut bytes: Vec<u8> = vec![0; capacity];
    unsafe { __hint_store_vec(bytes.as_mut_ptr(), num_words as u32) }
    // SAFETY: We populate a `Vec<u8>` by hintstore-ing `num_words` 4 byte words. We set the
    // length to `len` and don't care about the extra `capacity - len` bytes stored.
    unsafe {
        bytes.set_len(len);
    }
    bytes
}

pub fn read_word() -> u32 {
    let mut bytes = [0u8; 4];
    unsafe { __hint_store_vec(bytes.as_mut_ptr(), 1) }
    u32::from_le_bytes(bytes)
}

pub fn main() {
    let s = "Hello, world!";
    unsafe {
        __debug_print(s.as_ptr(), s.len() as u32);
    }

    let v = read_vec();
    assert_eq!(v.len(), 4);
    assert_eq!(v[0], 0xcc);
    assert_eq!(v[1], 0xbb);
    assert_eq!(v[2], 0xaa);
    assert_eq!(v[3], 0xff);

    let v = read_vec();
    assert_eq!(v.len(), 4);
    assert_eq!(v[0], 0x66);
    assert_eq!(v[1], 0x00);
    assert_eq!(v[2], 0xdd);
    assert_eq!(v[3], 0xee);
}
