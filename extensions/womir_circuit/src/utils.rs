/// Sign-extend a u32 value to `[u8; N]`.
/// For `N = 4`, this is equivalent to `c.to_le_bytes()`.
/// For `N > 4`, the upper bytes are sign-extended from bit 31.
#[inline(always)]
pub(crate) fn sign_extend_u32<const N: usize>(c: u32) -> [u8; N] {
    let sign_byte = if c & 0x8000_0000 != 0 { 0xFF } else { 0x00 };
    let le = c.to_le_bytes();
    std::array::from_fn(|i| if i < 4 { le[i] } else { sign_byte })
}
