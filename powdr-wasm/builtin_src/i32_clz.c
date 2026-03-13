_Static_assert(sizeof(unsigned int) == 4,   "int must be 4 bytes");

typedef unsigned int uint32_t;

__attribute__((export_name("i32_clz")))
uint32_t i32_clz(uint32_t x) {
    if (x == 0) return 32;

    uint32_t n = 0;
    if (x <= 0x0000FFFFu) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFFu) { n += 8;  x <<= 8;  }
    if (x <= 0x0FFFFFFFu) { n += 4;  x <<= 4;  }
    if (x <= 0x3FFFFFFFu) { n += 2;  x <<= 2;  }
    if (x <= 0x7FFFFFFFu) { n += 1; }
    return n;
}
