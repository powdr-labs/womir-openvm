_Static_assert(sizeof(unsigned long long) == 8,   "long must be 8 bytes");

typedef unsigned long long uint64_t;

__attribute__((export_name("i64_clz")))
uint64_t i64_clz(uint64_t x) {
    if (x == 0) return 64;

    uint64_t n = 0;
    if (x <= 0x00000000FFFFFFFFull) { n += 32; x <<= 32; }
    if (x <= 0x0000FFFFFFFFFFFFull) { n += 16; x <<= 16; }
    if (x <= 0x00FFFFFFFFFFFFFFull) { n += 8;  x <<= 8;  }
    if (x <= 0x0FFFFFFFFFFFFFFFull) { n += 4;  x <<= 4;  }
    if (x <= 0x3FFFFFFFFFFFFFFFull) { n += 2;  x <<= 2;  }
    if (x <= 0x7FFFFFFFFFFFFFFFull) { n += 1; }
    return n;
}
