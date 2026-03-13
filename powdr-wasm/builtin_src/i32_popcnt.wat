;; Based on https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

(module
  (func $i32_popcnt (param $x i32) (result i32)
    ;; x = x - ((x >> 1) & 0x5555_5555)
    local.get $x
    local.get $x
    i32.const 1
    i32.shr_u
    i32.const 0x55555555
    i32.and
    i32.sub
    local.tee $x

    ;; x = (x & 0x3333_3333) + ((x >> 2) & 0x3333_3333)
    i32.const 2
    i32.shr_u
    i32.const 0x33333333
    i32.and
    local.get $x
    i32.const 0x33333333
    i32.and
    i32.add
    local.tee $x

    ;; x = (x + (x >> 4)) & 0x0F0F_0F0F
    i32.const 4
    i32.shr_u
    local.get $x
    i32.add
    i32.const 0x0F0F0F0F
    i32.and

    ;; return (x * 0x0101_0101) >> 24
    i32.const 0x01010101
    i32.mul
    i32.const 24
    i32.shr_u
  )
  (export "i32_popcnt" (func $i32_popcnt))
)
