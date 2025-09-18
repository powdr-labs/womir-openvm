;; Based on https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

(module
  (func $i64_popcnt (param $x i64) (result i64)
    ;; x = x - ((x >> 1) & 0x5555...5555)
    local.get $x
    local.get $x
    i64.const 1
    i64.shr_u
    i64.const 0x5555555555555555
    i64.and
    i64.sub
    local.tee $x

    ;; x = (x & 0x3333...3333) + ((x >> 2) & 0x3333...3333)
    i64.const 2
    i64.shr_u
    i64.const 0x3333333333333333
    i64.and
    local.get $x
    i64.const 0x3333333333333333
    i64.and
    i64.add
    local.tee $x

    ;; x = (x + (x >> 4)) & 0x0F0F...0F0F
    i64.const 4
    i64.shr_u
    local.get $x
    i64.add
    i64.const 0x0F0F0F0F0F0F0F0F
    i64.and

    ;; return (x * 0x0101...0101) >> 56
    i64.const 0x0101010101010101
    i64.mul
    i64.const 56
    i64.shr_u
  )
  (export "i64_popcnt" (func $i64_popcnt))
)
