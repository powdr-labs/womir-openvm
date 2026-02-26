;; f64 operations test
;; f64 results are returned as (hi, lo) i32 pairs using i64 intermediary

(module
  (memory (export "memory") 1)

  ;; f64 add: 1.5 + 2.5 = 4.0
  (func (export "f64_add_test") (result i32 i32)
    (local $r i64)
    f64.const 1.5
    f64.const 2.5
    f64.add
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )

  ;; f64 sub: 5.0 - 3.0 = 2.0
  (func (export "f64_sub_test") (result i32 i32)
    (local $r i64)
    f64.const 5.0
    f64.const 3.0
    f64.sub
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )

  ;; f64 mul: 3.0 * 4.0 = 12.0
  (func (export "f64_mul_test") (result i32 i32)
    (local $r i64)
    f64.const 3.0
    f64.const 4.0
    f64.mul
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )

  ;; f64 div: 10.0 / 4.0 = 2.5
  (func (export "f64_div_test") (result i32 i32)
    (local $r i64)
    f64.const 10.0
    f64.const 4.0
    f64.div
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )

  ;; f64 neg: -(3.0) = -3.0
  (func (export "f64_neg_test") (result i32 i32)
    (local $r i64)
    f64.const 3.0
    f64.neg
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )

  ;; f64 eq: 3.0 == 3.0 -> 1
  (func (export "f64_eq_test") (result i32)
    f64.const 3.0
    f64.const 3.0
    f64.eq
  )

  ;; f64 lt: 2.0 < 3.0 -> 1
  (func (export "f64_lt_test") (result i32)
    f64.const 2.0
    f64.const 3.0
    f64.lt
  )

  ;; f64 floor: floor(3.7) = 3.0
  (func (export "f64_floor_test") (result i32 i32)
    (local $r i64)
    f64.const 3.7
    f64.floor
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )

  ;; f64 promote f32: 2.5f -> 2.5
  (func (export "f64_promote_f32_test") (result i32 i32)
    (local $r i64)
    f32.const 2.5
    f64.promote_f32
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )

  ;; f32 demote f64: 2.5 -> 2.5f
  (func (export "f32_demote_f64_test") (result i32)
    f64.const 2.5
    f32.demote_f64
    i32.reinterpret_f32
  )

  ;; i32_trunc_f64_s: trunc(-3.7) = -3
  (func (export "i32_trunc_f64_s_test") (result i32)
    f64.const -3.7
    i32.trunc_f64_s
  )

  ;; f64_convert_i32_s: convert(42) = 42.0
  (func (export "f64_convert_i32_s_test") (result i32 i32)
    (local $r i64)
    i32.const 42
    f64.convert_i32_s
    i64.reinterpret_f64
    local.tee $r
    i64.const 32
    i64.shr_u
    i32.wrap_i64
    local.get $r
    i32.wrap_i64
  )
)

;; f64 add: 4.0 = 0x4010000000000000 -> hi=0x40100000, lo=0x00000000
(assert_return (invoke "f64_add_test") (i32.const 0x40100000) (i32.const 0x00000000))

;; f64 sub: 2.0 = 0x4000000000000000 -> hi=0x40000000, lo=0x00000000
(assert_return (invoke "f64_sub_test") (i32.const 0x40000000) (i32.const 0x00000000))

;; f64 mul: 12.0 = 0x4028000000000000 -> hi=0x40280000, lo=0x00000000
(assert_return (invoke "f64_mul_test") (i32.const 0x40280000) (i32.const 0x00000000))

;; f64 div: 2.5 = 0x4004000000000000 -> hi=0x40040000, lo=0x00000000
(assert_return (invoke "f64_div_test") (i32.const 0x40040000) (i32.const 0x00000000))

;; f64 neg: -3.0 = 0xC008000000000000 -> hi=0xC0080000, lo=0x00000000
(assert_return (invoke "f64_neg_test") (i32.const 0xC0080000) (i32.const 0x00000000))

;; f64 eq
(assert_return (invoke "f64_eq_test") (i32.const 1))

;; f64 lt
(assert_return (invoke "f64_lt_test") (i32.const 1))

;; f64 floor: 3.0 = 0x4008000000000000 -> hi=0x40080000, lo=0x00000000
(assert_return (invoke "f64_floor_test") (i32.const 0x40080000) (i32.const 0x00000000))

;; f64 promote f32: 2.5 = 0x4004000000000000 -> hi=0x40040000, lo=0x00000000
(assert_return (invoke "f64_promote_f32_test") (i32.const 0x40040000) (i32.const 0x00000000))

;; f32 demote f64: 2.5f = 0x40200000
(assert_return (invoke "f32_demote_f64_test") (i32.const 0x40200000))

;; i32_trunc_f64_s
(assert_return (invoke "i32_trunc_f64_s_test") (i32.const -3))

;; f64_convert_i32_s: 42.0 = 0x4045000000000000 -> hi=0x40450000, lo=0x00000000
(assert_return (invoke "f64_convert_i32_s_test") (i32.const 0x40450000) (i32.const 0x00000000))
