;; f32 operations test

(module
  (memory (export "memory") 1)

  ;; f32 add: 1.5 + 2.5 = 4.0
  (func (export "f32_add_test") (result i32)
    f32.const 1.5
    f32.const 2.5
    f32.add
    i32.reinterpret_f32
  )

  ;; f32 sub: 5.0 - 3.0 = 2.0
  (func (export "f32_sub_test") (result i32)
    f32.const 5.0
    f32.const 3.0
    f32.sub
    i32.reinterpret_f32
  )

  ;; f32 mul: 3.0 * 4.0 = 12.0
  (func (export "f32_mul_test") (result i32)
    f32.const 3.0
    f32.const 4.0
    f32.mul
    i32.reinterpret_f32
  )

  ;; f32 div: 10.0 / 4.0 = 2.5
  (func (export "f32_div_test") (result i32)
    f32.const 10.0
    f32.const 4.0
    f32.div
    i32.reinterpret_f32
  )

  ;; f32 neg: -(3.0) = -3.0
  (func (export "f32_neg_test") (result i32)
    f32.const 3.0
    f32.neg
    i32.reinterpret_f32
  )

  ;; f32 abs: |(-5.0)| = 5.0
  (func (export "f32_abs_test") (result i32)
    f32.const -5.0
    f32.abs
    i32.reinterpret_f32
  )

  ;; f32 floor: floor(3.7) = 3.0
  (func (export "f32_floor_test") (result i32)
    f32.const 3.7
    f32.floor
    i32.reinterpret_f32
  )

  ;; f32 ceil: ceil(3.2) = 4.0
  (func (export "f32_ceil_test") (result i32)
    f32.const 3.2
    f32.ceil
    i32.reinterpret_f32
  )

  ;; f32 trunc: trunc(3.7) = 3.0
  (func (export "f32_trunc_test") (result i32)
    f32.const 3.7
    f32.trunc
    i32.reinterpret_f32
  )

  ;; f32 nearest: nearest(3.5) = 4.0 (round to even)
  (func (export "f32_nearest_test") (result i32)
    f32.const 3.5
    f32.nearest
    i32.reinterpret_f32
  )

  ;; f32 eq: 3.0 == 3.0 -> 1
  (func (export "f32_eq_test") (result i32)
    f32.const 3.0
    f32.const 3.0
    f32.eq
  )

  ;; f32 lt: 2.0 < 3.0 -> 1
  (func (export "f32_lt_test") (result i32)
    f32.const 2.0
    f32.const 3.0
    f32.lt
  )

  ;; f32 min: min(2.0, 3.0) = 2.0
  (func (export "f32_min_test") (result i32)
    f32.const 2.0
    f32.const 3.0
    f32.min
    i32.reinterpret_f32
  )

  ;; f32 max: max(2.0, 3.0) = 3.0
  (func (export "f32_max_test") (result i32)
    f32.const 2.0
    f32.const 3.0
    f32.max
    i32.reinterpret_f32
  )

  ;; f32 sqrt: sqrt(4.0) = 2.0
  (func (export "f32_sqrt_test") (result i32)
    f32.const 4.0
    f32.sqrt
    i32.reinterpret_f32
  )

  ;; i32_trunc_f32_s: trunc(-3.7) = -3
  (func (export "i32_trunc_f32_s_test") (result i32)
    f32.const -3.7
    i32.trunc_f32_s
  )

  ;; f32_convert_i32_s: convert(42) = 42.0
  (func (export "f32_convert_i32_s_test") (result i32)
    i32.const 42
    f32.convert_i32_s
    i32.reinterpret_f32
  )
)

;; f32 add: 4.0 = 0x40800000
(assert_return (invoke "f32_add_test") (i32.const 0x40800000))

;; f32 sub: 2.0 = 0x40000000
(assert_return (invoke "f32_sub_test") (i32.const 0x40000000))

;; f32 mul: 12.0 = 0x41400000
(assert_return (invoke "f32_mul_test") (i32.const 0x41400000))

;; f32 div: 2.5 = 0x40200000
(assert_return (invoke "f32_div_test") (i32.const 0x40200000))

;; f32 neg: -3.0 = 0xC0400000
(assert_return (invoke "f32_neg_test") (i32.const 0xC0400000))

;; f32 abs: 5.0 = 0x40A00000
(assert_return (invoke "f32_abs_test") (i32.const 0x40A00000))

;; f32 floor: 3.0 = 0x40400000
(assert_return (invoke "f32_floor_test") (i32.const 0x40400000))

;; f32 ceil: 4.0 = 0x40800000
(assert_return (invoke "f32_ceil_test") (i32.const 0x40800000))

;; f32 trunc: 3.0 = 0x40400000
(assert_return (invoke "f32_trunc_test") (i32.const 0x40400000))

;; f32 nearest: 4.0 = 0x40800000 (3.5 rounds to 4.0)
(assert_return (invoke "f32_nearest_test") (i32.const 0x40800000))

;; f32 eq
(assert_return (invoke "f32_eq_test") (i32.const 1))

;; f32 lt
(assert_return (invoke "f32_lt_test") (i32.const 1))

;; f32 min: 2.0 = 0x40000000
(assert_return (invoke "f32_min_test") (i32.const 0x40000000))

;; f32 max: 3.0 = 0x40400000
(assert_return (invoke "f32_max_test") (i32.const 0x40400000))

;; f32 sqrt: 2.0 = 0x40000000
(assert_return (invoke "f32_sqrt_test") (i32.const 0x40000000))

;; i32_trunc_f32_s: -3
(assert_return (invoke "i32_trunc_f32_s_test") (i32.const -3))

;; f32_convert_i32_s: 42.0 = 0x42280000
(assert_return (invoke "f32_convert_i32_s_test") (i32.const 0x42280000))
