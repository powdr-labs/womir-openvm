;; Tests for unaligned memory access.
;;
;; These functions load/store at addresses that are NOT naturally aligned
;; for the given operation width. Without support_unaligned_memory=true,
;; the translation would use loadw/loadhu/storew/storeh which require
;; aligned addresses and would produce wrong results or fail.
;;
;; All functions take the address as a parameter to prevent constant folding.

(module
  (memory 1)
  ;; Bytes 0..15: 0x01 0x02 0x03 ... 0x10
  (data (i32.const 0) "\01\02\03\04\05\06\07\08\09\0a\0b\0c\0d\0e\0f\10")

  ;; --- Single byte load (uses loadbu, no alignment issue) ---
  ;; This verifies that loadbu works with non-zero base addresses.

  (func (export "byte_at") (param $addr i32) (result i32)
    (i32.load8_u (local.get $addr))
  )

  ;; --- i32 loads at unaligned addresses ---

  (func (export "i32_load") (param $addr i32) (result i32)
    (i32.load (local.get $addr))
  )

  ;; --- i32 store/load roundtrip ---

  (func (export "i32_roundtrip") (param $addr i32) (param $val i32) (result i32)
    (i32.store (local.get $addr) (local.get $val))
    (i32.load (local.get $addr))
  )

  ;; --- i64 loads at unaligned addresses ---

  (func (export "i64_load") (param $addr i32) (result i64)
    (i64.load (local.get $addr))
  )

  ;; --- i64 store/load roundtrip ---

  (func (export "i64_roundtrip") (param $addr i32) (param $lo i32) (param $hi i32) (result i64)
    (i64.store (local.get $addr)
      (i64.or
        (i64.extend_i32_u (local.get $lo))
        (i64.shl (i64.extend_i32_u (local.get $hi)) (i64.const 32))
      )
    )
    (i64.load (local.get $addr))
  )

  ;; --- 16-bit loads at unaligned addresses ---

  (func (export "i32_load16u") (param $addr i32) (result i32)
    (i32.load16_u (local.get $addr))
  )

  (func (export "i32_load16s") (param $addr i32) (result i32)
    (i32.load16_s (local.get $addr))
  )

  ;; --- 16-bit store/load roundtrip ---

  (func (export "i32_store16_roundtrip") (param $addr i32) (param $val i32) (result i32)
    (i32.store16 (local.get $addr) (local.get $val))
    (i32.load16_u (local.get $addr))
  )

  ;; --- i64 16-bit loads ---

  (func (export "i64_load16u") (param $addr i32) (result i64)
    (i64.load16_u (local.get $addr))
  )

  (func (export "i64_load16s") (param $addr i32) (result i64)
    (i64.load16_s (local.get $addr))
  )

  ;; --- i64 32-bit loads ---

  (func (export "i64_load32u") (param $addr i32) (result i64)
    (i64.load32_u (local.get $addr))
  )

  (func (export "i64_load32s") (param $addr i32) (result i64)
    (i64.load32_s (local.get $addr))
  )

  ;; --- i64 store16/load16 roundtrip ---

  (func (export "i64_store16_roundtrip") (param $addr i32) (param $lo i32) (param $hi i32) (result i64)
    (i64.store16 (local.get $addr)
      (i64.or
        (i64.extend_i32_u (local.get $lo))
        (i64.shl (i64.extend_i32_u (local.get $hi)) (i64.const 32))
      )
    )
    (i64.load16_u (local.get $addr))
  )

  ;; --- i64 store32/load32 roundtrip ---

  (func (export "i64_store32_roundtrip") (param $addr i32) (param $lo i32) (param $hi i32) (result i64)
    (i64.store32 (local.get $addr)
      (i64.or
        (i64.extend_i32_u (local.get $lo))
        (i64.shl (i64.extend_i32_u (local.get $hi)) (i64.const 32))
      )
    )
    (i64.load32_u (local.get $addr))
  )

  ;; --- i32 store (covers I32Store + I64Store32 + F32Store path) ---

  (func (export "i32_store_roundtrip") (param $addr i32) (param $val i32) (result i32)
    (i32.store (local.get $addr) (local.get $val))
    (i32.load (local.get $addr))
  )

  ;; --- i64 store/load roundtrip ---

  (func (export "i64_store_roundtrip") (param $addr i32) (param $lo i32) (param $hi i32) (result i64)
    (i64.store (local.get $addr)
      (i64.or
        (i64.extend_i32_u (local.get $lo))
        (i64.shl (i64.extend_i32_u (local.get $hi)) (i64.const 32))
      )
    )
    (i64.load (local.get $addr))
  )
)

;; ===== Verify byte-level addressing works =====
(assert_return (invoke "byte_at" (i32.const 0)) (i32.const 1))    ;; 0x01
(assert_return (invoke "byte_at" (i32.const 1)) (i32.const 2))    ;; 0x02
(assert_return (invoke "byte_at" (i32.const 2)) (i32.const 3))    ;; 0x03
(assert_return (invoke "byte_at" (i32.const 3)) (i32.const 4))    ;; 0x04

;; ===== i32 loads from pre-initialized memory at unaligned addresses =====

;; i32.load at addr 1: bytes [0x02, 0x03, 0x04, 0x05] = 0x05040302 = 84148994
(assert_return (invoke "i32_load" (i32.const 1)) (i32.const 84148994))
;; i32.load at addr 3: bytes [0x04, 0x05, 0x06, 0x07] = 0x07060504 = 117835012
(assert_return (invoke "i32_load" (i32.const 3)) (i32.const 117835012))

;; ===== i32 store/load roundtrip at unaligned address =====
(assert_return (invoke "i32_roundtrip" (i32.const 65) (i32.const 305419896)) (i32.const 305419896))

;; ===== i64 loads from pre-initialized memory at unaligned addresses =====

;; i64.load at addr 1: bytes [0x02..0x09] = 0x0908070605040302
(assert_return (invoke "i64_load" (i32.const 1)) (i64.const 650777868590383874))
;; i64.load at addr 3: bytes [0x04..0x0b] = 0x0b0a090807060504
(assert_return (invoke "i64_load" (i32.const 3)) (i64.const 795458214266537220))

;; ===== i64 store/load roundtrip at unaligned address =====
(assert_return (invoke "i64_roundtrip" (i32.const 65) (i32.const 305419896) (i32.const 2596069104)) (i64.const 11150031900141442680))

;; ===== 16-bit loads at unaligned (odd) address =====

;; i32.load16_u at addr 1: bytes [0x02, 0x03] = 0x0302 = 770
(assert_return (invoke "i32_load16u" (i32.const 1)) (i32.const 770))
;; i32.load16_s at addr 1: bytes [0x02, 0x03] = 0x0302 = 770 (positive)
(assert_return (invoke "i32_load16s" (i32.const 1)) (i32.const 770))

;; ===== 16-bit store/load roundtrip at unaligned address =====
(assert_return (invoke "i32_store16_roundtrip" (i32.const 65) (i32.const 48879)) (i32.const 48879))

;; ===== i64 16-bit loads at unaligned address =====
(assert_return (invoke "i64_load16u" (i32.const 1)) (i64.const 770))
(assert_return (invoke "i64_load16s" (i32.const 1)) (i64.const 770))

;; ===== i64 32-bit loads at unaligned address =====
;; i64.load32_u at addr 1: same as i32 = 0x05040302 = 84148994
(assert_return (invoke "i64_load32u" (i32.const 1)) (i64.const 84148994))
;; i64.load32_s at addr 1: positive value, same as unsigned
(assert_return (invoke "i64_load32s" (i32.const 1)) (i64.const 84148994))

;; ===== i64 store16/load16 roundtrip at unaligned address =====
(assert_return (invoke "i64_store16_roundtrip" (i32.const 65) (i32.const 48879) (i32.const 0)) (i64.const 48879))

;; ===== i64 store32/load32 roundtrip at unaligned address =====
(assert_return (invoke "i64_store32_roundtrip" (i32.const 65) (i32.const 305419896) (i32.const 0)) (i64.const 305419896))

;; ===== i32 store/load roundtrip =====
(assert_return (invoke "i32_store_roundtrip" (i32.const 65) (i32.const 305419896)) (i32.const 305419896))

;; ===== i64 store/load roundtrip =====
(assert_return (invoke "i64_store_roundtrip" (i32.const 65) (i32.const 305419896) (i32.const 2596069104)) (i64.const 11150031900141442680))
