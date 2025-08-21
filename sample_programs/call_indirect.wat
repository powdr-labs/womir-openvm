(module
  ;; function type: (i32, i32) -> i32
  (type $bin (func (param i32 i32) (result i32)))

  ;; two tiny funcs
  (func $add (type $bin) (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add)

  (func $sub (type $bin) (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.sub)

  ;; default table 0 with two funcrefs
  (table 2 funcref)
  (elem (i32.const 0) $add $sub)

  ;; indirect call: index 0 = add, 1 = sub
  (func (export "call_op") (param $index i32) (param $x i32) (param $y i32) (result i32)
    local.get $x
    local.get $y
    local.get $index
    call_indirect (type $bin))

  ;; tiny self-check: 1 if ok, else 0
  (func (export "test") (result i32)
    ;; add via 0
    i32.const 2
    i32.const 3
    i32.const 0
    call_indirect (type $bin)
    i32.const 5
    i32.eq

    ;; sub via 1
    i32.const 7
    i32.const 4
    i32.const 1
    call_indirect (type $bin)
    i32.const 3
    i32.eq

    i32.and)
)
