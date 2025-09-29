
;; Sum all the natural numbers up to n (inclusive).
(func (export "n_first_sum") (param $n i64) (result i64)
    (local $sum i64)
    block $done
        loop $loop
            local.get $n
            i64.const 0
            i64.gt_u
            if
                local.get $n
                local.get $sum
                i64.add
                local.set $sum
                local.get $n
                i64.const 1
                i64.sub
                local.set $n
                br $loop
            else
                br $done
            end
        end
    end
    local.get $sum
)
