;; Syntax and Scoping

(module
  (type (func (param (ref 0)) (result (ref 0))))
  (rec
    (type (func (param (ref 2))))
    (type (func (result (ref 1))))
  )

  (rec)
  (rec (type (func)))
  (rec (type $t (func)))
  (rec (type $t1 (func)) (type (func)) (type $t2 (func)))
  (rec (type $g (func (param (ref $g)) (result (ref $g)))))
  (rec
    (type $h (func (param (ref $k))))
    (type $k (func (result (ref $h))))
  )
)

(assert_invalid
  (module
    (type (func (param (ref 1))))
    (type (func))
  )
  "unknown type"
)
(assert_invalid
  (module
    (rec (type (func (param (ref 1)))))
    (rec (type (func)))
  )
  "unknown type"
)


(module
  (rec (type $ft (func)))
  (func $f)  ;; the implicit type of $f is $ft
  (global (ref $ft) (ref.func $f))
)

(assert_invalid
  (module
    (rec (type $ft (func)) (type (func)))
    (func $f)  ;; the implicit type of $f is not $ft
    (global (ref $ft) (ref.func $f))
  )
  "type mismatch"
)
(assert_invalid
  (module
    (rec (type (func)) (type $ft (func)))
    (func $f)  ;; the implicit type of $f is not $ft
    (global (ref $ft) (ref.func $f))
  )
  "type mismatch"
)
