;; https://github.com/WebAssembly/gc/issues/516
(assert_invalid
  (module
    (type $t (func))
    (func $f (param (ref null $t)) (result funcref) (local.get 0))
    (func (param funcref) (result funcref funcref)
      (ref.null $t)
      (local.get 0)
      (br_on_cast 0 funcref (ref $t))  ;; only leaves two funcref's on the stack
      (drop)
      (call $f)
      (local.get 0)
    )
  )
  "type mismatch"
)
