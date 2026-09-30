(module
  (type $node (struct (field i32) (field (mut exnref))))
  (tag $item (param (ref $node)))
  (global $state (mut i32) (i32.const 0))
  (func (export "init") (param i32) (result i32)
    local.get 0 global.set $state local.get 0)
  (func (export "step") (param i32) (result i32)
    (local $round i32) (local $value i32)
    (local $node (ref null $node)) (local $exception exnref)
    global.get $state local.get 0 i32.xor local.set $value
    loop $again
      local.get $value i32.const 5 i32.rotl
      i32.const -1640531535 i32.mul
      local.get $round i32.add local.set $value
      local.get $value ref.null exn struct.new $node local.set $node
      block $caught (result (ref $node) exnref)
        try_table (catch_ref $item $caught)
          local.get $node ref.as_non_null throw $item
        end unreachable
      end
      local.set $exception local.set $node
      local.get $node local.get $exception struct.set $node 1
      local.get $node struct.get $node 0 global.get $state i32.xor global.set $state
      ;; Drop both roots. The unreachable node/exception cycle must be reclaimed.
      ref.null $node local.set $node
      ref.null exn local.set $exception
      local.get $round i32.const 1 i32.add local.tee $round
      i32.const 64 i32.lt_u br_if $again
    end
    global.get $state)
  (func (export "digest") (param i32) (result i32)
    global.get $state local.get 0 i32.xor))
