(module
  (type $values (array (mut i32)))
  (type $node (struct (field (ref $values)) (field (mut i32))))
  (func (export "calculate") (param $input i32) (result i32)
    (local $value i32) (local $round i32) (local $index i32) (local $next i32)
    (local $data (ref null $values)) (local $node (ref null $node))
    local.get $input local.set $value
    loop $rounds
      local.get $value local.get $round i32.xor
      i32.const 32 array.new $values local.set $data
      local.get $data ref.as_non_null local.get $value
      struct.new $node local.set $node
      i32.const 0 local.set $index
      loop $elements
        local.get $node struct.get $node 0
        local.get $index array.get $values
        local.get $index i32.xor
        i32.const 0x9e3779b1 i32.mul
        local.get $round i32.add local.set $next
        local.get $input local.get $index i32.xor i32.const 1 i32.and
        i32.eqz
        if (result i32)
          local.get $next i32.const 5 i32.rotl
        else
          local.get $next i32.const 0x85ebca6b i32.xor
        end
        local.set $next
        local.get $node struct.get $node 0
        local.get $index local.get $next array.set $values
        local.get $node
        local.get $node struct.get $node 1 local.get $next i32.add
        i32.const 3 i32.rotl struct.set $node 1
        local.get $index i32.const 1 i32.add local.tee $index
        i32.const 32 i32.ne br_if $elements
      end
      local.get $node struct.get $node 1
      local.get $node struct.get $node 0
      local.get $input local.get $round i32.add i32.const 31 i32.and
      array.get $values i32.xor local.set $value
      ;; Release this round's graph. Subsequent allocations must collect it.
      ref.null $node local.set $node
      ref.null $values local.set $data
      local.get $round i32.const 1 i32.add local.tee $round
      i32.const 64 i32.ne br_if $rounds
    end
    local.get $value))
