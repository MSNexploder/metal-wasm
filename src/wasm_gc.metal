// One lane owns its heap and roots. References encode a stable block offset;
// neither collection nor allocation moves a live object.
constant ulong GC_ALLOCATED = 1ul << 63u;
constant ulong GC_MARKED = 1ul << 62u;
constant ulong GC_SCANNED = 1ul << 61u;
constant ulong GC_SIZE_MASK = 0xfffffffful;
constant ulong GC_OBJECT_TAG = 0x300000000ul;

struct ModuleState;

struct GcState {
    device ulong* heap;
    uint capacity;
    device ulong* roots;
    uint root_capacity;
    uint root_top;
    thread ModuleState* states;
};

// Generated from the module types and instance state, after ModuleState is
// defined. Field indices count payload words, including both words of v128.
inline bool gc_ref_word(uint type, uint payload_index);
inline void gc_mark_state(thread GcState& gc);

inline uint gc_block_size(ulong header) {
    return uint(header & GC_SIZE_MASK);
}

struct GcRootFrame {
    thread GcState* state;
    uint base;

    GcRootFrame(thread GcState& gc, uint slots, thread uint& trap)
        : state(&gc), base(gc.root_top) {
        if (gc.root_top > gc.root_capacity || slots > gc.root_capacity - gc.root_top) {
            trap = 15u;
            return;
        }
        gc.root_top += slots;
        for (uint i = base; i < gc.root_top; ++i) gc.roots[i] = 0ul;
    }

    GcRootFrame(const thread GcRootFrame&) = delete;
    thread GcRootFrame& operator=(const thread GcRootFrame&) = delete;

    ~GcRootFrame() {
        state->root_top = base;
    }
};

// Operand stack roots can contain scalar words. A matching tag alone must not
// cause an interior pointer, a free block, or an out-of-range value to be traced.
inline uint gc_object_offset(thread GcState& gc, ulong reference) {
    if ((reference >> 32u) != (GC_OBJECT_TAG >> 32u) || uint(reference) == 0u)
        return gc.capacity;
    uint wanted = uint(reference) - 1u;
    if (wanted >= gc.capacity) return gc.capacity;
    uint offset = 0u;
    while (offset <= wanted) {
        ulong header = gc.heap[offset];
        uint words = gc_block_size(header);
        if (words == 0u || words > gc.capacity - offset) return gc.capacity;
        if (offset == wanted)
            return (header & GC_ALLOCATED) != 0ul ? offset : gc.capacity;
        offset += words;
    }
    return gc.capacity;
}

inline void gc_mark(thread GcState& gc, ulong reference) {
    uint offset = gc_object_offset(gc, reference);
    if (offset != gc.capacity) gc.heap[offset] |= GC_MARKED;
}

inline void gc_collect(thread GcState& gc) {
    for (uint offset = 0u; offset < gc.capacity;) {
        ulong header = gc.heap[offset];
        gc.heap[offset] = header & ~(GC_MARKED | GC_SCANNED);
        offset += gc_block_size(header);
    }
    for (uint i = 0u; i < gc.root_top; ++i) gc_mark(gc, gc.roots[i]);
    gc_mark_state(gc);

    // Repeated scans replace a separate bounded mark worklist. Mark before
    // visiting children so cycles terminate, including references to earlier
    // blocks that will be visited in the next scan.
    bool visited;
    do {
        visited = false;
        for (uint offset = 0u; offset < gc.capacity;) {
            ulong header = gc.heap[offset];
            uint words = gc_block_size(header);
            if ((header & (GC_ALLOCATED | GC_MARKED | GC_SCANNED))
                == (GC_ALLOCATED | GC_MARKED)) {
                gc.heap[offset] = header | GC_SCANNED;
                uint type = uint(gc.heap[offset + 1u]);
                for (uint i = 0u; i < words - 3u; ++i) {
                    if (gc_ref_word(type, i)) gc_mark(gc, gc.heap[offset + 3u + i]);
                }
                visited = true;
            }
            offset += words;
        }
    } while (visited);

    // Free unreachable blocks and join adjacent free blocks in the same pass.
    uint previous_free = gc.capacity;
    for (uint offset = 0u; offset < gc.capacity;) {
        ulong header = gc.heap[offset];
        uint words = gc_block_size(header);
        if ((header & (GC_ALLOCATED | GC_MARKED)) == (GC_ALLOCATED | GC_MARKED)) {
            gc.heap[offset] = ulong(words) | GC_ALLOCATED;
            previous_free = gc.capacity;
        } else {
            gc.heap[offset] = ulong(words);
            if (previous_free == gc.capacity) previous_free = offset;
            else gc.heap[previous_free] += ulong(words);
        }
        offset += words;
    }
}

inline ulong gc_try_alloc(thread GcState& gc, uint type, uint length, uint words) {
    for (uint offset = 0u; offset < gc.capacity;) {
        ulong header = gc.heap[offset];
        uint available = gc_block_size(header);
        if ((header & GC_ALLOCATED) == 0ul && available >= words) {
            uint used = words;
            // Every block must have enough space for an object header.
            if (available - words >= 3u) gc.heap[offset + words] = ulong(available - words);
            else used = available;
            gc.heap[offset] = ulong(used) | GC_ALLOCATED;
            gc.heap[offset + 1u] = ulong(type);
            gc.heap[offset + 2u] = ulong(length);
            for (uint i = 3u; i < used; ++i) gc.heap[offset + i] = 0ul;
            return GC_OBJECT_TAG | ulong(offset + 1u);
        }
        offset += available;
    }
    return 0ul;
}

inline ulong gc_alloc(thread GcState& gc, uint type, uint length,
                      uint payload_words, thread uint& trap) {
    if (trap != 0u) return 0ul;
    if (gc.capacity < 3u || payload_words > gc.capacity - 3u) {
        trap = 15u;
        return 0ul;
    }
    uint words = payload_words + 3u;
    ulong reference = gc_try_alloc(gc, type, length, words);
    if (reference == 0ul) {
        gc_collect(gc);
        reference = gc_try_alloc(gc, type, length, words);
    }
    if (reference == 0ul) trap = 15u;
    return reference;
}

// Compiled reference operations receive validated references. Check the tag and
// bounds here without walking every block on each field access. The returned
// pointer addresses payload; type and logical length are at [-2] and [-1].
inline device ulong* gc_ptr(thread GcState& gc, ulong reference, thread uint& trap) {
    if (reference == 0ul) {
        trap = 9u;
        return nullptr;
    }
    uint offset = uint(reference) - 1u;
    if ((reference >> 32u) != (GC_OBJECT_TAG >> 32u) || uint(reference) == 0u
        || gc.capacity < 3u || offset > gc.capacity - 3u
        || (gc.heap[offset] & GC_ALLOCATED) == 0ul) {
        trap = 15u;
        return nullptr;
    }
    return gc.heap + offset + 3u;
}
