#include "buddy_allocator.hpp"

size_t sssp::buddy_allocator_chunk::alloc_node(int layer) {
    // Note the <= to reach the virtual root node!
    if (layer <= buddy_allocator_layers) {
        if (m_layers[layer].empty()) {
            // No free nodes for this layer, try to allocate one higher up
            // and return one half, and keep the buddy as free.
            size_t offset = alloc_node(layer + 1);
            if (offset == size_t(-1)) {
                return size_t(-1);
            }
            BOOST_ASSERT(m_nodes[offset].free_in_layer == -1);

            size_t buddy = offset ^ (size_t(1) << layer);
            buddy_allocator_node& buddy_node = m_nodes[buddy];
            BOOST_ASSERT(buddy_node.free_in_layer >= layer);
            buddy_node.free_in_layer = layer;
            m_layers[layer].push_front(buddy_node);

            return offset;
        } else {
            // We found a free node at this layer. Just return it.
            buddy_allocator_node& free_node = m_layers[layer].front();
            BOOST_ASSERT(free_node.free_in_layer == layer);
            m_layers[layer].pop_front();
            free_node.free_in_layer = -1;

            size_t offset = &free_node - &m_nodes[0];
            return offset;
        }
    } else {
        // The request can't be satisfied.
        return size_t(-1);
    }
}

void sssp::buddy_allocator_chunk::free_node(int layer, size_t offset) {
    buddy_allocator_node& this_node = m_nodes[offset];
    BOOST_ASSERT(this_node.free_in_layer == -1);

    if (layer < buddy_allocator_layers) {
        // If the buddy is free, free the parent otherwise
        // just push the new free node to the free list.
        size_t buddy = offset ^ (size_t(1) << layer);
        buddy_allocator_node& buddy_node = m_nodes[buddy];

        if (buddy_node.free_in_layer == layer) {
            m_layers[layer].erase(m_layers[layer].iterator_to(buddy_node));
            size_t parent = std::min(offset, buddy);
            if (parent == buddy) {
                buddy_node.free_in_layer = -1;
                this_node.free_in_layer = layer;
            } else {
                this_node.free_in_layer = -1;
                buddy_node.free_in_layer = layer;
            }
            free_node(layer + 1, parent);
        } else {
            BOOST_ASSERT(buddy_node.free_in_layer < layer);
            this_node.free_in_layer = layer;
            m_layers[layer].push_front(this_node);
        }
    } else {
        // We are at the root node, just push the free node. No more merging.
        this_node.free_in_layer = layer;
        m_layers[layer].push_front(this_node);
    }
}

char* sssp::buddy_allocator_memory::alloc(int layer) {
    size_t offset = -1;
    buddy_allocator_chunk* chosen_chunk = nullptr;

    if (m_last_success != nullptr) {
        auto end = m_chunks.end();
        auto mid = m_chunks.iterator_to(*m_last_success);
        for (auto iter = mid; iter != end; ++iter) {
            offset = iter->alloc_node(layer);
            if (offset != size_t(-1)) {
                chosen_chunk = &*iter;
                break;
            }
        }
        if (chosen_chunk == nullptr) {
            for (auto iter = m_chunks.begin(); iter != mid; ++iter) {
                offset = iter->alloc_node(layer);
                if (offset != size_t(-1)) {
                    chosen_chunk = &*iter;
                    break;
                }
            }
        }
    }

    if (chosen_chunk == nullptr) {
        chosen_chunk = new buddy_allocator_chunk();
        m_chunks.insert(*chosen_chunk);
        offset = chosen_chunk->alloc_node(layer);
    }

    m_last_success = chosen_chunk;
    BOOST_ASSERT(offset != size_t(-1) && chosen_chunk != nullptr);
    return chosen_chunk->base_pointer() + (offset << buddy_allocator_missing_layers);
}

void sssp::buddy_allocator_memory::free(int layer, char* address) {
    buddy_allocator_chunk& chunk = *std::prev(m_chunks.upper_bound(
        address, [](char* ptr, const buddy_allocator_chunk& chunk) { return ptr < chunk.base_pointer(); }));
    size_t offset = static_cast<size_t>(address - chunk.base_pointer()) >> buddy_allocator_missing_layers;
    chunk.free_node(layer, offset);
}
