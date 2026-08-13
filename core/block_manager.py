"""
Paged KV cache allocator. Owns the pool of fixed-size pages and each
sequence's block table (logical position -> physical page).
"""
