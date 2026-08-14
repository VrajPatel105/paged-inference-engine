import pytest
from core.block_manager import BlockManager, AllocStatus

def test_interleaved_allocation():
    bm = BlockManager(num_blocks=1000, block_size=20)

    seen_blocks = set()
    owned = {}  

    def do_allocate(seq_id, n, expect_status):
        result = bm.allocate(seq_id, n)
        assert result.status == expect_status
        if expect_status == AllocStatus.SUCCESS:
            for b in result.block_ids:
                assert b not in seen_blocks, f"block {b} double-allocated"
                seen_blocks.add(b)
            owned[seq_id] = result.block_ids
        # we also do invariant check every time
        assert bm.num_free_blocks() + len(seen_blocks) == bm.num_blocks

    def do_release(seq_id, expect_status):
        result = bm.release_blocks(seq_id)
        assert result == expect_status
        if expect_status == AllocStatus.SUCCESS:
            for b in owned[seq_id]:
                seen_blocks.discard(b)
            del owned[seq_id]
        assert bm.num_free_blocks() + len(seen_blocks) == bm.num_blocks

    do_allocate(0, 200, AllocStatus.SUCCESS)
    do_allocate(1, 130, AllocStatus.SUCCESS)
    do_allocate(2, 320, AllocStatus.SUCCESS)
    do_release(1, AllocStatus.SUCCESS)
    do_allocate(3, 500, AllocStatus.INSUFFICIENT_SPACE)
    do_allocate(3, 300, AllocStatus.SUCCESS)
    do_release(6, AllocStatus.FAILURE)