import numpy as np

from cq.validation import walk_forward_splits


def test_splits_are_chronological_with_gap_and_cover_tail():
    n, train_min, test_size, gap = 100, 40, 15, 2
    splits = list(walk_forward_splits(n, train_min, test_size, gap))
    assert splits
    tested = np.concatenate([te for _, te in splits])
    # Test blocks do not overlap and reach the end of the series.
    assert len(tested) == len(np.unique(tested))
    assert tested[-1] == n - 1
    for tr, te in splits:
        assert tr[0] == 0 and len(tr) >= train_min
        assert te[0] - tr[-1] == gap + 1  # embargo of exactly `gap` bars
