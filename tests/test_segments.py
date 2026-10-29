import numpy as np
import pandas as pd
import pytest

from biscit.segments import add_boundary_regions, filter_short_stretches, summarize_path


COLUMNS = ["start_pos", "end_pos", "n_markers", "state"]


def segments(*rows):
    """
    Build a segment table from ``(start_pos, end_pos, n_markers, state)`` tuples.

    Parameters
    ----------
    *rows : tuple of int
        One tuple per segment.

    Returns
    -------
    pandas.DataFrame
        Segment table with columns `COLUMNS`.
    """
    return pd.DataFrame(list(rows), columns=COLUMNS)


def as_rows(df):
    """
    Convert a segment table to a list of tuples in `COLUMNS` order.

    Parameters
    ----------
    df : pandas.DataFrame
        Segment table with at least the columns in `COLUMNS`.

    Returns
    -------
    list of tuple of int
        One ``(start_pos, end_pos, n_markers, state)`` tuple per row.
    """
    return [tuple(int(v) for v in row) for row in df[COLUMNS].itertuples(index=False)]


# summarize_path

def test_summarize_path_collapses_runs():
    path = [0, 0, 1, 1, 1, 0]
    pos = [10, 20, 30, 40, 50, 60]
    assert as_rows(summarize_path(path, pos)) == [
        (10, 20, 2, 0),
        (30, 50, 3, 1),
        (60, 60, 1, 0),
    ]


def test_summarize_path_single_marker():
    assert as_rows(summarize_path([2], [500])) == [(500, 500, 1, 2)]


def test_summarize_path_empty():
    result = summarize_path([], [])
    assert len(result) == 0
    assert set(result.columns) == set(COLUMNS)


def test_summarize_path_length_mismatch():
    with pytest.raises(ValueError):
        summarize_path([0, 1], [10])


# filter_short_stretches

def test_short_segment_between_identical_states_is_masked_not_merged():
    """A short BB between two long AA segments becomes AA / 3 / AA, not one AA."""
    df = segments((1, 500, 5, 0), (501, 550, 2, 2), (551, 1000, 5, 0))
    assert as_rows(filter_short_stretches(df, min_length=100)) == [
        (1, 500, 5, 0),
        (501, 550, 2, 3),
        (551, 1000, 5, 0),
    ]


def test_masked_segment_merges_with_neighbouring_state_3():
    df = segments((1, 500, 5, 0), (501, 600, 3, 3), (601, 620, 2, 2))
    assert as_rows(filter_short_stretches(df, min_length=100)) == [
        (1, 500, 5, 0),
        (501, 620, 5, 3),
    ]


def test_min_length_threshold_is_inclusive():
    """Length is measured from first to last marker inclusive; exactly `min_length` is kept."""
    df = segments((1, 100, 2, 0), (101, 199, 2, 2))
    assert filter_short_stretches(df, min_length=100)["state"].tolist() == [0, 3]


def test_filter_preserves_markers_and_drops_helper_column():
    df = segments((1, 500, 5, 0), (501, 510, 2, 1), (511, 520, 3, 2), (521, 1000, 7, 0))
    result = filter_short_stretches(df, min_length=100)
    assert result["n_markers"].sum() == df["n_markers"].sum()
    assert "genomic_length" not in result.columns


def test_filter_does_not_modify_input():
    df = segments((1, 500, 5, 0), (501, 550, 2, 2))
    original = df.copy()
    filter_short_stretches(df, min_length=100)
    pd.testing.assert_frame_equal(df, original)


def test_filter_empty():
    result = filter_short_stretches(segments())
    assert len(result) == 0


# add_boundary_regions

def test_boundary_regions_added_at_both_ends():
    df = segments((100, 200, 5, 0), (300, 400, 5, 2))
    assert as_rows(add_boundary_regions(df, chr_length=1000)) == [
        (1, 99, 0, 3),
        (100, 200, 5, 0),
        (300, 400, 5, 2),
        (401, 1000, 0, 3),
    ]


def test_no_boundary_regions_when_segments_reach_ends():
    df = segments((1, 200, 5, 0), (300, 1000, 5, 2))
    assert as_rows(add_boundary_regions(df, chr_length=1000)) == as_rows(df)


def test_boundary_regions_empty_input_spans_chromosome():
    assert as_rows(add_boundary_regions(segments(), chr_length=1000)) == [(1, 1000, 0, 3)]


# Full chain

@pytest.mark.parametrize("seed", range(5))
def test_chain_tiles_chromosome(seed):
    """summarize -> filter -> pad gives ordered, non-overlapping, complete segments."""
    rng = np.random.default_rng(seed)
    # Runs of random states and lengths, on irregularly spaced markers
    path = np.repeat(rng.integers(0, 3, size=30), rng.integers(1, 20, size=30))
    pos = np.cumsum(rng.integers(100, 5000, size=len(path)))
    chr_length = int(pos[-1]) + 1000

    filtered = filter_short_stretches(summarize_path(path, pos), min_length=10_000)
    result = add_boundary_regions(filtered, chr_length=chr_length)

    # Adjacent segments never share a state after merging
    assert (np.diff(filtered["state"].to_numpy()) != 0).all()

    starts = result["start_pos"].to_numpy()
    ends = result["end_pos"].to_numpy()
    assert starts[0] == 1
    assert ends[-1] == chr_length
    assert (starts <= ends).all()
    assert (starts[1:] > ends[:-1]).all()
    assert result["n_markers"].sum() == len(path)
