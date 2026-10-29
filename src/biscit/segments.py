import numpy as np
import pandas as pd


def summarize_path(path, pos):
    """
    Collapse a per-marker state path into contiguous segments.

    Parameters
    ----------
    path : array_like of int
        Hidden state at each marker, shape ``(n_loci,)``, as returned by
        `biscit.hmm.viterbi_min_duration`.
    pos : array_like of int
        Marker positions in base pairs, shape ``(n_loci,)``.

    Returns
    -------
    pandas.DataFrame
        One row per run of identical consecutive states, with columns
        ``start_pos``, ``end_pos``, ``n_markers`` and ``state``. Segment
        boundaries are the positions of the first and last marker in the run,
        so there are gaps between adjacent segments. Empty input gives an
        empty DataFrame with these columns.

    Raises
    ------
    ValueError
        If `path` and `pos` have different lengths.
    """
    path = np.asarray(path)
    pos = np.asarray(pos)
    if len(path) != len(pos):
        raise ValueError("path and pos must have the same length")
    if len(path) == 0:
        return pd.DataFrame(columns=["state", "start_pos", "end_pos", "n_markers"])
    runs = []
    current_state = path[0]
    start_idx = 0
    for i in range(1, len(path)):
        # State change: close the current run at the previous marker
        if path[i] != current_state:
            runs.append({"start_pos": pos[start_idx], "end_pos": pos[i - 1], "n_markers": i - start_idx, "state": current_state})
            current_state = path[i]
            start_idx = i
    # Close the final run, which the loop never does
    runs.append({"start_pos": pos[start_idx], "end_pos": pos[-1], "n_markers": len(path) - start_idx, "state": current_state})
    return pd.DataFrame(runs)


def filter_short_stretches(summary_df, min_length=50000):
    """
    Mark short segments as unassigned and merge adjacent identical segments.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Segment table with columns ``start_pos``, ``end_pos``, ``n_markers``
        and ``state``, as returned by `summarize_path`.
    min_length : int, optional
        Minimum segment length in base pairs, measured from the first to the
        last marker. Shorter segments are set to state 3 (unassigned).
        Default is 50000.

    Returns
    -------
    pandas.DataFrame
        Segment table with the same columns, in which consecutive segments
        with the same state (including newly unassigned ones) have been merged
        and their ``n_markers`` summed. Empty input is returned as a copy.

    Notes
    -----
    The HMM only emits states 0-2; state 3 (unassigned) is introduced here and
    in `add_boundary_regions`.
    """
    if len(summary_df) == 0:
        return summary_df.copy()
    df = summary_df.copy()
    # Mask segments spanning less than min_length bp
    df["genomic_length"] = df["end_pos"] - df["start_pos"] + 1
    df.loc[df["genomic_length"] < min_length, "state"] = 3
    # Merge runs of segments that now share a state
    merged = []
    current_state = df.iloc[0]["state"]
    start_pos = df.iloc[0]["start_pos"]
    end_pos = df.iloc[0]["end_pos"]
    total_n_markers = df.iloc[0]["n_markers"]
    for i in range(1, len(df)):
        if df.iloc[i]["state"] == current_state:
            # Extend the current merged segment
            end_pos = df.iloc[i]["end_pos"]
            total_n_markers += df.iloc[i]["n_markers"]
        else:
            # Close it and start a new one
            merged.append({"state": current_state, "start_pos": start_pos, "end_pos": end_pos, "n_markers": total_n_markers})
            current_state = df.iloc[i]["state"]
            start_pos = df.iloc[i]["start_pos"]
            end_pos = df.iloc[i]["end_pos"]
            total_n_markers = df.iloc[i]["n_markers"]
    merged.append({"start_pos": start_pos, "end_pos": end_pos, "n_markers": total_n_markers, "state": current_state})
    return pd.DataFrame(merged)


def add_boundary_regions(summary_df, chr_length):
    """
    Pad a segment table so it covers the whole chromosome.

    Adds unassigned (state 3) segments from position 1 to the first segment,
    and from the last segment to the chromosome end. Gaps between internal
    segments are not filled.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Segment table with columns ``start_pos``, ``end_pos``, ``n_markers``
        and ``state``, sorted by position.
    chr_length : int
        Chromosome length in base pairs.

    Returns
    -------
    pandas.DataFrame
        Segment table with boundary segments added where needed. Boundary
        segments have ``n_markers`` of 0. Empty input gives a single
        unassigned segment spanning the whole chromosome.
    """
    if len(summary_df) == 0:
        return pd.DataFrame([{"start_pos": 1, "end_pos": chr_length, "n_markers": 0, "state": 3}])
    rows = []
    # Region before the first marker
    first_start = summary_df.iloc[0]["start_pos"]
    if first_start > 1:
        rows.append({"start_pos": 1, "end_pos": first_start - 1, "n_markers": 0, "state": 3})
    for _, row in summary_df.iterrows():
        rows.append(row.to_dict())
    # Region after the last marker
    last_end = summary_df.iloc[-1]["end_pos"]
    if last_end < chr_length:
        rows.append({"start_pos": last_end + 1, "end_pos": chr_length, "n_markers": 0, "state": 3})
    return pd.DataFrame(rows)
