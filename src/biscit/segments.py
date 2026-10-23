import numpy as np
import pandas as pd


def summarize_path(path, pos):
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
        if path[i] != current_state:
            runs.append({"start_pos": pos[start_idx], "end_pos": pos[i - 1], "n_markers": i - start_idx, "state": current_state})
            current_state = path[i]
            start_idx = i
    runs.append({"start_pos": pos[start_idx], "end_pos": pos[-1], "n_markers": len(path) - start_idx, "state": current_state})
    return pd.DataFrame(runs)


def filter_short_stretches(summary_df, min_length=50000):
    if len(summary_df) == 0:
        return summary_df.copy()
    df = summary_df.copy()
    df["genomic_length"] = df["end_pos"] - df["start_pos"] + 1
    df.loc[df["genomic_length"] < min_length, "state"] = 3
    merged = []
    current_state = df.iloc[0]["state"]
    start_pos = df.iloc[0]["start_pos"]
    end_pos = df.iloc[0]["end_pos"]
    total_n_markers = df.iloc[0]["n_markers"]
    for i in range(1, len(df)):
        if df.iloc[i]["state"] == current_state:
            end_pos = df.iloc[i]["end_pos"]
            total_n_markers += df.iloc[i]["n_markers"]
        else:
            merged.append({"state": current_state, "start_pos": start_pos, "end_pos": end_pos, "n_markers": total_n_markers})
            current_state = df.iloc[i]["state"]
            start_pos = df.iloc[i]["start_pos"]
            end_pos = df.iloc[i]["end_pos"]
            total_n_markers = df.iloc[i]["n_markers"]
    merged.append({"start_pos": start_pos, "end_pos": end_pos, "n_markers": total_n_markers, "state": current_state})
    return pd.DataFrame(merged)


def add_boundary_regions(summary_df, chr_length):
    if len(summary_df) == 0:
        return pd.DataFrame([{"start_pos": 1, "end_pos": chr_length, "n_markers": 0, "state": 3}])
    rows = []
    first_start = summary_df.iloc[0]["start_pos"]
    if first_start > 1:
        rows.append({"start_pos": 1, "end_pos": first_start - 1, "n_markers": 0, "state": 3})
    for _, row in summary_df.iterrows():
        rows.append(row.to_dict())
    last_end = summary_df.iloc[-1]["end_pos"]
    if last_end < chr_length:
        rows.append({"start_pos": last_end + 1, "end_pos": chr_length, "n_markers": 0, "state": 3})
    return pd.DataFrame(rows)
