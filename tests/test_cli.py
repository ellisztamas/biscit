import argparse
import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biscit.cli import DEFAULT_CHR_LENGTHS, build_parser, main, parse_chr_lengths, process_chromosome
from biscit.io import extract_aligned_pl
from conftest import PL_BY_GENOTYPE, write_h5


OUTPUT_COLUMNS = ["chrom", "start_pos", "end_pos", "state", "state_label", "n_markers"]
SRC = Path(__file__).resolve().parents[1] / "src"


def run_to_table(cross, tmp_path, *extra):
    """
    Run `main` on a simulated cross and read the output table.

    Parameters
    ----------
    cross : conftest.Cross
        Simulated cross to decode.
    tmp_path : pathlib.Path
        Directory for the output file.
    *extra : str
        Additional command-line arguments.

    Returns
    -------
    pandas.DataFrame
        The output TSV.
    """
    out = tmp_path / "segments.tsv"
    assert main(cross.argv("-o", str(out), *extra)) == 0
    return pd.read_csv(out, sep="\t")


def states_at(result, positions):
    """
    Look up the decoded state at each marker from a segment table.

    Parameters
    ----------
    result : pandas.DataFrame
        Output table for a single chromosome.
    positions : numpy.ndarray of int
        Marker positions, shape ``(n_loci,)``.

    Returns
    -------
    numpy.ndarray of int
        State of the segment containing each marker, shape ``(n_loci,)``.
    """
    idx = np.searchsorted(result["start_pos"].to_numpy(), positions, side="right") - 1
    # Segment edges are marker positions, so every marker lies inside one
    assert (positions <= result["end_pos"].to_numpy()[idx]).all()
    return result["state"].to_numpy()[idx]


# parse_chr_lengths and build_parser

def test_parse_chr_lengths_default_is_a_copy():
    lengths = parse_chr_lengths(None)
    assert lengths == DEFAULT_CHR_LENGTHS
    lengths["Chr1"] = 1
    assert DEFAULT_CHR_LENGTHS["Chr1"] == 30427671


def test_parse_chr_lengths_tolerates_whitespace_and_empty_entries():
    assert parse_chr_lengths(" Chr1 = 100 ,, Chr2=200, ") == {"Chr1": 100, "Chr2": 200}


@pytest.mark.parametrize("value", ["Chr1", "Chr1=100,Chr2", "Chr1=abc", "Chr1=1.5"])
def test_parse_chr_lengths_rejects_malformed(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_chr_lengths(value)


REQUIRED = ["--input", "a.h5", "--reference", "b.h5", "--offspring", "O", "--parent1", "A", "--parent2", "B"]


def test_parser_defaults():
    args = build_parser().parse_args(REQUIRED)
    assert args.n_generations_selfing == 10
    assert args.min_markers == 50
    assert args.min_base_pairs == 50000
    assert args.output is None
    assert args.chromosomes is None
    assert args.chr_lengths is None


@pytest.mark.parametrize("drop", range(0, len(REQUIRED), 2))
def test_parser_requires_arguments(drop):
    argv = REQUIRED[:drop] + REQUIRED[drop + 2:]
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_parser_rejects_bad_chr_lengths():
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(REQUIRED + ["--chr-lengths", "Chr1"])
    assert exc.value.code == 2


# process_chromosome

def test_process_chromosome_tiles_chromosome(simulated_ril):
    cross = simulated_ril
    args = build_parser().parse_args(cross.argv())
    data = extract_aligned_pl(cross.input, cross.reference, cross.offspring, cross.parent1, cross.parent2)[cross.chrom]

    result = process_chromosome(cross.chrom, data, cross.chr_length, args)

    assert set(result.columns) == {"chrom", "start_pos", "end_pos", "n_markers", "state"}
    assert result.columns[0] == "chrom"
    assert result["start_pos"].iloc[0] == 1
    assert result["end_pos"].iloc[-1] == cross.chr_length
    assert (result["start_pos"].to_numpy()[1:] > result["end_pos"].to_numpy()[:-1]).all()


def test_end_to_end_recovers_simulated_ril(simulated_ril, tmp_path):
    """A P1-then-P2 offspring decodes to one P1 and one P2 segment."""
    cross = simulated_ril
    out = tmp_path / "segments.tsv"

    assert main(cross.argv("-o", str(out))) == 0
    result = pd.read_csv(out, sep="\t")

    assert list(result.columns) == OUTPUT_COLUMNS
    assert (result["chrom"] == cross.chrom).all()

    # Segments tile the chromosome from position 1 to its end, in order
    assert result["start_pos"].iloc[0] == 1
    assert result["end_pos"].iloc[-1] == cross.chr_length
    assert (result["start_pos"].to_numpy()[1:] > result["end_pos"].to_numpy()[:-1]).all()
    assert result["n_markers"].sum() == len(cross.positions)

    # Labels use the parent names given on the command line
    labels = {0: cross.parent1, 1: "heterozygous", 2: cross.parent2, 3: "Unassigned"}
    assert (result["state_label"] == result["state"].map(labels)).all()

    # Only the padding at the chromosome ends is unassigned
    assert result["state"].tolist() == [3, 0, 2, 3]
    assert result["n_markers"].tolist()[0] == result["n_markers"].tolist()[-1] == 0

    # The breakpoint falls within a couple of markers of the truth
    true_switch = np.flatnonzero(np.diff(cross.ancestry))[0]
    p1, p2 = result.iloc[1], result.iloc[2]
    assert p1["start_pos"] == cross.positions[0]
    assert p2["end_pos"] == cross.positions[-1]
    assert abs(p1["n_markers"] - (true_switch + 1)) <= 2
    assert p1["n_markers"] + p2["n_markers"] == len(cross.positions)


def test_end_to_end_recovers_heterozygous_segment(simulated_ril_het, tmp_path):
    """An AA / AB / BB offspring decodes to three segments in that order."""
    cross = simulated_ril_het
    result = run_to_table(cross, tmp_path)

    assert result["state"].tolist() == [3, 0, 1, 2, 3]
    assert result["state_label"].tolist()[1:4] == [cross.parent1, "heterozygous", cross.parent2]
    # Allow a couple of misplaced markers at each of the two breakpoints
    decoded = states_at(result, cross.positions)
    assert (decoded != cross.ancestry).sum() <= 4


def test_end_to_end_breakpoint_in_uninformative_stretch(simulated_ril_mixed_informativeness, tmp_path):
    """
    Where the parents agree, ancestry cannot be seen, so the breakpoint may
    fall anywhere in that stretch; informative markers are still decoded
    correctly.
    """
    cross = simulated_ril_mixed_informativeness
    result = run_to_table(cross, tmp_path)

    assert result["state"].tolist() == [3, 0, 2, 3]
    decoded = states_at(result, cross.positions)
    np.testing.assert_array_equal(decoded[cross.informative], cross.ancestry[cross.informative])

    # The P1 -> P2 switch lies between the last informative marker before the
    # true breakpoint and the first one after it
    true_switch = np.flatnonzero(np.diff(cross.ancestry))[0]
    informative_idx = np.flatnonzero(cross.informative)
    last_before = cross.positions[informative_idx[informative_idx <= true_switch][-1]]
    first_after = cross.positions[informative_idx[informative_idx > true_switch][0]]
    p1, p2 = result.iloc[1], result.iloc[2]
    assert last_before <= p1["end_pos"] < p2["start_pos"] <= first_after


def test_end_to_end_writes_to_stdout(simulated_ril, capsys):
    cross = simulated_ril
    assert main(cross.argv()) == 0
    captured = capsys.readouterr()

    result = pd.read_csv(io.StringIO(captured.out), sep="\t")
    assert list(result.columns) == OUTPUT_COLUMNS
    assert len(result) == 4
    # Progress goes to stderr so stdout stays a clean table
    assert f"Chromosome {cross.chrom}: {len(cross.positions)} comparable markers" in captured.err
    assert "Wrote" not in captured.err


def test_end_to_end_no_shared_markers(tmp_path, capsys):
    ref, alt = PL_BY_GENOTYPE[0], PL_BY_GENOTYPE[2]
    reference = tmp_path / "reference.h5"
    offspring = tmp_path / "offspring.h5"
    write_h5(reference, ["P1", "P2"], ["Chr1"] * 2, [10, 20], [[ref, alt]] * 2)
    write_h5(offspring, ["F10"], ["Chr1"] * 2, [15, 25], [[ref]] * 2)
    out = tmp_path / "segments.tsv"

    argv = [
        "--input", str(offspring), "--reference", str(reference),
        "--offspring", "F10", "--parent1", "P1", "--parent2", "P2", "-o", str(out),
    ]
    assert main(argv) == 1
    assert "No comparable markers" in capsys.readouterr().err
    assert not out.exists()


def test_end_to_end_skips_unknown_requested_chromosome(simulated_ril, tmp_path, capsys):
    cross = simulated_ril
    result = run_to_table(cross, tmp_path, "--chr", f"{cross.chrom},Chr9")
    assert set(result["chrom"]) == {cross.chrom}
    assert "chromosome 'Chr9' not found" in capsys.readouterr().err


def test_end_to_end_no_requested_chromosome_found(simulated_ril, tmp_path, capsys):
    cross = simulated_ril
    out = tmp_path / "segments.tsv"
    assert main(cross.argv("-o", str(out), "--chr", "Chr9")) == 1
    assert "Nothing to output" in capsys.readouterr().err
    assert not out.exists()


def test_end_to_end_missing_chromosome_length_uses_last_marker(simulated_ril, tmp_path, capsys):
    """Without a known length, the table ends at the last marker with no trailing pad."""
    cross = simulated_ril
    # A later --chr-lengths overrides the one set by cross.argv()
    result = run_to_table(cross, tmp_path, "--chr-lengths", "Chr2=1000")

    assert f"no length for chromosome '{cross.chrom}'" in capsys.readouterr().err
    assert result["end_pos"].iloc[-1] == cross.positions[-1]
    assert result["state"].tolist() == [3, 0, 2]


def test_module_help_runs():
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    proc = subprocess.run(
        [sys.executable, "-m", "biscit.cli", "--help"], capture_output=True, text=True, env=env
    )
    assert proc.returncode == 0
    assert "usage: biscit" in proc.stdout


@pytest.mark.skipif(shutil.which("biscit") is None, reason="biscit entry point not installed")
def test_entry_point_help_runs():
    proc = subprocess.run(["biscit", "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    assert "usage: biscit" in proc.stdout
