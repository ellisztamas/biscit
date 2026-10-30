import argparse
import sys

import pandas as pd

from .io import extract_aligned_pl
from .hmm import get_emission_probs, viterbi_min_duration
from .segments import summarize_path, filter_short_stretches, add_boundary_regions


# Default A. thaliana (TAIR10) chromosome lengths.
DEFAULT_CHR_LENGTHS = {
    "Chr1": 30427671,
    "Chr2": 19698289,
    "Chr3": 23459830,
    "Chr4": 18585056,
    "Chr5": 26975502,
}

def parse_chr_lengths(value):
    """
    Parse a chromosome-lengths string into a dict.

    Used as the argparse ``type`` for ``--chr-lengths``.

    Parameters
    ----------
    value : str or None
        Comma-separated ``NAME=LENGTH`` pairs, e.g.
        ``"Chr1=30427671,Chr2=19698289"``. Empty entries are ignored. If None,
        the TAIR10 defaults are returned.

    Returns
    -------
    dict of str to int
        Mapping from chromosome name to length in base pairs.

    Raises
    ------
    argparse.ArgumentTypeError
        If an entry lacks ``=`` or its length is not an integer.
    """
    if value is None:
        return dict(DEFAULT_CHR_LENGTHS)
    lengths = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"Invalid chr-lengths entry '{item}'. Expected NAME=LENGTH."
            )
        name, length = item.split("=", 1)
        try:
            lengths[name.strip()] = int(length)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid length '{length}' for chromosome '{name}'."
            )
    return lengths


def parse_mapping_error(value):
    """
    Parse and validate a per-read mapping error rate.

    Used as the argparse ``type`` for ``--mapping-error``.

    Parameters
    ----------
    value : str
        Error rate as a decimal string, e.g. ``"0.05"``.

    Returns
    -------
    float
        The error rate.

    Raises
    ------
    argparse.ArgumentTypeError
        If `value` is not a number strictly between 0 and 0.5.
    """
    try:
        rate = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid mapping error '{value}'.")
    # At 0 a single discordant read rules out a genotype; at 0.5 reads carry
    # no information
    if not 0 < rate < 0.5:
        raise argparse.ArgumentTypeError(
            f"Mapping error must be between 0 and 0.5 (exclusive), got {value}."
        )
    return rate


def build_parser():
    """
    Build the command-line argument parser for ``biscit``.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the input files, sample names, output path, chromosome
        selection and lengths, the read mapping error, and the HMM and
        filtering parameters.
    """
    parser = argparse.ArgumentParser(
        prog="biscit",
        description=(
            "HMM-based imputation of offspring genotypes from offspring allelic "
            "depths and parental PL arrays (A. thaliana RIL crosses)."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to the test/offspring HDF5 file. Must contain calldata/AD.")
    parser.add_argument(
        "--reference", required=True, 
        help=(
            "Path to the reference/parental HDF5 file. Must contain calldata/PL. "
            "REF/ALT alleles must be coded identically to those in --input."
            )
    )
    parser.add_argument("--offspring", required=True, help="Offspring sample name in --input.")
    parser.add_argument("--parent1", required=True, help="First parent sample name in --reference.")
    parser.add_argument("--parent2", required=True, help="Second parent sample name in --reference.")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output BED-like TSV path. Writes to stdout if omitted.",
    )
    parser.add_argument(
        "--chr", dest="chromosomes", default=None,
        help="Comma-separated chromosomes to process (default: all aligned chromosomes).",
    )
    parser.add_argument(
        "--chr-lengths", type=parse_chr_lengths, default=None,
        help=(
            "Chromosome lengths as 'Chr1=30427671,Chr2=19698289,...'. "
            "Defaults to TAIR10 lengths."
        ),
    )
    parser.add_argument(
        "--mapping-error", type=parse_mapping_error, default=0.05,
        help=(
            "Probability that an offspring read shows the wrong allele, used to "
            "convert allelic depths to genotype likelihoods (default: 0.05)."
        ),
    )
    parser.add_argument(
        "--n-generations-selfing", type=int, default=10,
        help="Number of selfing generations for emission probabilities (default: 10).",
    )
    parser.add_argument(
        "-N", "--min-markers", type=int, default=50,
        help="Minimum state duration (markers) for the Viterbi decoder (default: 50).",
    )
    parser.add_argument(
        "--min-base-pairs", type=int, default=50000,
        help="Minimum genomic length (bp) before a stretch is reassigned to Unassigned (default: 50000).",
    )
    parser.add_argument(
        "--allow-identical-sites", action="store_true",
        help=(
            "Keep sites where both parents are most likely the same homozygote "
            "(default: drop them)."
        ),
    )
    return parser





def process_chromosome(chrom, data, chr_length, args):
    """
    Decode ancestry segments for one chromosome.

    Computes emission probabilities, runs the minimum-duration Viterbi
    decoder, collapses the path into segments, masks short segments, and pads
    to the chromosome ends.

    Parameters
    ----------
    chrom : str
        Chromosome name, written to the ``chrom`` column.
    data : dict of str to numpy.ndarray
        Aligned data for this chromosome, with keys ``"positions"``,
        ``"parent1"``, ``"parent2"`` and ``"offspring"``, as returned per
        chromosome by `biscit.io.extract_aligned_pl`.
    chr_length : int
        Chromosome length in base pairs.
    args : argparse.Namespace
        Parsed arguments; uses ``n_generations_selfing``, ``min_markers`` and
        ``min_base_pairs``.

    Returns
    -------
    pandas.DataFrame
        Segment table with columns ``chrom``, ``start_pos``, ``end_pos``,
        ``n_markers`` and ``state``.
    """
    emission_probs = get_emission_probs(
        data, n_generations_selfing=args.n_generations_selfing
    )
    best_path = viterbi_min_duration(
        pos=data["positions"],
        emissions=emission_probs,
        N=args.min_markers,
        n_self=args.n_generations_selfing,
        initial=None,
    )
    # Collapse to segments, mask short ones, then pad to chromosome ends
    bed_table_long = summarize_path(best_path, data["positions"])
    bed_table = filter_short_stretches(bed_table_long, min_length=args.min_base_pairs)
    bed_table = add_boundary_regions(bed_table, chr_length=chr_length)
    bed_table = bed_table.copy()
    bed_table.insert(0, "chrom", chrom)
    return bed_table


def main(argv=None):
    """
    Run ``biscit`` from the command line.

    Converts the offspring allelic depths to PL, aligns them with the
    parental PL arrays, decodes each
    selected chromosome, and writes a BED-like TSV with columns ``chrom``,
    ``start_pos``, ``end_pos``, ``state``, ``state_label`` and ``n_markers``.
    Progress messages and warnings go to stderr.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. If None, ``sys.argv[1:]`` is used.

    Returns
    -------
    int
        Exit status: 0 on success, 1 if an input file lacks a required field
        or sample, there are no comparable markers, or there is nothing to
        output.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # argparse does not apply `type` to a None default, so fill it in here
    chr_lengths = args.chr_lengths or dict(DEFAULT_CHR_LENGTHS)

    try:
        pl_arrays = extract_aligned_pl(
            args.input, args.reference, args.offspring, args.parent1, args.parent2,
            mapping_error=args.mapping_error,
            allow_identical_sites=args.allow_identical_sites,
        )
    except KeyError as e:
        # Missing fields or samples are input problems, not bugs, so report
        # the message without a traceback
        print(f"Error: {e.args[0]}", file=sys.stderr)
        return 1

    if not pl_arrays:
        print("No comparable markers found between input and reference.", file=sys.stderr)
        return 1

    # Restrict to requested chromosomes, warning about any with no data
    if args.chromosomes:
        wanted = [c.strip() for c in args.chromosomes.split(",") if c.strip()]
        chroms = [c for c in wanted if c in pl_arrays]
        missing = [c for c in wanted if c not in pl_arrays]
        for c in missing:
            print(f"Warning: chromosome '{c}' not found in aligned data; skipping.", file=sys.stderr)
    else:
        chroms = list(pl_arrays.keys())

    tables = []
    for chrom in chroms:
        data = pl_arrays[chrom]
        print(
            f"Chromosome {chrom}: {len(data['positions'])} comparable markers",
            file=sys.stderr,
        )
        # Fall back to the last marker if the length is unknown, so no
        # trailing boundary segment is added
        if chrom not in chr_lengths:
            print(
                f"Warning: no length for chromosome '{chrom}'; using last marker position.",
                file=sys.stderr,
            )
            chr_length = int(data["positions"][-1])
        else:
            chr_length = chr_lengths[chrom]
        tables.append(process_chromosome(chrom, data, chr_length, args))

    if not tables:
        print("Nothing to output after processing.", file=sys.stderr)
        return 1
    
    result = pd.concat(tables, ignore_index=True)
    # Hidden-state labels used in the BED output (state index -> name).
    STATE_LABELS = {
        0: args.parent1,
        1: "heterozygous",
        2: args.parent2,
        3: "Unassigned"
        }
    result["state_label"] = result["state"].map(STATE_LABELS).fillna("Unknown")
    columns = ["chrom", "start_pos", "end_pos", "state", "state_label", "n_markers"]
    # Fix column order for the output
    result = result[[c for c in columns if c in result.columns]]

    if args.output:
        result.to_csv(args.output, sep="\t", index=False)
        print(f"Wrote {len(result)} segments to {args.output}", file=sys.stderr)
    else:
        result.to_csv(sys.stdout, sep="\t", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
