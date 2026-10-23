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
    """Parse a ``Chr1=30427671,Chr2=19698289`` style string into a dict."""
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


def build_parser():
    parser = argparse.ArgumentParser(
        prog="biscit",
        description=(
            "HMM-based imputation of offspring genotypes from parental PL arrays "
            "(A. thaliana RIL crosses)."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to the test/offspring HDF5 file.")
    parser.add_argument("--reference", required=True, help="Path to the reference/parental HDF5 file.")
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
        "--n-generations-selfing", type=int, default=10,
        help="Number of selfing generations for emission probabilities (default: 10).",
    )
    parser.add_argument(
        "-N", "--min-markers", type=int, default=50,
        help="Minimum state duration (markers) for the Viterbi decoder (default: 50).",
    )
    parser.add_argument(
        "--min-base-pairs", type=int, default=50000,
        help="Minimum genomic length (bp) before a stretch is reassigned to Novel (default: 50000).",
    )
    return parser





def process_chromosome(chrom, data, chr_length, args):
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
    bed_table_long = summarize_path(best_path, data["positions"])
    bed_table = filter_short_stretches(bed_table_long, min_length=args.min_base_pairs)
    bed_table = add_boundary_regions(bed_table, chr_length=chr_length)
    bed_table = bed_table.copy()
    bed_table.insert(0, "chrom", chrom)
    return bed_table


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    chr_lengths = args.chr_lengths or dict(DEFAULT_CHR_LENGTHS)

    pl_arrays = extract_aligned_pl(
        args.input, args.reference, args.offspring, args.parent1, args.parent2
    )

    if not pl_arrays:
        print("No comparable markers found between input and reference.", file=sys.stderr)
        return 1

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
    result = result[[c for c in columns if c in result.columns]]

    if args.output:
        result.to_csv(args.output, sep="\t", index=False)
        print(f"Wrote {len(result)} segments to {args.output}", file=sys.stderr)
    else:
        result.to_csv(sys.stdout, sep="\t", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
