# Booyah! Intra-Swedish-Crosses Imputation Tool (BISCIT)

HMM-based imputation of offspring genotypes from parental PL arrays, designed for
*Arabidopsis thaliana* RIL (recombinant inbred line) crosses.

`biscit` reads Phred-scaled genotype likelihoods (`calldata/PL`) from HDF5-converted
VCFs for an offspring and its two parents, builds per-marker emission probabilities,
and decodes the most likely ancestry path (parent 1 homozygous, heterozygous,
parent 2 homozygous, or a "novel"/private state) with a minimum-duration Viterbi
algorithm. The result is a BED-like table of genotype segments per chromosome.

## Installation

```bash
cd biscit
pip install -e .
```

This installs the `biscit` command and the importable `biscit` package
(dependencies: `numpy`, `pandas`, `h5py`).

## Usage

```bash
biscit \
  --input   path/to/test.hdf5 \
  --reference path/to/reference.hdf5 \
  --offspring ISC_029.1 \
  --parent1 1552 \
  --parent2 1158 \
  --output  segments.tsv
```

By default all aligned chromosomes are processed. Restrict with `--chr Chr1,Chr2`.
If `--output` is omitted the table is written to stdout (progress messages go to
stderr).

### Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `--n-generations-selfing` | 10 | Selfing generations used for emission probabilities |
| `--n-self` | 10 | Selfing generations used in the transition model |
| `-N`, `--min-duration` | 50 | Minimum state duration (in markers) in the decoder |
| `--min-length` | 50000 | Minimum segment length (bp) before a stretch is marked Novel |
| `--chr-lengths` | TAIR10 | `Chr1=30427671,Chr2=19698289,...` chromosome lengths |

### Output columns

`chrom`, `start_pos`, `end_pos`, `state`, `state_label`, `n_markers`

State labels: `0=AA` (parent 1), `1=AB` (heterozygous), `2=BB` (parent 2),
`3=Novel`.

## Library use

```python
from biscit import extract_aligned_pl, get_emission_probs, viterbi_min_duration

pl_arrays = extract_aligned_pl("test.hdf5", "reference.hdf5", "ISC_029.1", "1552", "1158")
emissions = get_emission_probs(pl_arrays["Chr1"], n_generations_selfing=10)
path = viterbi_min_duration(pl_arrays["Chr1"]["positions"], emissions, N=50, n_self=10)
```
