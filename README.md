# Booyah! Intra-Swedish-Crosses Imputation Tool (BISCIT)

HMM-based imputation of offspring genotypes from parental genotype data, designed for *Arabidopsis thaliana* RIL (recombinant inbred line) crosses.

`biscit` reads allelic depths (`calldata/AD`) for an offspring and Phred-scaled genotype likelihoods (`calldata/PL`) for its two parents from HDF5-converted VCFs. It converts the offspring's read counts to genotype likelihoods using a per-read mapping error rate, builds per-marker emission probabilities, and decodes the most likely ancestry path (parent 1 homozygous, heterozygous, or parent 2 homozygous) with a minimum-duration Viterbi algorithm. The result is a BED-like table of genotype segments per chromosome.

See [here](docs/how_biscit_works.md) for a detailed explanation of what the program does.

## Installation

```bash
cd biscit
pip install -e .
```

This installs the `biscit` command and the importable `biscit` package
(dependencies: `numpy`, `pandas`, `h5py`).

Alternatively, install a conda/mamba environment:

```bash
cd biscit
mamba env create -f environment.yml
```

## Input data 

`biscit` requires VCF files be converted to HDF5 format.
The `--input` (offspring) file must contain the `AD` field and the `--reference` (parental) file the `PL` field.

See [here](docs/input_data.md) for full details on the input data.

## Usage

Check input parameters:
```bash
biscit -h
```

If genotype data for your offspring sample `test_individual` are in `test.hdf5` and data for the parental lines `parentA` and `parentB` are in `reference.hdf5`, run:

```bash
biscit \
  --input   path/to/test.hdf5 \
  --reference path/to/reference.hdf5 \
  --offspring test_individual \
  --parent1 parentA \
  --parent2 parentB \
  --output  segments.tsv
```

By default all aligned chromosomes are processed. Restrict with `--chr Chr1,Chr2`.
If `--output` is omitted the table is written to stdout (progress messages go to
stderr).

### Key options

| Option | Default | Meaning |
| --- | --- | --- |
| `--mapping-error` | 0.05 | Probability that an offspring read shows the wrong allele |
| `--n-generations-selfing` | 10 | Selfing generations used for emission probabilities |
| `--cM-per-Mb` | 4.0 | Recombination rate (cM/Mb) used for transition probabilities |
| `-N`, `--min-markers` | 50 | Minimum state duration (in markers) in the decoder |
| `--min-base-pairs` | 50000 | Minimum segment length (bp) before a stretch is marked Unassigned |
| `--chr-lengths` | TAIR10 | `Chr1=30427671,Chr2=19698289,...` chromosome lengths |
| `--allow-identical-sites` | off | Keep sites where both parents are most likely the same homozygote |

By default, `biscit` drops sites where both parents are most likely the same
homozygote (both have PL 0 for hom-REF, or both for hom-ALT). These sites
cannot tell the parents apart. Pass `--allow-identical-sites` to keep them (for example, if the parents share long tracts of genome identical by descent).

`--cM-per-Mb` sets the probability of switching state between neighbouring markers.
One rate is used for the whole genome. The default of 4 cM/Mb is a rough genome-wide
average for *A. thaliana*. Raise it to allow more breakpoints, or lower it if you
see spurious short switches. See [docs/how_biscit_works.md](docs/how_biscit_works.md)
for how the rate enters the transition matrix.

### Output columns

`chrom`, `start_pos`, `end_pos`, `state`, `state_label`, `n_markers`

State labels: `0=AA` (parent 1), `1=AB` (heterozygous), `2=BB` (parent 2),
`3=Unassigned` (segments shorter than `--min-base-pairs` and stretches at
chromosome ends).


## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
