# Booyah! Intra-Swedish-Crosses Imputation Tool (BISCIT)

HMM-based imputation of offspring genotypes from parental PL arrays, designed for
*Arabidopsis thaliana* RIL (recombinant inbred line) crosses.

`biscit` reads Phred-scaled genotype likelihoods (`calldata/PL`) from HDF5-converted VCFs for an offspring and its two parents, builds per-marker emission probabilities, and decodes the most likely ancestry path (parent 1 homozygous, heterozygous, parent 2 homozygous, or a "novel"/private state) with a minimum-duration Viterbi algorithm. The result is a BED-like table of genotype segments per chromosome.

## Installation

```bash
cd biscit
pip install -e .
```

This installs the `biscit` command and the importable `biscit` package
(dependencies: `numpy`, `pandas`, `h5py`).

Alteratively, install a conda/mamba environment:

```bash
cd biscit
mamba env create -f environment.yml
```

## Input data 

`biscit` requires VCF files be converted to HDF5 format.
The reason for using HDF5 is that it allows for loading data in chunks, which is much quicker than loading an entire VCF file into memory every time you want to check a single sample.

I recommend creating this using [`vcf_to_hdf5`](https://scikit-allel.readthedocs.io/en/latest/io.html#allel.vcf_to_hdf5) from `scikit-allel` in Python. For example:

```Python
import allel
allel.vcf_to_hdf5('example.vcf', 'example.h5', fields='*', overwrite=True)
```

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
| `--n-generations-selfing` | 10 | Selfing generations used for emission probabilities |
| `-N`, `--min-markers` | 50 | Minimum state duration (in markers) in the decoder |
| `--min-length` | 50000 | Minimum segment length (bp) before a stretch is marked Novel |
| `--chr-lengths` | TAIR10 | `Chr1=30427671,Chr2=19698289,...` chromosome lengths |

### Output columns

`chrom`, `start_pos`, `end_pos`, `state`, `state_label`, `n_markers`

State labels: `0=AA` (parent 1), `1=AB` (heterozygous), `2=BB` (parent 2),
`3=Novel`.


## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
