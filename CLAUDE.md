# biscit

HMM-based imputation of offspring genotypes from offspring allelic depths
(AD) and parental PL arrays in *A. thaliana* RIL crosses. Source is in `src/biscit/`:

- `io.py`: load parental PL and offspring AD from scikit-allel-style HDF5
  files, convert the offspring AD to PL, and align offspring with parents
  per chromosome
- `hmm.py`: emission probabilities, transition matrix, and minimum-duration
  Viterbi decoding
- `segments.py`: collapse the decoded path into segments, mask short ones,
  and pad to chromosome ends
- `cli.py`: the `biscit` entry point; writes a BED-like table

`build/` holds a stale copy of the package. Edit only `src/`.

## Documentation conventions

- Every function gets a NumPy-style docstring: a summary line, an optional
  extended description, then `Parameters`, `Returns`, and `Raises` /
  `Notes` where relevant.
- Give array shapes in double backticks, e.g. ``(n_loci, 3)``, and state
  dict keys explicitly. Refer to other parameters or functions with single
  backticks.
- Spell out encodings: genotype columns are 0 = hom-REF, 1 = het,
  2 = hom-ALT; hidden states are 0 = AA (parent 1), 1 = AB, 2 = BB
  (parent 2). State 3 is not an HMM state: it is added in `segments.py` for
  short and boundary segments, and is labelled "Unassigned" in the output.
- Inline comments explain *why* or the biological meaning, not what the
  syntax does. Keep them short, one per logical step, and skip the obvious
  lines. Trivial one-line helpers need no inline comments.
- When annotating, do not change behaviour. Report suspected bugs or
  inconsistencies separately instead of fixing them silently.
