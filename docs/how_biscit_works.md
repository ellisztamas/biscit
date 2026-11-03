# How `biscit` works

`biscit` infers the ancestry of each region of an offspring genome in a
biparental cross. It does this with a hidden Markov model (HMM): at each
marker it compares the offspring's reads with the genotypes of the two
parents, and then finds the most likely sequence of ancestry states along
each chromosome.

This document covers what happens to the data once it is loaded. For how to
prepare the input files (HDF5 conversion, required `AD` and `PL` fields,
consistent REF/ALT coding, and optional subsetting to genic SNPs), see
[Create the data files for `biscit`](input_data.md).

The pipeline for each offspring sample is:

1. Load parental `PL` and offspring `AD`, and convert offspring `AD` to `PL`
   (`io.py`).
2. Match markers between offspring and parents, and drop uninformative or
   unreliable sites (`io.py`).
3. Compute emission probabilities for each hidden state at each marker
   (`hmm.py`).
4. Decode the most likely state path using a Viterbi algorithm with a
   minimum segment length (`hmm.py`).
5. Collapse the path into segments, mask short segments, and pad to the
   chromosome ends (`segments.py`).
6. Write a BED-like table (`cli.py`).

## Encodings

Two different encodings are used throughout, and it is worth keeping them
apart:

| Code | Genotype (columns of PL/GL arrays) | Hidden state (HMM)   |
|------|------------------------------------|----------------------|
| 0    | hom-REF                            | AA: homozygous for parent 1 |
| 1    | het                                | AB: heterozygous     |
| 2    | hom-ALT                            | BB: homozygous for parent 2 |
| 3    | –                                  | Unassigned (added after decoding) |

Genotypes describe alleles relative to the reference assembly; hidden states
describe ancestry relative to the parents. Which genotype corresponds to
which state depends on the parents' genotypes at that marker. State 3 is not
an HMM state: it is introduced only when post-processing segments.

## Loading the data

`biscit` reads HDF5 files in the layout written by `allel.vcf_to_hdf5`.
Only the columns for the requested samples are read from `calldata/AD` or
`calldata/PL`, so the full genotype matrix is never held in memory.

- **Parents**: Phred-scaled genotype likelihoods (`PL`) are taken directly
  from the reference file.
- **Offspring**: allelic depths (`AD`) are read from the input file and
  converted to `PL` (below). Using read counts directly means the offspring
  genotype likelihoods do not depend on the variant caller's genotype model,
  which may assume Hardy-Weinberg proportions or diploid-outbred priors that
  do not hold in an inbred line.

### Converting allelic depths to genotype likelihoods

For parental genotype data `biscit` extracts genotype likelihoods that a marker is homozygous for the REF allele, heterozygous, or homozygous for the ALT allele directly from the `PL` of the HDF5 file.

For offspring genotype data, `biscit` instead uses the read counts with a generous error rate, anticipating that coverage will be much lower and calls will be messier.
For a marker with $r$ REF reads and $a$ ALT reads, and a per-read error rate
$\varepsilon$ (`--mapping-error`, default 0.05), the likelihood of each
genotype is

$$
\begin{aligned}
L(\text{hom-REF}) &= (1-\varepsilon)^r \, \varepsilon^a \\
L(\text{het})     &= 0.5^{\,r+a} \\
L(\text{hom-ALT}) &= \varepsilon^r \, (1-\varepsilon)^a
\end{aligned}
$$

That is, each read independently shows the true allele with probability
$1-\varepsilon$, and a heterozygote produces either allele with probability
one half. The error rate absorbs sequencing errors and mismapped reads; it
must lie strictly between 0 and 0.5. At 0, a single discordant read would
rule out a homozygous genotype entirely; at 0.5, reads carry no information.

The calculation is done in log10 space to avoid underflow at high depth, and
then rescaled to the usual `PL` convention: $\mathrm{PL} = -10 \log_{10} L$,
shifted so that the most likely genotype has PL 0. Only the first ALT allele
is used. Missing `AD` (negative values) is coded as PL −1, and markers with
no reads end up with all three PLs equal to 0; both are removed by the
filters below.

## Aligning offspring and parents

Markers are matched between the two files per chromosome, on chromosome name
and position only. **REF and ALT alleles are not compared**, so the two files
must code alleles identically (see [input_data.md](input_data.md)). Only
chromosomes present in both files are processed.

A marker is then dropped if, in any of the three individuals:

- any PL value is negative (missing data), or
- all PL values are 0 (no information; for the offspring, no reads),

or if either parent's most likely genotype is heterozygous (`PL[1] == 0`).
The parents are expected to be inbred, so an apparently heterozygous
parental call most likely reflects mismapping, paralogous sequence or a
structural variant, rather than real heterozygosity.

By default, markers where both parents have the same most likely homozygous
genotype are also dropped, because they cannot distinguish the parents.
`--allow-identical-sites` keeps them (see the next section for how they are
handled).

## Emission probabilities

Parental and offspring PLs are first converted to genotype likelihoods
normalised to sum to one over the three genotypes:

$$
g(k) = \frac{10^{-\mathrm{PL}_k/10}}{\sum_j 10^{-\mathrm{PL}_j/10}}
$$

Write $g_1$, $g_2$ and $o$ for the likelihoods of parent 1, parent 2 and
the offspring.

### Expected heterozygosity

After $n$ generations of selfing (`--n-generations-selfing`, default 10),
the expected proportion of the genome that is still heterozygous is

$$
h = 0.5^{\,n}
$$

and each homozygous state has expected frequency $(1-h)/2$. These
frequencies are used as weights on the emission for each state.

### Summing over parental genotypes

The emission for each state sums over the four combinations in which both
parents are homozygous. For each combination it multiplies the likelihood of
the parents' data by the likelihood of the offspring data given the genotype
that the state implies:

$$
\begin{aligned}
e_{AA} &= \tfrac{1-h}{2}\,\big[
  g_1(0)g_2(2)\,o(0) + g_1(2)g_2(0)\,o(2)
  + g_1(0)g_2(0)\,o(0) + g_1(2)g_2(2)\,o(2) \big] \\
e_{AB} &= h\,\big[
  g_1(0)g_2(2)\,o(1) + g_1(2)g_2(0)\,o(1)
  + g_1(0)g_2(0)\,o(0) + g_1(2)g_2(2)\,o(2) \big] \\
e_{BB} &= \tfrac{1-h}{2}\,\big[
  g_1(0)g_2(2)\,o(2) + g_1(2)g_2(0)\,o(0)
  + g_1(0)g_2(0)\,o(0) + g_1(2)g_2(2)\,o(2) \big]
\end{aligned}
$$

- The first two terms are **segregating sites**, where the parents carry
  different alleles. Here the offspring genotype tells us about ancestry:
  under AA the offspring should match parent 1, under BB it should match
  parent 2, and under AB it should be heterozygous.
- The last two terms are **identical sites**, where both parents carry the
  same homozygous genotype. The offspring is expected to match both parents
  whatever its ancestry, so these terms are the same in every state and only
  differ by the state weight. With the default filtering these sites have
  already been removed, but if they are kept they carry some weight when the
  parental genotypes are uncertain.

Parental heterozygous genotypes are not included in the sum. Markers where a
parent is most likely heterozygous have already been removed, so any
remaining heterozygous likelihood is simply ignored.

The result is an array of shape `(n_loci, 3)`, one column per hidden state.
The rows are not normalised at this stage.

## Transitions between states

Transition probabilities depend on the physical distance between adjacent
markers. The distance $d$ (bp) is converted to a genetic distance in Morgans
assuming a uniform recombination rate $c$ (`--cM-per-Mb`, default 4.0):

$$
\rho = \frac{c \, d}{10^6 \times 100}
$$

This is used directly as the probability of leaving the current state, split
equally between the other two states:

$$
T = \begin{pmatrix}
1-\rho & \rho/2 & \rho/2 \\
\rho/2 & 1-\rho & \rho/2 \\
\rho/2 & \rho/2 & 1-\rho
\end{pmatrix}
$$

All three states, including the heterozygous one, are treated alike. The
model therefore does not encode, for example, that in a RIL a switch from AA
to BB normally passes through a short AB stretch; that kind of structure is
instead handled by the minimum segment length below.

Some points to be aware of:

- No map function (e.g. Haldane) is applied. This is fine for the short
  distances between adjacent markers, but the matrix would stop being a
  valid probability matrix beyond 1 Morgan (25 Mb at the default rate, or
  $100/c$ Mb in general).
- A single recombination rate is used for the whole genome. The default of
  4 cM/Mb is a rough genome-wide average for *A. thaliana*; it can be changed
  with `--cM-per-Mb`, but local variation, such as suppressed recombination
  around centromeres, is not accounted for.
- The transition matrix does not depend on the number of generations of
  selfing.

## Decoding: Viterbi with a minimum segment length

The most likely state path is found with the Viterbi algorithm, modified so
that every segment (except the first and last) must span at least `N`
markers (`-N/--min-markers`, default 50). This prevents the decoder from
switching state on the basis of a handful of markers, which in practice are
often caused by mismapped reads or mis-called SNPs (see the discussion of
subsetting in [input_data.md](input_data.md)).

### Preparing the emissions

Each row of the emission array is first normalised to sum to one. Rows that
sum to zero (no state supports the data) are replaced with a uniform
distribution. Because this rescales all states at a marker by the same
factor, it does not change which path is most likely; it only keeps the
values in a well-behaved numerical range. All calculations are then done in
log space, with zero probabilities clipped to the smallest positive float.

The initial state distribution is uniform. Note that the expected genotype
frequencies are already included in the emissions, so they act as a
per-marker prior.

### Expanded state space

To enforce the minimum length, each hidden state $s$ is paired with a
counter $c \in \{1, \dots, N\}$ giving how many consecutive markers have been
spent in $s$. This gives $3N$ augmented states. At each step:

- **Staying** in state $s$ advances the counter, $c \to \min(c+1, N)$, with
  probability $T_{ss}$. Once the counter reaches $N$ it stays there.
- **Switching** from $s$ to $s' \ne s$ is allowed only when $c = N$, with
  probability $T_{ss'}$. The new segment starts with counter 1.

Every state starts with its counter at $N$, so the first segment can be
shorter than $N$ markers. The final segment can also be shorter than $N$,
because nothing forces the path to finish with a full counter.

After the last marker, the path is traced back from the augmented state with
the highest score, and the counter is dropped to give one hidden state
(0, 1 or 2) per marker.

The run time scales as `O(n_loci × N)`, and the decoder is implemented as
plain Python loops, so large values of `N` on dense marker sets will be
slow.

## Post-processing segments

### Collapsing the path

The per-marker path is collapsed into runs of identical consecutive states.
Each segment records its state, the positions of its first and last markers,
and the number of markers it contains. Segment boundaries are marker
positions, so there are gaps between adjacent segments where the breakpoint
lies somewhere between two markers.

### Masking short segments

Segments whose physical length (last marker − first marker + 1) is shorter
than `--min-base-pairs` (default 50 000 bp) are set to state 3
(Unassigned). Adjacent segments that now share a state, including runs of
newly unassigned segments, are merged and their marker counts summed.

This filter works in base pairs, whereas the minimum-duration constraint in
the decoder works in markers. The two complement each other: a segment of 50
markers can still be very short physically where markers are dense.

### Padding to chromosome ends

Finally, Unassigned segments are added from position 1 to just before the
first marker, and from just after the last marker to the chromosome end.
These have `n_markers` of 0. Chromosome lengths default to TAIR10 and can be
set with `--chr-lengths`; if a chromosome's length is unknown, the last
marker is used and no trailing segment is added. Gaps between internal
segments are not filled.

## Output

The output is a tab-separated, BED-like table with one row per segment:

| Column        | Meaning |
|---------------|---------|
| `chrom`       | Chromosome name |
| `start_pos`   | First position of the segment |
| `end_pos`     | Last position of the segment |
| `state`       | 0, 1, 2 or 3 (see [Encodings](#encodings)) |
| `state_label` | Name of parent 1, `heterozygous`, name of parent 2, or `Unassigned` |
| `n_markers`   | Number of markers in the segment |

Positions are 1-based and inclusive, as in the VCF, not 0-based half-open as
in a true BED file.

## Parameters at a glance

| Option | Default | Used in |
|--------|---------|---------|
| `--mapping-error` | 0.05 | AD to PL conversion |
| `--allow-identical-sites` | off | Marker filtering |
| `--n-generations-selfing` | 10 | Emission weights ($h = 0.5^n$) |
| `-N`, `--min-markers` | 50 | Minimum segment length in the decoder |
| `--min-base-pairs` | 50 000 | Masking short segments |
| `--chr-lengths` | TAIR10 | Padding to chromosome ends |
| `--cM-per-Mb` | 4.0 | Transition matrix (recombination rate) |
