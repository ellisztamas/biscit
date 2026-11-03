# Create the data files for `biscit`

## Genotype data

### HDF5 files

`biscit` works with HDF5 files created from VCF files using `scikit-allel`.
The reason for this is that HDF5 files allow us to access subsets of the
genotype data much more easily without having to read the entire VCF file into
memory, or parse the whole file.

I recommend creating this using [vcf_to_hdf5](https://scikit-allel.readthedocs.io/en/latest/io.html#allel.vcf_to_hdf5) from scikit-allel in Python. For example:

```python
import allel
allel.vcf_to_hdf5('example.vcf', 'example.h5', fields='*', overwrite=True)
```

There are three important things to be aware of when preparing the input files:

1. It is essential that the parental and offspring input files are based on 
alignments to the same reference assemblies.
2. `biscit` identifies which SNP positions match between the two input files,
but it does not check that of the REF and ALT alleles were defined in the same
way at each site. One way to enforce this is to use a targets file with BCFtools
that gives the allelic state explicitly.
3. For offspring genotypes, `biscit` uses the read counts directly from field `AD`.
The input file with progeny genotypes therefore needs to contain field `AD`,
which is not always included by default. For the parental genotypes, `biscit`
uses the PHRED-scaled likelihoods from the `PL` field of the VCF file, which 
usually is included.

### Subsetting the genotypes

Note that this is not mandatory, but may help if you find the output looks like
complete garbage.

VCF files usually involves aligning reads from multiple samples to a single
reference genome assembly.
We should expect this to cause reads to aligning incorrectly.
For example, reads coming from structural variants not present in the reference
assembly will align incorrectly to similar regions, leading to well-supported 
but incorrect heterozygous calls.
That in turn causes the HMM to return incorrect haplotypes with high confidence.
Note that this issue is essentially impossible to fix at the HMM stage - the
problem is with the data.

For this reason it may be sufficient to restrict the SNPs used for imputation
to a subset of high quality, reliable markers.
In *Arabidopsis thaliana*, regions between genes are messy, but coding genes 
are highly syntenic and contain SNPs that type reliably.
The genome is also pretty compact, so you don't lose too much information by 
the lack in resolution caused by recombination breakpoints that fall between 
genes.
This seems to be effective in *A. thaliana*; it may not be necessary or sufficient
in other systems.

Here is an outline of how to subset a VCF file with genome wide SNPs to include
only those with genic SNPs.

1. Download the annotation file TAIR10_GFF3_genes.gff giving positions of coding genes in the *A. thaliana* reference assembly TAIR10:
```sh
genes=https://www.arabidopsis.org/download/file?path=Genes/TAIR10_genome_release/TAIR10_gff3/TAIR10_GFF3_genes.gff
wget -P $genes
```
2. Get a list of SNPs segregating in the progeny, and filter that list for positions inside genes. Here I'm starting from the progeny because I know that the VCF file for the progeny has fewer SNPs than the parents, but in principle this ought to work if you start from the parents as well.
```sh
# Get a list of SNP positions in the VCF for the progeny
bcftools view --min-ac 160 progeny.vcf.gz | 
    bcftools query -f"%CHROM\t%POS\n"  > progeny_snp_positions.txt
# Convert to BED format. Positions are given as Chr, Position, Position +1
awk '{print $1"\t"$2"\t"$2+1}' progeny_snp_positions.txt > $progeny_snp_positions.bed

# Select only features labelled as coding genes in the GFF, and convert to BED
grep "Note=protein_coding_gene" TAIR10_GFF3_genes.gff | awk -F'\t' '{print $1"\t"$4"\t"$5}' > annotated_genes.bed

# Get only those SNPs that are inside genes.
# Bedtools intersect finds the SNPs inside genes, and awk reformats for bcftools
# Example output: "Chr1 6063"
bedtools intersect -a $all_snps_bed -b annotated_genes.bed -wb | \
awk -F'\t' '{print $1"\t"$2}' | \
bgzip > snps_in_genes.bed
```
3. Use bcftools to subset full VCF files for the progent and parents to only contain SNPs inside genes. It is also recommended to remove variants with duplicate positions (usually where a SNP and indel occur at the same location).
```sh
# Subset the VCF file parents.vcf.gz
bcftools view \
    -R snps_in_genes.bed \
    -v snps \
    -O z  \
    -o parents_subset.vcf.gz \
    parents.vcf.gz

# Subset the VCF file for the progeny
bcftools view \
    -R snps_in_genes.bed \
    -v snps \
    -O z \
    -o progeny_subset.vcf.gz \
    progeny.vcf.gz
```


## Chromosome sizes

`biscit` needs the length of each chromosome so that it can extend the
output segments out to the chromosome ends (the stretches before the first
marker and after the last marker are labelled Unassigned). Lengths are given
with `--chr-lengths` as comma-separated `NAME=LENGTH` pairs.

By default `biscit` uses the lengths of the five nuclear chromosomes in the
TAIR10 assembly of *A. thaliana*:

```
Chr1=30427671,Chr2=19698289,Chr3=23459830,Chr4=18585056,Chr5=26975502
```

These defaults are only correct if your reads were aligned to TAIR10. If you
aligned to a different assembly, such as TAIR12 or the assembly of another
accession, both the lengths and possibly the chromosome names will differ,
and you should supply them yourself. Two things matter:

1. The lengths must come from the same assembly that the reads were aligned
to.
2. The names must match the `CHROM` column of your VCF files exactly
(e.g. `Chr1` is not the same as `chr1` or `1`).

The simplest place to get both is the FASTA index of the reference assembly.
For example, if you aligned to TAIR12 and the reference is in `TAIR12.fa`:

```sh
# Create TAIR12.fa.fai if it does not already exist.
# Columns 1 and 2 are the sequence name and length.
samtools faidx TAIR12.fa
cut -f1,2 TAIR12.fa.fai

# Keep the nuclear chromosomes and format them for --chr-lengths
cut -f1,2 TAIR12.fa.fai | grep -E '^Chr[1-5]\b' | \
    awk '{print $1"="$2}' | paste -sd, -
```

Adjust the `grep` pattern to whatever the chromosomes are called in your
assembly. Pass the resulting string to `biscit` with
`--chr-lengths "Chr1=...,Chr2=...,..."`.

If you do not have the FASTA file, the same information is usually in the
`##contig` lines of the VCF header, which record the reference the VCF was
called against:

```sh
bcftools view -h progeny.vcf.gz | grep '^##contig'
# e.g. ##contig=<ID=Chr1,length=30427671>
```

Only the chromosomes you process need a length; extra entries, such as the
chloroplast and mitochondrion, are ignored. If a chromosome has no length,
`biscit` prints a warning and uses the position of the last marker instead,
so no Unassigned stretch is added at the end of that chromosome.
