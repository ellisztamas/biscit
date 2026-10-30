"""Shared fixtures: synthetic HDF5 inputs with known ancestry."""

from dataclasses import dataclass

import h5py
import numpy as np
import pytest


# PL rows for a confidently called genotype, indexed by genotype code
# 0 = hom-REF, 1 = het, 2 = hom-ALT
PL_BY_GENOTYPE = np.array([
    [0, 60, 60],
    [60, 0, 60],
    [60, 60, 0],
])

# AD rows (REF count, ALT count) for a confidently called genotype, indexed
# by genotype code as for `PL_BY_GENOTYPE`
AD_BY_GENOTYPE = np.array([
    [20, 0],
    [10, 10],
    [0, 20],
])

CHROM = "Chr1"
SPACING = 1000


@dataclass
class Cross:
    """
    A simulated cross written to disk, with its true ancestry.

    Attributes
    ----------
    input : pathlib.Path
        HDF5 file containing the offspring sample, with AD only.
    reference : pathlib.Path
        HDF5 file containing both parental samples.
    offspring, parent1, parent2 : str
        Sample names in `input` and `reference`.
    chrom : str
        Name of the single simulated chromosome.
    chr_length : int
        Chromosome length in base pairs, past the last marker so a trailing
        boundary segment is expected.
    positions : numpy.ndarray of int
        Marker positions, shape ``(n_loci,)``.
    genotypes : dict of str to numpy.ndarray of int
        True genotype codes for keys ``"parent1"``, ``"parent2"`` and
        ``"offspring"``, each shape ``(n_loci,)``, coded 0 = hom-REF,
        1 = het, 2 = hom-ALT.
    ancestry : numpy.ndarray of int
        True hidden state at each marker, shape ``(n_loci,)``, coded
        0 = AA (parent 1), 1 = AB, 2 = BB (parent 2).
    informative : numpy.ndarray of bool
        Whether the parents carry opposing homozygous genotypes at each
        marker, shape ``(n_loci,)``.
    """

    input: object
    reference: object
    offspring: str
    parent1: str
    parent2: str
    chrom: str
    chr_length: int
    positions: np.ndarray
    genotypes: dict
    ancestry: np.ndarray
    informative: np.ndarray

    def argv(self, *extra):
        """
        Build a `biscit.cli.main` argument list for this cross.

        Parameters
        ----------
        *extra : str
            Additional arguments appended to the list.

        Returns
        -------
        list of str
            Arguments naming the files, samples and chromosome length.
        """
        return [
            "--input", str(self.input),
            "--reference", str(self.reference),
            "--offspring", self.offspring,
            "--parent1", self.parent1,
            "--parent2", self.parent2,
            "--chr-lengths", f"{self.chrom}={self.chr_length}",
            *extra,
        ]


def write_h5(path, samples, chrom, pos, pl=None, ad=None):
    """
    Write a minimal HDF5 file in scikit-allel layout.

    Only the calldata fields that are given are written, so a file can hold
    PL, AD, both or neither.

    Parameters
    ----------
    path : str or path-like
        Output file.
    samples : list of str
        Sample names, stored as UTF-8 byte strings.
    chrom : array_like of str
        Chromosome of each variant, shape ``(n_variants,)``.
    pos : array_like of int
        Position of each variant, shape ``(n_variants,)``.
    pl : array_like of int, optional
        Phred-scaled likelihoods, shape ``(n_variants, n_samples, 3)``.
    ad : array_like of int, optional
        Allelic depths (REF, ALT), shape ``(n_variants, n_samples, 2)``.
    """
    with h5py.File(path, "w") as h5:
        h5["samples"] = np.array([s.encode("utf-8") for s in samples])
        h5.create_dataset("variants/CHROM", data=np.array(chrom, dtype="S"))
        h5["variants/POS"] = np.asarray(pos)
        if pl is not None:
            h5["calldata/PL"] = np.asarray(pl, dtype=np.int32)
        if ad is not None:
            h5["calldata/AD"] = np.asarray(ad, dtype=np.int32)


def offspring_genotypes(gt1, gt2, ancestry):
    """
    Derive offspring genotypes from parental genotypes and true ancestry.

    Parameters
    ----------
    gt1, gt2 : numpy.ndarray of int
        Homozygous parental genotype codes (0 or 2), shape ``(n_loci,)``.
    ancestry : numpy.ndarray of int
        Hidden state at each marker, shape ``(n_loci,)``, coded 0 = AA,
        1 = AB, 2 = BB.

    Returns
    -------
    numpy.ndarray of int
        Offspring genotype codes, shape ``(n_loci,)``.
    """
    # An AB segment is only visibly het where the parents differ; where they
    # share a genotype, every state gives that same genotype
    het = np.where(gt1 == gt2, gt1, 1)
    return np.choose(ancestry, [gt1, het, gt2])


def make_cross(directory, gt1, gt2, ancestry):
    """
    Write parental and offspring HDF5 files for one simulated chromosome.

    Markers are placed every `SPACING` bp starting at `SPACING`, and every
    genotype is called confidently: parents with `PL_BY_GENOTYPE` and the
    offspring with `AD_BY_GENOTYPE`.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to write ``reference.h5`` and ``offspring.h5`` into.
    gt1, gt2 : array_like of int
        Homozygous parental genotype codes (0 or 2), shape ``(n_loci,)``.
    ancestry : array_like of int
        True hidden state of the offspring at each marker, shape
        ``(n_loci,)``, coded 0 = AA, 1 = AB, 2 = BB.

    Returns
    -------
    Cross
        File paths, sample names and the true genotypes and ancestry.
    """
    gt1, gt2, ancestry = (np.asarray(a) for a in (gt1, gt2, ancestry))
    n = len(ancestry)
    positions = SPACING * np.arange(1, n + 1)
    chrom = [CHROM] * n
    gt_off = offspring_genotypes(gt1, gt2, ancestry)

    reference = directory / "reference.h5"
    input = directory / "offspring.h5"
    write_h5(
        reference, ["P1", "P2"], chrom, positions,
        np.stack([PL_BY_GENOTYPE[gt1], PL_BY_GENOTYPE[gt2]], axis=1),
    )
    write_h5(input, ["F10"], chrom, positions, ad=AD_BY_GENOTYPE[gt_off][:, None, :])

    return Cross(
        input=input,
        reference=reference,
        offspring="F10",
        parent1="P1",
        parent2="P2",
        chrom=CHROM,
        chr_length=int(positions[-1]) + 10 * SPACING,
        positions=positions,
        genotypes={"parent1": gt1, "parent2": gt2, "offspring": gt_off},
        ancestry=ancestry,
        informative=gt1 != gt2,
    )


@pytest.fixture
def simulated_ril(tmp_path):
    """
    Offspring is parent 1 for the first half of Chr1 and parent 2 after.

    400 markers 1 kb apart; parents are hom-REF and hom-ALT at every site.
    """
    n = 400
    ancestry = np.repeat([0, 2], n // 2)
    return make_cross(tmp_path, np.zeros(n, int), np.full(n, 2), ancestry)


@pytest.fixture
def simulated_ril_het(tmp_path):
    """
    Offspring is AA, then AB, then BB, each over a third of Chr1.

    600 markers 1 kb apart; parents are hom-REF and hom-ALT at every site.
    """
    n = 600
    ancestry = np.repeat([0, 1, 2], n // 3)
    return make_cross(tmp_path, np.zeros(n, int), np.full(n, 2), ancestry)


@pytest.fixture
def simulated_ril_mixed_informativeness(tmp_path):
    """
    Parents differ, then agree, then alternate, each over a third of Chr1.

    600 markers 1 kb apart. In the first third the parents carry opposing
    homozygous genotypes, in the second third identical ones, and in the final
    third a repeating cycle of opposing and identical sites. Both orientations
    (P1 REF / P2 ALT and the reverse) and both shared genotypes are used.
    The offspring is parent 1 for the first half and parent 2 for the second,
    so the true breakpoint lies in the uninformative middle third.
    """
    n = 600
    third = n // 3
    # Cycles of (parent1, parent2) genotypes for each third
    opposing = [(0, 2), (2, 0)]
    identical = [(0, 0), (2, 2)]
    mixed = [(0, 2), (0, 0), (2, 0), (2, 2)]
    pairs = np.array(
        [opposing[i % 2] for i in range(third)]
        + [identical[i % 2] for i in range(third)]
        + [mixed[i % 4] for i in range(third)]
    )
    ancestry = np.repeat([0, 2], n // 2)
    return make_cross(tmp_path, pairs[:, 0], pairs[:, 1], ancestry)
