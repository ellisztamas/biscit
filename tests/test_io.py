import numpy as np
import pytest

from biscit.io import extract_aligned_pl, load_vcf_hdf5
from conftest import PL_BY_GENOTYPE, write_h5


# Confident hom-REF and hom-ALT calls
REF, ALT = PL_BY_GENOTYPE[0], PL_BY_GENOTYPE[2]


def write_pair(directory, ref_chrom, ref_pos, ref_pl, off_chrom, off_pos, off_pl):
    """
    Write a parental file (samples P1, P2) and an offspring file (sample F10).

    Parameters
    ----------
    directory : pathlib.Path
        Directory to write ``reference.h5`` and ``offspring.h5`` into.
    ref_chrom, ref_pos : array_like
        Chromosome and position of each parental variant.
    ref_pl : array_like of int
        Parental PL, shape ``(n_variants, 2, 3)``.
    off_chrom, off_pos : array_like
        Chromosome and position of each offspring variant.
    off_pl : array_like of int
        Offspring PL, shape ``(n_variants, 3)``.

    Returns
    -------
    tuple of pathlib.Path
        Paths to the offspring and reference files.
    """
    reference = directory / "reference.h5"
    offspring = directory / "offspring.h5"
    write_h5(reference, ["P1", "P2"], ref_chrom, ref_pos, ref_pl)
    write_h5(offspring, ["F10"], off_chrom, off_pos, np.asarray(off_pl)[:, None, :])
    return offspring, reference


def align(offspring, reference):
    """Run `extract_aligned_pl` with the sample names used by `write_pair`."""
    return extract_aligned_pl(offspring, reference, "F10", "P1", "P2")


# load_vcf_hdf5

@pytest.fixture
def three_sample_file(tmp_path):
    """Three samples whose PL values encode sample and variant index."""
    path = tmp_path / "three.h5"
    pl = np.arange(4 * 3 * 3).reshape(4, 3, 3)
    write_h5(path, ["A", "B", "C"], ["Chr1", "Chr1", "Chr2", "Chr2"], [10, 20, 5, 15], pl)
    return path, pl


def test_load_returns_requested_samples(three_sample_file):
    path, pl = three_sample_file
    chrom, pos, loaded = load_vcf_hdf5(path, ["C", "A"])
    assert chrom.tolist() == ["Chr1", "Chr1", "Chr2", "Chr2"]
    assert pos.tolist() == [10, 20, 5, 15]
    assert list(loaded) == ["C", "A"]
    np.testing.assert_array_equal(loaded["A"], pl[:, 0, :])
    np.testing.assert_array_equal(loaded["C"], pl[:, 2, :])


def test_load_removes_duplicate_names(three_sample_file):
    path, _ = three_sample_file
    _, _, loaded = load_vcf_hdf5(path, ["B", "A", "B"])
    assert list(loaded) == ["B", "A"]


def test_load_unknown_sample_raises(three_sample_file):
    path, _ = three_sample_file
    with pytest.raises(KeyError, match="Z"):
        load_vcf_hdf5(path, ["A", "Z"])


# extract_aligned_pl

def test_only_shared_chromosomes_and_positions(tmp_path):
    ref_pl = np.array([[REF, ALT]] * 4)
    files = write_pair(
        tmp_path,
        ["Chr1", "Chr1", "Chr1", "Chr3"], [10, 20, 30, 10], ref_pl,
        ["Chr1", "Chr1", "Chr2", "Chr3"], [20, 30, 10, 99], [REF] * 4,
    )
    result = align(*files)
    # Chr2 is absent from the reference; Chr3 shares no positions
    assert list(result) == ["Chr1"]
    assert result["Chr1"]["positions"].tolist() == [20, 30]
    for key in ("parent1", "parent2", "offspring"):
        assert result["Chr1"][key].shape == (2, 3)


def test_rows_aligned_when_positions_unsorted(tmp_path):
    """Each returned row belongs to the position it is reported at."""
    def pl_for(p, alt=False):
        # Distinct, valid PLs that identify their position
        return [p + 1, p, 0] if alt else [0, p, p + 1]

    ref_pos = [50, 10, 40, 20, 30]
    off_pos = [30, 50, 20, 10, 40]
    # Parents are opposite homozygotes so no locus is dropped as identical
    ref_pl = np.array([[pl_for(p), pl_for(p + 1000, alt=True)] for p in ref_pos])
    off_pl = np.array([pl_for(p + 2000) for p in off_pos])
    files = write_pair(tmp_path, ["Chr1"] * 5, ref_pos, ref_pl, ["Chr1"] * 5, off_pos, off_pl)

    data = align(*files)["Chr1"]
    pos = data["positions"]
    assert pos.tolist() == [10, 20, 30, 40, 50]
    np.testing.assert_array_equal(data["parent1"][:, 1], pos)
    np.testing.assert_array_equal(data["parent2"][:, 1], pos + 1000)
    np.testing.assert_array_equal(data["offspring"][:, 1], pos + 2000)


@pytest.mark.parametrize(
    "p1, p2, off",
    [
        pytest.param([-1, -1, -1], ALT, REF, id="parent1-all-missing"),
        pytest.param(REF, [-1, 60, 0], REF, id="parent2-partly-missing"),
        pytest.param(REF, ALT, [0, -1, 60], id="offspring-partly-missing"),
        pytest.param([0, 0, 0], ALT, REF, id="parent1-all-zero"),
        pytest.param(REF, ALT, [0, 0, 0], id="offspring-all-zero"),
        pytest.param([60, 0, 60], ALT, REF, id="parent1-het"),
        pytest.param(REF, [30, 0, 60], REF, id="parent2-het"),
    ],
)
def test_filters_drop_bad_locus(tmp_path, p1, p2, off):
    """The bad locus at position 20 is dropped; its good neighbours are kept."""
    ref_pl = np.array([[REF, ALT], [p1, p2], [REF, ALT]])
    off_pl = np.array([REF, off, ALT])
    files = write_pair(tmp_path, ["Chr1"] * 3, [10, 20, 30], ref_pl, ["Chr1"] * 3, [10, 20, 30], off_pl)
    assert align(*files)["Chr1"]["positions"].tolist() == [10, 30]


@pytest.mark.parametrize("shared", [REF, ALT], ids=["hom-REF", "hom-ALT"])
@pytest.mark.parametrize("allow", [False, True])
def test_identical_parent_homozygotes(tmp_path, shared, allow):
    """Loci where the parents share a homozygote are dropped unless allowed."""
    ref_pl = np.array([[REF, ALT], [shared, shared], [REF, ALT]])
    off_pl = np.array([REF, shared, ALT])
    files = write_pair(tmp_path, ["Chr1"] * 3, [10, 20, 30], ref_pl, ["Chr1"] * 3, [10, 20, 30], off_pl)
    result = extract_aligned_pl(*files, "F10", "P1", "P2", allow_identical_sites=allow)
    assert result["Chr1"]["positions"].tolist() == ([10, 20, 30] if allow else [10, 30])


def test_het_offspring_is_kept(tmp_path):
    """Only parental heterozygosity is filtered, not offspring heterozygosity."""
    ref_pl = np.array([[REF, ALT]])
    files = write_pair(tmp_path, ["Chr1"], [10], ref_pl, ["Chr1"], [10], [PL_BY_GENOTYPE[1]])
    assert align(*files)["Chr1"]["positions"].tolist() == [10]


def test_fully_filtered_chromosome_is_omitted(tmp_path):
    ref_pl = np.array([[REF, ALT], [[-1, -1, -1], ALT]])
    files = write_pair(
        tmp_path,
        ["Chr1", "Chr2"], [10, 10], ref_pl,
        ["Chr1", "Chr2"], [10, 10], [REF, REF],
    )
    assert list(align(*files)) == ["Chr1"]


def test_returned_pl_are_non_negative(simulated_ril, tmp_path):
    """No negative PL reaches `pl_to_gl`, even when the input contains some."""
    cross = simulated_ril
    rng = np.random.default_rng(0)
    pl = np.stack([PL_BY_GENOTYPE[cross.genotypes["parent1"]], PL_BY_GENOTYPE[cross.genotypes["parent2"]]], axis=1)
    # Knock out random PL entries across both parents
    pl[rng.random(pl.shape) < 0.05] = -1
    reference = tmp_path / "masked_reference.h5"
    write_h5(reference, ["P1", "P2"], [cross.chrom] * len(cross.positions), cross.positions, pl)

    data = extract_aligned_pl(cross.input, reference, cross.offspring, cross.parent1, cross.parent2)[cross.chrom]
    assert 0 < len(data["positions"]) < len(cross.positions)
    for key in ("parent1", "parent2", "offspring"):
        assert (data[key] >= 0).all()
