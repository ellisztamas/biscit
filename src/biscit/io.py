import h5py
import numpy as np


def _decode(values):
    """
    Decode an array of UTF-8 byte strings.

    Parameters
    ----------
    values : array_like of bytes
        Byte strings, e.g. the ``samples`` dataset of an HDF5 file.

    Returns
    -------
    numpy.ndarray
        Array of decoded Python strings.
    """
    return np.array([value.decode("utf-8") for value in values])


def load_vcf_hdf5(path, sample_names):
    """
    Read variant coordinates and genotype likelihoods for selected samples.

    Only the PL slices for the requested samples are read, so the full
    ``calldata/PL`` matrix is never loaded into memory.

    Parameters
    ----------
    path : str or path-like
        Path to an HDF5 file in scikit-allel layout, containing ``samples``,
        ``variants/CHROM``, ``variants/POS`` and ``calldata/PL``.
    sample_names : iterable of str
        Names of samples to load. Duplicates are removed, preserving order.

    Returns
    -------
    chrom : numpy.ndarray of str
        Chromosome name for each variant, shape ``(n_variants,)``.
    pos : numpy.ndarray of int
        Position of each variant, shape ``(n_variants,)``.
    pl : dict of str to numpy.ndarray
        Mapping from sample name to its PL array, shape
        ``(n_variants, n_genotypes)``.

    Raises
    ------
    KeyError
        If any of `sample_names` are not present in the file.
    """
    # Deduplicate while preserving order
    sample_names = list(dict.fromkeys(sample_names))
    with h5py.File(path, "r") as h5:
        samples = _decode(h5["samples"][:])
        sample_index = {name: i for i, name in enumerate(samples)}
        missing = [name for name in sample_names if name not in sample_index]
        if missing:
            raise KeyError(f"Samples not found in {path}: {missing}")
        chrom = h5["variants/CHROM"].asstr()[:]
        pos = h5["variants/POS"][:]
        # Slice one sample at a time so the full PL matrix is never loaded
        pl_dataset = h5["calldata/PL"]
        pl = {name: pl_dataset[:, sample_index[name], :] for name in sample_names}
    return chrom, pos, pl


def extract_aligned_pl(
    input, reference, offspring_name, parent1_name, parent2_name,
    allow_identical_sites=False,
):
    """
    Align offspring and parental PL arrays per chromosome.

    Loci are matched on chromosome and position only (alleles are not
    compared). A locus is dropped if, for any of the three individuals, any PL
    value is negative (missing, -1) or all are zero, or if either parent has
    ``PL[1] == 0`` (i.e. the heterozygote is the most likely genotype). Unless
    `allow_identical_sites` is True, a locus is also dropped if both parents
    are most likely the same homozygote (both ``PL[0] == 0`` or both
    ``PL[2] == 0``).

    Parameters
    ----------
    input : str or path-like
        HDF5 file containing the offspring sample.
    reference : str or path-like
        HDF5 file containing both parental samples.
    offspring_name : str
        Name of the offspring sample in `input`.
    parent1_name : str
        Name of the first parent in `reference`.
    parent2_name : str
        Name of the second parent in `reference`.
    allow_identical_sites : bool, optional
        If True, keep loci where both parents are most likely the same
        homozygote. These cannot distinguish the parents. Default is False.

    Returns
    -------
    dict of str to dict
        Mapping from chromosome name to a dict with keys:

        ``"positions"`` : numpy.ndarray of int
            Retained positions, shape ``(n_loci,)``.
        ``"parent1"``, ``"parent2"``, ``"offspring"`` : numpy.ndarray
            Row-aligned PL arrays, shape ``(n_loci, n_genotypes)``.

        Chromosomes with no shared or retained loci are omitted.

    Raises
    ------
    KeyError
        If any named sample is not found in its file.
    """
    test_chrom, test_pos, test_pl = load_vcf_hdf5(input, [offspring_name])
    ref_chrom, ref_pos, ref_pl = load_vcf_hdf5(
        reference, [parent1_name, parent2_name]
    )
    offspring_pl = test_pl[offspring_name]
    parent1_pl = ref_pl[parent1_name]
    parent2_pl = ref_pl[parent2_name]
    results = {}

    # Only chromosomes present in both files can be aligned
    for chrom in sorted(set(test_chrom) & set(ref_chrom)):
        test_mask = test_chrom == chrom
        ref_mask = ref_chrom == chrom
        # Match on position only; REF/ALT alleles are not compared
        common_pos, test_idx, ref_idx = np.intersect1d(
            test_pos[test_mask], ref_pos[ref_mask], return_indices=True
        )
        if common_pos.size == 0:
            continue

        # Subset each PL array to the shared positions, in matching row order
        parent1 = parent1_pl[ref_mask][ref_idx]
        parent2 = parent2_pl[ref_mask][ref_idx]
        offspring = offspring_pl[test_mask][test_idx]

        # Drop loci that are wholly or partly missing (any negative PL) or
        # uninformative (all 0) in any individual, or where either parent is
        # most likely heterozygous. Negative PLs would otherwise become
        # likelihoods above 1 in `pl_to_gl`.
        keep = ~(
            np.any(parent1 < 0, axis=1)
            | np.any(parent2 < 0, axis=1)
            | np.any(offspring < 0, axis=1)
            | np.all(parent1 == 0, axis=1)
            | np.all(parent2 == 0, axis=1)
            | np.all(offspring == 0, axis=1)
            | (parent1[:, 1] == 0)
            | (parent2[:, 1] == 0)
        )
        # Parents sharing a homozygous genotype carry no ancestry information
        if not allow_identical_sites:
            keep &= ~(
                ((parent1[:, 0] == 0) & (parent2[:, 0] == 0))
                | ((parent1[:, 2] == 0) & (parent2[:, 2] == 0))
            )
        if not np.any(keep):
            continue

        results[chrom] = {
            "positions": common_pos[keep],
            "parent1": parent1[keep],
            "parent2": parent2[keep],
            "offspring": offspring[keep],
        }

    return results
