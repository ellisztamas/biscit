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


def load_vcf_hdf5(path, sample_names, field="PL"):
    """
    Read variant coordinates and a per-sample calldata field for selected samples.

    Only the slices for the requested samples are read, so the full
    ``calldata/<field>`` matrix is never loaded into memory.

    Parameters
    ----------
    path : str or path-like
        Path to an HDF5 file in scikit-allel layout, containing ``samples``,
        ``variants/CHROM``, ``variants/POS`` and ``calldata/<field>``.
    sample_names : iterable of str
        Names of samples to load. Duplicates are removed, preserving order.
    field : str, optional
        Name of the calldata field to read, e.g. ``"PL"`` or ``"AD"``.
        Default ``"PL"``.

    Returns
    -------
    chrom : numpy.ndarray of str
        Chromosome name for each variant, shape ``(n_variants,)``.
    pos : numpy.ndarray of int
        Position of each variant, shape ``(n_variants,)``.
    data : dict of str to numpy.ndarray
        Mapping from sample name to its `field` array, shape
        ``(n_variants, n_values)``; ``n_values`` is the number of genotypes
        for PL and the number of alleles for AD.

    Raises
    ------
    KeyError
        If the file has no ``calldata/<field>`` dataset, or if any of
        `sample_names` are not present in the file.
    """
    # Deduplicate while preserving order
    sample_names = list(dict.fromkeys(sample_names))
    with h5py.File(path, "r") as h5:
        if f"calldata/{field}" not in h5:
            raise KeyError(
                f"{path} has no 'calldata/{field}' dataset. Make sure the "
                f"{field} FORMAT field is present in the VCF and is included "
                f"when converting it to HDF5 (e.g. with "
                f"allel.vcf_to_hdf5(..., fields=['*']))."
            )
        samples = _decode(h5["samples"][:])
        sample_index = {name: i for i, name in enumerate(samples)}
        missing = [name for name in sample_names if name not in sample_index]
        if missing:
            raise KeyError(f"Samples not found in {path}: {missing}")
        chrom = h5["variants/CHROM"].asstr()[:]
        pos = h5["variants/POS"][:]
        # Slice one sample at a time so the full matrix is never loaded
        dataset = h5[f"calldata/{field}"]
        data = {name: dataset[:, sample_index[name], :] for name in sample_names}
    return chrom, pos, data


def ad_to_pl(ad, mapping_error=0.05):
    """
    Convert allelic depths to Phred-scaled genotype likelihoods.

    Each read is assumed to show the true allele with probability
    ``1 - mapping_error``. For a locus with ``r`` REF and ``a`` ALT reads:

    - hom-REF: ``(1 - mapping_error) ** r * mapping_error ** a``
    - het: ``0.5 ** r * 0.5 ** a``
    - hom-ALT: ``mapping_error ** r * (1 - mapping_error) ** a``

    Parameters
    ----------
    ad : numpy.ndarray
        Allelic depths, shape ``(n_loci, n_alleles)``. Column 0 is the REF
        count and column 1 the first ALT count; further columns are ignored.
        Negative values mark missing data.
    mapping_error : float, optional
        Probability that a read shows the wrong allele. Must satisfy
        ``0 < mapping_error < 0.5``. Default 0.05.

    Returns
    -------
    numpy.ndarray of float
        PL array, shape ``(n_loci, 3)``, with columns 0 = hom-REF, 1 = het,
        2 = hom-ALT. Each row is scaled so its most likely genotype is 0.
        Rows with missing AD are set to -1, and rows with no reads are all
        0, so `extract_aligned_pl` filters both out as it would for PL.

    Raises
    ------
    ValueError
        If `mapping_error` is not strictly between 0 and 0.5.
    """
    if not 0 < mapping_error < 0.5:
        raise ValueError(f"mapping_error must be in (0, 0.5), got {mapping_error}")
    ref = ad[:, 0].astype(float)
    alt = ad[:, 1].astype(float)
    # Work in log10 space; at high depth the raw products underflow to 0
    log_correct = np.log10(1 - mapping_error)
    log_error = np.log10(mapping_error)
    loglik = np.stack(
        [
            ref * log_correct + alt * log_error,
            (ref + alt) * np.log10(0.5),
            ref * log_error + alt * log_correct,
        ],
        axis=1,
    )
    # Rescale so the best genotype is 0, matching the convention for PL
    pl = -10 * (loglik - loglik.max(axis=1, keepdims=True))
    # Mark missing depths so the negative-PL filter drops them
    pl[np.any(ad[:, :2] < 0, axis=1)] = -1
    return pl


def extract_aligned_pl(
    input, reference, offspring_name, parent1_name, parent2_name,
    mapping_error=0.05, allow_identical_sites=False,
):
    """
    Align offspring and parental PL arrays per chromosome.

    Parental PLs are read directly from `reference`. Offspring PLs are
    computed from allelic depths (``calldata/AD``) in `input` using
    `ad_to_pl`. Loci are matched on chromosome and position only (alleles
    are not compared). A locus is dropped if, for any of the three
    individuals, any PL value is negative (missing, -1) or all are zero
    (for the offspring, no reads), or if either parent has ``PL[1] == 0``
    (i.e. the heterozygote is the most likely genotype). Unless
    `allow_identical_sites` is True, a locus is also dropped if both parents
    are most likely the same homozygote (both ``PL[0] == 0`` or both
    ``PL[2] == 0``).

    Parameters
    ----------
    input : str or path-like
        HDF5 file containing the offspring sample, with a ``calldata/AD``
        dataset.
    reference : str or path-like
        HDF5 file containing both parental samples, with a ``calldata/PL``
        dataset.
    offspring_name : str
        Name of the offspring sample in `input`.
    parent1_name : str
        Name of the first parent in `reference`.
    parent2_name : str
        Name of the second parent in `reference`.
    mapping_error : float, optional
        Per-read error rate passed to `ad_to_pl`. Default 0.05.
    allow_identical_sites : bool, optional
        If True, keep loci where both parents are most likely the same
        homozygote. These cannot distinguish the parents. Default is False.

    Returns
    -------
    dict of str to dict
        Mapping from chromosome name to a dict with keys:

        ``"positions"`` : numpy.ndarray of int
            Retained positions, shape ``(n_loci,)``.
        ``"parent1"``, ``"parent2"`` : numpy.ndarray
            Row-aligned parental PL arrays, shape ``(n_loci, n_genotypes)``.
        ``"offspring"`` : numpy.ndarray of float
            Row-aligned offspring PL array derived from AD, shape
            ``(n_loci, 3)``.

        Chromosomes with no shared or retained loci are omitted.

    Raises
    ------
    KeyError
        If `input` has no ``calldata/AD`` dataset, `reference` has no
        ``calldata/PL`` dataset, or any named sample is not found in its
        file.
    ValueError
        If `mapping_error` is not strictly between 0 and 0.5.
    """
    test_chrom, test_pos, test_ad = load_vcf_hdf5(input, [offspring_name], field="AD")
    ref_chrom, ref_pos, ref_pl = load_vcf_hdf5(
        reference, [parent1_name, parent2_name]
    )
    offspring_pl = ad_to_pl(test_ad[offspring_name], mapping_error)
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
