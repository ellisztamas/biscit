import h5py
import numpy as np


def _decode(values):
    return np.array([value.decode("utf-8") for value in values])


def load_vcf_hdf5(path, sample_names):
    """Read coordinates and only the requested sample slices of calldata/PL."""
    sample_names = list(dict.fromkeys(sample_names))
    with h5py.File(path, "r") as h5:
        samples = _decode(h5["samples"][:])
        sample_index = {name: i for i, name in enumerate(samples)}
        missing = [name for name in sample_names if name not in sample_index]
        if missing:
            raise KeyError(f"Samples not found in {path}: {missing}")
        chrom = h5["variants/CHROM"].asstr()[:]
        pos = h5["variants/POS"][:]
        pl_dataset = h5["calldata/PL"]
        pl = {name: pl_dataset[:, sample_index[name], :] for name in sample_names}
    return chrom, pos, pl


def extract_aligned_pl(input, reference, offspring_name, parent1_name, parent2_name):
    """Align offspring and parental PL arrays per chromosome, dropping uninformative loci."""
    test_chrom, test_pos, test_pl = load_vcf_hdf5(input, [offspring_name])
    ref_chrom, ref_pos, ref_pl = load_vcf_hdf5(
        reference, [parent1_name, parent2_name]
    )
    offspring_pl = test_pl[offspring_name]
    parent1_pl = ref_pl[parent1_name]
    parent2_pl = ref_pl[parent2_name]
    results = {}

    for chrom in sorted(set(test_chrom) & set(ref_chrom)):
        test_mask = test_chrom == chrom
        ref_mask = ref_chrom == chrom
        common_pos, test_idx, ref_idx = np.intersect1d(
            test_pos[test_mask], ref_pos[ref_mask], return_indices=True
        )
        if common_pos.size == 0:
            continue

        parent1 = parent1_pl[ref_mask][ref_idx]
        parent2 = parent2_pl[ref_mask][ref_idx]
        offspring = offspring_pl[test_mask][test_idx]

        keep = ~(
            np.all(parent1 == -1, axis=1)
            | np.all(parent2 == -1, axis=1)
            | np.all(offspring == -1, axis=1)
            | np.all(parent1 == 0, axis=1)
            | np.all(parent2 == 0, axis=1)
            | np.all(offspring == 0, axis=1)
            | (parent1[:, 1] == 0)
            | (parent2[:, 1] == 0)
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
