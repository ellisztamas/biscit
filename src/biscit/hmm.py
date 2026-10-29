import numpy as np


def pl_to_gl(pl):
    """
    Convert Phred-scaled genotype likelihoods to normalized likelihoods.

    Parameters
    ----------
    pl : numpy.ndarray
        Phred-scaled likelihoods (PL), shape ``(..., 3)`` with genotypes
        ordered as hom-REF, het, hom-ALT.

    Returns
    -------
    numpy.ndarray
        Genotype likelihoods with the same shape as `pl`, normalized to sum
        to one along the last axis.
    """
    l = 10 ** (-pl / 10.0)
    return l / l.sum(axis=-1, keepdims=True)


def get_emission_probs(pl_array, n_generations_selfing):
    """
    Compute per-locus emission probabilities for the four hidden states.

    Combines the genotype likelihoods of the offspring and both parents to
    give the probability of the observed data under each ancestry state.

    Parameters
    ----------
    pl_array : dict of str to numpy.ndarray
        PL arrays for a single chromosome, with keys ``"offspring"``,
        ``"parent1"`` and ``"parent2"``, each of shape ``(n_loci, 3)``, as
        returned per chromosome by `biscit.io.extract_aligned_pl`.
    n_generations_selfing : int
        Number of generations of selfing. Used to compute the expected
        residual heterozygosity, ``0.5 ** n_generations_selfing``.

    Returns
    -------
    numpy.ndarray
        Emission probabilities, shape ``(n_loci, 4)``. Columns correspond to
        homozygous for parent 1, heterozygous, homozygous for parent 2, and
        a private allele not explained by either parent. Rows are not
        normalized.
    """
    # Genotype likelihoods; column 0 = hom-REF, 1 = het, 2 = hom-ALT
    gl_f10 = pl_to_gl(pl_array["offspring"])
    gl_p1 = pl_to_gl(pl_array["parent1"])
    gl_p2 = pl_to_gl(pl_array["parent2"])

    # Expected residual heterozygosity after selfing
    pr_het_base = 0.5 ** n_generations_selfing

    # Each state sums over the possible parental genotype combinations. The
    # first two terms are sites where the parents differ, and are informative
    # about ancestry. The last two are sites where the parents share the same
    # homozygous genotype and the offspring matches it; these cannot
    # distinguish states, so their weight is split between AA, AB and BB in
    # proportion to the expected genotype frequencies.

    # Homozygous for parent 1: offspring matches parent 1 at segregating sites
    pr_hom1 = (
        gl_p1[:  , 0] * gl_p2[:, 2] * gl_f10[:, 0]
        + gl_p1[:, 2] * gl_p2[:, 0] * gl_f10[:, 2]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 0]
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 2]
    ) * (1 - pr_het_base) * 0.5

    # Heterozygous: offspring is het at segregating sites
    pr_het = (
        gl_p1[  :, 0] * gl_p2[:, 2] * gl_f10[:, 1]
        + gl_p1[:, 2] * gl_p2[:, 0] * gl_f10[:, 1]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 0]
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 2]
    ) * pr_het_base

    # Homozygous for parent 2: offspring matches parent 2 at segregating sites
    pr_hom2 = (
        gl_p1[  :, 0] * gl_p2[:, 2] * gl_f10[:, 2]
        + gl_p1[:, 2] * gl_p2[:, 0] * gl_f10[:, 0]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 0]
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 2]
    ) * (1 - pr_het_base) * 0.5

    # Private allele: parents agree but the offspring carries the other allele
    # (homozygous or het), e.g. a new mutation or a genotyping error
    pr_private = (
        gl_p1[  :, 0] * gl_p2[:, 0] * gl_f10[:, 2]
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 0]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 1]
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 1]
    )

    # Stack into shape (n_loci, 4)
    emission_probs = np.array([pr_hom1, pr_het, pr_hom2, pr_private]).T
    return emission_probs


def transition_matrix(pos1, pos2, cM_per_Mb=4.0, n_self=10, eps_novel=1e-12):
    """
    Compute the transition matrix between two adjacent markers.

    The physical distance is converted to a recombination fraction using a
    constant recombination rate and Haldane's map function, then to the
    cumulative recombination fraction for a recombinant inbred line,
    ``2r / (1 + 2r)``.

    Parameters
    ----------
    pos1, pos2 : int or float
        Positions of the two markers in base pairs.
    cM_per_Mb : float, optional
        Recombination rate in centimorgans per megabase. Default is 4.0.
    n_self : int, optional
        Number of generations of selfing. Sets the probability of moving to or
        from the heterozygous state, ``0.5 ** (n_self - 1)``. Default is 10.
    eps_novel : float, optional
        Probability of moving into or out of the private-allele state.
        Default is 1e-12.

    Returns
    -------
    numpy.ndarray
        Transition matrix of shape ``(4, 4)``, where entry ``[i, j]`` is the
        probability of moving from state ``i`` to state ``j``. States are
        ordered AA (parent 1), AB (heterozygous), BB (parent 2), Novel.
    """
    # Genetic distance in Morgans, assuming a constant recombination rate
    d_M = cM_per_Mb * abs(pos2 - pos1) / 1e6 / 100
    # Haldane's map function: distance -> recombination fraction per meiosis
    r = 0.5 * (1 - np.exp(-2 * d_M))
    # Cumulative recombination fraction in a RIL derived by selfing
    r_ril = 2 * r / (1 + 2 * r)
    # Probability of moving to or from the heterozygous state
    h = 0.5 ** (n_self - 1)

    # Transitions among AA, AB, BB; rows are the state at pos1
    T = np.array([
        [1 - r_ril - h, h,     r_ril],
        [h / 2,         1 - h, h / 2],
        [r_ril,         h,     1 - r_ril - h],
    ])

    # Add the Novel state, which is rarely entered or left
    T4 = np.zeros((4, 4))
    T4[:3, :3] = T * (1 - eps_novel)
    T4[:3, 3] = eps_novel
    T4[3, :3] = eps_novel / 3
    T4[3, 3] = 1 - eps_novel
    return T4


def viterbi_min_duration(pos, emissions, N, n_self, initial=None):
    """
    Find the most likely state path with a minimum segment length.

    Runs the Viterbi algorithm on an expanded state space that pairs each
    hidden state with a counter of how many consecutive markers have been
    spent in it. A change of state is only allowed once the current state
    has lasted at least `N` markers. The first segment starts with the counter
    full, so it may be shorter than `N`; the last segment is also not
    constrained.

    Parameters
    ----------
    pos : array_like of int
        Marker positions in base pairs, shape ``(n_loci,)``.
    emissions : numpy.ndarray
        Emission probabilities, shape ``(n_loci, 4)``, as returned by
        `get_emission_probs`. Each row is normalized to sum to one; rows that
        sum to zero are replaced by a uniform distribution.
    N : int
        Minimum number of consecutive markers in a state before a transition
        to a different state is allowed.
    n_self : int
        Number of generations of selfing, passed to `transition_matrix`.
    initial : array_like of float, optional
        Initial state probabilities, length 4. Normalized internally. If
        None, all states are equally likely.

    Returns
    -------
    numpy.ndarray of int
        Most likely state at each marker, shape ``(n_loci,)``, coded as
        0 = AA (parent 1), 1 = AB (heterozygous), 2 = BB (parent 2),
        3 = Novel.

    Notes
    -----
    Transition probabilities use the default recombination rate of
    `transition_matrix`. Run time scales as ``O(n_loci * N)``.
    """
    # Normalize emissions per locus; loci with no support become uniform
    totals = emissions.sum(axis=1, keepdims=True)
    emissions = np.divide(
        emissions, totals, out=np.full_like(emissions, 0.25), where=totals > 0
    )

    n = len(emissions)
    # Work in log space, clipping zeros to avoid log(0)
    tiny = np.finfo(float).tiny

    def logp(p):
        return np.log(np.clip(p, tiny, 1.0))

    # Initial state distribution, uniform by default
    pi = np.ones(4) if initial is None else np.asarray(initial, dtype=float)
    pi = pi / pi.sum()

    # Augmented state space: each hidden state s is paired with a counter
    # c in 1..N giving how many consecutive markers have been spent in s.
    # The counter stops at N, which means "free to switch".
    n_aug = 4 * N

    def aug(s, c):
        # Index of (state s, counter c) in the augmented state vector
        return s * N + (c - 1)

    # Start every state with a full counter so the first segment is
    # unconstrained
    score = np.full(n_aug, -np.inf)
    for s in range(4):
        score[aug(s, N)] = logp(pi[s]) + logp(emissions[0, s])

    # Backpointers: best previous augmented state for each marker and state
    back = np.zeros((n, n_aug), dtype=int)

    for t in range(1, n):
        # Transition probabilities depend on the distance between markers
        T = np.asarray(transition_matrix(pos[t - 1], pos[t], n_self=n_self), dtype=float)
        logT = logp(T)
        new_score = np.full(n_aug, -np.inf)
        new_back = np.zeros(n_aug, dtype=int)

        for s_prev in range(4):
            for c_prev in range(1, N + 1):
                idx_prev = aug(s_prev, c_prev)
                sp = score[idx_prev]
                if sp == -np.inf:
                    # Unreachable state
                    continue

                # Stay in the same state and advance the counter
                c_new = min(c_prev + 1, N)
                idx_stay = aug(s_prev, c_new)
                cand = sp + logT[s_prev, s_prev]
                if cand > new_score[idx_stay]:
                    new_score[idx_stay] = cand
                    new_back[idx_stay] = idx_prev

                # Switch to a different state only once the minimum duration
                # is reached; the new segment starts with counter 1
                if c_prev == N:
                    for s_new in range(4):
                        if s_new == s_prev:
                            continue
                        idx_new = aug(s_new, 1)
                        cand = sp + logT[s_prev, s_new]
                        if cand > new_score[idx_new]:
                            new_score[idx_new] = cand
                            new_back[idx_new] = idx_prev

        # Add the emission for marker t to every reachable augmented state
        for s in range(4):
            e = logp(emissions[t, s])
            for c in range(1, N + 1):
                idx = aug(s, c)
                if new_score[idx] != -np.inf:
                    new_score[idx] += e

        score = new_score
        back[t] = new_back

    # Traceback from the best final augmented state
    aug_path = np.empty(n, dtype=int)
    aug_path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        aug_path[t - 1] = back[t, aug_path[t]]

    # Drop the counter to recover the hidden state
    path = aug_path // N
    return path
