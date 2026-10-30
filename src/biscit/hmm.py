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
    Compute per-locus emission probabilities for the three hidden states.

    Combines the genotype likelihoods of the offspring and both parents to
    give the probability of the observed data under each ancestry state.

    Parameters
    ----------
    pl_array : dict of str to numpy.ndarray
        PL arrays for a single chromosome, with keys ``"offspring"``,
        ``"parent1"`` and ``"parent2"``, each of shape ``(n_loci, 3)``, as
        returned per chromosome by `biscit.io.extract_aligned_pl`. The
        ``"offspring"`` array is float, computed from allelic depths by
        `biscit.io.ad_to_pl`.
    n_generations_selfing : int
        Number of generations of selfing. Used to compute the expected
        residual heterozygosity, ``0.5 ** n_generations_selfing``.

    Returns
    -------
    numpy.ndarray
        Emission probabilities, shape ``(n_loci, 3)``. Columns correspond to
        0 = AA (homozygous for parent 1), 1 = AB (heterozygous) and
        2 = BB (homozygous for parent 2). Rows are not normalized.
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

    # Stack into shape (n_loci, 3)
    emission_probs = np.array([pr_hom1, pr_het, pr_hom2]).T
    return emission_probs


def transition_matrix(pos1, pos2, cM_per_Mb=4.0):
    """
    Compute the transition matrix between two adjacent markers.

    The physical distance is converted to a genetic distance in Morgans
    assuming a constant recombination rate, and this is used directly as the
    probability of leaving the current state. That probability is split
    equally between the two other states, so all states, including the
    heterozygous one, are treated alike.

    Parameters
    ----------
    pos1, pos2 : int or float
        Positions of the two markers in base pairs.
    cM_per_Mb : float, optional
        Recombination rate in centimorgans per megabase. Default is 4.0.

    Returns
    -------
    numpy.ndarray
        Transition matrix of shape ``(3, 3)``, where entry ``[i, j]`` is the
        probability of moving from state ``i`` to state ``j``. States are
        ordered 0 = AA (parent 1), 1 = AB (heterozygous), 2 = BB (parent 2).

    Notes
    -----
    The genetic distance is not passed through a map function, so the matrix
    is only a valid probability matrix while it is at most 1 Morgan (25 Mb at
    the default rate). Beyond that the diagonal becomes negative.
    """
    # Genetic distance in Morgans, used directly as the switching probability
    rate = cM_per_Mb * abs(pos2 - pos1) / 1e6 / 100

    # Stay with probability 1 - rate; otherwise move to either other state
    T = np.full((3, 3), rate / 2)
    np.fill_diagonal(T, 1 - rate)
    return T


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
        Emission probabilities, shape ``(n_loci, 3)``, as returned by
        `get_emission_probs`. Each row is normalized to sum to one; rows that
        sum to zero are replaced by a uniform distribution.
    N : int
        Minimum number of consecutive markers in a state before a transition
        to a different state is allowed.
    n_self : int
        Number of generations of selfing. Currently unused, since
        `transition_matrix` no longer depends on it; kept so that existing
        callers still work.
    initial : array_like of float, optional
        Initial state probabilities, length 3. Normalized internally. If
        None, all states are equally likely.

    Returns
    -------
    numpy.ndarray of int
        Most likely state at each marker, shape ``(n_loci,)``, coded as
        0 = AA (parent 1), 1 = AB (heterozygous), 2 = BB (parent 2).

    Notes
    -----
    Transition probabilities use the default recombination rate of
    `transition_matrix`. Run time scales as ``O(n_loci * N)``.
    """
    # Normalize emissions per locus; loci with no support become uniform
    n, n_states = emissions.shape
    totals = emissions.sum(axis=1, keepdims=True)
    emissions = np.divide(
        emissions, totals, out=np.full_like(emissions, 1 / n_states), where=totals > 0
    )

    # Work in log space, clipping zeros to avoid log(0)
    tiny = np.finfo(float).tiny

    def logp(p):
        return np.log(np.clip(p, tiny, 1.0))

    # Initial state distribution, uniform by default
    pi = np.ones(n_states) if initial is None else np.asarray(initial, dtype=float)
    pi = pi / pi.sum()

    # Augmented state space: each hidden state s is paired with a counter
    # c in 1..N giving how many consecutive markers have been spent in s.
    # The counter stops at N, which means "free to switch".
    n_aug = n_states * N

    def aug(s, c):
        # Index of (state s, counter c) in the augmented state vector
        return s * N + (c - 1)

    # Start every state with a full counter so the first segment is
    # unconstrained
    score = np.full(n_aug, -np.inf)
    for s in range(n_states):
        score[aug(s, N)] = logp(pi[s]) + logp(emissions[0, s])

    # Backpointers: best previous augmented state for each marker and state
    back = np.zeros((n, n_aug), dtype=int)

    for t in range(1, n):
        # Transition probabilities depend on the distance between markers
        T = np.asarray(transition_matrix(pos[t - 1], pos[t]), dtype=float)
        logT = logp(T)
        new_score = np.full(n_aug, -np.inf)
        new_back = np.zeros(n_aug, dtype=int)

        for s_prev in range(n_states):
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
                    for s_new in range(n_states):
                        if s_new == s_prev:
                            continue
                        idx_new = aug(s_new, 1)
                        cand = sp + logT[s_prev, s_new]
                        if cand > new_score[idx_new]:
                            new_score[idx_new] = cand
                            new_back[idx_new] = idx_prev

        # Add the emission for marker t to every reachable augmented state
        for s in range(n_states):
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
