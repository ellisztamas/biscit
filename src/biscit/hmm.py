import numpy as np


def pl_to_gl(pl):
    """Convert Phred-scaled likelihoods (..., 3) to normalized genotype likelihoods."""
    l = 10 ** (-pl / 10.0)
    return l / l.sum(axis=-1, keepdims=True)


def get_emission_probs(pl_array, n_generations_selfing):
    gl_f10 = pl_to_gl(pl_array["offspring"])
    gl_p1 = pl_to_gl(pl_array["parent1"])
    gl_p2 = pl_to_gl(pl_array["parent2"])

    pr_het_base = 0.5 ** n_generations_selfing

    pr_hom1 = (
        gl_p1[:, 0] * gl_p2[:, 2] * gl_f10[:, 0]
        + gl_p1[:, 2] * gl_p2[:, 0] * gl_f10[:, 2]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 0] * (1 - pr_het_base) / 2
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 2] * (1 - pr_het_base) / 2
    )

    pr_het = (
        gl_p1[:, 0] * gl_p2[:, 2] * gl_f10[:, 1]
        + gl_p1[:, 2] * gl_p2[:, 0] * gl_f10[:, 1]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 0] * pr_het_base
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 2] * pr_het_base
    )

    pr_hom2 = (
        gl_p1[:, 0] * gl_p2[:, 2] * gl_f10[:, 2]
        + gl_p1[:, 2] * gl_p2[:, 0] * gl_f10[:, 0]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 0] * (1 - pr_het_base) / 2
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 2] * (1 - pr_het_base) / 2
    )

    pr_private = (
        gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 2]
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 0]
        + gl_p1[:, 0] * gl_p2[:, 0] * gl_f10[:, 1]
        + gl_p1[:, 2] * gl_p2[:, 2] * gl_f10[:, 1]
    )

    emission_probs = np.array([pr_hom1, pr_het, pr_hom2, pr_private]).T
    return emission_probs


def transition_matrix(pos1, pos2, cM_per_Mb=4.0, n_self=10, eps_novel=1e-12):
    """4x4 transition matrix (AA, AB, BB, Novel) between two marker positions (bp)."""
    d_M = cM_per_Mb * abs(pos2 - pos1) / 1e6 / 100
    r = 0.5 * (1 - np.exp(-2 * d_M))
    r_ril = 2 * r / (1 + 2 * r)
    h = 0.5 ** (n_self - 1)

    T = np.array([
        [1 - r_ril - h, h,     r_ril],
        [h / 2,         1 - h, h / 2],
        [r_ril,         h,     1 - r_ril - h],
    ])

    T4 = np.zeros((4, 4))
    T4[:3, :3] = T * (1 - eps_novel)
    T4[:3, 3] = eps_novel
    T4[3, :3] = eps_novel / 3
    T4[3, 3] = 1 - eps_novel
    return T4


def viterbi(pos, emissions, initial=None):
    totals = emissions.sum(axis=1, keepdims=True)
    emissions = np.divide(
        emissions, totals, out=np.full_like(emissions, 0.25), where=totals > 0
    )

    pi = np.ones(4) if initial is None else np.asarray(initial, dtype=float)
    if (pi.shape != (4,) or not np.isfinite(pi).all()
            or (pi < 0).any() or pi.sum() <= 0):
        raise ValueError("initial needs four nonnegative weights with positive sum.")
    pi = pi / pi.sum()

    tiny = np.finfo(float).tiny

    def logp(p):
        return np.log(np.clip(p, tiny, 1.0))

    loge = logp(emissions)
    score = logp(pi) + loge[0]
    back = np.zeros((len(emissions), 4), dtype=int)

    for t in range(1, len(emissions)):
        T = np.asarray(transition_matrix(pos[t - 1], pos[t]), dtype=float)
        candidates = score[:, None] + logp(T)
        back[t] = candidates.argmax(axis=0)
        score = candidates[back[t], np.arange(4)] + loge[t]

    path = np.empty(len(emissions), dtype=int)
    path[-1] = score.argmax()
    for t in range(len(emissions) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    return path


def viterbi_min_duration(pos, emissions, N, n_self, initial=None):
    totals = emissions.sum(axis=1, keepdims=True)
    emissions = np.divide(
        emissions, totals, out=np.full_like(emissions, 0.25), where=totals > 0
    )

    n = len(emissions)
    tiny = np.finfo(float).tiny

    def logp(p):
        return np.log(np.clip(p, tiny, 1.0))

    pi = np.ones(4) if initial is None else np.asarray(initial, dtype=float)
    pi = pi / pi.sum()

    n_aug = 4 * N

    def aug(s, c):
        return s * N + (c - 1)

    score = np.full(n_aug, -np.inf)
    for s in range(4):
        score[aug(s, N)] = logp(pi[s]) + logp(emissions[0, s])

    back = np.zeros((n, n_aug), dtype=int)

    for t in range(1, n):
        T = np.asarray(transition_matrix(pos[t - 1], pos[t], n_self=n_self), dtype=float)
        logT = logp(T)
        new_score = np.full(n_aug, -np.inf)
        new_back = np.zeros(n_aug, dtype=int)

        for s_prev in range(4):
            for c_prev in range(1, N + 1):
                idx_prev = aug(s_prev, c_prev)
                sp = score[idx_prev]
                if sp == -np.inf:
                    continue

                c_new = min(c_prev + 1, N)
                idx_stay = aug(s_prev, c_new)
                cand = sp + logT[s_prev, s_prev]
                if cand > new_score[idx_stay]:
                    new_score[idx_stay] = cand
                    new_back[idx_stay] = idx_prev

                if c_prev == N:
                    for s_new in range(4):
                        if s_new == s_prev:
                            continue
                        idx_new = aug(s_new, 1)
                        cand = sp + logT[s_prev, s_new]
                        if cand > new_score[idx_new]:
                            new_score[idx_new] = cand
                            new_back[idx_new] = idx_prev

        for s in range(4):
            e = logp(emissions[t, s])
            for c in range(1, N + 1):
                idx = aug(s, c)
                if new_score[idx] != -np.inf:
                    new_score[idx] += e

        score = new_score
        back[t] = new_back

    aug_path = np.empty(n, dtype=int)
    aug_path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        aug_path[t - 1] = back[t, aug_path[t]]

    path = aug_path // N
    return path
