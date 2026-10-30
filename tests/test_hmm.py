import itertools

import numpy as np
import pytest

from biscit.hmm import get_emission_probs, pl_to_gl, transition_matrix, viterbi_min_duration
from conftest import PL_BY_GENOTYPE


# pl_to_gl

def test_pl_to_gl_flat_pl_is_uniform():
    np.testing.assert_allclose(pl_to_gl(np.array([0, 0, 0])), [1 / 3] * 3)


def test_pl_to_gl_ratios_follow_phred_scale():
    """Each 10 PL units is a tenfold drop in likelihood."""
    gl = pl_to_gl(np.array([0, 10, 20]))
    np.testing.assert_allclose(gl / gl[0], [1, 0.1, 0.01])


def test_pl_to_gl_rows_normalized():
    pl = np.random.default_rng(0).integers(0, 100, size=(50, 3))
    gl = pl_to_gl(pl)
    assert gl.shape == pl.shape
    np.testing.assert_allclose(gl.sum(axis=1), 1)


# get_emission_probs

def emissions_for(p1, p2, off, n_generations_selfing=10, pl_table=PL_BY_GENOTYPE):
    """
    Emission probabilities for loci given as genotype codes.

    Parameters
    ----------
    p1, p2, off : array_like of int
        Genotype codes of parent 1, parent 2 and the offspring, shape
        ``(n_loci,)``, coded 0 = hom-REF, 1 = het, 2 = hom-ALT.
    n_generations_selfing : int, optional
        Passed to `get_emission_probs`.
    pl_table : numpy.ndarray, optional
        PL row for each genotype code, shape ``(3, 3)``.

    Returns
    -------
    numpy.ndarray
        Emission probabilities, shape ``(n_loci, 3)``.
    """
    pl = {
        "parent1": pl_table[np.atleast_1d(p1)],
        "parent2": pl_table[np.atleast_1d(p2)],
        "offspring": pl_table[np.atleast_1d(off)],
    }
    return get_emission_probs(pl, n_generations_selfing)


@pytest.mark.parametrize(
    "p1, p2, off, expected",
    [
        (0, 2, 0, 0),
        (0, 2, 2, 2),
        (0, 2, 1, 1),
        (2, 0, 2, 0),
        (2, 0, 0, 2),
        (2, 0, 1, 1),
    ],
)
def test_emission_argmax(p1, p2, off, expected):
    """The most likely state matches the genotype configuration."""
    assert emissions_for(p1, p2, off).argmax(axis=1)[0] == expected


def test_emission_shape():
    rng = np.random.default_rng(0)
    em = emissions_for(rng.choice([0, 2], 25), rng.choice([0, 2], 25), rng.integers(0, 3, 25))
    assert em.shape == (25, 3)


@pytest.mark.parametrize("shared", [0, 2])
@pytest.mark.parametrize("n_gen", [1, 3, 10])
def test_monomorphic_site_split_by_expected_frequencies(shared, n_gen):
    """
    Where both parents and offspring share a genotype, AA and BB are equal and
    AA, AB and BB split the weight by expected genotype frequency.
    """
    # Near-certain calls, so genotype likelihoods are effectively 0 or 1
    exact = np.array([[0, 300, 300], [300, 0, 300], [300, 300, 0]])
    em = emissions_for(shared, shared, shared, n_gen, pl_table=exact)[0]
    h = 0.5 ** n_gen
    np.testing.assert_allclose(em, [(1 - h) / 2, h, (1 - h) / 2], atol=1e-12)


# transition_matrix

@pytest.mark.parametrize("distance", [0, 1_000, 1_000_000, 25_000_000])
def test_transition_rows_are_distributions(distance):
    T = transition_matrix(0, distance)
    assert T.shape == (3, 3)
    assert (T >= 0).all()
    np.testing.assert_allclose(T.sum(axis=1), 1)


@pytest.mark.xfail(
    strict=True,
    reason="No map function: the diagonal 1 - rate is negative beyond 1 Morgan",
)
def test_transition_non_negative_beyond_one_morgan():
    assert (transition_matrix(0, 30_000_000) >= 0).all()


def test_transition_no_recombination_at_zero_distance():
    np.testing.assert_array_equal(transition_matrix(500, 500), np.eye(3))


def test_transition_order_of_positions_irrelevant():
    np.testing.assert_array_equal(transition_matrix(100, 90_000), transition_matrix(90_000, 100))


def test_transition_rate_is_linear_in_distance():
    """1 Mb at 4 cM/Mb is 0.04 Morgans, split equally between the other two states."""
    T = transition_matrix(0, 1_000_000)
    assert T[0, 0] == pytest.approx(0.96)
    assert T[0, 2] == pytest.approx(0.02)
    assert transition_matrix(0, 2_000_000)[0, 2] == pytest.approx(0.04)
    assert transition_matrix(0, 1_000_000, cM_per_Mb=8.0)[0, 2] == pytest.approx(0.04)


def test_transition_all_states_treated_alike():
    T = transition_matrix(0, 2_000_000)
    assert len(set(np.diag(T))) == 1
    assert len(set(T[~np.eye(3, dtype=bool)])) == 1


# viterbi_min_duration

def strong_emissions(states, p=0.997):
    """
    Emissions that strongly favour the given state at each marker.

    Parameters
    ----------
    states : array_like of int
        Favoured state at each marker, shape ``(n_loci,)``.
    p : float, optional
        Probability of the favoured state; the rest is split evenly.

    Returns
    -------
    numpy.ndarray
        Emission probabilities, shape ``(n_loci, 3)``.
    """
    states = np.asarray(states)
    em = np.full((len(states), 3), (1 - p) / 2)
    em[np.arange(len(states)), states] = p
    return em


def marker_positions(n, spacing=1000):
    """Evenly spaced marker positions, shape ``(n,)``."""
    return spacing * np.arange(1, n + 1)


def test_viterbi_recovers_clean_breakpoint():
    truth = np.repeat([0, 2], 30)
    decoded = viterbi_min_duration(marker_positions(60), strong_emissions(truth), N=10, n_self=10)
    np.testing.assert_array_equal(decoded, truth)


def test_viterbi_smooths_internal_burst_shorter_than_N():
    """A short BB burst inside AA survives with N=1 but not with N=10."""
    observed = np.zeros(60, dtype=int)
    observed[30:35] = 2
    pos = marker_positions(60)
    em = strong_emissions(observed)
    np.testing.assert_array_equal(viterbi_min_duration(pos, em, N=1, n_self=10), observed)
    np.testing.assert_array_equal(viterbi_min_duration(pos, em, N=10, n_self=10), np.zeros(60))


@pytest.mark.parametrize("burst", [slice(0, 5), slice(55, 60)], ids=["first", "last"])
def test_viterbi_allows_short_first_and_last_segments(burst):
    """The minimum duration does not apply to the segments at either end."""
    observed = np.zeros(60, dtype=int)
    observed[burst] = 2
    decoded = viterbi_min_duration(marker_positions(60), strong_emissions(observed), N=10, n_self=10)
    np.testing.assert_array_equal(decoded, observed)


def test_viterbi_zero_emission_rows_are_uninformative():
    em = strong_emissions(np.zeros(20, dtype=int))
    em[[5, 12]] = 0
    decoded = viterbi_min_duration(marker_positions(20), em, N=5, n_self=10)
    np.testing.assert_array_equal(decoded, np.zeros(20))


def test_viterbi_single_locus():
    em = np.array([[0.1, 0.2, 0.6, 0.1]])
    assert viterbi_min_duration([1000], em, N=10, n_self=10).tolist() == [2]


def _logp(p):
    """Log of `p`, clipped as in `viterbi_min_duration` to avoid log(0)."""
    return np.log(np.clip(p, np.finfo(float).tiny, 1.0))


def run_lengths(path):
    """
    Lengths of runs of identical consecutive states.

    Parameters
    ----------
    path : sequence of int
        Hidden state at each marker.

    Returns
    -------
    list of int
        Length of each run, in order.
    """
    return [len(list(group)) for _, group in itertools.groupby(path)]


def is_feasible(path, N):
    """
    Check the minimum-duration constraint of `viterbi_min_duration`.

    Every segment except the first and last must span at least `N` markers.

    Parameters
    ----------
    path : sequence of int
        Hidden state at each marker.
    N : int
        Minimum number of markers per internal segment.

    Returns
    -------
    bool
        True if `path` satisfies the constraint.
    """
    return all(length >= N for length in run_lengths(path)[1:-1])


def path_log_prob(path, pos, emissions, initial=None):
    """
    Joint log probability of a state path and the observations.

    Uses the same emission normalization, initial distribution and transition
    matrices as `viterbi_min_duration`.

    Parameters
    ----------
    path : sequence of int
        Hidden state at each marker, shape ``(n_loci,)``.
    pos : numpy.ndarray of int
        Marker positions, shape ``(n_loci,)``.
    emissions : numpy.ndarray
        Unnormalized emission probabilities, shape ``(n_loci, 3)``.
    initial : array_like of float, optional
        Initial state probabilities, length 3. Uniform if None.

    Returns
    -------
    float
        Log probability of `path`.
    """
    totals = emissions.sum(axis=1, keepdims=True)
    log_e = _logp(np.divide(
        emissions, totals, out=np.full_like(emissions, 1 / 3), where=totals > 0
    ))
    pi = np.ones(3) if initial is None else np.asarray(initial, dtype=float)
    pi = pi / pi.sum()

    score = _logp(pi[path[0]]) + log_e[0, path[0]]
    for t in range(1, len(path)):
        log_t = _logp(transition_matrix(pos[t - 1], pos[t]))
        score += log_t[path[t - 1], path[t]] + log_e[t, path[t]]
    return score


@pytest.mark.parametrize("initial", [None, [0.7, 0.1, 0.2]])
@pytest.mark.parametrize("N", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_viterbi_matches_brute_force(seed, N, initial):
    """Viterbi finds the best of all 3**n paths that respect the minimum duration."""
    rng = np.random.default_rng(seed)
    n_loci = 6
    n_self = 3
    # Multi-megabase gaps make transitions between all states plausible, and sharply peaked emissions favour
    # frequent switches, so the duration constraint binds for N > 1
    pos = np.cumsum(rng.integers(100_000, 5_000_000, size=n_loci))
    emissions = rng.dirichlet(np.full(3, 0.1), size=n_loci)

    feasible_scores = [
        path_log_prob(path, pos, emissions, initial)
        for path in itertools.product(range(3), repeat=n_loci)
        if is_feasible(path, N)
    ]

    decoded = viterbi_min_duration(pos, emissions, N=N, n_self=n_self, initial=initial)

    assert decoded.shape == (n_loci,)
    assert is_feasible(decoded, N)
    # Compare scores rather than paths, since ties may be broken differently
    assert path_log_prob(decoded, pos, emissions, initial) == pytest.approx(
        max(feasible_scores)
    )
