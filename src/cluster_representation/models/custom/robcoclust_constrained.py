"""
Trimmed, constrained Classification EM (CEM) for robust Gaussian co-clustering
of three-way data.

Model
-----
The data are an array ``x`` of shape ``(n, p, d)``: ``n`` rows, ``p`` columns
and a ``d``-dimensional observation in every cell ``(i, j)``. Rows are split
into ``g`` row clusters (one-hot matrix ``z``, shape ``(n, g)``) and columns
into ``m`` column clusters (one-hot matrix ``w``, shape ``(p, m)``). Given the
partitions, every cell of block ``(k, l)`` is Gaussian::

    x[i, j] | z[i, k] = 1, w[j, l] = 1  ~  N_d(mu[k, l], sigma[k, l])

with row-cluster proportions ``pi`` (``g``,) and column-cluster proportions
``rho`` (``m``,). This is a Gaussian latent block model (LBM) with
multivariate cells.

Criterion
---------
The algorithm maximises the trimmed classification log-likelihood::

    sum_k n_k log pi_k + sum_l p_l log rho_l
      + sum_{i,j,k,l} z_ik * w_jl * mask_ij * log phi_d(x_ij; mu_kl, sigma_kl)

where ``mask_ij = 1`` if cell ``(i, j)`` is used and ``0`` if it is missing
(NaN in the input) or has been trimmed.

Algorithm (one CEM iteration)
-----------------------------
1. Impartial trimming: the ``trim_lvl`` fraction of the *observed* cells with
   the largest squared Mahalanobis distance to their own block is masked out.
2. CE step on rows: each row is hard-assigned to a row cluster, subject to
   ``size_min <= |row cluster| <= size_max``. The constrained assignment is
   solved exactly as a min-cost-flow problem (OR-Tools), following
   k-means-constrained (https://github.com/joshlk/k-means-constrained).
3. M step: update ``pi``, ``mu`` and ``sigma``.
4. CE step on columns: each column is hard-assigned to a column cluster,
   subject to ``col_size_min <= |column cluster| <= col_size_max``, with the
   same min-cost-flow solver.
5. M step: update ``rho``, ``mu`` and ``sigma``.

A run stops when it reaches a fixed point (row partition, column partition
and mask all unchanged), when the relative change of the criterion falls
below ``tol``, after ``iter_max`` iterations, or when it degenerates.

Options for the M step:

* consistency correction of the trimmed covariance matrices;
* an eigenvalue-ratio constraint ``max(lambda) / min(lambda) <= c`` across
  all blocks (Fritz, Garcia-Escudero & Mayo-Iscar, 2012), which keeps the
  likelihood bounded and avoids spurious solutions.

Initial block parameters come from robust estimators applied to random
partitions: the Minimum Covariance Determinant (MCD) for d > 1, the median
and the Qn scale for d = 1, each computed on at most ``init_max_cells``
randomly subsampled cells per block. Several random initialisations are
run; the best non-degenerate one is kept.

Missing values
--------------
Cells with any NaN component are missing. They get ``mask = 0`` for the whole
fit, never count towards the trimming budget, and are zero-filled internally
so that NaNs never enter the arithmetic (their value is irrelevant because
they are always masked).

Shape conventions used throughout
---------------------------------
* ``x``      : (n, p, d)   data
* ``z``      : (n, g)      one-hot row partition
* ``w``      : (p, m)      one-hot column partition
* ``mask0``  : (n, p)      1 = cell observed, 0 = missing
* ``mask``   : (n, p)      1 = cell used, 0 = missing or trimmed
* ``mu``     : (g, m, d)   block means
* ``sigma``  : (g, m, d, d) block covariances
* ``gamma``  : tuple ``(mu, sigma)``
"""

import warnings
import numpy as np
from itertools import product
from typing import Tuple, Union
from scipy.stats import chi2
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import MinCovDet
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow


def tbcem_constrained(
        x,
        g,
        m,
        row_size_min = None,
        row_size_max = None,
        col_size_min = None,
        col_size_max = None,
        trim_lvl = 0.10,
        c = None,
        consistency = True,
        n_init = 10,
        until_converged = True,
        iter_max: int = 50,
        tol: Union[float, None] = 1e-10,
        max_init: Union[int, None] = None,
        max_init_attempts: int = 50,
        init_max_cells: Union[int, None] = 500,
        seed = None,
        verbose: int = False,
):
    """
    Fit a trimmed Gaussian latent block model with size-constrained row
    clusters by Classification EM.

    Parameters
    ----------
    x : array-like, shape (n, p, d) or (n, p)
        Three-way data array. A 2-D array is treated as univariate cells
        (d = 1) and reshaped to (n, p, 1); the outputs then have
        ``mu`` of shape (g, m, 1) and ``sigma`` of shape (g, m, 1, 1).
        Cells with any NaN component are treated as missing.
    g : int
        Number of row clusters.
    m : int
        Number of column clusters.
    row_size_min : int or None, default None
        Minimum number of rows per row cluster. ``None`` means 1 (no empty
        row cluster).
    row_size_max : int or None, default None
        Maximum number of rows per row cluster. ``None`` means ``n``.
    col_size_min : int or None, default None
        Minimum number of columns per column cluster. ``None`` means 1 (no
        empty column cluster). An empty cluster would make the run
        degenerate, so 1 is the least restrictive useful value.
    col_size_max : int or None, default None
        Maximum number of columns per column cluster. ``None`` means ``p``.
    trim_lvl : float in [0, 1), default 0.10
        Fraction of the observed cells trimmed at each iteration (impartial
        trimming on Mahalanobis distance). Missing cells do not count.
    c : float >= 1 or None, default None
        Eigenvalue-ratio constraint for the block covariances. ``None``
        means unconstrained.
    consistency : bool, default True
        Whether to apply the consistency factor to the trimmed covariance
        estimates (see ``consistency_factor``).
    n_init : int, default 10
        Number of random initialisations.
    until_converged : bool, default True
        If True, keep running new initialisations after ``n_init`` until at
        least one run finishes without degeneracy, up to ``max_init`` runs
        in total.
    iter_max : int, default 50
        Maximum number of CEM iterations per initialisation.
    tol : float or None, default 1e-10
        A run also stops when
        ``|score_t - score_{t-1}| <= tol * max(1, |score_{t-1}|)``.
        ``None`` disables this test; the fixed-point test (partitions and
        mask unchanged) always applies.
    max_init : int or None, default None
        Maximum total number of initialisations when ``until_converged`` is
        True. ``None`` means ``10 * n_init``. Values below ``n_init`` are
        raised to ``n_init``.
    max_init_attempts : int, default 1000
        Maximum number of random partitions drawn for one initialisation
        while looking for a non-degenerate starting point.
    init_max_cells : int or None, default 500
        Maximum number of cells per block used for the initial robust
        estimates (MCD for d > 1, median and Qn for d = 1). Larger blocks
        are randomly subsampled. ``None`` uses all cells (slow for large
        blocks).
    seed : int or None, default None
        Seed for ``numpy.random.default_rng`` (random initial partitions).
    verbose : int, default False
        0/False is silent. A positive integer ``k`` prints a progress line
        every ``k`` CEM iterations, plus degeneracy / convergence messages.

    Returns
    -------
    dict
        ``mu`` (g, m, d), ``sigma`` (g, m, d, d), ``pi`` (g,), ``rho`` (m,),
        ``z`` (n, g), ``w`` (p, m), ``mask`` (n, p): final state of the best
        run; ``score`` its classification log-likelihood;
        ``best_init`` index of the best run;
        ``converged`` whether the best run met a stopping criterion before
        ``iter_max``; ``n_iter`` its number of completed iterations;
        ``n_success`` / ``n_fail`` number of non-degenerate / degenerate runs;
        ``runs`` per-run summaries (dicts with ``success``, ``converged``,
        ``n_iter``, ``score``, ``msg``);
        ``evolution`` list of per-run histories (one list per
        initialisation; each element is the tuple
        ``(pi, rho, gamma, z, w, mask, score)`` after a full iteration).

    Raises
    ------
    ValueError
        If the inputs are invalid, if there are too few observed cells for
        ``g * m`` blocks of at least ``d + 1`` cells, if the size constraints
        are infeasible, or if no run completed a single iteration.
    RuntimeError
        If no valid initial partition is found within ``max_init_attempts``.

    Notes
    -----
    The best run is chosen among the non-degenerate runs by its final score.
    If every run degenerated, the run with the highest last recorded
    (non-degenerate) score is returned with a warning.
    """

    rng = np.random.default_rng(seed=seed)

    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        # Univariate cells: (n, p) -> (n, p, 1), i.e. d = 1
        x = x[..., np.newaxis]
    if x.ndim != 3:
        raise ValueError("x must have shape (n, p) or (n, p, d)")
    if not (0 <= trim_lvl < 1):
        raise ValueError("trim_lvl must be in [0, 1)")
    if c is not None and c < 1:
        raise ValueError("c must be >= 1")

    n, p, d = x.shape

    # Observed-cell mask: 1 where all d components are present. Missing cells
    # are zero-filled so that NaNs never enter the arithmetic; they are
    # always masked, so the fill value is irrelevant.
    mask0 = (~np.any(np.isnan(x), axis=2)).astype(int)
    x = np.where(mask0[..., None] == 1, x, 0.0)

    # Feasibility: every block needs d + 1 used cells after trimming
    n_obs = int(mask0.sum())
    n_kept = n_obs - int(np.floor(trim_lvl * n_obs))
    if g * m * (d + 1) > n_kept:
        raise ValueError(
            f"Too few observed cells: {n_kept} cells are kept after trimming, "
            f"but {g}x{m} blocks need at least {d + 1} cells each."
        )

    # Size bounds (rows and columns); None -> no empty cluster, no upper bound
    if row_size_min is None:
        row_size_min = 1
    if row_size_max is None:
        row_size_max = n
    if col_size_min is None:
        col_size_min = 1
    if col_size_max is None:
        col_size_max = p
    _check_size_bounds(n, g, row_size_min, row_size_max, "rows")
    _check_size_bounds(p, m, col_size_min, col_size_max, "columns")
    if max_init is None:
        max_init = 10 * n_init
    max_init = max(max_init, n_init)

    init_num = 0
    any_success = False
    n_success, n_fail = 0, 0
    results_list = []  # one `evolution` list per initialisation
    runs = []          # one summary dict per initialisation

    # Outer loop: independent random initialisations
    while _condition0(init_num, any_success, n_init, until_converged, max_init):
        if verbose != 0: print(f"\n[Init. number: {init_num+1}/{n_init}]")

        # -------------- #
        # INITIALISATION #
        # -------------- #
        # History of this run: (pi, rho, gamma, z, w, mask, score) per iteration
        evolution = []
        # Draw random partitions until (a) every block has at least d + 1
        # observed cells and (b) the MCD block estimates are well conditioned.
        for _ in range(max_init_attempts):
            z, w = _partition_matrices_init(n, p, g, m, 0, 0, rng)

            if not np.all((z.T @ mask0 @ w) >= d + 1):
                continue

            gamma = _rob_init_per_block(x, z, w, mask0, init_max_cells, rng)

            is_deg, msg = _is_degenerate(gamma, z, w, mask0)
            if not is_deg:
                break
        else:
            raise RuntimeError(
                f"No non-degenerate initial partition found in "
                f"{max_init_attempts} attempts; try fewer clusters."
            )

        # Initial proportions from the random partitions
        pi = z.sum(axis=0) / z.sum()
        rho = w.sum(axis=0) / w.sum()

        # -------------- #
        # CEM ITERATIONS #
        # -------------- #
        iter_num = 0
        degenerated, converged = False, False
        msg = ""
        prev = None  # (z, w, mask, score) of the previous iteration

        while iter_num < iter_max:
            if (verbose > 0) and (iter_num % verbose == 0):
                print(f"CEM iter: {iter_num+1}/{iter_max}")

            # Trimming: recompute the mask from mask0 using current parameters
            mask = _impartial_trimming(x, z, w, gamma[0], gamma[1], mask0, trim_lvl)

            # CE step (rows): posteriors, then size-constrained hard assignment
            s = _row_posterior_mn(x, w, pi, gamma, mask)
            z = _constrained_ce_step(s, size_min=row_size_min, size_max=row_size_max)
            degenerated, msg = _is_degenerate(gamma, z, w, mask)
            if degenerated:
                break

            # M step (1/2): update pi and block parameters given new z
            pi = z.sum(axis=0) / z.sum()
            gamma = _block_estimates_mn(x, z, w, mask, c, consistency, mask0)
            degenerated, msg = _is_degenerate(gamma, z, w, mask)
            if degenerated:
                break

            # CE step (columns): posteriors, then size-constrained hard assignment
            t = _col_posterior_mn(x, z, rho, gamma, mask)
            w = _constrained_ce_step(t, size_min=col_size_min, size_max=col_size_max)
            degenerated, msg = _is_degenerate(gamma, z, w, mask)
            if degenerated:
                break

            # M step (2/2): update rho and block parameters given new w
            rho = w.sum(axis=0) / w.sum()
            gamma = _block_estimates_mn(x, z, w, mask, c, consistency, mask0)
            degenerated, msg = _is_degenerate(gamma, z, w, mask)
            if degenerated:
                break

            # Classification log-likelihood of the current state
            score = _log_likelihood(x, z, w, pi, rho, gamma[0], gamma[1], mask)
            evolution.append((pi, rho, gamma, z, w, mask, score))
            iter_num += 1

            # Convergence: fixed point (the next iteration would reproduce
            # the same state) or negligible change of the criterion
            if prev is not None:
                fixed_point = (np.array_equal(z, prev[0])
                               and np.array_equal(w, prev[1])
                               and np.array_equal(mask, prev[2]))
                small_change = (tol is not None and
                                abs(score - prev[3]) <= tol * max(1.0, abs(prev[3])))
                if fixed_point or small_change:
                    converged = True
                    break
            prev = (z, w, mask, score)

        # A run succeeds if it never degenerated and completed >= 1 iteration
        success = (not degenerated) and len(evolution) > 0
        any_success = any_success or success
        n_success += int(success)
        n_fail += int(not success)
        if verbose:
            if degenerated:
                print(f"Degenerate solution ({msg}) after {iter_num} iterations")
            elif converged:
                print(f"Converged after {iter_num} iterations")
        init_num += 1

        results_list.append(evolution)
        runs.append({
            "success": success,
            "converged": converged,
            "n_iter": iter_num,
            "score": evolution[-1][-1] if evolution else None,
            "msg": msg,
        })

    # Keep the best run, preferring non-degenerate ones
    successes = [r["success"] for r in runs]
    best_final_state, _, best_score, best_idx = _get_best_init(results_list, successes)
    pi, rho, gamma, z, w, mask, score = best_final_state
    mu, sigma = gamma

    return {
        "mu": mu,
        "sigma": sigma,
        "pi": pi,
        "rho": rho,
        "z": z,
        "w": w,
        "mask": mask,
        "score": score,
        "best_init": best_idx,
        "converged": runs[best_idx]["converged"],
        "n_iter": runs[best_idx]["n_iter"],
        "n_success": n_success,
        "n_fail": n_fail,
        "runs": runs,
        "evolution": results_list
    }


# ------------- #
# AUX FUNCTIONS #
# ------------- #

def _condition0(init_num, any_success, n_init, until_converged, max_init=None):
    """
    Decide whether to run another initialisation.

    Parameters
    ----------
    init_num : int
        Number of initialisations already run.
    any_success : bool
        Whether at least one run ended without degeneracy.
    n_init : int
        Requested number of initialisations.
    until_converged : bool
        If True, continue past ``n_init`` while no run has succeeded.
    max_init : int or None
        Hard cap on the total number of initialisations when
        ``until_converged`` is True. ``None`` means no cap.

    Returns
    -------
    bool
        True if another initialisation should be run.
    """

    if init_num < n_init:
        return True
    if until_converged and not any_success:
        return max_init is None or init_num < max_init
    return False


def _log_likelihood(x, z, w, pi, rho, mu, sigma, mask = None) -> float:
    """
    Trimmed classification log-likelihood of the Gaussian LBM.

    Computes::

        sum_k n_k log pi_k + sum_l p_l log rho_l
          - 0.5 * d * N_used * log(2 pi)
          - 0.5 * sum_{used cells} log det sigma_{kl}
          - 0.5 * sum_{used cells} Mahalanobis^2(x_ij; mu_kl, sigma_kl)

    where "used cells" are the cells with ``mask_ij = 1``, each taken in its
    own block ``(k, l)``.

    Parameters
    ----------
    x : ndarray, shape (n, p, d)
    z : ndarray, shape (n, g)
    w : ndarray, shape (p, m)
    pi : ndarray, shape (g,)
    rho : ndarray, shape (m,)
    mu : ndarray, shape (g, m, d)
    sigma : ndarray, shape (g, m, d, d)
    mask : ndarray, shape (n, p), optional
        Defaults to all ones.

    Returns
    -------
    float
        Value of the criterion.
    """

    if mask is None:
        mask = np.ones(x.shape[:2])

    # (n, p, g, m): distance of every cell to every block
    mahalanobis2_all = _mahalanobis_distance_squared(x, mu, sigma)

    log_l = np.dot(np.sum(z, axis=0), np.log(pi)) + \
            np.dot(np.sum(w, axis=0), np.log(rho)) - \
            0.5 * np.size(x, 2) * np.einsum("ik,ij,jl->", z, mask, w) * np.log(2 * np.pi) - \
            0.5 * np.einsum("ik,ij,kl,jl->", z, mask, np.log(np.linalg.det(sigma)), w) - \
            0.5 * np.einsum("ik,ij,ijkl,jl->", z, mask, mahalanobis2_all, w)

    return float(log_l)


def _get_best_init(results_list, success=None):
    """
    Select the best initialisation.

    Candidates are the non-degenerate runs (``success[idx]`` True) with a
    non-empty history; among them, the one whose last state has the highest
    score wins. If there is no such run, every run with a non-empty history
    is a candidate, and a warning is issued.

    Parameters
    ----------
    results_list : list of list
        One history per initialisation; each history is a list of tuples
        ``(pi, rho, gamma, z, w, mask, score)``.
    success : list of bool, optional
        Success flag of each run. ``None`` treats every run as successful.

    Returns
    -------
    best_final_state : tuple
        Last state of the best run.
    best_evolution : list
        History of the best run (the same object as in ``results_list``;
        do not modify it).
    best_score : float
        Score of ``best_final_state``.
    best_init_idx : int
        Index of the best run in ``results_list``.

    Raises
    ------
    ValueError
        If every history is empty.
    """

    if success is None:
        success = [True] * len(results_list)

    def _pick(candidates):
        best_score, best_idx = -np.inf, None
        for idx in candidates:
            final_score = results_list[idx][-1][-1]  # last iteration, score is last element
            if final_score > best_score:
                best_score, best_idx = final_score, idx
        return best_idx, best_score

    non_empty = [i for i, ev in enumerate(results_list) if ev]
    if not non_empty:
        raise ValueError("No valid initialisations found in results_list.")

    best_init_idx, best_score = _pick([i for i in non_empty if success[i]])
    if best_init_idx is None:
        warnings.warn(
            "Every initialisation degenerated; returning the best state "
            "recorded before degeneracy. Consider more initialisations, "
            "fewer clusters, a lower trim_lvl or setting c."
        )
        best_init_idx, best_score = _pick(non_empty)

    best_evolution = results_list[best_init_idx]
    best_final_state = best_evolution[-1]

    return best_final_state, best_evolution, best_score, best_init_idx


def _partition_matrices_init(n, p, g, m, a, b, rng):
    """
    Draw random one-hot row and column partitions.

    Each row (column) is assigned to a cluster uniformly at random. A
    fraction ``a`` of rows and ``b`` of columns can be left unassigned
    (all-zero rows of ``z`` / ``w``), e.g. to start with some units trimmed.

    Parameters
    ----------
    n, p : int
        Number of rows and columns.
    g, m : int
        Number of row and column clusters.
    a, b : float in [0, 1]
        Fraction of rows / columns left unassigned. ``tbcem_constrained``
        uses ``a = b = 0``.
    rng : numpy.random.Generator

    Returns
    -------
    z : ndarray, shape (n, g), float
    w : ndarray, shape (p, m), float

    Notes
    -----
    Nothing prevents empty clusters; the caller checks block sizes.
    """
    z = np.zeros((n, g))
    l_row = rng.integers(0, g, n)
    out_row = rng.choice(range(n), size=int(np.ceil(a * n)), replace=False)
    for i, position in enumerate(l_row):
        if i not in out_row:
            z[i, position] = 1

    w = np.zeros((p, m))
    l_col = rng.integers(0, m, p)
    out_col = rng.choice(range(p), size=int(np.ceil(b * p)), replace=False)
    for i, position in enumerate(l_col):
        if i not in out_col:
            w[i, position] = 1

    return z, w


def _get_block_values(
        x: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        k: int,
        l: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the cells of block ``(k, l)``.

    Parameters
    ----------
    x : ndarray, shape (n, p, d) or (n, p)
    z : ndarray, shape (n, g)
    w : ndarray, shape (p, m)
    k, l : int
        Row-cluster and column-cluster index.

    Returns
    -------
    block_vals : ndarray, shape (n_k, p_l, d) or (n_k, p_l)
        Values of the block (possibly with zero-length axes if the block is
        empty).
    ij : ndarray, shape (n_k * p_l, 2)
        (row, column) indices of the block cells, row-major.

    Notes
    -----
    The mask is ignored: missing and trimmed cells are included.
    """

    i = np.flatnonzero(z[:, k])
    j = np.flatnonzero(w[:, l])
    ij = _cartesian([i, j])
    block_vals = x[np.ix_(i, j)]

    return block_vals, ij


def _rob_init_per_block(x, z, w, mask=None, max_cells=500, rng=None):
    """
    Robust initial block parameters.

    For every block ``(k, l)``, the used cells (``mask = 1``) are stacked
    into an ``(N_kl, d)`` matrix. If ``N_kl > max_cells``, a random subsample
    of ``max_cells`` cells is drawn first. Then:

    * ``d > 1``: ``sklearn.covariance.MinCovDet`` (MCD) is fitted;
    * ``d = 1``: location = median, variance = Qn^2 (see ``qn_scale``).

    Subsampling keeps the cost per block bounded: the MCD gets much
    slower as the sample grows, and Qn needs all pairwise
    differences (O(N^2) memory). On a random initial partition a
    subsample is sufficient, since every block mixes several true blocks
    anyway.

    Parameters
    ----------
    x : ndarray, shape (n, p, d)
    z : ndarray, shape (n, g)
    w : ndarray, shape (p, m)
    mask : ndarray, shape (n, p), optional
        Cells to use. Defaults to all ones.
    max_cells : int or None, default 500
        Maximum number of cells per block used for the estimates. ``None``
        uses all cells.
    rng : numpy.random.Generator, optional
        Random generator for the subsampling. Defaults to
        ``np.random.default_rng(0)``.

    Returns
    -------
    locations : ndarray, shape (g, m, d)
        Robust location of each block.
    covariances : ndarray, shape (g, m, d, d)
        Robust covariance of each block (MCD: reweighted and
        consistency-corrected; d = 1: squared Qn).

    Notes
    -----
    The MCD uses ``random_state=0``, so given the partition and the
    subsample it is deterministic.
    """

    g, m = z.shape[1], w.shape[1]
    d = x.shape[2]

    if mask is None:
        mask = np.ones(x.shape[:2])
    if rng is None:
        rng = np.random.default_rng(0)

    locations = np.zeros((g, m , d))
    covariances = np.zeros((g, m , d, d))

    # TODO: parallelize?
    for k, l in product(range(g), range(m)):
        i = np.flatnonzero(z[:, k])
        j = np.flatnonzero(w[:, l])
        # (n_k, p_l, d) -> (n_k * p_l, d), keeping only used cells
        block_vals_2d = x[np.ix_(i, j)].reshape(-1, d)
        block_vals_2d = block_vals_2d[mask[np.ix_(i, j)].ravel() == 1]

        # Random subsample to bound the cost
        if max_cells is not None and block_vals_2d.shape[0] > max_cells:
            idx = rng.choice(block_vals_2d.shape[0], size=max_cells, replace=False)
            block_vals_2d = block_vals_2d[idx]

        if d == 1:
            v = block_vals_2d[:, 0]
            locations[k, l, 0] = np.median(v)
            covariances[k, l, 0, 0] = _qn_scale(v) ** 2
        else:
            mcd = MinCovDet(random_state=0).fit(block_vals_2d)
            locations[k, l] = mcd.location_
            covariances[k, l] = mcd.covariance_

    return locations, covariances


# TODO: check with original paper
def _qn_scale(v):
    """
    Qn robust scale estimator (Rousseeuw & Croux, 1993).

    ``Qn = c * d_n * {|v_i - v_j|; i < j}_(k)``, the ``k``-th smallest pairwise
    absolute difference, with ``k = C(h, 2)`` and ``h = floor(n / 2) + 1``.
    ``c = 2.2219`` makes it consistent for the standard deviation at the
    normal distribution, and ``d_n`` is the finite-sample correction of
    Croux & Rousseeuw (1992). Breakdown point 50%, Gaussian efficiency 82%.

    Parameters
    ----------
    v : array-like, shape (N,)

    Returns
    -------
    float
        Qn scale estimate (0 if ``N < 2``).

    Notes
    -----
    Direct O(N^2) implementation (all pairwise differences). Use it on
    moderate N, e.g. after subsampling as in ``mcd_per_block``.
    """
    v = np.asarray(v, dtype=float).ravel()
    n = v.size
    if n < 2:
        return 0.0

    h = n // 2 + 1
    k = h * (h - 1) // 2
    i, j = np.triu_indices(n, k=1)
    diffs = np.abs(v[i] - v[j])
    q = np.partition(diffs, k - 1)[k - 1]

    # Finite-sample correction factors (Croux & Rousseeuw, 1992)
    if n <= 9:
        dn = [0.399, 0.994, 0.512, 0.844, 0.611, 0.857, 0.669, 0.872][n - 2]
    elif n % 2 == 1:
        dn = n / (n + 1.4)
    else:
        dn = n / (n + 3.8)

    return 2.2219 * dn * q


def _is_degenerate(
        params: Tuple[np.ndarray, np.ndarray],
        z: np.ndarray,
        w: np.ndarray,
        mask: Union[np.ndarray, None] = None
) -> Tuple[bool, str]:
    """
    Check whether the current solution is degenerate.

    A solution is degenerate if

    * some block has fewer than ``d + 1`` used (unmasked) cells
      (``"ClustDim"``), or
    * some parameter is not finite (``"NonFinite"``), or
    * some block covariance is not positive definite (``"NonPD"``), or
    * some block covariance has condition number above ``1e6``
      (``"CondNum"``).

    Parameters
    ----------
    params : tuple (mu, sigma)
        ``mu`` (g, m, d), ``sigma`` (g, m, d, d).
    z : ndarray, shape (n, g)
    w : ndarray, shape (p, m)
    mask : ndarray, shape (n, p), optional
        Defaults to all ones.

    Returns
    -------
    is_deg : bool
    msg : str
        Reason code (``"ClustDim"``, ``"NonFinite"``, ``"NonPD"``,
        ``"CondNum"``) or ``""``.
    """

    n, p, d = z.shape[0], w.shape[0], params[0].shape[2]

    if mask is None:
        mask = np.ones((n, p))

    # Check on clusters' dimensions: (g, m) matrix of used cells per block
    if np.any((z.T @ mask @ w) < d + 1):
        return True, "ClustDim"

    # Check for NaN / inf in the parameters
    if not (np.all(np.isfinite(params[0])) and np.all(np.isfinite(params[1]))):
        return True, "NonFinite"

    # Check positive definiteness (catches zero variances, e.g. Qn = 0 with
    # many ties when d = 1, where the condition number is not informative)
    if np.any(np.linalg.eigvalsh(params[1]) <= 0):
        return True, "NonPD"

    # Check on condition number (batched over the g x m blocks)
    cond = np.linalg.cond(params[1])
    if np.any(cond > 1.e6):
        return True, "CondNum"

    return False, ""


def _row_posterior_mn(
        x: np.ndarray,
        w: np.ndarray,
        pi: np.ndarray,
        gamma: Tuple[np.ndarray, np.ndarray],
        mask: np.ndarray = None
) -> np.ndarray:
    """
    Posterior row-cluster membership given the column partition.

    For row ``i`` and row cluster ``k``::

        log s_ik = log pi_k
                   + sum_j sum_l w_jl * mask_ij * log phi_d(x_ij; mu_kl, sigma_kl)
                   - log normaliser

    i.e. the row's used cells are scored against the blocks of row cluster
    ``k``, each cell in the block of its own column cluster.

    Parameters
    ----------
    x : ndarray, shape (n, p, d)
    w : ndarray, shape (p, m)
        Current column partition.
    pi : ndarray, shape (g,)
    gamma : tuple (mu, sigma)
    mask : ndarray, shape (n, p), optional
        Defaults to all ones.

    Returns
    -------
    s : ndarray, shape (n, g)
        Row-normalised posterior probabilities (computed with logsumexp).

    Notes
    -----
    A small ridge ``1e-12 * I`` is added to every covariance for numerical
    safety; ``1e-12`` is also added inside the logs to avoid log(0).
    """

    const = 1.e-12
    n, p, d = x.shape

    sigma_reg = gamma[1] + const * np.eye(d)
    dist_m = _mahalanobis_distance_squared(x, gamma[0], sigma_reg)  # (n, p, g, m)

    if mask is None:
        mask = np.ones((n, p))

    # Unnormalised log-posterior, shape (n, g):
    #   log prior - 0.5 * sum log det - 0.5 * sum Mahalanobis^2 - 0.5 * d * N_i * log(2 pi)
    # where N_i = number of used cells in row i.
    s = np.repeat(np.log(pi + const)[np.newaxis, :], n, axis=0) - \
        0.5 * np.einsum("jl,ij,kl->ik", w, mask, np.log(np.linalg.det(sigma_reg) + const)) - \
        0.5 * np.einsum("jl,ij,ijkl->ik", w, mask, dist_m) - \
        0.5 * d * np.log(2 * np.pi) * np.expand_dims(np.einsum("jl,ij->i", w, mask), axis=1)

    # Normalise in log space
    log_den = np.expand_dims(logsumexp(s, axis=1), 1)
    s = np.exp(s - log_den)

    return s


def _col_posterior_mn(
        x: np.ndarray,
        z: np.ndarray,
        rho: np.ndarray,
        gamma: Tuple[np.ndarray, np.ndarray],
        mask: np.ndarray = None
) -> np.ndarray:
    """
    Posterior column-cluster membership given the row partition.

    The column counterpart of ``row_posterior_mn``::

        log t_jl = log rho_l
                   + sum_i sum_k z_ik * mask_ij * log phi_d(x_ij; mu_kl, sigma_kl)
                   - log normaliser

    Parameters
    ----------
    x : ndarray, shape (n, p, d)
    z : ndarray, shape (n, g)
        Current row partition.
    rho : ndarray, shape (m,)
    gamma : tuple (mu, sigma)
    mask : ndarray, shape (n, p), optional
        Defaults to all ones.

    Returns
    -------
    t : ndarray, shape (p, m)
        Column-normalised posterior probabilities.

    Notes
    -----
    Same regularisation as in ``row_posterior_mn``.
    """

    const = 1.e-12
    n, p, d = x.shape

    sigma_reg = gamma[1] + const * np.eye(d)
    mal_dist = _mahalanobis_distance_squared(x, gamma[0], sigma_reg)  # (n, p, g, m)

    if mask is None:
        mask = np.ones((n, p))

    # Unnormalised log-posterior, shape (p, m)
    t = np.repeat(np.log(rho + const)[:, np.newaxis], p, axis=1).T - \
        0.5 * np.einsum("ik,ij,kl->jl", z, mask, np.log(np.linalg.det(sigma_reg) + const)) - \
        0.5 * np.einsum("ik,ij,ijkl->jl", z, mask, mal_dist) - \
        0.5 * d * np.log(2 * np.pi) * np.expand_dims(np.einsum("ik,ij->j", z, mask), axis=1)

    log_den = np.expand_dims(logsumexp(t, axis=1), 1)
    t = np.exp(t - log_den)

    return t


def _block_estimates_mn(
        x: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        mask: Union[np.ndarray, None] = None,
        c: Union[float, None] = None,
        consistency: bool = True,
        mask0: Union[np.ndarray, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M step for the block parameters: trimmed means and covariances.

    For every block ``(k, l)``, using only the used cells (``mask = 1``):

    * ``mu_kl``    = sample mean;
    * ``sigma_kl`` = unbiased sample covariance (divisor ``N_kl - 1``);
    * optionally scaled by the consistency factor for trimming;
    * optionally projected onto the eigenvalue-ratio constraint set.

    Parameters
    ----------
    x : ndarray, shape (n, p, d)
        Must be NaN-free (missing cells zero-filled and masked).
    z : ndarray, shape (n, g)
    w : ndarray, shape (p, m)
    mask : ndarray, shape (n, p), optional
        Used cells. Defaults to all ones.
    c : float or None
        Eigenvalue-ratio constraint. ``None`` means no constraint.
    consistency : bool
        Apply ``consistency_factor`` with the per-block retained fraction
        ``(used cells) / (observed cells)``.
    mask0 : ndarray, shape (n, p), optional
        Observed cells (before trimming). Defaults to all ones. Missing
        cells are excluded from the denominator of the retained fraction,
        so they are not treated as trimmed.

    Returns
    -------
    mu : ndarray, shape (g, m, d)
    sigma : ndarray, shape (g, m, d, d)
        Symmetrised covariance matrices.

    Notes
    -----
    Computed block by block, so memory is ``O(n * p * d)`` rather than
    ``O(g * m * n * p * d)``.
    """

    n, p, d = x.shape
    g, m = z.shape[1], w.shape[1]

    if mask is None:
        mask = np.ones((n, p))
    if mask0 is None:
        mask0 = np.ones((n, p))

    # Number of used cells per block, shape (g, m)
    den = np.einsum("ik,ij,jl->kl", z, mask, w)

    mu = np.zeros((g, m, d))
    sigma = np.zeros((g, m, d, d))
    for k, l in product(range(g), range(m)):
        # (n, p) weights: 1 for used cells of block (k, l), 0 elsewhere
        wt = np.outer(z[:, k], w[:, l]) * mask
        mu[k, l] = np.einsum("ij,ijd->d", wt, x) / den[k, l]
        x_c = x - mu[k, l]
        sigma[k, l] = np.einsum("ij,ijd,ijt->dt", wt, x_c, x_c) / (den[k, l] - 1)

    if consistency:
        # Observed cells per block (used + trimmed, excluding missing)
        n_observed = np.einsum("ik,ij,jl->kl", z, mask0, w)
        sigma = sigma * _consistency_factor(den / n_observed, d)[..., None, None]

    if c is not None:
        sigma_batch = sigma.reshape(g * m, d, d)

        # Batched eigen-decomposition
        eig_vals, eig_vects = np.linalg.eigh(sigma_batch)  # shapes: (g*m, d), (g*m, d, d)
        eigenvalues_list = eig_vals.T                      # shape:  (d, g*m)

        # Enforce constraints; blocks weighted by their number of used cells
        eigenvalues_list = _eigen_restriction(eigenvalues_list, den.ravel(), c=c)

        # Reconstruct sigma = V diag(lambda) V^T for each block
        pre_sigma_new = np.array([np.diag(eigenvalues_list[:, i]) for i in range(g * m)])
        sigma_new = eig_vects @ pre_sigma_new @ eig_vects.transpose(0, 2, 1)

        sigma[:] = sigma_new.reshape(g, m, d, d)

    # Enforce symmetry of the covariance matrix
    sigma = 0.5 * (sigma + sigma.swapaxes(-1, -2))

    return mu, sigma


def _mahalanobis_distance_squared(x, mu, sigma):
    """
    Squared Mahalanobis distance of every cell to every block.

    Parameters
    ----------
    x : ndarray, shape (n, p, d)
    mu : ndarray, shape (g, m, d)
    sigma : ndarray, shape (g, m, d, d)

    Returns
    -------
    ndarray, shape (n, p, g, m)
        ``out[i, j, k, l] = (x_ij - mu_kl)^T sigma_kl^{-1} (x_ij - mu_kl)``.

    Notes
    -----
    Computed block by block; peak extra memory is ``O(n * p * d)``.
    """

    n, p, d = x.shape
    g, m = mu.shape[:2]
    sigma_inv = np.linalg.inv(sigma)

    out = np.empty((n, p, g, m))
    for k, l in product(range(g), range(m)):
        x_c = x - mu[k, l]
        out[:, :, k, l] = np.einsum("ijd,dt,ijt->ij", x_c, sigma_inv[k, l], x_c)

    return out


def _impartial_trimming(
        x: np.ndarray,
        z: np.ndarray,
        w: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        mask0: np.ndarray,
        trim_lvl: float = 0
) -> np.ndarray:
    """
    Impartial trimming of cells based on distance to their own block.

    Each observed cell ``(i, j)`` gets the squared Mahalanobis distance to
    the block it currently belongs to, ``(k(i), l(j))``. The
    ``floor(trim_lvl * N_observed)`` observed cells with the largest
    distance get ``mask = 0``. Trimming starts from ``mask0`` at every call,
    so a cell trimmed at one iteration can come back at the next.

    Parameters
    ----------
    x : ndarray, shape (n, p, d)
        NaN-free data.
    z : ndarray, shape (n, g)
    w : ndarray, shape (p, m)
    mu : ndarray, shape (g, m, d)
    sigma : ndarray, shape (g, m, d, d)
    mask0 : ndarray, shape (n, p)
        Observed cells (0 for missing cells, which are never selected and do
        not count towards the budget).
    trim_lvl : float, default 0
        Fraction of observed cells to trim.

    Returns
    -------
    mask : ndarray, shape (n, p)
    """

    mask = mask0.copy()

    observed = mask0.ravel() == 1
    k = int(np.floor(trim_lvl * observed.sum()))  # number of cells to trim
    if k == 0:
        return mask

    # Distance to own block only: z and w are one-hot, so the sum over (k, l)
    # selects the cell's own block. Shape (n, p).
    dist_all = _mahalanobis_distance_squared(x, mu, sigma)
    dist_m_squared = np.einsum("ik,ijkl,jl->ij", z, dist_all, w)

    # Missing cells can never be selected
    flat_dist = dist_m_squared.ravel().astype(float)
    flat_dist[~observed] = -np.inf

    # get indices of k largest elements (no full sort needed)
    trim_idx = np.argpartition(flat_dist, -k)[-k:]

    # convert back to multi-index
    multi_idx = np.unravel_index(trim_idx, mask.shape)

    mask[multi_idx] = 0

    return mask


def _eigen_restriction(eigenvalues, block_sizes, c, tol=1e-8):
    """
    Optimal truncation of eigenvalues under an eigenvalue-ratio constraint.

    Solves, for the eigenvalues ``lambda_{r,K}`` of all block covariances
    (``r`` = eigen-index, ``K`` = block), the problem of finding the
    threshold ``m`` such that the truncated eigenvalues::

        e_{r,K} = min(max(lambda_{r,K}, m), c * m)

    minimise the weighted objective::

        sum_K (n_K / n) * sum_r [ log e_{r,K} + lambda_{r,K} / e_{r,K} ]

    which ensures ``max(e) / min(e) <= c`` across all blocks. The optimal
    ``m`` is searched over the midpoints of the sorted values
    ``{lambda, lambda / c}``, following Fritz, Garcia-Escudero & Mayo-Iscar
    (2012), "A fast algorithm for robust constrained clustering", CSDA.

    Parameters
    ----------
    eigenvalues : ndarray, shape (v, K)
        Eigenvalues; column ``K`` holds the ``v = d`` eigenvalues of block
        ``K``.
    block_sizes : ndarray, shape (K,)
        Block weights (number of used cells per block).
    c : float >= 1
        Maximum allowed ratio between the largest and smallest eigenvalue.
    tol : float, default 1e-8
        If all eigenvalues of non-empty blocks are <= ``tol``, they are
        returned unchanged.

    Returns
    -------
    ndarray, shape (v, K)
        Constrained eigenvalues.

    Notes
    -----
    * If the constraint already holds, eigenvalues are returned unchanged
      (empty blocks are set to the mean eigenvalue).
    * Inside the function ``d`` is the (K, v) eigenvalue matrix, not the
      data dimension.
    """

    if c < 1:
        raise ValueError("The restriction factor c cannot be smaller than 1.")

    d = eigenvalues.T.copy()                   # (k, v); copy to avoid mutating caller's array
    v, k = eigenvalues.shape                   # v = dimension, k = number of blocks
    n = np.sum(block_sizes)
    weights = (block_sizes / n)[:, np.newaxis] # (k, 1), broadcasts over v

    # Candidate thresholds: sorted union of {lambda} and {lambda / c}
    eig_vector = eigenvalues.ravel(order="F")
    d_sort = np.sort(np.concatenate((eig_vector / c, eig_vector)))

    kv = k * v
    dim_sort = kv * 2

    # Midpoints between consecutive sorted values (plus the two ends)
    d1 = np.concatenate((d_sort, [2.0 * d_sort[-1]]))
    d2 = np.concatenate(([0], d_sort))
    ed = (d1 + d2) / 2
    dim_sort += 1

    d_nis = d[block_sizes > 0]                 # (k_nonzero, v)
    max_d_nis = np.max(d_nis)

    if max_d_nis <= tol:
        # All eigenvalues numerically zero: nothing to do
        out = eigenvalues
    else:
        if np.abs(max_d_nis / np.min(d_nis)) <= c:
            # Constraint already satisfied
            if np.any(block_sizes == 0):
                d[block_sizes == 0] = np.mean(d_nis)
            out = d.T
        else:
            # For each candidate interval mp, eigenvalues below ed_mp are set to
            # m and those above c * ed_mp to c * m; the optimal m in closed form
            # is
            #   m = sum_K w_K (s_K + t_K / c) / sum_K w_K r_K
            # with r = count, s = sum of the low eigenvalues and t = sum of the
            # high eigenvalues.
            t = np.zeros((k, dim_sort))
            s = np.zeros((k, dim_sort))
            r = np.zeros((k, dim_sort))
            sol = np.zeros(dim_sort)
            obj = np.zeros(dim_sort)

            for mp in range(dim_sort):
                ed_mp = ed[mp]
                ed_mp_c = ed_mp * c
                r[:, mp] = np.sum(d < ed_mp, axis=1) + np.sum(d > ed_mp_c, axis=1)
                s[:, mp] = np.sum(d * (d < ed_mp), axis=1)
                t[:, mp] = np.sum(d * (d > ed_mp_c), axis=1)
                sol_mp = np.sum(weights * (s[:, mp:mp+1] + t[:, mp:mp+1] / c)) / np.sum(weights * r[:, mp:mp+1])
                sol[mp] = sol_mp
                # Truncated eigenvalues for this candidate m
                e = sol_mp * (d < sol_mp) + d * (d >= sol_mp) * (d <= c * sol_mp) + (c * sol_mp) * (d > c * sol_mp)
                obj[mp] = np.sum(weights * (np.log(e) + d / e))

            # Pick the candidate minimising the objective and truncate
            ind_max = np.argmin(obj)
            m = sol[ind_max]
            out = (m * (d < m) + d * (d >= m) * (d <= c * m) + (c * m) * (d > c * m)).T
    return out


def _match_clusters(z_prev: np.ndarray, z_curr: np.ndarray) -> np.ndarray:
    """
    Match the clusters of two partitions to fix label switching.

    Builds the Jaccard similarity matrix between the clusters of ``z_prev``
    and ``z_curr`` and finds the assignment with the highest total
    similarity (Hungarian algorithm). Public utility, e.g. to align the
    labels of different runs or of an estimate with the true partition.

    Parameters
    ----------
    z_prev, z_curr : ndarray, shape (n, g)
        One-hot partitions with the same number of clusters.

    Returns
    -------
    perm : ndarray, shape (g,)
        ``perm[k]`` is the cluster of ``z_curr`` matched to cluster ``k`` of
        ``z_prev``; ``z_curr[:, perm]`` is aligned with ``z_prev``.
    """

    intersection = z_prev.T @ z_curr                      # (g, g)
    sizes_prev = z_prev.sum(axis=0)[:, None]              # (g, 1)
    sizes_curr = z_curr.sum(axis=0)[None, :]              # (1, g)
    union = np.maximum(sizes_prev + sizes_curr - intersection, 1e-12)
    jaccard = intersection / union

    row_ind, col_ind = linear_sum_assignment(-jaccard)
    # row_ind is sorted [0, 1, ..., g-1] by construction of linear_sum_assignment
    # when the cost matrix is square, so col_ind directly gives the permutation.
    perm = np.empty_like(col_ind)
    perm[row_ind] = col_ind
    return perm


def _consistency_factor(retained_fraction, d):
    """
    Consistency factor for covariance matrices estimated on trimmed data.

    If the fraction ``q`` of points of a ``N_d(mu, Sigma)`` sample closest in
    Mahalanobis distance is kept, the covariance of the retained points is
    ``Sigma * P(chi2_{d+2} <= chi2_{d, q}) / q``. Multiplying by::

        q / P(chi2_{d+2} <= chi2_{d, q})

    makes the estimator consistent for ``Sigma``.

    Parameters
    ----------
    retained_fraction : float or ndarray
        Retained fraction ``q`` in (0, 1]; for ``q = 1`` the factor is 1.
    d : int
        Dimension of the observations.

    Returns
    -------
    float or ndarray
        Factor (>= 1), same shape as ``retained_fraction``.

    Notes
    -----
    Trimming is global, not per block, so applying the formula with the
    per-block retained fraction is an approximation.
    """
    return retained_fraction / chi2.cdf(chi2.ppf(retained_fraction, d), d + 2)



def _cartesian(arrays):
    """Cartesian product of two 1-D index arrays -> shape (len(a)*len(b), 2).

    Ordering is row-major in the first array: (a0, b0), (a0, b1), ...
    ``solve_min_cost_flow_graph`` relies on this ordering.
    """
    a, b = arrays
    return np.array(np.meshgrid(a, b, indexing='ij')).reshape(2, -1).T


def _posterior_to_cost(s, eps=1e-300):
    """
    Convert an (n_samples, n_components) posterior matrix into a cost matrix
    suitable for the min-cost-flow assignment.

    The cost is ``-log(s_ik)``. Minimising the total cost of a hard
    assignment means maximising ``sum_i log s_ik(i)``. Since
    ``log s_ik`` differs from the classification log-likelihood
    contribution only by a row-specific constant (the log normaliser), the
    constrained CE step maximises the classification likelihood. Values are
    clipped to ``[eps, 1]`` to avoid ``log(0)``.
    """
    return -np.log(np.clip(s, eps, 1.0))


def _minimum_cost_flow_problem_graph(cost, size_min, size_max):
    """
    Build the min-cost-flow graph for a constrained assignment problem.

    Source: https://github.com/joshlk/k-means-constrained

    Graph structure (node indices in this order):

    * ``X``  : ``n_X`` sample nodes, supply +1 each;
    * ``C'`` : ``n_C`` dummy cluster nodes, supply 0;
    * ``C``  : ``n_C`` cluster nodes, demand ``size_min`` each;
    * ``art``: one artificial sink, demand ``n_X - n_C * size_min``.

    Arcs:

    * ``X_i -> C'_k`` for all ``(i, k)``: capacity 1, cost ``cost[i, k]``;
    * ``C'_k -> C_k``: capacity ``size_max`` (upper bound on cluster size), cost 0;
    * ``C_k -> art``: capacity ``n_X`` (unbounded), cost 0.

    Each ``C_k`` must keep ``size_min`` units of flow, giving the lower bound;
    extra flow up to ``size_max`` goes on to the sink. An integer optimal
    flow sends each sample to exactly one cluster.

    Parameters
    ----------
    cost : ndarray, shape (n_samples, n_components)
        Cost of assigning sample i to component k (e.g. -log posterior).
    size_min, size_max : int
        Per-component size bounds (same semantics as k-means-constrained).

    Returns
    -------
    edges : ndarray, shape (n_arcs, 2), int64
        (tail, head) of each arc; the first ``n_X * n_C`` arcs are the
        ``X -> C'`` arcs in row-major order.
    costs : ndarray, shape (n_arcs,), int64
        Arc costs scaled by ``SCALE`` and rounded (the solver needs integers).
    capacities : ndarray, shape (n_arcs,), int64
    supplies : ndarray, shape (n_nodes,), int64
    n_C, n_X : int
        Number of components and samples.
    """
    n_X, n_C = cost.shape

    # Node indices
    X_ix = np.arange(n_X)
    C_dummy_ix = np.arange(X_ix[-1] + 1, X_ix[-1] + 1 + n_C)
    C_ix = np.arange(C_dummy_ix[-1] + 1, C_dummy_ix[-1] + 1 + n_C)
    art_ix = C_ix[-1] + 1

    # Arcs
    edges_X_C_dummy = _cartesian([X_ix, C_dummy_ix])       # all X -> all C'
    edges_C_dummy_C = np.stack([C_dummy_ix, C_ix], axis=1)  # each C' -> its C
    edges_C_art = np.stack([C_ix, art_ix * np.ones(n_C)], axis=1)  # all C -> artificial
    edges = np.concatenate([edges_X_C_dummy, edges_C_dummy_C, edges_C_art])

    # Costs: only the X -> C' arcs have a cost
    costs_X_C_dummy = cost.reshape(cost.size)
    costs = np.concatenate(
        [costs_X_C_dummy, np.zeros(edges.shape[0] - len(costs_X_C_dummy))]
    )

    # Capacities
    capacities_C_dummy_C = size_max * np.ones(n_C)
    cap_non = n_X  # won't restrict flow
    capacities = np.concatenate([
        np.ones(edges_X_C_dummy.shape[0]),
        capacities_C_dummy_C,
        cap_non * np.ones(n_C),
    ])

    # Supplies (positive) / demands (negative); they sum to zero
    supplies_X = np.ones(n_X)
    supplies_C = -1 * size_min * np.ones(n_C)
    supplies_art = -1 * (n_X - n_C * size_min)
    supplies = np.concatenate([supplies_X, np.zeros(n_C), supplies_C, [supplies_art]])

    # -log(s) with eps=1e-300 clipping is O(1)-O(700), so a finer scale than
    # k-means-constrained's 1000x buys extra precision with no real overflow
    # risk in int64.
    SCALE = 1e6
    edges = edges.astype('int64')
    costs = np.around(costs * SCALE, 0).astype('int64')
    capacities = capacities.astype('int64')
    supplies = supplies.astype('int64')

    return edges, costs, capacities, supplies, n_C, n_X


def _solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C, n_X):
    """
    Solve the min-cost-flow problem and read off the cluster labels.

    Source: https://github.com/joshlk/k-means-constrained

    Parameters
    ----------
    edges, costs, capacities, supplies, n_C, n_X
        Output of ``minimum_cost_flow_problem_graph``.

    Returns
    -------
    labels : ndarray, shape (n_X,)
        Cluster index of each sample: the ``C'`` node that receives its unit
        of flow.

    Raises
    ------
    RuntimeError
        If OR-Tools does not return an optimal solution (e.g. infeasible
        size bounds).
    """
    min_cost_flow = SimpleMinCostFlow()
    min_cost_flow.add_arcs_with_capacity_and_unit_cost(
        edges[:, 0], edges[:, 1], capacities, costs
    )
    min_cost_flow.set_nodes_supplies(np.arange(len(supplies)), supplies)

    status = min_cost_flow.solve()
    if status != min_cost_flow.OPTIMAL:
        raise RuntimeError(f"Min-cost-flow did not find an optimal solution (status={status})")

    # Flows on the first n_X * n_C arcs (X -> C', row-major) form the
    # (n_X, n_C) 0/1 assignment matrix
    labels_M = np.array(
        [min_cost_flow.flow(i) for i in range(n_X * n_C)]
    ).reshape(n_X, n_C).astype('int32')

    return labels_M.argmax(axis=1)


def _check_size_bounds(n_units, n_clusters, size_min, size_max, what="samples"):
    """
    Check that cluster-size bounds are valid and feasible.

    Parameters
    ----------
    n_units : int
        Number of units to assign (rows, columns, ...).
    n_clusters : int
        Number of clusters.
    size_min, size_max : int
        Bounds on the cluster sizes.
    what : str
        Name of the units, used in error messages.

    Raises
    ------
    ValueError
        If the bounds are outside [0, n_units], inverted, or infeasible
        (``size_min * n_clusters > n_units`` or
        ``size_max * n_clusters < n_units``).
    """
    if not (0 <= size_min <= n_units and 0 <= size_max <= n_units):
        raise ValueError(f"Size bounds for {what} must be between 0 and {n_units}")
    if size_max < size_min:
        raise ValueError(f"Size bounds for {what}: max ({size_max}) < min ({size_min})")
    if size_min * n_clusters > n_units:
        raise ValueError(
            f"Size bounds for {what}: min size {size_min} x {n_clusters} clusters "
            f"exceeds {n_units} {what}")
    if size_max * n_clusters < n_units:
        raise ValueError(
            f"Size bounds for {what}: max size {size_max} x {n_clusters} clusters "
            f"cannot cover {n_units} {what}")


def _constrained_ce_step(s, size_min=None, size_max=None):
    """
    Constrained classification E-step: hard-assign each row of the posterior
    matrix `s` (n_samples, n_components) to exactly one component, subject
    to size_min <= |cluster_k| <= size_max for every k.

    Source: https://github.com/joshlk/k-means-constrained
    (adaptation of _labels_constrained)

    Parameters
    ----------
    s : ndarray, shape (n_samples, n_components)
        Posterior membership probabilities (e.g. from ``row_posterior_mn``).
    size_min : int or None
        Minimum cluster size; ``None`` means 0.
    size_max : int or None
        Maximum cluster size; ``None`` means ``n_samples``.

    Returns
    -------
    z : ndarray, shape (n_samples, n_components), int
        One-hot hard assignment matrix, same shape/semantics as
        `np.eye(s.shape[1], dtype=int)[s.argmax(axis=1)]`.

    Raises
    ------
    ValueError
        If the bounds are out of range or infeasible
        (``size_min * n_components > n_samples`` or
        ``size_max * n_components < n_samples``).

    Notes
    -----
    With ``size_min = 0`` and ``size_max = n_samples`` the result equals the
    unconstrained argmax assignment (apart from ties and rounding of costs).
    """
    n_X, n_C = s.shape

    if size_min is None:
        size_min = 0
    if size_max is None:
        size_max = n_X

    # Feasibility checks on the size bounds
    _check_size_bounds(n_X, n_C, size_min, size_max, "samples")

    cost = _posterior_to_cost(s)
    edges, costs, capacities, supplies, n_C_, n_X_ = _minimum_cost_flow_problem_graph(cost, size_min, size_max)
    labels = _solve_min_cost_flow_graph(edges, costs, capacities, supplies, n_C_, n_X_)

    z = np.eye(n_C, dtype=int)[labels]
    return z