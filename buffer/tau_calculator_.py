# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 01:07:45 2026

@author: 18307
"""

import numpy as np

def estimate_q_from_connectivity(
    matrix,
    method="mad",          # "mad" or "percentile"
    alpha=0.05,
    lower_bound=60,
    upper_bound=90,
    verbose=True
):
    """
    Estimate recommended percentile q from a global connectivity matrix.

    Parameters
    ----------
    matrix : (N, N) array
        Global average PLV connectivity matrix.
    method : str
        "mad" (robust statistical) or "percentile" (simple threshold).
    alpha : float
        Significance level (used in percentile mode).
    lower_bound : int
        Minimum allowed q.
    upper_bound : int
        Maximum allowed q.
    verbose : bool
        Print debug information.

    Returns
    -------
    q : float
        Recommended percentile parameter.
    r_sig : float
        Ratio of significant edges.
    threshold : float
        Estimated significance threshold.
    """

    # ---- 1. Extract upper triangle (excluding diagonal) ----
    N = matrix.shape[0]
    iu = np.triu_indices(N, k=1)
    values = matrix[iu]

    # ---- 2. Estimate significance threshold ----
    if method == "mad":
        # Robust statistics (median + MAD)
        median = np.median(values)
        mad = np.median(np.abs(values - median))

        # Convert MAD to std approximation
        robust_std = 1.4826 * mad

        # threshold = median + z * std
        z = 2.0   # ~95% confidence
        threshold = median + z * robust_std

    elif method == "percentile":
        # Top (1-alpha) percentile as threshold
        threshold = np.percentile(values, 100 * (1 - alpha))

    else:
        raise ValueError("method must be 'mad' or 'percentile'")

    # ---- 3. Significant edges ----
    significant = values > threshold
    r_sig = np.mean(significant)

    # ---- 4. Convert to q ----
    q = 100 * (1 - r_sig)

    # ---- 5. Clip to reasonable range ----
    q = np.clip(q, lower_bound, upper_bound)

    if verbose:
        print("----- q estimation -----")
        print(f"Threshold: {threshold:.4f}")
        print(f"Significant ratio (r_sig): {r_sig:.4f}")
        print(f"Recommended q: {q:.2f}")

    return q, r_sig, threshold

import numpy as np
from typing import Dict, Any, Optional

def estimate_q_from_global_matrix_surrogate(
    matrix: np.ndarray,
    surrogate_matrices: np.ndarray,
    alpha: float = 0.05,
    use_upper_only: bool = True,
    clip_q_range: Optional[tuple[float, float]] = (50.0, 95.0),
    symmetric_check: bool = True,
) -> Dict[str, Any]:
    """
    Estimate the recommended percentile parameter q from a global average
    connectivity matrix using surrogate-based statistical significance.

    Parameters
    ----------
    matrix : np.ndarray
        Real global average connectivity matrix of shape (N, N).
        Expected to be symmetric for undirected connectivity.

    surrogate_matrices : np.ndarray
        Surrogate global average connectivity matrices of shape (S, N, N),
        where S is the number of surrogate repetitions.

        Each surrogate matrix should be computed in the same way as the real
        global matrix, i.e. it should represent a surrogate global-average
        connectivity matrix over the same development subset.

    alpha : float, optional
        Significance level. Default is 0.05.
        The surrogate threshold is taken as the (1 - alpha) percentile.

    use_upper_only : bool, optional
        If True, only the upper triangular off-diagonal edges are used for
        significance counting. This is recommended for symmetric connectivity
        matrices. Default is True.

    clip_q_range : tuple[float, float] or None, optional
        If not None, clip the final q estimate into this range.
        Example: (50.0, 95.0). Default is (50.0, 95.0).

    symmetric_check : bool, optional
        If True, check whether the input real matrix is approximately symmetric.
        Default is True.

    Returns
    -------
    result : dict
        A dictionary containing:
            - "q_raw": float
                Raw estimated q before clipping.
            - "q_clipped": float
                Clipped q if clip_q_range is provided, else same as q_raw.
            - "significant_ratio": float
                Ratio of significant edges among counted edges.
            - "num_significant_edges": int
                Number of significant edges.
            - "num_total_edges": int
                Number of counted edges.
            - "significance_mask": np.ndarray
                Boolean matrix of shape (N, N), True where the real matrix
                exceeds the surrogate threshold.
            - "threshold_matrix": np.ndarray
                Edge-wise surrogate threshold matrix of shape (N, N).
            - "pvalue_matrix": np.ndarray
                Empirical one-sided p-value matrix of shape (N, N).
    """
    matrix = np.asarray(matrix, dtype=float)
    surrogate_matrices = np.asarray(surrogate_matrices, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("`matrix` must be a square 2D array of shape (N, N).")

    if surrogate_matrices.ndim != 3:
        raise ValueError("`surrogate_matrices` must have shape (S, N, N).")

    s, n1, n2 = surrogate_matrices.shape
    n = matrix.shape[0]

    if (n1, n2) != (n, n):
        raise ValueError(
            "`surrogate_matrices` shape must be (S, N, N) with the same N as `matrix`."
        )

    if symmetric_check and not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("`matrix` is not symmetric within tolerance.")

    if not (0.0 < alpha < 1.0):
        raise ValueError("`alpha` must be between 0 and 1.")

    # Build edge-wise surrogate threshold: (1 - alpha) percentile
    threshold_matrix = np.percentile(surrogate_matrices, 100.0 * (1.0 - alpha), axis=0)

    # Significance mask: real matrix > surrogate threshold
    significance_mask = matrix > threshold_matrix

    # Optional: exclude diagonal
    np.fill_diagonal(significance_mask, False)

    # Empirical one-sided p-value:
    # p = (count(surr >= real) + 1) / (S + 1)
    pvalue_matrix = (np.sum(surrogate_matrices >= matrix[None, :, :], axis=0) + 1.0) / (s + 1.0)
    np.fill_diagonal(pvalue_matrix, 1.0)

    # Count only unique undirected edges if requested
    if use_upper_only:
        tri_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        counted_mask = tri_mask
    else:
        counted_mask = ~np.eye(n, dtype=bool)

    counted_significance = significance_mask[counted_mask]
    num_total_edges = int(np.sum(counted_mask))
    num_significant_edges = int(np.sum(counted_significance))

    if num_total_edges == 0:
        raise ValueError("No valid edges available for counting.")

    significant_ratio = num_significant_edges / num_total_edges

    # Convert significant-edge ratio to q
    # If significant edges occupy r_sig of all edges,
    # then recommended q is approximately 100 * (1 - r_sig)
    q_raw = 100.0 * (1.0 - significant_ratio)

    if clip_q_range is not None:
        q_min, q_max = clip_q_range
        q_clipped = float(np.clip(q_raw, q_min, q_max))
    else:
        q_clipped = float(q_raw)

    return {
        "q_raw": float(q_raw),
        "q_clipped": q_clipped,
        "significant_ratio": float(significant_ratio),
        "num_significant_edges": num_significant_edges,
        "num_total_edges": num_total_edges,
        "significance_mask": significance_mask,
        "threshold_matrix": threshold_matrix,
        "pvalue_matrix": pvalue_matrix,
    }

from utils import utils_feature_loading

plv = utils_feature_loading.read_fcs_global_average('seed', 'plv', sub_range=range(1, 6))

plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']
np.fill_diagonal(plv_alpha, 0)
np.fill_diagonal(plv_beta, 0)
np.fill_diagonal(plv_gamma, 0)

# mad
q_1 = estimate_q_from_connectivity(plv_alpha)
q_2 = estimate_q_from_connectivity(plv_beta)
q_3 = estimate_q_from_connectivity(plv_gamma)

# surrogat
surrogat_1 = np.mean(utils_feature_loading.read_fcs('seed', 'sub1ex1', 'plv')['alpha'], axis=0)
surrogat_2 = np.mean(utils_feature_loading.read_fcs('seed', 'sub2ex1', 'plv')['alpha'], axis=0)
surrogat_3 = np.mean(utils_feature_loading.read_fcs('seed', 'sub3ex1', 'plv')['alpha'], axis=0)
surrogat_4 = np.mean(utils_feature_loading.read_fcs('seed', 'sub4ex1', 'plv')['alpha'], axis=0)
surrogat_5 = np.mean(utils_feature_loading.read_fcs('seed', 'sub5ex1', 'plv')['alpha'], axis=0)

for sub_matrix in surrogat_1:
    np.fill_diagonal(surrogat_1, 0)
for sub_matrix in surrogat_2:
    np.fill_diagonal(surrogat_2, 0)
for sub_matrix in surrogat_3:
    np.fill_diagonal(surrogat_3, 0)
for sub_matrix in surrogat_4:
    np.fill_diagonal(surrogat_4, 0)
for sub_matrix in surrogat_5:
    np.fill_diagonal(surrogat_5, 0)
    
surrogat_matrices = np.stack([surrogat_1, surrogat_2, surrogat_3, surrogat_4, surrogat_5])
q_1 = estimate_q_from_global_matrix_surrogate(plv_alpha, surrogat_matrices)

surrogat_1 = np.mean(utils_feature_loading.read_fcs('seed', 'sub1ex1', 'plv')['beta'], axis=0)
surrogat_2 = np.mean(utils_feature_loading.read_fcs('seed', 'sub2ex1', 'plv')['beta'], axis=0)
surrogat_3 = np.mean(utils_feature_loading.read_fcs('seed', 'sub3ex1', 'plv')['beta'], axis=0)
surrogat_4 = np.mean(utils_feature_loading.read_fcs('seed', 'sub4ex1', 'plv')['beta'], axis=0)
surrogat_5 = np.mean(utils_feature_loading.read_fcs('seed', 'sub5ex1', 'plv')['beta'], axis=0)

for sub_matrix in surrogat_1:
    np.fill_diagonal(surrogat_1, 0)
for sub_matrix in surrogat_2:
    np.fill_diagonal(surrogat_2, 0)
for sub_matrix in surrogat_3:
    np.fill_diagonal(surrogat_3, 0)
for sub_matrix in surrogat_4:
    np.fill_diagonal(surrogat_4, 0)
for sub_matrix in surrogat_5:
    np.fill_diagonal(surrogat_5, 0)
    
surrogat_matrices = np.stack([surrogat_1, surrogat_2, surrogat_3, surrogat_4, surrogat_5])
q_2 = estimate_q_from_global_matrix_surrogate(plv_beta, surrogat_matrices)

surrogat_1 = np.mean(utils_feature_loading.read_fcs('seed', 'sub1ex1', 'plv')['gamma'], axis=0)
surrogat_2 = np.mean(utils_feature_loading.read_fcs('seed', 'sub2ex1', 'plv')['gamma'], axis=0)
surrogat_3 = np.mean(utils_feature_loading.read_fcs('seed', 'sub3ex1', 'plv')['gamma'], axis=0)
surrogat_4 = np.mean(utils_feature_loading.read_fcs('seed', 'sub4ex1', 'plv')['gamma'], axis=0)
surrogat_5 = np.mean(utils_feature_loading.read_fcs('seed', 'sub5ex1', 'plv')['gamma'], axis=0)

for sub_matrix in surrogat_1:
    np.fill_diagonal(surrogat_1, 0)
for sub_matrix in surrogat_2:
    np.fill_diagonal(surrogat_2, 0)
for sub_matrix in surrogat_3:
    np.fill_diagonal(surrogat_3, 0)
for sub_matrix in surrogat_4:
    np.fill_diagonal(surrogat_4, 0)
for sub_matrix in surrogat_5:
    np.fill_diagonal(surrogat_5, 0)
    
surrogat_matrices = np.stack([surrogat_1, surrogat_2, surrogat_3, surrogat_4, surrogat_5])
q_3 = estimate_q_from_global_matrix_surrogate(plv_gamma, surrogat_matrices)
