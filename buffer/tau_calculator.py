# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 15:06:27 2026

@author: usouu
"""

import numpy as np

def estimate_tau_from_matrix_percentile(matrix, percentile=75):
    """
    Estimate tau (PLV gating threshold) from PLV distribution using percentile.

    Parameters
    ----------
    plv_matrices : ndarray
        PLV matrices with shape (n_samples, n_channels, n_channels)

    percentile : float
        Percentile used to estimate tau (e.g., 50, 75, 80, 90)

    Returns
    -------
    tau : float
        Estimated PLV threshold
    """

    n_channels, _ = matrix.shape

    # extract upper triangle indices
    iu = np.triu_indices(n_channels, k=1)
    
    # collect all PLV edges
    values = matrix[iu[0], iu[1]].reshape(-1)

    # compute percentile
    tau = np.percentile(values, percentile)

    return tau

def estimate_tau_from_matrices_percentile(matrices, percentile=75):
    """
    Estimate tau (PLV gating threshold) from PLV distribution using percentile.

    Parameters
    ----------
    plv_matrices : ndarray
        PLV matrices with shape (n_samples, n_channels, n_channels)

    percentile : float
        Percentile used to estimate tau (e.g., 50, 75, 80, 90)

    Returns
    -------
    tau : float
        Estimated PLV threshold
    """

    n_samples, n_channels, _ = matrices.shape

    # extract upper triangle indices
    iu = np.triu_indices(n_channels, k=1)
    
    # collect all PLV edges
    values = matrices[:, iu[0], iu[1]].reshape(-1)

    # compute percentile
    tau = np.percentile(values, percentile)

    return tau

from utils import utils_feature_loading

 # %% PLV
# Experiments
# pcc = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
# plv = utils_feature_loading.read_fcs('seed', 'sub1ex3', 'plv')
    
# pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']
# plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']

# tau = estimate_tau_from_matrices_percentile(plv_gamma)
# print('tau:', tau)

# Avg
plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')

plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']

tau_1 = estimate_tau_from_matrix_percentile(plv_alpha)
print('tau:', tau_1) # 0.43926288553892395

tau_2 = estimate_tau_from_matrix_percentile(plv_beta)
print('tau:', tau_2) # 0.36658529865891054

tau_3 = estimate_tau_from_matrix_percentile(plv_gamma)
print('tau:', tau_3) # 0.32064941325729696

 # %% PLI
# Experiments
# pcc = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
# plv = utils_feature_loading.read_fcs('seed', 'sub1ex3', 'plv')
    
# pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']
# plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']

# tau = estimate_tau_from_matrices_percentile(plv_gamma)
# print('tau:', tau)

# Avg
pli = utils_feature_loading.read_fcs_global_average('seed', 'pli', sub_range=range(1, 6))

pli_alpha, pli_beta, pli_gamma = pli['alpha'], pli['beta'], pli['gamma']

tau_1 = estimate_tau_from_matrix_percentile(pli_alpha, percentile=25)
print('tau:', tau_1) # 0.20876658154913813

tau_2 = estimate_tau_from_matrix_percentile(pli_beta, percentile=25)
print('tau:', tau_2) # 0.13241194180064195

tau_3 = estimate_tau_from_matrix_percentile(pli_gamma, percentile=25)
print('tau:', tau_3) # 0.10950132586918054

 # %% PCC
# Experiments
# pcc = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
# plv = utils_feature_loading.read_fcs('seed', 'sub1ex3', 'plv')

# pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']
# plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']

# tau = estimate_tau_from_matrices_percentile(plv_gamma)
# print('tau:', tau)

# Avg
# pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')

# pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']

# import feature_engineering
# pcc_alpha_norm = feature_engineering.normalize_matrix(pcc_alpha, 'minmax', param={'target_range': (0, 1)})
# pcc_beta_norm = feature_engineering.normalize_matrix(pcc_beta, 'minmax', param={'target_range': (0, 1)})
# pcc_gamma_norm = feature_engineering.normalize_matrix(pcc_gamma, 'minmax', param={'target_range': (0, 1)})

# tau_1 = estimate_tau_from_matrix_percentile(pcc_alpha_norm)
# print('tau:', tau_1) # 0.6081250356762182

# tau_2 = estimate_tau_from_matrix_percentile(pcc_beta_norm)
# print('tau:', tau_2) # 0.5148656557079884

# tau_3 = estimate_tau_from_matrix_percentile(pcc_gamma_norm)
# print('tau:', tau_3) # 0.42120015206841854