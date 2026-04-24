# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 23:24:35 2026

This code is for parameter optimazation of k and tau for PG-AC model.

@author: 18307
"""
import numpy as np
import feature_fusion

# %% Defination of index
def redundancy_4_matrix(matrix, absolute=True):
    matrix = np.array(matrix)
    width_1, width_2 = matrix.shape
    repeat = width_1

    corr = []
    for i in range(repeat):
        for j in range(repeat):
            if i != j:
                edge_1, edge_2 = matrix[i], matrix[j]
                _corr = np.corrcoef(edge_1, edge_2)[0, 1]
                
                if absolute:
                   _corr = np.abs(_corr)
                    
                corr.append(_corr)
                
    redundancy_ = (1/(width_1*(width_1 - 1))) * np.sum(corr)
    return redundancy_

def spectral_entropy_4_matrix(matrix, eps=1e-12):
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.maximum(eigvals, 0)
    p = eigvals / (np.sum(eigvals) + eps)
    H = -np.sum(p * np.log(p + eps))
    Hmax = np.log(len(p))
    redundancy = 1 - H / Hmax
    return H, redundancy

def spectral_energy_compaction_4_matrix(matrix, k_ratio=0.1, eps=1e-12):
    """
    Compute spectral energy compaction of a matrix.

    Parameters
    ----------
    A : ndarray (N, N)
        Input matrix (symmetric preferred).
    k_ratio : float
        Ratio of top eigenvalues to accumulate (0 < k_ratio <= 1).
    eps : float
        Numerical stability constant.

    Returns
    -------
    EC : float
        Energy compaction ratio in [0, 1].
    """
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.abs(eigvals)
    eigvals = np.sort(eigvals)[::-1]

    k = max(1, int(len(eigvals) * k_ratio))
    return np.sum(eigvals[:k]) / (np.sum(eigvals) + eps)

# %% Index for implementation
def redundancy_4_fns(k, percentile,
                    pcc, plv):
    params = {'k': float(k), 'percentile': float(percentile)}

    fused_feature = feature_fusion.feature_fusion_sigmoid_gating(pcc, plv, params=params)
    
    r = redundancy_4_matrix(fused_feature)
    
    return r

def spectral_entropy_4_fns(k, percentile,
                           pcc, plv):
    params = {'k': float(k), 'percentile': float(percentile)}

    fused_feature = feature_fusion.feature_fusion_sigmoid_gating(pcc, plv, params=params)
    
    h, _ = spectral_entropy_4_matrix(fused_feature)

    return h

def spectral_energy_compaction_4_fns(k, percentile,
                                     pcc, plv):
    params = {'k': float(k), 'percentile': float(percentile)}

    fused_feature = feature_fusion.feature_fusion_sigmoid_gating(pcc, plv, params=params)
    
    h = spectral_energy_compaction_4_matrix(fused_feature)
    
    return h

# %% Grid search (loss function embedded)
def grid_search_p1_p2(p1_list, p2_list, boundary, loss_func, *args):
    loss_function = loss_func
    best = {"p1": None, "p2": None, "loss": np.inf if boundary == "lower" else -np.inf}
    loss_map = np.zeros([len(p1_list), len(p2_list)])
    for index_p1, p1 in enumerate(p1_list):
        for index_p2, p2 in enumerate(p2_list):
            loss = loss_function(p1, p2, *args)
            loss_map[index_p1][index_p2] = loss
            if boundary == "lower":
                if loss < best["loss"]:
                    best.update(p1=float(p1), p2=float(p2), loss=float(loss))
            elif boundary == "upper":
                if loss > best["loss"]:
                    best.update(p1=float(p1), p2=float(p2), loss=float(loss))
            
    return best, loss_map

if __name__ == '__main__':
    #%% Data Preparation
    from utils import utils_feature_loading
    from utils import utils_visualization
    pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')
    plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')
    
    pcc_alpha, pcc_beta, pcc_gamma = pcc['alpha'], pcc['beta'], pcc['gamma']
    plv_alpha, plv_beta, plv_gamma = plv['alpha'], plv['beta'], plv['gamma']
    
    labels = utils_feature_loading.read_labels('seed', header=True)
    
    # %% Optimization
    # optimization parameters
    k_list   = np.linspace(1, 100, 100)     # 你按实际调整范围/步长
    percentile_list = np.linspace(1, 100, 100)  # tau 通常希望 >0
    
    """
    Optimal Succussed
    Optimal Parameters by Classification Results Guided Grid Search: k, tau = [79.16, 0.01], [100, 0.17], [100, 0.43]
    """
    # optimization; spectral entropy
    best_se, loss_map_1 = grid_search_p1_p2(k_list, percentile_list, "lower", spectral_entropy_4_fns, pcc_alpha, plv_alpha)
    print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": 100, 1.0, fail; "lower": 76, 1
    utils_visualization.draw_projection(-loss_map_1, "loss map")
    
    best_se, loss_map_2 = grid_search_p1_p2(k_list, percentile_list, "lower", spectral_entropy_4_fns, pcc_beta, plv_beta)
    print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": 100. 1.0, fail; "lower": 100, 17
    utils_visualization.draw_projection(-loss_map_2, "loss map")
    
    best_se, loss_map_3 = grid_search_p1_p2(k_list, percentile_list, "lower", spectral_entropy_4_fns, pcc_gamma, plv_gamma)
    print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": 100, 1.0, fail; "lower": 100, 42
    utils_visualization.draw_projection(-loss_map_3, "loss map")
    
    # optimization; spectral energy compaction
    best_se, loss_map_1 = grid_search_p1_p2(k_list, percentile_list, "upper", spectral_energy_compaction_4_fns, pcc_alpha, plv_alpha)
    print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": 23, 1.0; "lower": 100, 1.0, fail
    utils_visualization.draw_projection(loss_map_1, "loss map")
    
    best_se, loss_map_2 = grid_search_p1_p2(k_list, percentile_list, "upper", spectral_energy_compaction_4_fns, pcc_beta, plv_beta)
    print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": 29. 27; "lower": 100, 1.0, fail
    utils_visualization.draw_projection(loss_map_2, "loss map")
    
    best_se, loss_map_3 = grid_search_p1_p2(k_list, percentile_list, "upper", spectral_energy_compaction_4_fns, pcc_gamma, plv_gamma)
    print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": 58, 41; "lower": 100, 1.0, fail
    utils_visualization.draw_projection(loss_map_3, "loss map")

    # # optimization; redundancy
    # best_se, loss_map_1 = grid_search_p1_p2(k_list, percentile_list, "upper", redundancy_4_fns, pcc_alpha, plv_alpha)
    # print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": fail; "lower": fail
    # utils_visualization.draw_projection(loss_map_1, "loss map")
    
    # best_se, loss_map_2 = grid_search_p1_p2(k_list, percentile_list, "upper", redundancy_4_fns, pcc_beta, plv_beta)
    # print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": fail; "lower": fail
    # utils_visualization.draw_projection(loss_map_2, "loss map")
    
    # best_se, loss_map_3 = grid_search_p1_p2(k_list, percentile_list, "upper", redundancy_4_fns, pcc_gamma, plv_gamma)
    # print(best_se["p1"], best_se["p2"], best_se["loss"]) # "upper": fail; "lower": fail
    # utils_visualization.draw_projection(loss_map_3, "loss map")
