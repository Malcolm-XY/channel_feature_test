# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 03:29:40 2026

@author: 18307
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 03:13:06 2026

@author: 18307
"""
import numpy as np
import pandas as pd

import feature_fusion
from utils import utils_feature_loading
from utils import utils_visualization
    
def similarity_matrices(A, B):
    sim = np.dot(A.flatten(), B.flatten()) / (
        np.linalg.norm(A.flatten()) * np.linalg.norm(B.flatten()))
    return sim

def rv_coefficient(A, B):
    """
    Compute RV coefficient between two matrices.

    Parameters
    ----------
    A : ndarray
    B : ndarray

    Returns
    -------
    float
        RV coefficient
    """

    A = np.asarray(A)
    B = np.asarray(B)

    AA = A @ A.T
    BB = B @ B.T

    numerator = np.trace(AA @ BB)

    denominator = np.sqrt(
        np.trace(AA @ AA) *
        np.trace(BB @ BB)
    )

    return numerator / denominator

def similarity_4_origin_fused(k, tau, nrr=0.5, details=False, similarity_function=rv_coefficient):
    params = {'k': float(k), 'tau': float(tau)}
    
    pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')
    plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')
    
    pcc_avg = (pcc['alpha']+pcc['beta']+pcc['gamma'])/3
    plv_avg = (plv['alpha']+plv['beta']+plv['gamma'])/3
    
    pcc_avg_flat = pd.DataFrame({'values': np.mean(pcc_avg, axis=0)})
    node_list_pcc_avg = pcc_avg_flat.sort_values('values')
    node_list_pcc_avg_prune_index = np.array(node_list_pcc_avg[int(np.ceil(len(node_list_pcc_avg)*nrr)):].index)
    
    pcc_avg_prune = pcc_avg.copy()
            
    pcc_avg_prune[node_list_pcc_avg_prune_index, :] = 0
    pcc_avg_prune[:, node_list_pcc_avg_prune_index] = 0
    
    fused_avg_prune = feature_fusion.feature_fusion_sigmoid_gating(pcc_avg_prune, plv_avg, params=params)
    
    sim = similarity_function(pcc_avg_prune, fused_avg_prune)
    
    if details:
        utils_visualization.draw_projection(pcc_avg)
        utils_visualization.draw_projection(pcc_avg_prune)
        utils_visualization.draw_projection(fused_avg_prune)
        print("Similarity:", sim)
    return sim

def spectral_energy_compaction_4_matrix(matrix, k_ratio=0.1, eps=1e-12):
    eigvals = np.linalg.eigvalsh(matrix)
    eigvals = np.abs(eigvals)
    eigvals = np.sort(eigvals)[::-1]

    k = max(1, int(len(eigvals) * k_ratio))
    return np.sum(eigvals[:k]) / (np.sum(eigvals) + eps)

def spectral_energy_compaction_4_fns(k, tau):
    params = {'k': float(k), 'tau': float(tau)}
    
    pcc = utils_feature_loading.read_fcs_global_average('seed', 'pcc')
    plv = utils_feature_loading.read_fcs_global_average('seed', 'plv')
    
    pcc_avg = (pcc['alpha']+pcc['beta']+pcc['gamma'])/3
    plv_avg = (plv['alpha']+plv['beta']+plv['gamma'])/3
    
    fused = feature_fusion.feature_fusion_sigmoid_gating(pcc_avg, plv_avg, params=params)

    h_ = spectral_energy_compaction_4_matrix(fused)
    
    return h_

# grid search (loss function embedded)
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

# weight center
def matrix_weight_center(W):
    W = np.asarray(W)

    rows, cols = W.shape

    y_idx, x_idx = np.indices((rows, cols))

    total_weight = W.sum()

    yc = (y_idx * W).sum() / total_weight
    xc = (x_idx * W).sum() / total_weight

    return yc, xc

def robust_peak_center(Z, x_values=None, y_values=None, eps_ratio=0.01, weighted=True):
    """
    在接近最大值的平台区域中，计算稳定中心
    Z: 2D matrix, shape=(ny, nx)
    eps_ratio: 相对容差，例如 0.01 表示保留 >= 99% max 的区域
    """
    Z = np.asarray(Z)
    ny, nx = Z.shape

    if x_values is None:
        x_values = np.arange(nx)
    if y_values is None:
        y_values = np.arange(ny)

    X, Y = np.meshgrid(x_values, y_values)

    zmax = np.max(Z)
    threshold = zmax * (1 - eps_ratio)
    mask = Z >= threshold

    if not np.any(mask):
        raise ValueError("No points found in plateau region.")

    if weighted:
        W = Z[mask]
        xc = np.sum(X[mask] * W) / np.sum(W)
        yc = np.sum(Y[mask] * W) / np.sum(W)
    else:
        xc = np.mean(X[mask])
        yc = np.mean(Y[mask])

    return yc, xc, mask

if __name__ == '__main__':
    # test    
    k, tau = 27, 0.2
    similarity_4_origin_fused(k, tau, details=True)
    
    k, tau = 50, 0.2
    similarity_4_origin_fused(k, tau, details=True)
    
    k, tau = 100, 0.2
    similarity_4_origin_fused(k, tau, details=True)
    
    # grid search
    k_list   = np.linspace(1, 100, 100)     # 你按实际调整范围/步长
    tau_list = np.linspace(0.01, 1, 100)  # tau 通常希望 >0
    
    best, loss_map = grid_search_p1_p2(k_list, tau_list, "upper", similarity_4_origin_fused)
    print(best)
    utils_visualization.draw_projection(loss_map, "loss map")
    
    # weight center
    yc, xc = matrix_weight_center(loss_map)
    k = np.max(k_list)*(yc/len(k_list))
    tau = np.max(tau_list)*(xc/len(tau_list))
    
    # peak center
    yc, xc, mask = robust_peak_center(loss_map)
    k = np.max(k_list)*(yc/len(k_list))
    tau = np.max(tau_list)*(xc/len(tau_list))
    
    # contour line
    def ridge_from_plateau(W, eps_ratio=0.01):
        W = np.asarray(W)
    
        zmax = np.max(W)
        mask = W >= zmax * (1 - eps_ratio)
    
        ridge_x = []
        ridge_y = []
    
        for i in range(W.shape[0]):
    
            cols = np.where(mask[i])[0]
    
            if len(cols) > 0:
                ridge_x.append(cols.mean())
                ridge_y.append(i)
    
        return np.array(ridge_x), np.array(ridge_y)
    
    ridge_x, ridge_y = ridge_from_plateau(loss_map)
    
    import matplotlib.pyplot as plt
    plt.imshow(loss_map, cmap="viridis")
    plt.plot(ridge_x, ridge_y, "r", linewidth=3)
    plt.colorbar()
    plt.show()
    
    # best, loss_map = grid_search_p1_p2(k_list, tau_list, "upper", spectral_energy_compaction_4_fns)
    # print(best)
    # utils_visualization.draw_projection(loss_map)