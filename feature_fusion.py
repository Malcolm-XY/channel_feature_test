# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 16:55:00 2026

@author: 18307
"""

import numpy as np
import warnings
import feature_engineering

def normalization_fixer(matrices_basis, matrices_modifier, params=None):
    """
    Apply optional normalization to basis and modifier matrices.
    """
    default_params = {
        'normalization_basis': None,
        'normalization_modifier': None,
        'scale': (0, 1)
    }

    if params is None:
        params = default_params
    else:
        # fill missing keys with defaults
        for key, val in default_params.items():
            params.setdefault(key, val)

    normalization_basis = params['normalization_basis']
    normalization_modifier = params['normalization_modifier']
    normalization_scale = params['scale']

    # Warning mechanism
    if normalization_basis is None:
        warnings.warn(
            "[normalization_fixer] 'normalization_basis' not explicitly specified. "
            "Defaulting to False (no normalization).",
            UserWarning
        )
        normalization_basis = False

    if normalization_modifier is None:
        warnings.warn(
            "[normalization_fixer] 'normalization_modifier' not explicitly specified. "
            "Defaulting to False (no normalization).",
            UserWarning
        )
        normalization_modifier = False

    # Normalization
    if normalization_basis:
        matrices_basis = feature_engineering.normalize_matrix(
            matrices_basis,
            method='minmax',
            param={'target_range': normalization_scale}
        )
    else:
        matrices_basis = np.asarray(matrices_basis)

    if normalization_modifier:
        matrices_modifier = feature_engineering.normalize_matrix(
            matrices_modifier,
            method='minmax',
            param={'target_range': normalization_scale}
        )
    else:
        matrices_modifier = np.asarray(matrices_modifier)

    matrices_basis = np.asarray(matrices_basis)
    matrices_modifier = np.asarray(matrices_modifier)

    return matrices_basis, matrices_modifier

class params_default:
    params_4_competitors = {'fusion_type': 'additive',
                            'normalization_basis': False,
                            'normalization_modifier': False,
                            'scale': (0,1)}

    params_4_PCAEC ={'fusion_type': 'sigmoid_gating',
                     'k': 10.0, # gate sharpness
                     'percentile': 25, # confidence threshold
                     'power': 1, # for power gating variant
                     'normalization_basis': False,
                     'normalization_modifier': False,
                     'scale': (0, 1)}

# competitors: additive, multiplicative, color_blocking
def feature_fusion_triangle_blocking(fn_basis, fn_modifier,
                                     params=params_default.params_4_competitors):
    fn_basis, fn_modifier = normalization_fixer(fn_basis, fn_modifier, params)

    upper = np.triu(fn_basis)
    lower = np.tril(fn_modifier)
    
    fn_fussed = upper + lower
    return fn_fussed

def feature_fusion_diagonal_blocking(fn_basis, fn_modifier,
                                     params=params_default.params_4_competitors):
    fn_basis, fn_modifier = normalization_fixer(fn_basis, fn_modifier, params)

    if len(fn_basis.shape) == 3:
        number_samples, length, _ = fn_basis.shape
        A = np.zeros([number_samples, 2*length, 2*length])
        B = A.copy()
        
        A[:, :length, :length] = fn_basis
        B[:, length:, length:] = fn_modifier
    elif len(fn_basis.shape) == 2:
        length, _ = fn_basis.shape
        A = np.zeros([2*length, 2*length])
        B = A.copy()
        
        A[:length, :length] = fn_basis
        B[length:, length:] = fn_modifier
    
    fn_fussed = A + B
    return fn_fussed

def feature_fusion_additive(fn_basis, fn_modifier,
                            params=params_default.params_4_competitors):
    fn_basis, fn_modifier = normalization_fixer(fn_basis, fn_modifier, params)
    
    fn_fussed = fn_basis + fn_modifier
    return fn_fussed

def feature_fusion_multiplicative(fn_basis, fn_modifier,
                                  params=params_default.params_4_competitors):
    fn_basis, fn_modifier = normalization_fixer(fn_basis, fn_modifier, params)
    
    fn_fussed = fn_basis * fn_modifier
    return fn_fussed

# proposed
from utils import utils_feature_loading
def feature_fusion_power_gating(fn_basis, fn_modifier,
                                params=params_default.params_4_PCAEC):
    power = params.get('power', 1)

    # normalization
    fn_basis, fn_modifier = normalization_fixer(fn_basis, fn_modifier, params)
    
    # operation
    alpha = fn_modifier ** power
    fn_fussed = fn_basis * alpha
    
    return fn_fussed

def feature_fusion_sigmoid_gating(fn_basis, fn_modifier,
                                  params=params_default.params_4_PCAEC):
    k = params.get('k', 10.0)
    percentile = params.get('percentile', 25)

    # normalization
    fn_basis, fn_modifier = normalization_fixer(fn_basis, fn_modifier, params)

    # -------- CASE 1 : single matrix (C,C) --------
    if fn_modifier.ndim == 2:
        C = fn_modifier.shape[0]
        tri_mask = np.triu(np.ones((C, C), dtype=bool), 1)
        
        values = fn_modifier[tri_mask]
        tau = np.percentile(values, percentile)
        
    # -------- CASE 2 : batch matrices (N,C,C) --------
    elif fn_modifier.ndim == 3:
        N, C, _ = fn_modifier.shape
        tri_mask = np.triu(np.ones((C, C), dtype=bool), 1)

        # extract upper triangle for each sample
        values = fn_modifier[:, tri_mask]
        # percentile for each sample
        tau = np.percentile(values, percentile, axis=1)
        # reshape for broadcasting
        tau = tau[:, None, None]
    
    else:
        raise ValueError("fn_modifier must be (C,C) or (N,C,C)")

    if k == 'heaviside':
        alpha = (fn_modifier > tau).astype(float)
    elif isinstance(k, (int, float)):
        alpha = 1.0 / (1.0 + np.exp(-k * (fn_modifier - tau)))
    else:
        raise ValueError(f"[feature_fusion] Invalid k: {k}. "
                        "Expected 'heaviside' or a numeric value (int/float).")

    fn_fused = fn_basis * alpha
    return fn_fused

# executor
def feature_fusion(fns_1, fns_2, params=params_default.params_4_competitors):
    fusion_type = params.get('fusion_type', None).lower()
    
    fusion_type_valid = {'triangle_blocking', 'diagonal_blocking',
                          'additive', 'multiplicative', 
                          'power_gating', 'sigmoid_gating',}
    if fusion_type not in fusion_type_valid:
        raise ValueError(f"Invalid filter '{fusion_type}'. Allowed filters: {fusion_type_valid}")
        
    # competitors: additive, multiplicative, triangle_blocking, diagonal_blocking
    elif fusion_type == 'additive':
        fn_fussed = feature_fusion_additive(fns_1, fns_2, params)
    elif fusion_type == 'multiplicative':
        fn_fussed = feature_fusion_multiplicative(fns_1, fns_2, params)
    elif fusion_type == 'triangle_blocking':
        fn_fussed = feature_fusion_triangle_blocking(fns_1, fns_2, params)
    elif fusion_type == 'diagonal_blocking':
        fn_fussed = feature_fusion_diagonal_blocking(fns_1, fns_2, params)
    
    # proposed power_gating
    elif fusion_type == 'power_gating':
        fn_fussed = feature_fusion_power_gating(fns_1, fns_2, params)
    
    # proposed PCC(sigmoid(PLV))
    elif fusion_type == 'sigmoid_gating':
        fn_fussed = feature_fusion_sigmoid_gating(fns_1, fns_2, params)

    else:
        raise ValueError(f"Invalid filter '{fusion_type}'. Allowed filters: {fusion_type_valid}")

    return fn_fussed

# %% Test
if __name__ == "__main__":
    # from utils import utils_feature_loading
    from utils import utils_visualization
    
    feature_basis='pcc'
    feature_modifier='plv'
    
    # Baselines
    fcs_basis_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_basis, 'joint', range(1,6))
    alpha_basis_global_averaged = fcs_basis_global_averaged['alpha']
    beta_basis_global_averaged = fcs_basis_global_averaged['beta']
    gamma_basis_global_averaged = fcs_basis_global_averaged['gamma']
    
    fcs_modifier_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_modifier, 'joint', range(1,6))
    alpha_modifier_global_averaged = fcs_modifier_global_averaged['alpha']
    beta_modifier_global_averaged = fcs_modifier_global_averaged['beta']
    gamma_modifier_global_averaged = fcs_modifier_global_averaged['gamma']
    
    # utils_visualization.draw_projection(alpha_basis_global_averaged)
    # utils_visualization.draw_projection(alpha_modifier_global_averaged)
    
    # %% Competitors
    # params = params_default.params_4_competitors.copy()
    # params['fusion_type'] = 'additive'
    # params['normalization_basis'] = True
    
    # alpha_fussed = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    # utils_visualization.draw_projection(alpha_fussed, "Additive, Alpha")
    
    # params = params_default.params_4_competitors.copy()
    # params['fusion_type'] = 'multiplicative'
    # alpha_fused = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    # utils_visualization.draw_projection(alpha_fused, "Multiplicative, Alpha")
    
    # params['fusion_type'] = 'triangle_blocking'
    # alpha_fused = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    # utils_visualization.draw_projection(alpha_fused, "Triangle Blocking, Alpha")
    
    # params['fusion_type'] = 'diagonal_blocking'
    # alpha_fused = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
    # utils_visualization.draw_projection(alpha_fused, "Diagonal Blocking, Alpha")
    
    # %% Proposed PG-AC
    # Sigmoid Gating
    params_a = params_default.params_4_PCAEC
    params_b, params_g = params_a.copy(), params_a.copy()
        
    alpha_fussed = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params_a)
    beta_fussed = feature_fusion(beta_basis_global_averaged, beta_modifier_global_averaged, params_b)    
    gamma_fussed = feature_fusion(gamma_basis_global_averaged, gamma_modifier_global_averaged, params_g)    
    
    utils_visualization.draw_projection(alpha_fussed, "Sigmoid Gating, Alpha")
    utils_visualization.draw_projection(beta_fussed, "Sigmoid Gating, Beta")
    utils_visualization.draw_projection(gamma_fussed, "Sigmoid Gating, Gamma")
    
    # Sigmoid Gating; Heaviside
    params_a['k'] = 'heaviside'
    params_b, params_g = params_a.copy(), params_a.copy()
    
    alpha_fussed = feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params_a)
    beta_fussed = feature_fusion(beta_basis_global_averaged, beta_modifier_global_averaged, params_b)    
    gamma_fussed = feature_fusion(gamma_basis_global_averaged, gamma_modifier_global_averaged, params_g)    
    
    utils_visualization.draw_projection(alpha_fussed, "Heaviside Gating, Alpha")
    utils_visualization.draw_projection(beta_fussed, "Heaviside Gating, Beta")
    utils_visualization.draw_projection(gamma_fussed, "Heaviside Gating, Gamma")