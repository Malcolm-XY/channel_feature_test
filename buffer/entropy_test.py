# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 22:27:11 2026

@author: 18307
"""

from utils import utils_feature_loading

pcc = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
pcc_alpha = pcc['alpha']
pcc_beta = pcc['beta']
pcc_gamma = pcc['gamma']

import feature_engineering

pcc_alpha_ = feature_engineering.normalize_matrix(pcc_alpha)
pcc_beta_ = feature_engineering.normalize_matrix(pcc_beta)
pcc_gamma_ = feature_engineering.normalize_matrix(pcc_gamma)

plv = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'plv')
plv_alpha = plv['alpha']
plv_beta = plv['beta']
plv_gamma = plv['gamma']

import scipy

pcc_e_a = scipy.stats.entropy(pcc_alpha_)
plv_e_a = scipy.stats.entropy(plv_alpha)

pcc_e_b = scipy.stats.entropy(pcc_beta_)
plv_e_b = scipy.stats.entropy(plv_beta)

pcc_e_g = scipy.stats.entropy(pcc_gamma_)
plv_e_g = scipy.stats.entropy(plv_gamma)

import numpy as np
print(np.mean(pcc_e_a))
print(np.mean(plv_e_a))

print(np.mean(pcc_e_b))
print(np.mean(plv_e_b))

print(np.mean(pcc_e_g))
print(np.mean(plv_e_g))