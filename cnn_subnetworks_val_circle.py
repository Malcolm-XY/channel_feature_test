# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:21:23 2025

@author: usouu
"""
import numpy as np
import pandas as pd

import torch

import cnn_validation
import feature_engineering
from models import models
from utils import utils_feature_loading
from utils import utils_tools

# from tool_read_params_save_xlsx import read_params
from tool_read_params_save_xlsx import save_to_xlsx_sheet
# from tool_read_params_save_xlsx import save_to_xlsx_fitting

# %% cnn subnetworks evaluation circle common
def cnn_subnetworks_evaluation_circle_original_cm(feature_cm='pcc', normalization_for_train=False,
                                                  subject_range=range(6,16), experiment_range=range(1,4),
                                                  node_retention_rate=1.0,
                                                  subnetworks_extract='read', subnetworks_extract_basis=range(1, 6),
                                                  save=False):
    if subnetworks_extract == 'read':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_cm, sub_range=subnetworks_extract_basis)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']
        
        strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=1)
        strength_beta = np.sum(np.abs(beta_global_averaged), axis=1)
        strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=1)
        
        channel_weights = {'alpha': strength_alpha, 
                           'beta': strength_beta,
                           'gamma': strength_gamma,
                           }

    elif subnetworks_extract == 'calculation':
        functional_node_strength = {'alpha': [], 'beta': [], 'gamma': []}
        
        for sub in subnetworks_extract_basis:
            for ex in experiment_range:
                subject_id = f"sub{sub}ex{ex}"
                print(f"Evaluating {subject_id}...")
                
                # CM/MAT
                # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
                # alpha = features['alpha']
                # beta = features['beta']
                # gamma = features['gamma']
    
                # CM/H5
                features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
                alpha = features['alpha']
                beta = features['beta']
                gamma = features['gamma']
                
                # Compute node strength
                strength_alpha = np.sum(np.abs(alpha), axis=1)
                strength_beta = np.sum(np.abs(beta), axis=1)
                strength_gamma = np.sum(np.abs(gamma), axis=1)
                
                # Save for further analysis
                functional_node_strength['alpha'].append(strength_alpha)
                functional_node_strength['beta'].append(strength_beta)
                functional_node_strength['gamma'].append(strength_gamma)
    
        channel_weights = {'gamma': np.mean(np.mean(functional_node_strength['gamma'], axis=0), axis=0),
                           'beta': np.mean(np.mean(functional_node_strength['beta'], axis=0), axis=0),
                           'alpha': np.mean(np.mean(functional_node_strength['alpha'], axis=0), axis=0)
                           }
    
    k = {'gamma': int(len(channel_weights['gamma']) * node_retention_rate),
         'beta': int(len(channel_weights['beta']) * node_retention_rate),
         'alpha': int(len(channel_weights['alpha']) * node_retention_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    
    # for traning and testing in CNN
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")
            
            # CM/MAT
            # features = utils_feature_loading.read_fcs_mat('seed', subject_id, feature_cm)
            # alpha = features['alpha']
            # beta = features['beta']
            # gamma = features['gamma']

            # CM/H5
            features = utils_feature_loading.read_fcs('seed', subject_id, feature_cm)
            alpha = features['alpha']
            beta = features['beta']
            gamma = features['gamma']

            # Selected CM           
            alpha_selected = alpha[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_selected = beta[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_selected = gamma[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            # Normalization before training
            if normalization_for_train:
                alpha_selected = feature_engineering.normalize_matrix(alpha_selected)
                beta_selected = feature_engineering.normalize_matrix(beta_selected)
                gamma_selected = feature_engineering.normalize_matrix(gamma_selected)
            
            x_selected = np.stack((alpha_selected, beta_selected, gamma_selected), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            
            # traning and testing
            # test
            # result_CM = cnn_validation.cnn_cross_validation(cnn_model, x_selected, y)
            result_CM = cnn_validation.cnn_sequential_validation(cnn_model, x_selected, y)
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_CM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    # Save
    if save:        
        folder_name = 'results_cnn_evaluation(stress_test)'
        file_name = f'cnn_evaluation(stress_test)_{feature_cm}_origin.xlsx'
        sheet_name = f'nrr_{node_retention_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)

    return df_results

import feature_fusion
def cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis='pcc', feature_modifier='plv',
                                                     params={'fusion_type': 'triangle_blocking',
                                                             'normalization_basis': False, 
                                                             'normalization_modifier': False,
                                                             'scale': (0, 1)},
                                                     normalization_for_train=False,
                                                     subject_range=range(6,16), experiment_range=range(1,4),
                                                     subnetworks_extract='separate_index', node_retention_rate=1.0,
                                                     subnets_extract_basis_sub=range(1, 6), subnets_extract_basis_ex=range(1, 4),
                                                     save=False):
    # subnetworks selects;channel selects------start
    # valid filters
    fusion_type = params.get('fusion_type')
    if fusion_type is None:
        raise ValueError("`fusion_type` must be provided and non-empty.")
    
    fusion_type = fusion_type.strip().lower()
    
    fusion_type_valid = {'triangle_blocking', 
                         'diagonal_blocking',
                         'additive', 'multiplicative',
                         'power_gating', 'sigmoid_gating'}
    
    if fusion_type not in fusion_type_valid:
        raise ValueError(f"Invalid filter '{fusion_type}'. Allowed filters: {fusion_type_valid}")
    
    # subnetwork extraction----start
    if subnetworks_extract == 'unify_index':
        fcs_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_basis, 'joint',
                                                                            subnets_extract_basis_sub)
        alpha_global_averaged = fcs_global_averaged['alpha']
        beta_global_averaged = fcs_global_averaged['beta']
        gamma_global_averaged = fcs_global_averaged['gamma']

    elif subnetworks_extract == 'separate_index':
        # basis feature
        fcs_basis_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_basis, 'joint',
                                                                                  subnets_extract_basis_sub)
        alpha_basis_global_averaged = fcs_basis_global_averaged['alpha']
        beta_basis_global_averaged = fcs_basis_global_averaged['beta']
        gamma_basis_global_averaged = fcs_basis_global_averaged['gamma']
        
        # modifier feature
        if feature_modifier is not None:
            fcs_modifier_global_averaged = utils_feature_loading.read_fcs_global_average('seed', feature_modifier, 'joint',
                                                                                         subnets_extract_basis_sub)
            alpha_modifier_global_averaged = fcs_modifier_global_averaged['alpha']
            beta_modifier_global_averaged = fcs_modifier_global_averaged['beta']
            gamma_modifier_global_averaged = fcs_modifier_global_averaged['gamma']
        elif feature_modifier is None:
            alpha_modifier_global_averaged = None
            beta_modifier_global_averaged = None
            gamma_modifier_global_averaged = None
            
        alpha_global_averaged = feature_fusion.feature_fusion(alpha_basis_global_averaged, alpha_modifier_global_averaged, params)
        beta_global_averaged = feature_fusion.feature_fusion(beta_basis_global_averaged, beta_modifier_global_averaged, params)
        gamma_global_averaged = feature_fusion.feature_fusion(gamma_basis_global_averaged, gamma_modifier_global_averaged, params)
        
    strength_alpha = np.sum(np.abs(alpha_global_averaged), axis=0)
    strength_beta = np.sum(np.abs(beta_global_averaged), axis=0)
    strength_gamma = np.sum(np.abs(gamma_global_averaged), axis=0)
        
    channel_weights = {'gamma': strength_gamma, 'beta': strength_beta, 'alpha': strength_alpha}
        
    k = {'gamma': int(len(channel_weights['gamma']) * node_retention_rate),
         'beta': int(len(channel_weights['beta']) * node_retention_rate),
         'alpha': int(len(channel_weights['alpha']) * node_retention_rate),
          }
    
    channel_selects = {'gamma': np.argsort(channel_weights['gamma'])[-k['gamma']:][::-1],
                       'beta': np.argsort(channel_weights['beta'])[-k['beta']:][::-1],
                       'alpha': np.argsort(channel_weights['alpha'])[-k['alpha']:][::-1]
                       }
    # subnetworks selects;channel selects------end
    
    # for training and testing in CNN------start
    # labels
    labels = utils_feature_loading.read_labels(dataset='seed', header=True)
    y = torch.tensor(np.array(labels)).view(-1)
    
    # data and evaluation circle
    all_results_list = []
    for sub in subject_range:
        for ex in experiment_range:
            subject_id = f"sub{sub}ex{ex}"
            print(f"Evaluating {subject_id}...")

            # FN/H5
            features_basis = utils_feature_loading.read_fcs('seed', subject_id, feature_basis)
            alpha_basis = features_basis['alpha']
            beta_basis = features_basis['beta']
            gamma_basis = features_basis['gamma']
            
            if feature_modifier is not None:
                features_modifier = utils_feature_loading.read_fcs('seed', subject_id, feature_modifier)
                alpha_modifier = features_modifier['alpha']
                beta_modifier = features_modifier['beta']
                gamma_modifier = features_modifier['gamma']
            elif feature_modifier is None:
                alpha_modifier = None
                beta_modifier = None
                gamma_modifier = None
                
            # fussed FN
            alpha_fussed = feature_fusion.feature_fusion(alpha_basis, alpha_modifier, params)
            beta_fussed = feature_fusion.feature_fusion(beta_basis, beta_modifier, params)
            gamma_fussed = feature_fusion.feature_fusion(gamma_basis, gamma_modifier, params)
            
            alpha_fussed = alpha_fussed[:,channel_selects['alpha'],:][:,:,channel_selects['alpha']]
            beta_fussed = beta_fussed[:,channel_selects['beta'],:][:,:,channel_selects['beta']]
            gamma_fussed = gamma_fussed[:,channel_selects['gamma'],:][:,:,channel_selects['gamma']]
            
            # Normalization before training
            if normalization_for_train:
                alpha_fussed = feature_engineering.normalize_matrix(alpha_fussed)
                beta_fussed = feature_engineering.normalize_matrix(beta_fussed)
                gamma_fussed = feature_engineering.normalize_matrix(gamma_fussed)
            
            x_rebuild = np.stack((alpha_fussed, beta_fussed, gamma_fussed), axis=1)
            
            # cnn model
            cnn_model = models.CNN_2layers_adaptive_maxpool_3()
            
            # training and testing
            # test
            # result_RCM = cnn_validation.cnn_cross_validation(cnn_model, x_rebuild, y)
            result_RCM = cnn_validation.cnn_sequential_validation(cnn_model, x_rebuild, y)            
            
            # Flatten result and add identifier
            result_flat = {'Identifier': subject_id, **result_RCM}
            all_results_list.append(result_flat)
            
    # Convert list of dicts to DataFrame
    df_results = pd.DataFrame(all_results_list)
    
    # Compute mean of all numeric columns (excluding Identifier)
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['Identifier'] = 'Average'
    
    # Std
    std_row = df_results.select_dtypes(include=[np.number]).std(ddof=0).to_dict()
    std_row['Identifier'] = 'Std'
    
    df_results = pd.concat([df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # Save
    if save:
        fusion_type = params.get('fusion_type', None).lower()
        if fusion_type == 'sigmoid_gating':
            folder_name = f'results_(stress_test)_{feature_basis.upper()}xSigmoid-{feature_modifier.upper()}-'
            params_desired = {'k': params['k'],
                              'p': params['percentile'],
                              'nm_basis': params['normalization_basis'],
                              'nm_modifier': params['normalization_modifier']}
        elif fusion_type == 'power_gating':
            folder_name = f'results_(stress_test)_{feature_basis.upper()}xSigmoid-{feature_modifier.upper()}-'
            params_desired = {'power': params['power'],
                              'nm_basis': params['normalization_basis'],
                              'nm_modifier': params['normalization_modifier']}
        elif fusion_type in {'triangle_blocking', 
                             'diagonal_blocking',
                             'additive', 'multiplicative'}:
            folder_name = 'results_(stress_test)_baselines'
            params_desired = {'type': fusion_type,
                              'nm_basis': params['normalization_basis'],
                              'nm_modifier': params['normalization_modifier']}
        else:
            raise ValueError("'Fusion Type' Error")
        
        suffix = "_".join(f"{k}-{v}" for k, v in params_desired.items())
        file_name = f"cnn_evaluation(stress_test)_{suffix}.xlsx"

        sheet_name = f'nrr_{node_retention_rate}'
        
        save_to_xlsx_sheet(df_results, folder_name, file_name, sheet_name)
        
        # Save Summary
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')
    
    return df_results

# %% Execute
def normal_evaluation_framework():
    # node retention rates
    nrr_list = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    nrr_list = [0.75]
    for nrr in nrr_list:
        # %% baseline: original functional networks
        cnn_subnetworks_evaluation_circle_original_cm(feature_cm='pli', # 'pcc', 'plv' or 'pli'
                                                      normalization_for_train=False, # always False
                                                      subject_range=range(6,16), experiment_range=range(1,4), 
                                                      node_retention_rate=nrr, 
                                                      subnetworks_extract='read', subnetworks_extract_basis=range(1,6),
                                                      save=True) # switch to True
        
        # -----------------------------------------------------------------------
        
        # %% competitors: additive, multiplicative, triangle_blocking, diagonal_blocking
        # cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis='pcc', # always 'pcc'
        #                                                  feature_modifier='plv', # 'plv' or 'pli'
        #                                                  params={'fusion_type': 'additive', 
        #                                                          # 'additive', 'multiplicative', 'triangle_blocking' or 'diagonal_blocking'
        #                                                          'normalization_basis': True, # 'addtive': True; others: False
        #                                                          'normalization_modifier': False, # always False
        #                                                          'scale': (0, 1)},
        #                                                  normalization_for_train=False, # always False 
        #                                                  subject_range=range(6,16), experiment_range=range(1,4),
        #                                                  subnetworks_extract='unify_index', node_retention_rate=nrr,
        #                                                  subnets_extract_basis_sub=range(1, 6), subnets_extract_basis_ex=range(1, 4),
        #                                                  save=False) # switch to True
        
        # -----------------------------------------------------------------------
        
        # %% Proposed Methods: PCCxSigmoid(PLV) or PCCxSigmoid(PLI)
        # params={'fusion_type': 'sigmoid_gating', # always 'sigmoid_gating'
        #         'k': None, # waiting for assignment
        #         'percentile': 30, # value 30 is recommended
        #         'normalization_basis': False, # True or False, depended on experiments
        #         'normalization_modifier': False} # always False
        
        # params['k'] = 'heaviside' # 'heaviside'or values ranges of [10, 200]
        # cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis='pcc', # always 'pcc'
        #                                                  feature_modifier='pli', # 'plv' or 'pli'
        #                                                  params=params,
        #                                                  normalization_for_train=False, # always False
        #                                                  subject_range=range(6,16), experiment_range=range(1,4),
        #                                                  subnetworks_extract='separate_index', # 'separate_index' is recommended
        #                                                  node_retention_rate=nrr, 
        #                                                  subnets_extract_basis_sub=range(1,6), subnets_extract_basis_ex=range(1,4),
        #                                                  save=True) # switch to True
        
        # ----------------------------------------------------------------------

        # %% Mirrors: PLVxSigmoid(PCC) or PLIxSigmoid(PCC)
        # params={'fusion_type': 'sigmoid_gating', # always 'sigmoid_gating'
        #         'k': None, # waiting for assignment
        #         'percentile': 30, # value 30 is recommended
        #         'normalization_basis': False, # always False
        #         'normalization_modifier': False} # True or False, depended on experiments
        
        # k = ['heaviside', 10, 20, 50, 200]
        # for k_ in k:
        #     params['k'] = k_
        #     cnn_subnetworks_evaluation_circle_feature_fusion(feature_basis='plv', # 'plv' or 'pli'
        #                                                      feature_modifier='pcc', # always 'pcc'
        #                                                      params=params,
        #                                                      normalization_for_train=False, # always False
        #                                                      subject_range=range(6,16), experiment_range=range(1,4),
        #                                                      subnetworks_extract='separate_index', # 'separate_index' is recommended
        #                                                      node_retention_rate=nrr, 
        #                                                      subnets_extract_basis_sub=range(1,6), subnets_extract_basis_ex=range(1,4),
        #                                                      save=True) # switch to True
        
        # ----------------------------------------------------------------------

# %% Execution
if __name__ == '__main__':
    normal_evaluation_framework()
    
    # end
    utils_tools.end_program_actions(play_sound=True, shutdown=False, countdown_seconds=120)