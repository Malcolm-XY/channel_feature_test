# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:01:19 2026

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
                                                  partition_ratio="cross_validation",
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
    y = torch.tensor(np.array(labels)).view(-1)
    
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
            if partition_ratio == "cross_validation":
                result_CM = cnn_validation.cnn_cross_validation(cnn_model, x_selected, y)
            elif isinstance(partition_ratio, float):
                result_CM = cnn_validation.cnn_validation(cnn_model, x_selected, y, partition_ratio=partition_ratio)
            
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
        
        # Save Summary
        df_summary = pd.DataFrame([mean_row, std_row])
        save_to_xlsx_sheet(df_summary, folder_name, file_name, 'summary')
        
    return df_results