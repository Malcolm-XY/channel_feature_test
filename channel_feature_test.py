# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:11:08 2026

@author: usouu
"""
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils import utils_feature_loading

def svm_validation_compact(dataset, identifier, feature_type="psd", ratio_train=0.7, category="valence"):
    # Samples
    feature = utils_feature_loading.read_cfs(dataset, identifier, feature_type)
    alpha, beta, gamma = feature["alpha"], feature["beta"], feature["gamma"]
    samples = np.hstack([alpha, beta, gamma])
    
    # Label
    if dataset.lower() == "seed":
        labels = utils_feature_loading.read_labels(dataset, header=True)["valence"]
    elif dataset.lower() == "dreamer":    
        labels = utils_feature_loading.read_labels(dataset, header=True)[category]
    
    # Classification
    acc, confusion_matrix_, report = svm_validation(samples, labels, ratio_train)
    summary = {
        "accuracy": acc,
        "confusion_matrix": confusion_matrix_,
        "report": report,
        }
    return summary
    
def svm_validation_compact_circle(feature_type, subject_range=range(6,16), experiment_range=range(1,4), save=False):
    # Inherent info
    dataset = "seed"

    # Data and evaluation circle
    summary_results = []
    for sub in subject_range:
        for ex in experiment_range:
            identifier = f"sub{sub}ex{ex}"
            print(f"Evaluating {identifier}...")
            
            result = svm_validation_compact(dataset, identifier, feature_type)
            summary_results.append(result)
    
    accuracies = []
    for result in summary_results:
        accuracy = result["accuracy"]
        accuracies.append(accuracy)
    accuracy_avg = np.mean(accuracies)
    print(f"The Average Accuracy: {accuracy_avg}")
    
    return accuracy_avg, summary_results

def svm_validation(samples, labels, ratio_train=0.7):
    indices_train = list(range(0,int(np.ceil(ratio_train * len(samples))),1))
    indices_test = list(range(int(np.ceil(ratio_train * len(samples))), int(len(samples)), 1))

    samples_train = samples[indices_train]
    samples_test = samples[indices_test]

    labels_train = labels[indices_train]
    labels_test = labels[indices_test]
    
    # =========================
    # SVM training and classification
    # =========================
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            decision_function_shape="ovr"
        ))
    ])

    svm_model.fit(samples_train, labels_train)

    labels_pred = svm_model.predict(samples_test)


    # =========================
    # Evaluation
    # =========================
    acc = accuracy_score(labels_test, labels_pred)
    confusion_matrix_ = confusion_matrix(labels_test, labels_pred)
    report = classification_report(labels_test, labels_pred)
    
    print(f"Train samples: {len(samples_train)}")
    print(f"Test samples : {len(samples_test)}")
    print(f"Accuracy     : {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix_)

    print("\nClassification Report:")
    print(report)
    
    return acc, confusion_matrix_, report

# Test; SVM
accuracy_avg, summary_results = svm_validation_compact_circle("de_lds", subject_range=range(1,2))

# Test; CNN
import torch
import cnn_validation

from models import models

labels = utils_feature_loading.read_labels(dataset="seed", header=True)
labels = torch.tensor(np.array(labels)).view(-1)

# 
feature = utils_feature_loading.read_cfs("seed", "sub1ex1", "de_lds")
alpha, beta, gamma = feature["alpha"], feature["beta"], feature["gamma"]
alpha, beta, gamma = [alpha] * 62, [beta] * 62, [gamma] * 62
alpha, beta, gamma = np.stack(alpha, axis=2), np.stack(beta, axis=2), np.stack(gamma, axis=2)
samples = np.stack([alpha,beta,gamma], axis=1)

# Model
# len_samples, len_features = samples.shape
# cnn_model = models.FC_2layers(input_len=len_features)
cnn_model = models.CNN_2layers_adaptive_maxpool_3()
result_CM_1 = cnn_validation.cnn_validation(cnn_model, samples, labels)

cnn_model = models.CNN_2layers_adaptive_maxpool_3()
result_CM_2 = cnn_validation.cnn_validation_reverse_division(cnn_model, samples, labels)