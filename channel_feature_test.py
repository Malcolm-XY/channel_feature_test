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
        label = utils_feature_loading.read_labels(dataset, header=True)["valence"]
    elif dataset.lower() == "dreamer":    
        label = utils_feature_loading.read_labels(dataset, header=True)[category]
    
    # Classification
    acc, confusion_matrix_, report = svm_validation(samples, label, ratio_train)
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

def svm_validation(feature, label, ratio_train=0.7):
    indices_train = list(range(0,int(np.ceil(ratio_train * len(feature))),1))
    indices_test = list(range(int(np.ceil(ratio_train * len(feature))), int(len(feature)), 1))

    feature_train = feature[indices_train]
    feature_test = feature[indices_test]

    label_train = label[indices_train]
    label_test = label[indices_test]
    
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

    svm_model.fit(feature_train, label_train)

    label_pred = svm_model.predict(feature_test)


    # =========================
    # Evaluation
    # =========================
    acc = accuracy_score(label_test, label_pred)
    confusion_matrix_ = confusion_matrix(label_test, label_pred)
    report = classification_report(label_test, label_pred)
    
    print(f"Train samples: {len(feature_train)}")
    print(f"Test samples : {len(feature_test)}")
    print(f"Accuracy     : {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix_)

    print("\nClassification Report:")
    print(report)
    
    return acc, confusion_matrix_, report

# Test
accuracy_avg, summary_results = svm_validation_compact_circle("de", subject_range=range(6,16))