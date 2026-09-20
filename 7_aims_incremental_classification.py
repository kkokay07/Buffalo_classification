#!/usr/bin/env python3
"""
Part a: Classification with top 10, 20, 30, ..., 200 AIMs
Using SVM and MLP with comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            matthews_corrcoef, roc_auc_score, confusion_matrix)
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

def extract_genotypes_for_snps(bed_prefix, snp_list, output_prefix):
    """Extract genotype data for specific SNPs using PLINK"""
    
    # Write SNP list to file
    snp_file = f"{output_prefix}_snps.txt"
    with open(snp_file, 'w') as f:
        for snp in snp_list:
            f.write(f"{snp}\n")
    
    # Extract using PLINK
    cmd = f"plink --bfile {bed_prefix} --extract {snp_file} --recode A --out {output_prefix}"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # Read raw file
    raw_file = f"{output_prefix}.raw"
    if not os.path.exists(raw_file):
        return None, None, None
    
    df = pd.read_csv(raw_file, sep=r'\s+')
    
    # Extract metadata and genotypes
    meta_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_cols = [c for c in df.columns if c not in meta_cols]
    
    y = df['FID'].values
    X = df[snp_cols].fillna(-1).values
    
    return X, y, snp_cols

def calculate_specificity(y_true, y_pred):
    """Calculate specificity for multi-class"""
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(len(cm)):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        if tn + fp > 0:
            specificity_per_class.append(tn / (tn + fp))
        else:
            specificity_per_class.append(0)
    return np.mean(specificity_per_class)

def evaluate_classifier(clf, X_train, X_test, y_train, y_test, le, cv_folds=5):
    """Evaluate classifier with all metrics"""
    
    # Train
    clf.fit(X_train, y_train)
    
    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
    
    n_classes = len(le.classes_)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['specificity'] = calculate_specificity(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    
    # AUC
    if y_prob is not None:
        try:
            y_test_bin = np.zeros((len(y_test), n_classes))
            for i, label in enumerate(y_test):
                y_test_bin[i, label] = 1
            metrics['auc'] = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    return metrics

def main():
    print("="*80)
    print("Part a: Incremental AIM Classification (SVM and MLP)")
    print("="*80)
    
    os.makedirs("incremental_aims", exist_ok=True)
    
    # Read combined AIM rankings
    aim_df = pd.read_csv("aim_results/top_aims_combined.csv")
    aim_df = aim_df.sort_values('avg_rank')
    
    # Input data
    bed_prefix = "refined_panel/refined_ref_panel"
    
    # Test different numbers of top AIMs
    aim_counts = list(range(10, 210, 10))  # 10, 20, 30, ..., 200
    
    results = []
    
    print(f"\nTesting with {len(aim_counts)} different AIM counts: {aim_counts[:5]}...{aim_counts[-5:]}")
    
    for n_aims in aim_counts:
        print(f"\n{'='*60}")
        print(f"Testing with top {n_aims} AIMs")
        print('='*60)
        
        # Get top N AIMs
        top_aims = aim_df.head(n_aims)['snp'].tolist()
        
        # Extract genotypes
        output_prefix = f"incremental_aims/aims_{n_aims}"
        X, y, snp_cols = extract_genotypes_for_snps(bed_prefix, top_aims, output_prefix)
        
        if X is None:
            print(f"Warning: Could not extract genotypes for {n_aims} AIMs")
            continue
        
        print(f"Extracted {X.shape[1]} SNPs for {X.shape[0]} individuals")
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. SVM Linear
        print(f"\n  SVM Linear:")
        svm = SVC(kernel='linear', probability=True, random_state=42)
        svm_metrics = evaluate_classifier(svm, X_train_scaled, X_test_scaled, y_train, y_test, le)
        
        for metric, value in svm_metrics.items():
            print(f"    {metric}: {value:.4f}")
        
        results.append({
            'n_AIMs': n_aims,
            'Classifier': 'SVM_Linear',
            **svm_metrics
        })
        
        # 2. MLP
        print(f"\n  MLP:")
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        mlp_metrics = evaluate_classifier(mlp, X_train_scaled, X_test_scaled, y_train, y_test, le)
        
        for metric, value in mlp_metrics.items():
            print(f"    {metric}: {value:.4f}")
        
        results.append({
            'n_AIMs': n_aims,
            'Classifier': 'MLP',
            **mlp_metrics
        })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("incremental_aims/incremental_classification_results.csv", index=False)
    
    # Find best configurations
    print("\n" + "="*80)
    print("SUMMARY: Best Configurations")
    print("="*80)
    
    # Best for SVM
    svm_results = results_df[results_df['Classifier'] == 'SVM_Linear']
    best_svm = svm_results.loc[svm_results['accuracy'].idxmax()]
    print(f"\nBest SVM: {best_svm['n_AIMs']} AIMs")
    print(f"  Accuracy: {best_svm['accuracy']:.4f}")
    print(f"  F1-Score: {best_svm['f1_score']:.4f}")
    print(f"  MCC: {best_svm['mcc']:.4f}")
    print(f"  AUC: {best_svm['auc']:.4f}")
    
    # Best for MLP
    mlp_results = results_df[results_df['Classifier'] == 'MLP']
    best_mlp = mlp_results.loc[mlp_results['accuracy'].idxmax()]
    print(f"\nBest MLP: {best_mlp['n_AIMs']} AIMs")
    print(f"  Accuracy: {best_mlp['accuracy']:.4f}")
    print(f"  F1-Score: {best_mlp['f1_score']:.4f}")
    print(f"  MCC: {best_mlp['mcc']:.4f}")
    print(f"  AUC: {best_mlp['auc']:.4f}")
    
    # Save best AIM lists
    best_n_aims_svm = int(best_svm['n_AIMs'])
    best_n_aims_mlp = int(best_mlp['n_AIMs'])
    
    best_aims_svm = aim_df.head(best_n_aims_svm)['snp'].tolist()
    best_aims_mlp = aim_df.head(best_n_aims_mlp)['snp'].tolist()
    
    with open("incremental_aims/best_aims_svm.txt", 'w') as f:
        for snp in best_aims_svm:
            f.write(f"{snp}\n")
    
    with open("incremental_aims/best_aims_mlp.txt", 'w') as f:
        for snp in best_aims_mlp:
            f.write(f"{snp}\n")
    
    print(f"\nBest SVM AIMs saved: incremental_aims/best_aims_svm.txt ({best_n_aims_svm} SNPs)")
    print(f"Best MLP AIMs saved: incremental_aims/best_aims_mlp.txt ({best_n_aims_mlp} SNPs)")
    
    # Create summary plot data
    print("\n" + "="*80)
    print("Results saved to incremental_aims/incremental_classification_results.csv")
    print("="*80)
    
    return best_n_aims_svm, best_n_aims_mlp, best_aims_svm, best_aims_mlp

if __name__ == "__main__":
    main()
