#!/usr/bin/env python3
"""
XGBoost and Naive Bayes Analysis for Buffalo Breed Classification
Using Gradient Boosting as XGBoost alternative (scikit-learn implementation)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            matthews_corrcoef, roc_auc_score, confusion_matrix,
                            classification_report)
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

def extract_genotypes(bed_prefix, output_prefix):
    """Extract genotype data using PLINK"""
    print(f"Extracting genotypes from {bed_prefix}...")
    
    cmd = f"plink --bfile {bed_prefix} --recode A --out {output_prefix}"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    raw_file = f"{output_prefix}.raw"
    df = pd.read_csv(raw_file, sep=r'\s+')
    
    meta_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_cols = [c for c in df.columns if c not in meta_cols]
    
    y = df['FID'].values
    X = df[snp_cols].fillna(-1).values
    
    return X, y, snp_cols, df[['FID', 'IID']]

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
    """Evaluate classifier with comprehensive metrics"""
    
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
    
    n_classes = len(le.classes_)
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['specificity'] = calculate_specificity(y_test, y_pred)
    metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
    
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
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    metrics['cv_accuracy_mean'] = cv_scores.mean()
    metrics['cv_accuracy_std'] = cv_scores.std()
    
    return metrics, y_pred, y_prob

def test_with_different_snps(X, y, snp_names, snp_counts, output_dir):
    """Test classifiers with different numbers of top SNPs"""
    
    results = []
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Get SNP importance using Gradient Boosting
    print("\nCalculating SNP importance using Gradient Boosting...")
    gb_temp = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_temp.fit(X, y_encoded)
    
    importance_df = pd.DataFrame({
        'snp': snp_names,
        'importance': gb_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(f"{output_dir}/snp_importance_ranking.csv", index=False)
    
    print(f"Top 10 most important SNPs:")
    print(importance_df.head(10).to_string(index=False))
    
    for n_snps in snp_counts:
        print(f"\n{'='*60}")
        print(f"Testing with top {n_snps} SNPs")
        print('='*60)
        
        # Select top N SNPs
        top_snps = importance_df.head(n_snps)['snp'].tolist()
        selected_indices = [snp_names.index(snp) for snp in top_snps if snp in snp_names]
        X_selected = X[:, selected_indices]
        
        print(f"Selected {X_selected.shape[1]} SNPs")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale for Naive Bayes
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. Gradient Boosting (XGBoost alternative)
        print(f"\n  Gradient Boosting (XGBoost-like):")
        gb_params = {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        }
        gb = GradientBoostingClassifier(**gb_params)
        gb_metrics, gb_pred, gb_prob = evaluate_classifier(gb, X_train, X_test, y_train, y_test, le)
        
        for metric, value in gb_metrics.items():
            print(f"    {metric}: {value:.4f}")
        
        results.append({
            'n_snps': n_snps,
            'classifier': 'GradientBoosting',
            **gb_metrics
        })
        
        # 2. Naive Bayes
        print(f"\n  Naive Bayes:")
        nb = GaussianNB()
        nb_metrics, nb_pred, nb_prob = evaluate_classifier(nb, X_train_scaled, X_test_scaled, y_train, y_test, le)
        
        for metric, value in nb_metrics.items():
            print(f"    {metric}: {value:.4f}")
        
        results.append({
            'n_snps': n_snps,
            'classifier': 'NaiveBayes',
            **nb_metrics
        })
        
        # Save predictions for best configuration
        if n_snps == 50:  # Save detailed results for mid-range SNP count
            pred_df = pd.DataFrame({
                'true_label': le.inverse_transform(y_test),
                'gb_pred': le.inverse_transform(gb_pred),
                'nb_pred': le.inverse_transform(nb_pred)
            })
            pred_df.to_csv(f"{output_dir}/predictions_{n_snps}snps.csv", index=False)
    
    return results, importance_df

def compare_with_all_snps(X, y, le, output_dir):
    """Compare classifiers using all SNPs with cross-validation"""
    
    print("\n" + "="*60)
    print("Comparison using ALL SNPs (1997)")
    print("="*60)
    
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    classifiers = {
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        'NaiveBayes': GaussianNB()
    }
    
    for name, clf in classifiers.items():
        print(f"\n{name}:")
        
        if name == 'NaiveBayes':
            metrics, y_pred, y_prob = evaluate_classifier(clf, X_train_scaled, X_test_scaled, y_train, y_test, le)
        else:
            metrics, y_pred, y_prob = evaluate_classifier(clf, X_train, X_test, y_train, y_test, le)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  MCC: {metrics['mcc']:.4f}")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  CV Accuracy: {metrics['cv_accuracy_mean']:.4f} (+/- {metrics['cv_accuracy_std']*2:.4f})")
        
        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))
        
        results.append({
            'classifier': name,
            'n_snps': X.shape[1],
            **metrics
        })
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
        cm_df.to_csv(f"{output_dir}/confusion_matrix_{name}_all_snps.csv")
    
    return results

def main():
    print("="*80)
    print("XGBoost (Gradient Boosting) and Naive Bayes Analysis")
    print("="*80)
    
    # Create output directory
    output_dir = "xgboost_naivebayes_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Input data
    bed_prefix = "refined_panel/refined_ref_panel"
    temp_prefix = f"{output_dir}/genotypes"
    
    # Extract genotypes
    X, y, snp_names, ind_info = extract_genotypes(bed_prefix, temp_prefix)
    
    print(f"\nDataset:")
    print(f"  Samples: {X.shape[0]}")
    print(f"  SNPs: {X.shape[1]}")
    print(f"  Breeds: {np.unique(y)}")
    
    breed_counts = pd.Series(y).value_counts()
    print(f"\nBreed distribution:")
    for breed, count in breed_counts.items():
        print(f"  {breed}: {count}")
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"\nClasses: {le.classes_}")
    
    # Test with different numbers of SNPs
    snp_counts = [10, 20, 50, 100, 200, X.shape[1]]
    results_snps, importance_df = test_with_different_snps(X, y, snp_names, snp_counts, output_dir)
    
    # Compare with all SNPs (detailed)
    results_all = compare_with_all_snps(X, y, le, output_dir)
    
    # Combine results
    all_results = results_snps + results_all
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{output_dir}/xgboost_naivebayes_results.csv", index=False)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: XGBoost and Naive Bayes Performance")
    print("="*80)
    
    print("\nGradient Boosting Results:")
    gb_results = results_df[results_df['classifier'] == 'GradientBoosting'].sort_values('accuracy', ascending=False)
    print(gb_results[['n_snps', 'accuracy', 'f1_score', 'mcc', 'auc']].to_string(index=False))
    
    print("\nNaive Bayes Results:")
    nb_results = results_df[results_df['classifier'] == 'NaiveBayes'].sort_values('accuracy', ascending=False)
    print(nb_results[['n_snps', 'accuracy', 'f1_score', 'mcc', 'auc']].to_string(index=False))
    
    # Find best configurations
    best_gb = gb_results.iloc[0]
    best_nb = nb_results.iloc[0]
    
    print(f"\n" + "="*80)
    print("BEST CONFIGURATIONS:")
    print("="*80)
    print(f"\nGradient Boosting:")
    print(f"  SNPs: {best_gb['n_snps']}")
    print(f"  Accuracy: {best_gb['accuracy']:.4f}")
    print(f"  F1-Score: {best_gb['f1_score']:.4f}")
    print(f"  MCC: {best_gb['mcc']:.4f}")
    print(f"  AUC: {best_gb['auc']:.4f}")
    
    print(f"\nNaive Bayes:")
    print(f"  SNPs: {best_nb['n_snps']}")
    print(f"  Accuracy: {best_nb['accuracy']:.4f}")
    print(f"  F1-Score: {best_nb['f1_score']:.4f}")
    print(f"  MCC: {best_nb['mcc']:.4f}")
    print(f"  AUC: {best_nb['auc']:.4f}")
    
    # Comparison with previous best (from main analysis)
    print("\n" + "="*80)
    print("COMPARISON WITH PREVIOUS BEST (Hard Voting Ensemble):")
    print("="*80)
    print(f"Previous best: 15 BIMs with 100% accuracy")
    print(f"Gradient Boosting: {best_gb['accuracy']*100:.1f}% accuracy with {best_gb['n_snps']} SNPs")
    print(f"Naive Bayes: {best_nb['accuracy']*100:.1f}% accuracy with {best_nb['n_snps']} SNPs")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}/")
    print("="*80)
    print("\nFiles generated:")
    print(f"  - {output_dir}/xgboost_naivebayes_results.csv")
    print(f"  - {output_dir}/snp_importance_ranking.csv")
    print(f"  - {output_dir}/predictions_50snps.csv")
    print(f"  - {output_dir}/confusion_matrix_*.csv")

if __name__ == "__main__":
    main()
