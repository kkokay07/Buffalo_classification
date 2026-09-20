#!/usr/bin/env python3
"""
Step 6 & 7: Machine Learning Classifiers and Ensembling for Breed Classification

Test set: 20%, Train set: 80%
5-fold cross-validation
Metrics: Accuracy, Precision, Recall, Specificity, F-score, MCC, AUC
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            matthews_corrcoef, roc_auc_score, confusion_matrix,
                            classification_report)
from sklearn.multiclass import OneVsRestClassifier
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

def extract_genotypes(bed_prefix, output_prefix):
    """Extract genotype data using PLINK"""
    print(f"Extracting genotypes from {bed_prefix}...")
    
    # Convert to raw format
    cmd = f"plink --bfile {bed_prefix} --recode A --out {output_prefix}"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    # Read raw file
    raw_file = f"{output_prefix}.raw"
    df = pd.read_csv(raw_file, sep=r'\s+')
    
    # Extract metadata and genotypes
    meta_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_cols = [c for c in df.columns if c not in meta_cols]
    
    # Get breed labels (FID)
    y = df['FID'].values
    
    # Get genotype matrix
    X = df[snp_cols].fillna(-1).values  # -1 for missing
    
    # Get SNP names
    snp_names = snp_cols
    
    return X, y, snp_names, df[['FID', 'IID']]

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

def calculate_metrics(y_true, y_pred, y_prob=None, n_classes=None):
    """Calculate all evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['specificity'] = calculate_specificity(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # AUC (One-vs-Rest)
    if y_prob is not None and n_classes is not None:
        try:
            # Binarize labels for multi-class AUC
            y_true_bin = np.zeros((len(y_true), n_classes))
            for i, label in enumerate(y_true):
                y_true_bin[i, label] = 1
            metrics['auc'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    return metrics

def get_classifiers():
    """Define all classifiers to test"""
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=200, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'SVM_RBF': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM_Linear': SVC(kernel='linear', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(),
        'DecisionTree': DecisionTreeClassifier(max_depth=20, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    return classifiers

def train_and_evaluate(X, y, classifiers, test_size=0.2, cv_folds=5):
    """Train and evaluate all classifiers"""
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f"Classes: {le.classes_}")
    print(f"Number of classes: {n_classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features for some classifiers
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    trained_models = {}
    
    print("\n" + "="*80)
    print("Training and Evaluating Classifiers")
    print("="*80)
    
    for name, clf in classifiers.items():
        print(f"\n{name}:")
        print("-" * 40)
        
        # Use scaled data for SVM, KNN, LogisticRegression, MLP, NaiveBayes
        if name in ['SVM_RBF', 'SVM_Linear', 'KNN', 'LogisticRegression', 'MLP', 'NaiveBayes']:
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train
            X_te = X_test
        
        # Train model
        clf.fit(X_tr, y_train)
        trained_models[name] = (clf, scaler if name in ['SVM_RBF', 'SVM_Linear', 'KNN', 'LogisticRegression', 'MLP', 'NaiveBayes'] else None)
        
        # Predictions
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te) if hasattr(clf, 'predict_proba') else None
        
        # Calculate metrics on test set
        test_metrics = calculate_metrics(y_test, y_pred, y_prob, n_classes)
        
        # 5-fold cross-validation on training set
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy')
        
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  Specificity: {test_metrics['specificity']:.4f}")
        print(f"  F1-score: {test_metrics['f1_score']:.4f}")
        print(f"  MCC: {test_metrics['mcc']:.4f}")
        print(f"  AUC: {test_metrics['auc']:.4f}")
        
        # Store results
        result = {
            'Classifier': name,
            'Test_Accuracy': test_metrics['accuracy'],
            'CV_Accuracy_Mean': cv_scores.mean(),
            'CV_Accuracy_Std': cv_scores.std(),
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'Specificity': test_metrics['specificity'],
            'F1_Score': test_metrics['f1_score'],
            'MCC': test_metrics['mcc'],
            'AUC': test_metrics['auc']
        }
        results.append(result)
    
    return results, trained_models, le, (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)

def create_ensembles(trained_models, X_train, X_test, y_train, y_test, le, X_train_scaled, X_test_scaled):
    """Create and evaluate ensemble methods"""
    
    print("\n" + "="*80)
    print("Ensembling Methods")
    print("="*80)
    
    n_classes = len(le.classes_)
    ensemble_results = []
    
    # Get top 3 models based on test accuracy for voting
    top_models = []
    for name, (clf, scaler) in trained_models.items():
        if scaler is not None:
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)
        else:
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        top_models.append((name, clf, scaler, acc))
    
    top_models.sort(key=lambda x: x[3], reverse=True)
    top_3 = top_models[:3]
    
    print(f"\nTop 3 models for voting: {[m[0] for m in top_3]}")
    
    # 1. Hard Voting (Majority Voting)
    print("\n1. Hard Voting (Majority Voting)")
    estimators = [(name, clf) for name, clf, _, _ in top_3]
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train, y_train)
    y_pred_hard = voting_hard.predict(X_test)
    
    # Calculate metrics
    metrics_hard = calculate_metrics(y_test, y_pred_hard, None, n_classes)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(voting_hard, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"  Test Accuracy: {metrics_hard['accuracy']:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f}")
    print(f"  Precision: {metrics_hard['precision']:.4f}")
    print(f"  Recall: {metrics_hard['recall']:.4f}")
    print(f"  Specificity: {metrics_hard['specificity']:.4f}")
    print(f"  F1-score: {metrics_hard['f1_score']:.4f}")
    print(f"  MCC: {metrics_hard['mcc']:.4f}")
    
    ensemble_results.append({
        'Ensemble_Method': 'Hard_Voting',
        'Test_Accuracy': metrics_hard['accuracy'],
        'CV_Accuracy_Mean': cv_scores.mean(),
        'CV_Accuracy_Std': cv_scores.std(),
        'Precision': metrics_hard['precision'],
        'Recall': metrics_hard['recall'],
        'Specificity': metrics_hard['specificity'],
        'F1_Score': metrics_hard['f1_score'],
        'MCC': metrics_hard['mcc'],
        'AUC': metrics_hard['auc']
    })
    
    # 2. Soft Voting (Weighted by probability)
    print("\n2. Soft Voting (Probability Averaging)")
    
    # For soft voting, we need models with predict_proba
    estimators_soft = []
    for name, clf, scaler, _ in top_3:
        if hasattr(clf, 'predict_proba'):
            estimators_soft.append((name, clf))
    
    if len(estimators_soft) >= 2:
        voting_soft = VotingClassifier(estimators=estimators_soft, voting='soft')
        voting_soft.fit(X_train, y_train)
        y_pred_soft = voting_soft.predict(X_test)
        y_prob_soft = voting_soft.predict_proba(X_test)
        
        metrics_soft = calculate_metrics(y_test, y_pred_soft, y_prob_soft, n_classes)
        cv_scores_soft = cross_val_score(voting_soft, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"  Test Accuracy: {metrics_soft['accuracy']:.4f}")
        print(f"  CV Accuracy: {cv_scores_soft.mean():.4f}")
        print(f"  Precision: {metrics_soft['precision']:.4f}")
        print(f"  Recall: {metrics_soft['recall']:.4f}")
        print(f"  Specificity: {metrics_soft['specificity']:.4f}")
        print(f"  F1-score: {metrics_soft['f1_score']:.4f}")
        print(f"  MCC: {metrics_soft['mcc']:.4f}")
        print(f"  AUC: {metrics_soft['auc']:.4f}")
        
        ensemble_results.append({
            'Ensemble_Method': 'Soft_Voting',
            'Test_Accuracy': metrics_soft['accuracy'],
            'CV_Accuracy_Mean': cv_scores_soft.mean(),
            'CV_Accuracy_Std': cv_scores_soft.std(),
            'Precision': metrics_soft['precision'],
            'Recall': metrics_soft['recall'],
            'Specificity': metrics_soft['specificity'],
            'F1_Score': metrics_soft['f1_score'],
            'MCC': metrics_soft['mcc'],
            'AUC': metrics_soft['auc']
        })
    
    # 3. Stacking (Meta-classifier)
    print("\n3. Stacking (Logistic Regression Meta-Classifier)")
    
    # Generate predictions from base models as features
    meta_features_train = []
    meta_features_test = []
    
    for name, clf, scaler, _ in top_3:
        if scaler is not None:
            prob_train = clf.predict_proba(X_train_scaled)
            prob_test = clf.predict_proba(X_test_scaled)
        else:
            prob_train = clf.predict_proba(X_train)
            prob_test = clf.predict_proba(X_test)
        meta_features_train.append(prob_train)
        meta_features_test.append(prob_test)
    
    X_meta_train = np.hstack(meta_features_train)
    X_meta_test = np.hstack(meta_features_test)
    
    # Train meta-classifier
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(X_meta_train, y_train)
    y_pred_meta = meta_clf.predict(X_meta_test)
    y_prob_meta = meta_clf.predict_proba(X_meta_test)
    
    metrics_meta = calculate_metrics(y_test, y_pred_meta, y_prob_meta, n_classes)
    
    # CV for stacking (simplified)
    cv_scores_meta = cross_val_score(meta_clf, X_meta_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"  Test Accuracy: {metrics_meta['accuracy']:.4f}")
    print(f"  CV Accuracy: {cv_scores_meta.mean():.4f}")
    print(f"  Precision: {metrics_meta['precision']:.4f}")
    print(f"  Recall: {metrics_meta['recall']:.4f}")
    print(f"  Specificity: {metrics_meta['specificity']:.4f}")
    print(f"  F1-score: {metrics_meta['f1_score']:.4f}")
    print(f"  MCC: {metrics_meta['mcc']:.4f}")
    print(f"  AUC: {metrics_meta['auc']:.4f}")
    
    ensemble_results.append({
        'Ensemble_Method': 'Stacking_LR',
        'Test_Accuracy': metrics_meta['accuracy'],
        'CV_Accuracy_Mean': cv_scores_meta.mean(),
        'CV_Accuracy_Std': cv_scores_meta.std(),
        'Precision': metrics_meta['precision'],
        'Recall': metrics_meta['recall'],
        'Specificity': metrics_meta['specificity'],
        'F1_Score': metrics_meta['f1_score'],
        'MCC': metrics_meta['mcc'],
        'AUC': metrics_meta['auc']
    })
    
    # 4. Weighted Average Ensemble
    print("\n4. Weighted Average Ensemble (by CV accuracy)")
    
    # Calculate weights based on CV performance
    weights = np.array([acc for _, _, _, acc in top_3])
    weights = weights / weights.sum()
    
    # Average probabilities
    avg_prob = np.zeros((X_test.shape[0], n_classes))
    for i, (name, clf, scaler, _) in enumerate(top_3):
        if scaler is not None:
            prob = clf.predict_proba(X_test_scaled)
        else:
            prob = clf.predict_proba(X_test)
        avg_prob += weights[i] * prob
    
    y_pred_weighted = np.argmax(avg_prob, axis=1)
    metrics_weighted = calculate_metrics(y_test, y_pred_weighted, avg_prob, n_classes)
    
    print(f"  Weights: {dict([(top_3[i][0], weights[i]) for i in range(len(top_3))])}")
    print(f"  Test Accuracy: {metrics_weighted['accuracy']:.4f}")
    print(f"  Precision: {metrics_weighted['precision']:.4f}")
    print(f"  Recall: {metrics_weighted['recall']:.4f}")
    print(f"  Specificity: {metrics_weighted['specificity']:.4f}")
    print(f"  F1-score: {metrics_weighted['f1_score']:.4f}")
    print(f"  MCC: {metrics_weighted['mcc']:.4f}")
    print(f"  AUC: {metrics_weighted['auc']:.4f}")
    
    ensemble_results.append({
        'Ensemble_Method': 'Weighted_Average',
        'Test_Accuracy': metrics_weighted['accuracy'],
        'CV_Accuracy_Mean': np.nan,  # Not applicable
        'CV_Accuracy_Std': np.nan,
        'Precision': metrics_weighted['precision'],
        'Recall': metrics_weighted['recall'],
        'Specificity': metrics_weighted['specificity'],
        'F1_Score': metrics_weighted['f1_score'],
        'MCC': metrics_weighted['mcc'],
        'AUC': metrics_weighted['auc']
    })
    
    return ensemble_results

def main():
    print("="*80)
    print("Step 6 & 7: Machine Learning Classifiers and Ensembling")
    print("="*80)
    
    # Create output directory
    os.makedirs("ml_results", exist_ok=True)
    
    # Input data
    bed_prefix = "refined_panel/refined_ref_panel"
    temp_prefix = "ml_results/genotypes"
    
    # Extract genotypes
    X, y, snp_names, ind_info = extract_genotypes(bed_prefix, temp_prefix)
    
    print(f"\nGenotype matrix shape: {X.shape}")
    print(f"Number of SNPs: {len(snp_names)}")
    print(f"Number of individuals: {len(y)}")
    
    # Get classifiers
    classifiers = get_classifiers()
    
    # Train and evaluate
    results, trained_models, le, data_splits = train_and_evaluate(X, y, classifiers)
    
    # Save individual classifier results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    results_df.to_csv("ml_results/classifier_results.csv", index=False)
    
    print("\n" + "="*80)
    print("Classifier Ranking (by Test Accuracy)")
    print("="*80)
    print(results_df[['Classifier', 'Test_Accuracy', 'CV_Accuracy_Mean', 'F1_Score', 'MCC', 'AUC']].to_string(index=False))
    
    # Create ensembles
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = data_splits
    ensemble_results = create_ensembles(
        trained_models, X_train, X_test, y_train, y_test, le, 
        X_train_scaled, X_test_scaled
    )
    
    # Save ensemble results
    ensemble_df = pd.DataFrame(ensemble_results)
    ensemble_df = ensemble_df.sort_values('Test_Accuracy', ascending=False)
    ensemble_df.to_csv("ml_results/ensemble_results.csv", index=False)
    
    print("\n" + "="*80)
    print("Ensemble Method Ranking (by Test Accuracy)")
    print("="*80)
    print(ensemble_df[['Ensemble_Method', 'Test_Accuracy', 'CV_Accuracy_Mean', 'F1_Score', 'MCC', 'AUC']].to_string(index=False))
    
    # Combined results
    combined_results = []
    for r in results:
        combined_results.append({
            'Method': r['Classifier'],
            'Type': 'Individual',
            **{k: v for k, v in r.items() if k != 'Classifier'}
        })
    for r in ensemble_results:
        combined_results.append({
            'Method': r['Ensemble_Method'],
            'Type': 'Ensemble',
            **{k: v for k, v in r.items() if k != 'Ensemble_Method'}
        })
    
    combined_df = pd.DataFrame(combined_results)
    combined_df = combined_df.sort_values('Test_Accuracy', ascending=False)
    combined_df.to_csv("ml_results/all_results.csv", index=False)
    
    # Print final summary
    print("\n" + "="*80)
    print("Final Summary - All Methods (Top 5)")
    print("="*80)
    print(combined_df.head()[['Method', 'Type', 'Test_Accuracy', 'Precision', 'Recall', 'F1_Score', 'MCC', 'AUC']].to_string(index=False))
    
    # Save best method info
    best_method = combined_df.iloc[0]
    print(f"\nBest Method: {best_method['Method']} ({best_method['Type']})")
    print(f"Test Accuracy: {best_method['Test_Accuracy']:.4f}")
    print(f"F1 Score: {best_method['F1_Score']:.4f}")
    print(f"MCC: {best_method['MCC']:.4f}")
    print(f"AUC: {best_method['AUC']:.4f}")
    
    print("\n" + "="*80)
    print("Results saved to ml_results/")
    print("="*80)
    print("\nStep 6 & 7 completed!")

if __name__ == "__main__":
    main()
