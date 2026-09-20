#!/usr/bin/env python3
"""
Part b: iHS Analysis (3SD threshold) and Union with Best AIMs
Then run all classifiers including ensembling
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
                            matthews_corrcoef, roc_auc_score, confusion_matrix)
import subprocess
import os
import glob
import warnings
warnings.filterwarnings('ignore')

def read_ihs_file(filepath, threshold=3.0):
    """
    Read iHS file and extract positions with |iHS| > threshold (3SD)
    Returns list of positions
    """
    positions = []
    ihs_values = []
    
    with open(filepath, 'r') as f:
        for line in f:
            cols = line.strip().split('\t')
            if len(cols) >= 7:
                try:
                    pos = int(cols[1])
                    ihs = float(cols[6])  # Standardized iHS is in column 7
                    if abs(ihs) > threshold:
                        positions.append(pos)
                        ihs_values.append(ihs)
                except ValueError:
                    continue
    
    return positions, ihs_values

def get_snp_at_position(bim_df, position):
    """Get SNP ID at a given position"""
    match = bim_df[bim_df['pos'] == position]
    if len(match) > 0:
        return match.iloc[0]['snp']
    return None

def extract_ihs_snps(ihs_files, bim_df, threshold=3.0):
    """
    Extract SNPs from all iHS files that exceed threshold
    Returns dictionary with breed-specific and union sets
    """
    breed_ihs = {}
    all_ihs_positions = set()
    
    for ihs_file in ihs_files:
        # Extract breed name from filename
        breed = os.path.basename(ihs_file).split('_')[0]
        print(f"\nProcessing {breed} iHS file...")
        
        positions, ihs_values = read_ihs_file(ihs_file, threshold)
        print(f"  Positions with |iHS| > {threshold}: {len(positions)}")
        
        # Map positions to SNP IDs
        snp_list = []
        for pos in positions:
            snp = get_snp_at_position(bim_df, pos)
            if snp:
                snp_list.append(snp)
                all_ihs_positions.add(pos)
        
        print(f"  SNPs found in our dataset: {len(snp_list)}")
        breed_ihs[breed] = {
            'positions': positions,
            'ihs_values': ihs_values,
            'snps': snp_list
        }
    
    # Get union of all iHS SNPs
    union_snps = set()
    for breed_data in breed_ihs.values():
        union_snps.update(breed_data['snps'])
    
    return breed_ihs, list(union_snps)

def extract_genotypes_for_snps(bed_prefix, snp_list, output_prefix):
    """Extract genotype data for specific SNPs using PLINK"""
    
    if len(snp_list) == 0:
        return None, None, None
    
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

def calculate_metrics(y_true, y_pred, y_prob=None, n_classes=None):
    """Calculate all evaluation metrics"""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['specificity'] = calculate_specificity(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    if y_prob is not None and n_classes is not None:
        try:
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

def train_and_evaluate_all(X, y, classifiers, test_size=0.2, cv_folds=5):
    """Train and evaluate all classifiers"""
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f"Classes: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, SNPs: {X.shape[1]}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    trained_models = {}
    
    print("\n" + "="*60)
    print("Individual Classifiers")
    print("="*60)
    
    for name, clf in classifiers.items():
        if name in ['SVM_RBF', 'SVM_Linear', 'KNN', 'LogisticRegression', 'MLP', 'NaiveBayes']:
            X_tr = X_train_scaled
            X_te = X_test_scaled
        else:
            X_tr = X_train
            X_te = X_test
        
        clf.fit(X_tr, y_train)
        trained_models[name] = (clf, scaler if name in ['SVM_RBF', 'SVM_Linear', 'KNN', 'LogisticRegression', 'MLP', 'NaiveBayes'] else None)
        
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te) if hasattr(clf, 'predict_proba') else None
        
        test_metrics = calculate_metrics(y_test, y_pred, y_prob, n_classes)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_tr, y_train, cv=cv, scoring='accuracy')
        
        print(f"{name}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1_score']:.4f}, MCC={test_metrics['mcc']:.4f}, AUC={test_metrics['auc']:.4f}")
        
        results.append({
            'Method': name,
            'Type': 'Individual',
            'Accuracy': test_metrics['accuracy'],
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'Specificity': test_metrics['specificity'],
            'F1': test_metrics['f1_score'],
            'MCC': test_metrics['mcc'],
            'AUC': test_metrics['auc']
        })
    
    return results, trained_models, le, (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler)

def create_ensembles(trained_models, X_train, X_test, y_train, y_test, le, X_train_scaled, X_test_scaled):
    """Create and evaluate ensemble methods"""
    
    print("\n" + "="*60)
    print("Ensemble Methods")
    print("="*60)
    
    n_classes = len(le.classes_)
    ensemble_results = []
    
    # Get top 3 models
    top_models = []
    for name, (clf, scaler) in trained_models.items():
        if scaler is not None:
            y_pred = clf.predict(X_test_scaled)
        else:
            y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        top_models.append((name, clf, scaler, acc))
    
    top_models.sort(key=lambda x: x[3], reverse=True)
    top_3 = top_models[:3]
    print(f"Top 3 models: {[m[0] for m in top_3]}")
    
    # Hard Voting
    estimators = [(name, clf) for name, clf, _, _ in top_3]
    voting_hard = VotingClassifier(estimators=estimators, voting='hard')
    voting_hard.fit(X_train, y_train)
    y_pred_hard = voting_hard.predict(X_test)
    metrics_hard = calculate_metrics(y_test, y_pred_hard, None, n_classes)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(voting_hard, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"Hard Voting: Acc={metrics_hard['accuracy']:.4f}, F1={metrics_hard['f1_score']:.4f}")
    ensemble_results.append({'Method': 'Hard_Voting', 'Type': 'Ensemble', 
                            'Accuracy': metrics_hard['accuracy'], 'CV_Mean': cv_scores.mean(),
                            'Precision': metrics_hard['precision'], 'Recall': metrics_hard['recall'],
                            'Specificity': metrics_hard['specificity'], 'F1': metrics_hard['f1_score'],
                            'MCC': metrics_hard['mcc'], 'AUC': metrics_hard['auc']})
    
    # Soft Voting
    estimators_soft = [(name, clf) for name, clf, _, _ in top_3 if hasattr(clf, 'predict_proba')]
    if len(estimators_soft) >= 2:
        voting_soft = VotingClassifier(estimators=estimators_soft, voting='soft')
        voting_soft.fit(X_train, y_train)
        y_pred_soft = voting_soft.predict(X_test)
        y_prob_soft = voting_soft.predict_proba(X_test)
        metrics_soft = calculate_metrics(y_test, y_pred_soft, y_prob_soft, n_classes)
        cv_scores_soft = cross_val_score(voting_soft, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"Soft Voting: Acc={metrics_soft['accuracy']:.4f}, F1={metrics_soft['f1_score']:.4f}")
        ensemble_results.append({'Method': 'Soft_Voting', 'Type': 'Ensemble',
                                'Accuracy': metrics_soft['accuracy'], 'CV_Mean': cv_scores_soft.mean(),
                                'Precision': metrics_soft['precision'], 'Recall': metrics_soft['recall'],
                                'Specificity': metrics_soft['specificity'], 'F1': metrics_soft['f1_score'],
                                'MCC': metrics_soft['mcc'], 'AUC': metrics_soft['auc']})
    
    # Stacking
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
    
    meta_clf = LogisticRegression(max_iter=1000, random_state=42)
    meta_clf.fit(X_meta_train, y_train)
    y_pred_meta = meta_clf.predict(X_meta_test)
    y_prob_meta = meta_clf.predict_proba(X_meta_test)
    metrics_meta = calculate_metrics(y_test, y_pred_meta, y_prob_meta, n_classes)
    cv_scores_meta = cross_val_score(meta_clf, X_meta_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"Stacking (LR): Acc={metrics_meta['accuracy']:.4f}, F1={metrics_meta['f1_score']:.4f}")
    ensemble_results.append({'Method': 'Stacking_LR', 'Type': 'Ensemble',
                            'Accuracy': metrics_meta['accuracy'], 'CV_Mean': cv_scores_meta.mean(),
                            'Precision': metrics_meta['precision'], 'Recall': metrics_meta['recall'],
                            'Specificity': metrics_meta['specificity'], 'F1': metrics_meta['f1_score'],
                            'MCC': metrics_meta['mcc'], 'AUC': metrics_meta['auc']})
    
    # Weighted Average
    weights = np.array([acc for _, _, _, acc in top_3])
    weights = weights / weights.sum()
    avg_prob = np.zeros((X_test.shape[0], n_classes))
    for i, (name, clf, scaler, _) in enumerate(top_3):
        if scaler is not None:
            prob = clf.predict_proba(X_test_scaled)
        else:
            prob = clf.predict_proba(X_test)
        avg_prob += weights[i] * prob
    y_pred_weighted = np.argmax(avg_prob, axis=1)
    metrics_weighted = calculate_metrics(y_test, y_pred_weighted, avg_prob, n_classes)
    
    print(f"Weighted Avg: Acc={metrics_weighted['accuracy']:.4f}, F1={metrics_weighted['f1_score']:.4f}")
    ensemble_results.append({'Method': 'Weighted_Average', 'Type': 'Ensemble',
                            'Accuracy': metrics_weighted['accuracy'], 'CV_Mean': np.nan,
                            'Precision': metrics_weighted['precision'], 'Recall': metrics_weighted['recall'],
                            'Specificity': metrics_weighted['specificity'], 'F1': metrics_weighted['f1_score'],
                            'MCC': metrics_weighted['mcc'], 'AUC': metrics_weighted['auc']})
    
    return ensemble_results

def main():
    print("="*80)
    print("Part b: iHS Analysis and Union with Best AIMs")
    print("="*80)
    
    os.makedirs("ihs_union_results", exist_ok=True)
    
    # Read BIM file to map positions to SNPs
    bim_cols = ['chrom', 'snp', 'cm', 'pos', 'a1', 'a2']
    bim_df = pd.read_csv("refined_panel/refined_ref_panel.bim", sep='\t', header=None, names=bim_cols)
    print(f"\nReference panel SNPs: {len(bim_df)}")
    
    # Find iHS files
    ihs_files = glob.glob("*_unstanderdized_iHS_file_3.ihs.out.100bins.norm")
    print(f"\niHS files found: {len(ihs_files)}")
    for f in ihs_files:
        print(f"  - {f}")
    
    # Extract iHS outliers (3SD threshold)
    print("\n" + "="*60)
    print("Extracting iHS outliers (|iHS| > 3)")
    print("="*60)
    
    breed_ihs, union_ihs_snps = extract_ihs_snps(ihs_files, bim_df, threshold=3.0)
    print(f"\nUnion of all iHS SNPs: {len(union_ihs_snps)}")
    
    # Save iHS results
    ihs_summary = []
    for breed, data in breed_ihs.items():
        ihs_summary.append({
            'Breed': breed,
            'iHS_Outliers': len(data['positions']),
            'SNPs_in_Dataset': len(data['snps'])
        })
        # Save SNP list
        with open(f"ihs_union_results/ihs_snps_{breed}.txt", 'w') as f:
            for snp in data['snps']:
                f.write(f"{snp}\n")
    
    ihs_summary_df = pd.DataFrame(ihs_summary)
    print("\niHS Summary per breed:")
    print(ihs_summary_df.to_string(index=False))
    
    # Save union
    with open("ihs_union_results/ihs_union_all.txt", 'w') as f:
        for snp in union_ihs_snps:
            f.write(f"{snp}\n")
    
    # Read best AIMs from part a
    best_aims_svm = [line.strip() for line in open("incremental_aims/best_aims_svm.txt")]
    best_aims_mlp = [line.strip() for line in open("incremental_aims/best_aims_mlp.txt")]
    
    print(f"\nBest AIMs from Part a:")
    print(f"  SVM: {len(best_aims_svm)} SNPs")
    print(f"  MLP: {len(best_aims_mlp)} SNPs")
    
    # Create union sets
    union_svm_ihs = list(set(best_aims_svm) | set(union_ihs_snps))
    union_mlp_ihs = list(set(best_aims_mlp) | set(union_ihs_snps))
    
    print(f"\nUnion sets:")
    print(f"  SVM AIMs + iHS: {len(union_svm_ihs)} SNPs ({len(best_aims_svm)} + {len(union_ihs_snps)} - overlap)")
    print(f"  MLP AIMs + iHS: {len(union_mlp_ihs)} SNPs ({len(best_aims_mlp)} + {len(union_ihs_snps)} - overlap)")
    
    # Find overlap
    overlap_svm = set(best_aims_svm) & set(union_ihs_snps)
    overlap_mlp = set(best_aims_mlp) & set(union_ihs_snps)
    print(f"  Overlap (SVM): {len(overlap_svm)} SNPs")
    print(f"  Overlap (MLP): {len(overlap_mlp)} SNPs")
    
    # Save union SNP lists
    with open("ihs_union_results/union_svm_ihs_snps.txt", 'w') as f:
        for snp in union_svm_ihs:
            f.write(f"{snp}\n")
    
    with open("ihs_union_results/union_mlp_ihs_snps.txt", 'w') as f:
        for snp in union_mlp_ihs:
            f.write(f"{snp}\n")
    
    # Use the larger union set for classification (or test both)
    # Here we'll use the SVM+ iHS union as it has fewer SNPs but both achieve 100%
    selected_union = union_svm_ihs  # Can change to union_mlp_ihs if needed
    print(f"\nSelected union for classification: {len(selected_union)} SNPs")
    
    # Extract genotypes
    bed_prefix = "refined_panel/refined_ref_panel"
    X, y, snp_cols = extract_genotypes_for_snps(bed_prefix, selected_union, "ihs_union_results/union_genotypes")
    
    if X is None:
        print("Error: Could not extract genotypes")
        return
    
    print(f"\nGenotype matrix: {X.shape}")
    
    # Get classifiers
    classifiers = get_classifiers()
    
    # Train and evaluate all classifiers
    results, trained_models, le, data_splits = train_and_evaluate_all(X, y, classifiers)
    
    # Create ensembles
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = data_splits
    ensemble_results = create_ensembles(trained_models, X_train, X_test, y_train, y_test, le, X_train_scaled, X_test_scaled)
    
    # Combine results
    all_results = results + ensemble_results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    # Save results
    results_df.to_csv("ihs_union_results/classification_results_union.csv", index=False)
    
    print("\n" + "="*80)
    print("Final Results (Ranked by Accuracy)")
    print("="*80)
    print(results_df[['Method', 'Type', 'Accuracy', 'F1', 'MCC', 'AUC']].head(15).to_string(index=False))
    
    # Identify best method
    best = results_df.iloc[0]
    print(f"\nBest Method: {best['Method']} ({best['Type']})")
    print(f"Accuracy: {best['Accuracy']:.4f}")
    print(f"F1-Score: {best['F1']:.4f}")
    print(f"MCC: {best['MCC']:.4f}")
    print(f"AUC: {best['AUC']:.4f}")
    
    # Save best model info for part c
    best_info = {
        'method': best['Method'],
        'type': best['Type'],
        'n_snps': len(selected_union),
        'accuracy': best['Accuracy'],
        'snp_list': selected_union
    }
    
    import pickle
    with open("ihs_union_results/best_model_info.pkl", 'wb') as f:
        pickle.dump(best_info, f)
    
    # Save SNP list with metadata
    snp_metadata = []
    for snp in selected_union:
        snp_info = bim_df[bim_df['snp'] == snp]
        if len(snp_info) > 0:
            is_aim = snp in best_aims_svm
            is_ihs = snp in union_ihs_snps
            source = []
            if is_aim:
                source.append("AIM")
            if is_ihs:
                source.append("iHS")
            
            snp_metadata.append({
                'snp': snp,
                'chrom': snp_info.iloc[0]['chrom'],
                'pos': snp_info.iloc[0]['pos'],
                'source': '+'.join(source)
            })
    
    snp_metadata_df = pd.DataFrame(snp_metadata)
    snp_metadata_df.to_csv("ihs_union_results/union_snp_metadata.csv", index=False)
    
    print(f"\nSNP breakdown:")
    print(f"  AIM only: {len([s for s in selected_union if s in best_aims_svm and s not in union_ihs_snps])}")
    print(f"  iHS only: {len([s for s in selected_union if s in union_ihs_snps and s not in best_aims_svm])}")
    print(f"  Both: {len([s for s in selected_union if s in best_aims_svm and s in union_ihs_snps])}")
    
    print("\n" + "="*80)
    print("Part b completed! Results saved to ihs_union_results/")
    print("="*80)

if __name__ == "__main__":
    main()
