#!/usr/bin/env python3
"""
Part c: Identify BIMs (Breed Informative Markers) and Freeze Best Model
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import subprocess
import os
import warnings
warnings.filterwarnings('ignore')

def extract_genotypes(bed_prefix, snp_list, output_prefix):
    """Extract genotype data for specific SNPs"""
    
    snp_file = f"{output_prefix}_snps.txt"
    with open(snp_file, 'w') as f:
        for snp in snp_list:
            f.write(f"{snp}\n")
    
    cmd = f"plink --bfile {bed_prefix} --extract {snp_file} --recode A --out {output_prefix}"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    raw_file = f"{output_prefix}.raw"
    df = pd.read_csv(raw_file, sep=r'\s+')
    
    meta_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_cols = [c for c in df.columns if c not in meta_cols]
    
    y = df['FID'].values
    X = df[snp_cols].fillna(-1).values
    
    return X, y, snp_cols, df[['FID', 'IID']]

def get_snp_importance(X, y, snp_names):
    """Get SNP importance using Random Forest"""
    rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'snp': snp_names,
        'importance': rf.feature_importances_
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df

def analyze_breed_discrimination(snp_df, bim_df):
    """Analyze how well each SNP discriminates between breeds"""
    
    # Get breed info
    breeds = snp_df['FID'].unique()
    
    # Get SNP columns
    meta_cols = ['FID', 'IID']
    snp_cols = [c for c in snp_df.columns if c not in ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']]
    
    discrimination_stats = []
    
    for snp in snp_cols:
        breed_genotypes = {}
        for breed in breeds:
            breed_data = snp_df[snp_df['FID'] == breed][snp]
            # Calculate mean genotype (0, 1, 2)
            mean_geno = breed_data.mean()
            # Calculate frequency of allele 1
            freq = breed_data.sum() / (2 * len(breed_data))
            breed_genotypes[breed] = {
                'mean_geno': mean_geno,
                'allele_freq': freq
            }
        
        # Calculate discrimination power (max difference in allele frequency between breeds)
        freqs = [breed_genotypes[b]['allele_freq'] for b in breeds]
        max_diff = max(freqs) - min(freqs)
        
        # Get SNP info from bim
        snp_info = bim_df[bim_df['snp'] == snp]
        chrom = snp_info.iloc[0]['chrom'] if len(snp_info) > 0 else 'NA'
        pos = snp_info.iloc[0]['pos'] if len(snp_info) > 0 else 'NA'
        
        discrimination_stats.append({
            'snp': snp,
            'chrom': chrom,
            'pos': pos,
            'max_freq_diff': max_diff,
            **{f'{b}_freq': breed_genotypes[b]['allele_freq'] for b in breeds},
            **{f'{b}_mean_geno': breed_genotypes[b]['mean_geno'] for b in breeds}
        })
    
    return pd.DataFrame(discrimination_stats)

def get_genotype_patterns(snp_df, bim_list):
    """Get genotype patterns for BIMs across all individuals"""
    
    patterns = []
    
    for _, row in snp_df.iterrows():
        pattern = {
            'FID': row['FID'],
            'IID': row['IID']
        }
        
        for snp in bim_list:
            if snp in row:
                val = row[snp]
                if pd.isna(val):
                    genotype = "NN"  # Missing
                elif val == 0:
                    genotype = "AA"  # Homozygous reference
                elif val == 1:
                    genotype = "AB"  # Heterozygous
                elif val == 2:
                    genotype = "BB"  # Homozygous alternate
                else:
                    genotype = "NN"
                pattern[snp] = genotype
        
        patterns.append(pattern)
    
    return pd.DataFrame(patterns)

def freeze_best_model(X, y, bim_list, output_dir):
    """Train and freeze the best model (Hard Voting)"""
    
    print("\n" + "="*60)
    print("Freezing Best Model (Hard Voting Ensemble)")
    print("="*60)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale data for some classifiers
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base classifiers
    clf1 = LogisticRegression(max_iter=1000, random_state=42)
    clf2 = SVC(kernel='linear', probability=True, random_state=42)
    clf3 = DecisionTreeClassifier(max_depth=20, random_state=42)
    
    # Train individual classifiers on scaled data
    clf1.fit(X_train_scaled, y_train)
    clf2.fit(X_train_scaled, y_train)
    clf3.fit(X_train, y_train)  # Decision tree doesn't need scaling
    
    # Create voting classifier
    voting = VotingClassifier(
        estimators=[
            ('lr', clf1),
            ('svm', clf2),
            ('dt', clf3)
        ],
        voting='hard'
    )
    voting.fit(X_train, y_train)
    
    # Evaluate
    y_pred = voting.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model components
    model_package = {
        'voting_classifier': voting,
        'scaler': scaler,
        'label_encoder': le,
        'bim_list': bim_list,
        'classes': list(le.classes_),
        'accuracy': accuracy,
        'n_snps': len(bim_list)
    }
    
    with open(f"{output_dir}/frozen_model.pkl", 'wb') as f:
        pickle.dump(model_package, f)
    
    # Save model info as JSON (for human readability)
    model_info = {
        'model_type': 'Hard Voting Ensemble',
        'base_classifiers': ['LogisticRegression', 'SVM_Linear', 'DecisionTree'],
        'classes': list(le.classes_),
        'n_classes': len(le.classes_),
        'n_snps': len(bim_list),
        'snps': bim_list,
        'accuracy': float(accuracy),
        'parameters': {
            'test_size': 0.2,
            'random_state': 42
        }
    }
    
    with open(f"{output_dir}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel saved to: {output_dir}/frozen_model.pkl")
    print(f"Model info saved to: {output_dir}/model_info.json")
    
    return voting, scaler, le, accuracy

def main():
    print("="*80)
    print("Part c: Identify BIMs and Freeze Best Model")
    print("="*80)
    
    os.makedirs("final_model", exist_ok=True)
    
    # Load union SNP list
    union_snps = [line.strip() for line in open("ihs_union_results/union_svm_ihs_snps.txt")]
    print(f"\nTotal union SNPs: {len(union_snps)}")
    
    # Read BIM info
    bim_cols = ['chrom', 'snp', 'cm', 'pos', 'a1', 'a2']
    bim_df = pd.read_csv("refined_panel/refined_ref_panel.bim", sep='\t', header=None, names=bim_cols)
    
    # Extract genotypes
    bed_prefix = "refined_panel/refined_ref_panel"
    X, y, snp_cols, ind_info = extract_genotypes(bed_prefix, union_snps, "final_model/all_union_snps")
    
    print(f"Genotype matrix: {X.shape}")
    
    # Get SNP importance
    print("\n" + "="*60)
    print("Calculating SNP Importance")
    print("="*60)
    
    importance_df = get_snp_importance(X, y, snp_cols)
    print("\nTop 20 SNPs by Random Forest importance:")
    print(importance_df.head(20).to_string(index=False))
    
    importance_df.to_csv("final_model/snp_importance.csv", index=False)
    
    # Analyze breed discrimination
    print("\n" + "="*60)
    print("Analyzing Breed Discrimination Power")
    print("="*60)
    
    # Read full genotype data
    raw_df = pd.read_csv("final_model/all_union_snps.raw", sep=r'\s+')
    
    discrimination_df = analyze_breed_discrimination(raw_df, bim_df)
    discrimination_df = discrimination_df.sort_values('max_freq_diff', ascending=False)
    
    print("\nTop 20 SNPs by allele frequency difference between breeds:")
    freq_cols = [c for c in discrimination_df.columns if '_freq' in c][:5]  # Get first 5 breed freq columns
    print_cols = ['snp', 'chrom', 'pos', 'max_freq_diff'] + freq_cols
    print(discrimination_df[print_cols].head(20).to_string(index=False))
    
    discrimination_df.to_csv("final_model/breed_discrimination.csv", index=False)
    
    # Identify top BIMs (combine importance and discrimination)
    print("\n" + "="*60)
    print("Identifying Top BIMs (Breed Informative Markers)")
    print("="*60)
    
    # Merge importance and discrimination
    bim_candidates = importance_df.merge(discrimination_df, on='snp')
    
    # Normalize scores
    bim_candidates['importance_norm'] = (bim_candidates['importance'] - bim_candidates['importance'].min()) / (bim_candidates['importance'].max() - bim_candidates['importance'].min())
    bim_candidates['freq_diff_norm'] = (bim_candidates['max_freq_diff'] - bim_candidates['max_freq_diff'].min()) / (bim_candidates['max_freq_diff'].max() - bim_candidates['max_freq_diff'].min())
    
    # Combined score
    bim_candidates['bim_score'] = 0.5 * bim_candidates['importance_norm'] + 0.5 * bim_candidates['freq_diff_norm']
    bim_candidates = bim_candidates.sort_values('bim_score', ascending=False)
    
    # Select top 10 and top 20 BIMs
    top_10_bims = bim_candidates.head(10)['snp'].tolist()
    top_20_bims = bim_candidates.head(20)['snp'].tolist()
    
    print(f"\nTop 10 BIMs:")
    print(bim_candidates[['snp', 'chrom', 'pos', 'importance', 'max_freq_diff', 'bim_score']].head(10).to_string(index=False))
    
    # Save BIM lists
    with open("final_model/top_10_bims.txt", 'w') as f:
        for snp in top_10_bims:
            f.write(f"{snp}\n")
    
    with open("final_model/top_20_bims.txt", 'w') as f:
        for snp in top_20_bims:
            f.write(f"{snp}\n")
    
    bim_candidates.head(20).to_csv("final_model/top_20_bims_detailed.csv", index=False)
    
    print(f"\nTop 10 BIMs saved: final_model/top_10_bims.txt")
    print(f"Top 20 BIMs saved: final_model/top_20_bims.txt")
    
    # Get genotype patterns for top 20 BIMs
    print("\n" + "="*60)
    print("Genotype Patterns for Top 20 BIMs")
    print("="*60)
    
    genotype_patterns = get_genotype_patterns(raw_df, top_20_bims)
    genotype_patterns.to_csv("final_model/bim_genotypes_all_individuals.csv", index=False)
    
    # Summarize genotypes by breed
    print("\nGenotype summary by breed:")
    unique_breeds = genotype_patterns['FID'].unique()[:3]  # Get first 3 breeds
    for breed in unique_breeds:
        breed_data = genotype_patterns[genotype_patterns['FID'] == breed]
        print(f"\n{breed} (n={len(breed_data)}):")
        
        for snp in top_10_bims[:5]:  # Show first 5
            geno_counts = breed_data[snp].value_counts()
            print(f"  {snp}: {dict(geno_counts)}")
    
    # Test classification with different numbers of BIMs
    print("\n" + "="*60)
    print("Testing Classification with Different BIM Counts")
    print("="*60)
    
    bim_counts = [10, 15, 20, 25]
    best_accuracy = 0
    best_bim_count = 0
    best_bim_list = []
    
    for count in bim_counts:
        selected_bims = top_20_bims[:count]
        
        # Get indices of selected SNPs
        selected_indices = [snp_cols.index(snp) for snp in selected_bims if snp in snp_cols]
        X_selected = X[:, selected_indices]
        
        # Quick test
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Simple test with Logistic Regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_scaled, y_train)
        acc = clf.score(X_test_scaled, y_test)
        
        print(f"  {count} BIMs: Accuracy = {acc:.4f}")
        
        if acc >= best_accuracy:
            best_accuracy = acc
            best_bim_count = count
            best_bim_list = selected_bims
    
    print(f"\nBest BIM count: {best_bim_count} (Accuracy: {best_accuracy:.4f})")
    
    # Use top 20 BIMs for final model (or best_bim_list if it has 100% accuracy)
    final_bim_list = best_bim_list if best_accuracy == 1.0 else top_20_bims
    print(f"\nUsing {len(final_bim_list)} BIMs for final model")
    
    # Use already extracted genotypes
    y_final = y
    
    # Get indices of final BIMs
    final_indices = [snp_cols.index(snp) for snp in final_bim_list if snp in snp_cols]
    X_final = X[:, final_indices]
    
    # Freeze the best model
    voting, scaler, le, final_accuracy = freeze_best_model(
        X_final, y_final, final_bim_list, "final_model"
    )
    
    # Create comprehensive summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - Breed Informative Markers (BIMs)")
    print("="*80)
    
    print(f"\nDataset:")
    print(f"  Reference breeds: {', '.join(le.classes_)}")
    print(f"  Total individuals: {len(y_final)}")
    
    print(f"\nBIMs identified:")
    print(f"  Top 10 BIMs: final_model/top_10_bims.txt")
    print(f"  Top 20 BIMs: final_model/top_20_bims.txt")
    
    print(f"\nBest Model (Hard Voting Ensemble):")
    print(f"  Base classifiers: Logistic Regression, SVM (Linear), Decision Tree")
    print(f"  Number of BIMs used: {len(final_bim_list)}")
    print(f"  Test accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    print(f"\nModel Files:")
    print(f"  Frozen model: final_model/frozen_model.pkl")
    print(f"  Model info: final_model/model_info.json")
    print(f"  SNP importance: final_model/snp_importance.csv")
    print(f"  Breed discrimination: final_model/breed_discrimination.csv")
    print(f"  BIM genotypes: final_model/bim_genotypes_all_individuals.csv")
    
    print("\n" + "="*80)
    print("Part c completed! Best model frozen and ready for classification.")
    print("="*80)

if __name__ == "__main__":
    main()
