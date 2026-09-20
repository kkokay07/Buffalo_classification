#!/usr/bin/env python3
"""
Create final visualization plots for BIM analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

def plot_bim_heatmap():
    """Create heatmap of allele frequencies by breed for top 10 BIMs"""
    
    # Read discrimination data
    disc_df = pd.read_csv("final_model/breed_discrimination.csv")
    top_10 = [line.strip() for line in open("final_model/top_10_bims.txt")]
    
    # Filter for top 10
    top_disc = disc_df[disc_df['snp'].isin(top_10)].copy()
    top_disc = top_disc.set_index('snp')
    
    # Get frequency columns
    freq_cols = [c for c in top_disc.columns if '_freq' in c]
    freq_data = top_disc[freq_cols]
    
    # Rename columns (order is based on actual column names)
    breed_order = [c.replace('_freq', '') for c in freq_cols]
    freq_data.columns = breed_order
    
    # Reorder to match BIM ranking
    freq_data = freq_data.reindex(top_10)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(freq_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0.5, vmin=0, vmax=1, 
                cbar_kws={'label': 'Allele Frequency'},
                linewidths=0.5, ax=ax)
    
    ax.set_title('Breed Informative Markers (BIMs)\nAllele Frequencies by Breed', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Breed', fontsize=12)
    ax.set_ylabel('SNP', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('final_model/bim_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: final_model/bim_heatmap.png")

def plot_accuracy_by_n_aims():
    """Plot accuracy vs number of AIMs"""
    
    results = pd.read_csv("incremental_aims/incremental_classification_results.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # SVM results
    svm_data = results[results['Classifier'] == 'SVM_Linear']
    ax1.plot(svm_data['n_AIMs'], svm_data['accuracy'], 'b-o', linewidth=2, markersize=6, label='Test Accuracy')
    ax1.plot(svm_data['n_AIMs'], svm_data['cv_accuracy_mean'], 'r--s', linewidth=2, markersize=6, label='CV Accuracy')
    ax1.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='Perfect Accuracy')
    ax1.set_xlabel('Number of AIMs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('SVM (Linear) Performance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.7, 1.05)
    
    # MLP results
    mlp_data = results[results['Classifier'] == 'MLP']
    ax2.plot(mlp_data['n_AIMs'], mlp_data['accuracy'], 'b-o', linewidth=2, markersize=6, label='Test Accuracy')
    ax2.plot(mlp_data['n_AIMs'], mlp_data['cv_accuracy_mean'], 'r--s', linewidth=2, markersize=6, label='CV Accuracy')
    ax2.axhline(y=1.0, color='g', linestyle=':', alpha=0.5, label='Perfect Accuracy')
    ax2.set_xlabel('Number of AIMs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('MLP Performance', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 1.05)
    
    plt.tight_layout()
    plt.savefig('incremental_aims/accuracy_by_n_aims.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: incremental_aims/accuracy_by_n_aims.png")

def plot_classifier_comparison():
    """Compare all classifiers from Part b"""
    
    results = pd.read_csv("ihs_union_results/classification_results_union.csv")
    results = results.sort_values('Accuracy', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['lightcoral' if t == 'Individual' else 'steelblue' for t in results['Type']]
    bars = ax.barh(results['Method'], results['Accuracy'], color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, results['Accuracy'])):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{acc:.3f}', va='center', fontsize=9)
    
    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Accuracy')
    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_title('Classifier Performance Comparison\n(25 SNPs: AIMs + iHS)', fontsize=13, fontweight='bold')
    ax.set_xlim(0.75, 1.05)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='lightcoral', label='Individual'),
                       Patch(facecolor='steelblue', label='Ensemble')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig('ihs_union_results/classifier_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: ihs_union_results/classifier_comparison.png")

def plot_bim_scores():
    """Plot BIM scores for top 20"""
    
    bim_df = pd.read_csv("final_model/top_20_bims_detailed.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # BIM Score
    top_10 = bim_df.head(10)
    ax1.barh(range(len(top_10)), top_10['bim_score'], color='darkgreen', alpha=0.7)
    ax1.set_yticks(range(len(top_10)))
    ax1.set_yticklabels([s.replace('_', '\n') for s in top_10['snp']], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('BIM Score', fontsize=12)
    ax1.set_title('Top 10 BIMs by Combined Score', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Allele frequency differences
    ax2.barh(range(len(top_10)), top_10['max_freq_diff'], color='darkblue', alpha=0.7)
    ax2.set_yticks(range(len(top_10)))
    ax2.set_yticklabels([s.replace('_', '\n') for s in top_10['snp']], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('Max Allele Frequency Difference', fontsize=12)
    ax2.set_title('Breed Discrimination Power', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_model/bim_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: final_model/bim_scores.png")

def main():
    print("="*60)
    print("Creating Final Visualization Plots")
    print("="*60)
    
    sns.set_style("whitegrid")
    
    print("\n1. Creating BIM heatmap...")
    plot_bim_heatmap()
    
    print("2. Creating accuracy by n_AIMs plot...")
    plot_accuracy_by_n_aims()
    
    print("3. Creating classifier comparison plot...")
    plot_classifier_comparison()
    
    print("4. Creating BIM scores plot...")
    plot_bim_scores()
    
    print("\n" + "="*60)
    print("All plots created successfully!")
    print("="*60)
    
    print("\nGenerated plots:")
    print("  - final_model/bim_heatmap.png")
    print("  - final_model/bim_scores.png")
    print("  - incremental_aims/accuracy_by_n_aims.png")
    print("  - ihs_union_results/classifier_comparison.png")

if __name__ == "__main__":
    main()
