#!/usr/bin/env python3
"""Generate ADMIXTURE bar plots for all K values"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

def read_q_file(q_file):
    return pd.read_csv(q_file, sep=' ', header=None)

def read_populations(pop_file):
    with open(pop_file, 'r') as f:
        return [line.strip() for line in f]

def plot_admixture(q_df, populations, k, output_file, title=None):
    pop_unique = sorted(set(populations))
    pop_array = np.array(populations)
    sorted_indices = []
    pop_boundaries = []
    current_idx = 0
    
    for pop in pop_unique:
        pop_indices = np.where(pop_array == pop)[0]
        sorted_indices.extend(pop_indices)
        current_idx += len(pop_indices)
        pop_boundaries.append(current_idx)
    
    q_sorted = q_df.values[sorted_indices, :]
    
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.tab20(np.linspace(0, 1, k))
    
    n_individuals = q_sorted.shape[0]
    x = np.arange(n_individuals)
    
    bottom = np.zeros(n_individuals)
    for i in range(k):
        ax.bar(x, q_sorted[:, i], bottom=bottom, color=colors[i], 
               width=1.0, edgecolor='none')
        bottom += q_sorted[:, i]
    
    for boundary in pop_boundaries[:-1]:
        ax.axvline(x=boundary - 0.5, color='black', linewidth=0.5)
    
    prev_boundary = 0
    for i, pop in enumerate(pop_unique):
        boundary = pop_boundaries[i]
        center = (prev_boundary + boundary) / 2
        ax.text(center, -0.05, pop, ha='center', va='top', fontsize=10, 
                rotation=0, transform=ax.get_xaxis_transform())
        prev_boundary = boundary
    
    ax.set_xlim(-0.5, n_individuals - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Ancestry Proportion')
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_file}")

def plot_cv_errors(cv_file, output_file):
    cv_data = []
    with open(cv_file, 'r') as f:
        for line in f:
            if line.startswith('K='):
                parts = line.strip().split()
                k = int(parts[0].split('=')[1])
                cv = float(parts[1].split('=')[1])
                cv_data.append((k, cv))
    
    if cv_data:
        ks, cvs = zip(*cv_data)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, cvs, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('K (Number of Ancestral Populations)')
        ax.set_ylabel('Cross-Validation Error')
        ax.set_title('ADMIXTURE Cross-Validation Error')
        ax.grid(True, alpha=0.3)
        
        best_k = ks[np.argmin(cvs)]
        ax.axvline(x=best_k, color='r', linestyle='--', alpha=0.5, 
                   label=f'Best K={best_k}')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {output_file}")
        print(f"Best K: {best_k}")

def main():
    print("="*60)
    print("ADMIXTURE Plot Generation")
    print("="*60)
    
    pop_file = "admixture_results/populations.txt"
    cv_file = "admixture_results/cv_errors.txt"
    
    os.makedirs("admixture_plots", exist_ok=True)
    
    populations = read_populations(pop_file)
    print(f"Loaded {len(populations)} individuals")
    
    k_values = []
    for f in os.listdir("admixture_results"):
        if f.endswith(".Q"):
            k = int(f.replace("K", "").replace(".Q", ""))
            k_values.append(k)
    k_values = sorted(k_values)
    
    print(f"Generating plots for K={k_values}")
    
    for k in k_values:
        q_file = f"admixture_results/K{k}.Q"
        q_df = read_q_file(q_file)
        
        output_file = f"admixture_plots/admixture_K{k}.png"
        plot_admixture(q_df, populations, k, output_file, 
                      title=f"ADMIXTURE K={k}")
    
    if os.path.exists(cv_file):
        plot_cv_errors(cv_file, "admixture_plots/cv_errors.png")
    
    print("\n" + "="*60)
    print("Ancestry Proportion Summary (K=5)")
    print("="*60)
    
    q_file = "admixture_results/K5.Q"
    q_df = read_q_file(q_file)
    
    pop_array = np.array(populations)
    pop_unique = sorted(set(populations))
    
    for pop in pop_unique:
        pop_mask = pop_array == pop
        pop_q = q_df.values[pop_mask, :]
        avg_ancestry = np.mean(pop_q, axis=0)
        print(f"{pop}: {avg_ancestry.round(3)}")
    
    print("\nPlots saved to admixture_plots/")

if __name__ == "__main__":
    main()
