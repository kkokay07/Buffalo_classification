#!/usr/bin/env python3
"""
Step 5: Refine Reference Panel
Apply Crum et al. (2019) criterion: Retain individuals with at least 80% uniqueness 
to their respective groups based on ADMIXTURE results.
"""

import numpy as np
import pandas as pd
import os
import subprocess

def read_q_file(q_file):
    return pd.read_csv(q_file, sep=' ', header=None)

def read_ind_pop(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None, names=['fid', 'iid'])
    return df

def refine_reference_panel(q_df, ind_pop_df, threshold=0.80):
    """Refine reference panel based on ancestry proportions."""
    
    populations = ind_pop_df['fid'].values
    individuals = ind_pop_df['iid'].values
    unique_pops = sorted(set(populations))
    
    k = q_df.shape[1]
    
    # Find dominant ancestral component for each population
    pop_dominant_component = {}
    for pop in unique_pops:
        pop_mask = populations == pop
        pop_q = q_df.values[pop_mask, :]
        avg_ancestry = np.mean(pop_q, axis=0)
        dominant = np.argmax(avg_ancestry)
        pop_dominant_component[pop] = dominant
        print(f"  {pop}: Dominant component = {dominant} (avg ancestry = {avg_ancestry[dominant]:.3f})")
    
    # Identify individuals to keep
    keep_indices = []
    removed_individuals = []
    
    for i, (pop, ind) in enumerate(zip(populations, individuals)):
        dominant = pop_dominant_component[pop]
        ancestry_to_dominant = q_df.iloc[i, dominant]
        
        if ancestry_to_dominant >= threshold:
            keep_indices.append(i)
        else:
            removed_individuals.append({
                'fid': pop,
                'iid': ind,
                'dominant_ancestry': ancestry_to_dominant
            })
    
    return keep_indices, removed_individuals, pop_dominant_component

def main():
    print("="*60)
    print("Step 5: Refine Reference Panel")
    print("="*60)
    print("Applying Crum et al. (2019) criterion:")
    print("Retaining individuals with >= 80% ancestry to their groups")
    print("")
    
    ind_pop_file = "admixture_results/ind_order.txt"
    q_file = "admixture_results/K5.Q"  # Use K=5 (best K, matches number of breeds)
    
    print("Reading ADMIXTURE results...")
    ind_pop_df = read_ind_pop(ind_pop_file)
    q_df = read_q_file(q_file)
    
    print(f"Total individuals: {len(ind_pop_df)}")
    print(f"Number of ancestral populations (K): {q_df.shape[1]}")
    
    breed_counts = ind_pop_df['fid'].value_counts()
    print(f"\nOriginal breed distribution:")
    for breed, count in breed_counts.items():
        print(f"  {breed}: {count}")
    
    print("\nIdentifying dominant ancestral components per breed...")
    keep_indices, removed, pop_components = refine_reference_panel(
        q_df, ind_pop_df, threshold=0.80
    )
    
    print(f"\nIndividuals to keep: {len(keep_indices)}")
    print(f"Individuals removed: {len(removed)}")
    
    if removed:
        print("\nRemoved individuals (admixture > 20%):")
        for ind in removed:
            print(f"  {ind['fid']} {ind['iid']}: {ind['dominant_ancestry']:.3f}")
    
    # Create filtered PLINK files
    print("\nCreating filtered PLINK dataset...")
    
    os.makedirs("refined_panel", exist_ok=True)
    
    keep_df = ind_pop_df.iloc[keep_indices]
    keep_file = "refined_panel/keep_individuals.txt"
    keep_df.to_csv(keep_file, sep='\t', header=False, index=False, columns=['fid', 'iid'])
    
    cmd = f"plink --bfile qc_results/step4_mind --keep {keep_file} --make-bed --out refined_panel/refined_ref_panel"
    subprocess.run(cmd, shell=True, check=True)
    
    print("\n" + "="*60)
    print("Refined Panel Summary")
    print("="*60)
    
    refined_breeds = keep_df['fid'].value_counts()
    print(f"\nRefined breed distribution:")
    for breed, count in refined_breeds.items():
        original_count = breed_counts[breed]
        pct_kept = (count / original_count) * 100
        print(f"  {breed}: {count}/{original_count} ({pct_kept:.1f}%)")
    
    print(f"\nTotal individuals kept: {len(keep_df)}/{len(ind_pop_df)} ({len(keep_df)/len(ind_pop_df)*100:.1f}%)")
    
    summary = {
        'original_count': len(ind_pop_df),
        'refined_count': len(keep_df),
        'removed_count': len(removed),
        'threshold': 0.80,
        'k_used': q_df.shape[1]
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("refined_panel/refinement_summary.csv", index=False)
    
    if removed:
        removed_df = pd.DataFrame(removed)
        removed_df.to_csv("refined_panel/removed_individuals.csv", index=False)
    
    print("\nRefined panel files:")
    print("  refined_panel/refined_ref_panel.{bed,bim,fam}")
    print("  refined_panel/keep_individuals.txt")
    print("  refined_panel/refinement_summary.csv")
    
    print("\nStep 5 completed!")

if __name__ == "__main__":
    main()
