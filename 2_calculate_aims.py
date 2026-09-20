#!/usr/bin/env python3
"""
Step 3: Calculate Ancestry Informative Markers (AIMs) statistics
- Wright's FST (Wright 1951)
- Informativeness for assignment (In) (Rosenberg et al. 2002)
- Delta values (Shriver et al. 2003)
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import subprocess

def read_bim(filename):
    """Read .bim file"""
    cols = ['chrom', 'snp', 'cm', 'pos', 'a1', 'a2']
    return pd.read_csv(filename, sep='\t', header=None, names=cols)

def read_fam(filename):
    """Read .fam file"""
    cols = ['fid', 'iid', 'pid', 'mid', 'sex', 'pheno']
    return pd.read_csv(filename, sep=r'\s+', header=None, names=cols)

def calculate_allele_frequencies(genotypes, populations, pop_ids):
    """
    Calculate allele frequencies for each SNP in each population
    genotypes: numpy array (n_individuals, n_snps) with values 0, 1, 2, -1 (missing)
    populations: array of population labels for each individual
    pop_ids: unique population IDs
    """
    n_snps = genotypes.shape[1]
    n_pops = len(pop_ids)
    
    freqs = np.zeros((n_pops, n_snps))
    
    for i, pop in enumerate(pop_ids):
        mask = populations == pop
        pop_geno = genotypes[mask, :]
        
        for j in range(n_snps):
            snp_geno = pop_geno[:, j]
            non_missing = snp_geno >= 0
            if np.sum(non_missing) > 0:
                allele_sum = np.sum(snp_geno[non_missing])
                total_alleles = 2 * np.sum(non_missing)
                freqs[i, j] = allele_sum / total_alleles
            else:
                freqs[i, j] = np.nan
    
    return freqs

def calculate_fst(freqs, pop_sizes):
    """Calculate Wright's FST (Weir-Cockerham estimator)"""
    n_pops, n_snps = freqs.shape
    fst_values = np.zeros(n_snps)
    
    for j in range(n_snps):
        p = freqs[:, j]
        valid = ~np.isnan(p)
        if np.sum(valid) < 2:
            fst_values[j] = 0
            continue
        p = p[valid]
        n = pop_sizes[valid]
        
        p_bar = np.sum(n * p) / np.sum(n)
        if p_bar == 0 or p_bar == 1:
            fst_values[j] = 0
            continue
        
        s2 = np.sum(n * (p - p_bar)**2) / np.sum(n)
        numerator = s2 - p_bar * (1 - p_bar) / (np.sum(n) - 1)
        denominator = p_bar * (1 - p_bar)
        
        if denominator > 0:
            fst = numerator / denominator
            fst_values[j] = max(0, fst)
        else:
            fst_values[j] = 0
    
    return fst_values

def calculate_in(freqs):
    """Calculate Informativeness for Assignment (In) - Rosenberg et al. 2003"""
    n_pops, n_snps = freqs.shape
    in_values = np.zeros(n_snps)
    
    for j in range(n_snps):
        p = freqs[:, j]
        valid = ~np.isnan(p)
        if np.sum(valid) < 2:
            in_values[j] = 0
            continue
        p = p[valid]
        
        p_avg = np.mean(p)
        q_avg = 1 - p_avg
        
        if p_avg <= 0 or p_avg >= 1:
            in_values[j] = 0
            continue
        
        h_s = sum(2 * freq * (1 - freq) for freq in p if 0 < freq < 1) / len(p)
        h_t = 2 * p_avg * q_avg
        
        if h_t > 0:
            in_values[j] = h_s / h_t
        else:
            in_values[j] = 0
    
    return in_values

def calculate_delta(freqs):
    """Calculate Delta values - Shriver et al. 1997"""
    n_pops, n_snps = freqs.shape
    delta_values = np.zeros(n_snps)
    
    for j in range(n_snps):
        p = freqs[:, j]
        valid = ~np.isnan(p)
        if np.sum(valid) < 2:
            delta_values[j] = 0
            continue
        p = p[valid]
        delta_values[j] = np.max(p) - np.min(p)
    
    return delta_values

def main():
    print("="*60)
    print("Step 3: Calculating AIM Statistics")
    print("="*60)
    
    bed_file = "qc_results/step4_mind.bed"
    bim_file = "qc_results/step4_mind.bim"
    fam_file = "qc_results/step4_mind.fam"
    
    print("\nReading PLINK files...")
    bim_df = read_bim(bim_file)
    fam_df = read_fam(fam_file)
    
    print(f"SNPs: {len(bim_df)}")
    print(f"Individuals: {len(fam_df)}")
    
    breeds = fam_df['fid'].values
    unique_breeds = np.unique(breeds)
    print(f"Breeds: {list(unique_breeds)}")
    
    breed_counts = pd.Series(breeds).value_counts()
    print(f"\nBreed counts:\n{breed_counts}")
    
    # Extract genotypes using PLINK
    print("\nExtracting genotype data...")
    base = bed_file.replace('.bed', '')
    cmd = f"plink --bfile {base} --recode A --out {base}_raw"
    subprocess.run(cmd, shell=True, capture_output=True)
    
    raw_file = f"{base}_raw.raw"
    geno_df = pd.read_csv(raw_file, sep=r'\s+')
    
    meta_cols = ['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE']
    snp_cols = [c for c in geno_df.columns if c not in meta_cols]
    
    print(f"Genotype matrix: {geno_df[snp_cols].shape}")
    
    genotypes = geno_df[snp_cols].fillna(-1).values
    populations = geno_df['FID'].values
    pop_ids = np.unique(populations)
    pop_sizes = np.array([np.sum(populations == pop) for pop in pop_ids])
    
    # Calculate allele frequencies
    print("\nCalculating allele frequencies...")
    freqs = calculate_allele_frequencies(genotypes, populations, pop_ids)
    
    # Calculate statistics
    print("Calculating Wright's FST...")
    fst_values = calculate_fst(freqs, pop_sizes)
    
    print("Calculating Informativeness for Assignment (In)...")
    in_values = calculate_in(freqs)
    
    print("Calculating Delta values...")
    delta_values = calculate_delta(freqs)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'snp': bim_df['snp'].values,
        'chrom': bim_df['chrom'].values,
        'pos': bim_df['pos'].values,
        'FST': fst_values,
        'In': in_values,
        'delta': delta_values
    })
    
    for i, pop in enumerate(pop_ids):
        results_df[f'freq_{pop}'] = freqs[i, :]
    
    os.makedirs('aim_results', exist_ok=True)
    results_df.to_csv('aim_results/aim_statistics.csv', index=False)
    print(f"\nSaved: aim_results/aim_statistics.csv")
    
    # Select top AIMs
    top_n = min(500, len(results_df))
    
    top_fst = results_df.nlargest(top_n, 'FST')
    top_fst.to_csv('aim_results/top_aims_fst.csv', index=False)
    
    top_in = results_df.nlargest(top_n, 'In')
    top_in.to_csv('aim_results/top_aims_in.csv', index=False)
    
    top_delta = results_df.nlargest(top_n, 'delta')
    top_delta.to_csv('aim_results/top_aims_delta.csv', index=False)
    
    # Combined ranking
    results_df['rank_fst'] = results_df['FST'].rank(ascending=False)
    results_df['rank_in'] = results_df['In'].rank(ascending=False)
    results_df['rank_delta'] = results_df['delta'].rank(ascending=False)
    results_df['avg_rank'] = (results_df['rank_fst'] + results_df['rank_in'] + results_df['rank_delta']) / 3
    
    top_combined = results_df.nsmallest(top_n, 'avg_rank')
    top_combined.to_csv('aim_results/top_aims_combined.csv', index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("AIM Statistics Summary")
    print("="*60)
    print(f"FST: min={fst_values.min():.4f}, max={fst_values.max():.4f}, mean={fst_values.mean():.4f}")
    print(f"In: min={in_values.min():.4f}, max={in_values.max():.4f}, mean={in_values.mean():.4f}")
    print(f"Delta: min={delta_values.min():.4f}, max={delta_values.max():.4f}, mean={delta_values.mean():.4f}")
    
    print("\nTop 10 AIMs by FST:")
    print(top_fst[['snp', 'chrom', 'pos', 'FST', 'In', 'delta']].head(10).to_string(index=False))
    
    # Create SNP list
    top_aims_snps = top_combined['snp'].tolist()
    with open('aim_results/top_aims_snplist.txt', 'w') as f:
        for snp in top_aims_snps:
            f.write(f"{snp}\n")
    
    print(f"\nSaved top {len(top_aims_snps)} AIM SNPs")
    print("\nStep 3 completed!")

if __name__ == "__main__":
    main()
