#!/bin/bash
# Step 1 & 2: PLINK Quality Control Pipeline
# Remove sex chromosomes (X, Y, MT), TOD individuals, apply call rate and mind filters

set -e

echo "============================================"
echo "Step 1 & 2: PLINK QC Pipeline"
echo "============================================"

INPUT="2000ml"
OUTPUT_PREFIX="buffalo_qc"

# Create output directory
mkdir -p qc_results
mkdir -p logs

echo ""
echo "Original data statistics:"
echo "Individuals: $(wc -l < ${INPUT}.fam)"
echo "SNPs: $(wc -l < ${INPUT}.bim)"

# Step 1: Remove TOD individuals
echo ""
echo "Step 1: Removing TOD individuals..."
awk '$1 != "TOD" {print $1"\t"$2}' ${INPUT}.fam > qc_results/keep_individuals_no_tod.txt
plink --bfile ${INPUT} --keep qc_results/keep_individuals_no_tod.txt --make-bed --out qc_results/step1_no_tod

echo "After removing TOD: $(wc -l < qc_results/step1_no_tod.fam) individuals"

# Step 2: Remove sex chromosomes and mitochondria
echo ""
echo "Step 2: Removing sex chromosomes (X, Y) and mitochondria (MT)..."

# Create list of chromosomes to exclude (X, Y, MT)
awk '$1 ~ /^[XxYy]$/ || $1 ~ /^[Mm][Tt]$/ || $1 == 0 || $1 == "X" || $1 == "Y" || $1 == "MT" {print $1"\t"$2}' qc_results/step1_no_tod.bim > qc_results/sex_mt_snps.txt

if [ -s qc_results/sex_mt_snps.txt ]; then
    echo "Found $(wc -l < qc_results/sex_mt_snps.txt) SNPs on sex chromosomes or mitochondria"
    plink --bfile qc_results/step1_no_tod --exclude qc_results/sex_mt_snps.txt --make-bed --out qc_results/step2_no_sex_mt
else
    echo "No sex/mt SNPs found by name. Proceeding with all autosomal SNPs..."
    cp qc_results/step1_no_tod.bed qc_results/step2_no_sex_mt.bed
    cp qc_results/step1_no_tod.bim qc_results/step2_no_sex_mt.bim
    cp qc_results/step1_no_tod.fam qc_results/step2_no_sex_mt.fam
fi

echo "After removing sex/mt SNPs: $(wc -l < qc_results/step2_no_sex_mt.bim) SNPs remaining"

# Step 3: Apply SNP call rate filter (>95%, i.e., --geno 0.05)
echo ""
echo "Step 3: Applying SNP call rate filter (>95%)..."
plink --bfile qc_results/step2_no_sex_mt --geno 0.05 --make-bed --out qc_results/step3_geno
echo "After CR>95%: $(wc -l < qc_results/step3_geno.bim) SNPs remaining"

# Step 4: Apply missing genotype filter (--mind 0.1, i.e., <10% missing)
echo ""
echo "Step 4: Applying missing genotype filter (--mind 0.1)..."
plink --bfile qc_results/step3_geno --mind 0.1 --make-bed --out qc_results/step4_mind
echo "After mind<0.1: $(wc -l < qc_results/step4_mind.fam) individuals remaining"
echo "After mind<0.1: $(wc -l < qc_results/step4_mind.bim) SNPs remaining"

# Final QC report
echo ""
echo "============================================"
echo "QC Summary"
echo "============================================"
echo "Original individuals: $(wc -l < 2000ml.fam)"
echo "Original SNPs: $(wc -l < 2000ml.bim)"
echo ""
echo "Final individuals: $(wc -l < qc_results/step4_mind.fam)"
echo "Final SNPs: $(wc -l < qc_results/step4_mind.bim)"
echo ""
echo "Breed distribution after QC:"
cut -f1 qc_results/step4_mind.fam | sort | uniq -c
echo ""
echo "Chromosome distribution after QC:"
cut -f1 qc_results/step4_mind.bim | sort | uniq -c

echo ""
echo "QC completed! Output: qc_results/step4_mind.{bed,bim,fam}"
