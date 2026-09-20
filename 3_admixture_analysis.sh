#!/bin/bash
# Step 4: ADMIXTURE Analysis from k=2 to n (n=number of breeds)

set -e

echo "============================================"
echo "Step 4: ADMIXTURE Analysis"
echo "============================================"

mkdir -p admixture_results
mkdir -p admixture_plots

INPUT_BED="qc_results/step4_mind"

# Get number of breeds
N_BREEDS=$(awk '{print $1}' ${INPUT_BED}.fam | sort | uniq | wc -l)
echo "Number of breeds detected: $N_BREEDS"

# Run ADMIXTURE for K=2 to N_BREEDS (max 7 for practical purposes)
MAX_K=$((N_BREEDS + 2))
if [ $MAX_K -gt 7 ]; then
    MAX_K=7
fi

echo ""
echo "Running ADMIXTURE for K=2 to K=$MAX_K..."
echo ""

for K in $(seq 2 $MAX_K); do
    echo "Running ADMIXTURE for K=$K..."
    
    admixture --cv ${INPUT_BED}.bed $K > admixture_results/log${K}.out 2>&1
    
    mv step4_mind.${K}.Q admixture_results/K${K}.Q
    mv step4_mind.${K}.P admixture_results/K${K}.P
    
    CV_ERROR=$(grep -h "CV error" admixture_results/log${K}.out | awk '{print $4}')
    echo "K=$K, CV error=$CV_ERROR"
done

echo ""
echo "Cross-validation errors:" > admixture_results/cv_errors.txt
for K in $(seq 2 $MAX_K); do
    CV_ERROR=$(grep -h "CV error" admixture_results/log${K}.out | awk '{print $4}')
    echo "K=$K CV=$CV_ERROR" >> admixture_results/cv_errors.txt
done

echo "CV errors summary:"
cat admixture_results/cv_errors.txt

# Create population order file
awk '{print $1}' ${INPUT_BED}.fam > admixture_results/populations.txt
paste ${INPUT_BED}.fam admixture_results/populations.txt > admixture_results/ind_order.txt

echo ""
echo "ADMIXTURE analysis completed!"
ls -la admixture_results/
