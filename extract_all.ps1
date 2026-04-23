# Extraction and Indexing Script for CV Assignment 4

Write-Host "========================================="
Write-Host "Starting Feature Extraction Pipeline"
Write-Host "========================================="

# 1. Feature: LBP
Write-Host "`n--- Extracting LBP ---"
python scripts/extract_features.py --dataset paris --method lbp
python scripts/build_index.py --dataset paris --method lbp

python scripts/extract_features.py --dataset food-101 --method lbp --limit_per_class 200
python scripts/build_index.py --dataset food-101 --method lbp

# 2. Feature: HOG
Write-Host "`n--- Extracting HOG ---"
python scripts/extract_features.py --dataset paris --method hog
python scripts/build_index.py --dataset paris --method hog

python scripts/extract_features.py --dataset food-101 --method hog --limit_per_class 200
python scripts/build_index.py --dataset food-101 --method hog

# 3. Feature: Color Histogram (Extremely fast, highly recommended for Food-101 comparison)
Write-Host "`n--- Extracting Color Histogram ---"
python scripts/extract_features.py --dataset paris --method color_hist
python scripts/build_index.py --dataset paris --method color_hist

python scripts/extract_features.py --dataset food-101 --method color_hist --limit_per_class 200
python scripts/build_index.py --dataset food-101 --method color_hist

# 4. Feature: Color SIFT BoVW (This will take the longest!)
Write-Host "`n--- Extracting Color SIFT BoVW ---"
python scripts/extract_features.py --dataset paris --method color_sift_bovw --vocab_size 200
python scripts/build_index.py --dataset paris --method color_sift_bovw

python scripts/extract_features.py --dataset food-101 --method color_sift_bovw --vocab_size 200 --limit_per_class 200
python scripts/build_index.py --dataset food-101 --method color_sift_bovw

Write-Host "`n========================================="
Write-Host "Done! All datasets are formally indexed."
Write-Host "========================================="
