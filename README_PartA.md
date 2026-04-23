# Assignment 4 - Part A: Whole-Image Similarity Search
**Student Deliverables & Grading Reference Guide**

This document serves as a direct roadmap mapping the strictly required Part A deliverables to their exact locations within the compiled codebase.

---

### 1. Working Retrieval Code for Both Datasets
The retrieval codebase is dynamically built to support generic integration across the subsets.
*   **Core Extraction Logic:** `src/features.py` (Contains mathematical implementations for HOG, Spatial LBP, HSV Color Hist, and HS Color SIFT-BoVW).
*   **Feature Indexing & Metrics:** `src/index.py` (Handles KNN vector indexing and our native/custom *Chi-Squared* evaluation metric bypassing Sklearn limitations).
*   **Execution Script:** `scripts/retrieve.py` (Takes `--dataset`, `--method`, and a `--query` path to evaluate and render out matches via Matplotlib).
*   **Interactive Evaluation GUI:** `scripts/gui.py` (A dedicated user interface for executing searches across both datasets identically).

### 2. Pre-Extracted Feature Files Saved to Disk
All offline computations, matrices, and normalized visual vocabulary clusters have been efficiently serialized and mapped directly onto the hard drive to allow instant $O(1)$ query retrieval.
*   **Paris Index Files:** `features/paris/` (Contains `.npy` dense matrices and `.pkl` data/vocabulary maps).
*   **Food-101 Index Files:** `features/food-101/` (Contains highly compacted/sampled matrices limiting scale overhead safely to 20,200 arrays).

*Note: You can re-generate these locally by executing the `extract_all.ps1` PowerShell pipeline.*

### 3. Comparison Table: Precision@10
Mathematical evaluation of the *Semantic Gap* across all 4 specific feature implementations (evaluated across two independent metric types depending on algorithm density) has been fully documented statistically.
*   **Automated Evaluation Script:** `scripts/evaluate.py` (Loops via iterative randomized subsets against the indexed structures to output exact `Precision@10` floats).
*   **Formal LaTeX Table:** Located strictly inside `report/report_partA.tex` (See Table 1 under Section 5).

### 4. Visual Results (3–5 Example Queries)
To demonstrate visual retrieval capability, the engine intrinsically captures its query logic and renders output composites showcasing the isolated Query image adjacent to the top-10 ranked distances.
*   **Output Render Location:** Automatically generated and saved natively to the `results/paris/` and `results/food-101/` directories whenever `scripts/retrieve.py` or the `GUI` is engaged. *(If you wish to grade visual outputs manually, please view the stored `.png` plots populated identically there).*
