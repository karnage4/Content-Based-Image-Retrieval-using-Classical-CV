# Content-Based Image Retrieval (CBIR) Using Classical Computer Vision Features

This repository contains the implementation of a Content-Based Image Retrieval (CBIR) system using strictly classical computer vision features (no deep learning). 

The system supports two datasets:
- **Food-101**: High intra-class variation, natural lighting, background clutter.
- **Paris Buildings**: Repetitive structural patterns, viewpoint/scale changes, rich edge content.

## Features Implemented

The system extracts visual characteristics utilizing the following handcrafted features:
- **SIFT + Bag of Visual Words (BoVW)**
- **Color Histograms** (across various color spaces like HSV, LAB)
- **HOG (Histogram of Oriented Gradients)** (for shape and outline comparison)
- **LBP (Local Binary Patterns)** (for extensive micro-texture analysis, useful for Paris dataset)
- **Color SIFT**

## Project Structure

```text
.
├── data/               # Place the datasets here ('food-101' and 'paris')
├── features/           # Extracted features and BoVW vocabulary will be stored here
├── results/            # Temporary or output results
├── src/                # Core logic
│   ├── dataset.py      # Dataset loading utilities
│   ├── features.py     # Extracted feature logic classes (SIFT, HOG, LBP, Color)
│   └── index.py        # K-NN / KD-tree / Indexing search classes
├── scripts/            # Command line pipelines and GUI scripts
│   ├── extract_features.py # Extraction pipeline
│   ├── build_index.py      # Index building
│   ├── retrieve.py         # Main retrieve functionality script
│   ├── evaluate.py         # Evaluates the dataset and computes Precision@10
│   ├── eda.py              # Exploratory Data Analysis
│   ├── gui.py              # Tkinter graphical user interface for retrieval
│   ├── gui_sift.py         # GUI for visualizing SIFT logic
│   └── visualize_sift.py   # Visualization utility script
└── requirements.txt    # Project dependencies
```

## Setup & Installation

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure datasets are correctly placed in `data/food-101` and `data/paris`. The PowerShell script `extract_all.ps1` can be used for extractions if the datasets are compressed.

## Execution Pipeline

The CBIR system works in a few distinct steps. It is recommended to run the scripts as modules from the root processing directory.

### 1. Exploratory Data Analysis (EDA)
You can optionally run the EDA script to visualize images, class distributions, and bounding box characteristics of the dataset.
```bash
python -m scripts.eda --dataset paris
```

### 2. Feature Extraction
Extract your chose descriptor set across the whole dataset. The features are saved onto the disk (`features/`) for fast indexing later.
```bash
python -m scripts.extract_features --dataset paris --feature_type lbp
python -m scripts.extract_features --dataset food-101 --feature_type color_hist
```

### 3. Build Index
Build a fast nearest-neighbor index using `NearestNeighbors` or FLANN over the saved feature arrays and dump it into `features/`.
```bash
python -m scripts.build_index --dataset paris --feature_type lbp --metric euclidean
```

### 4. Query Retrieval
Perform a test retrieval for a specific whole-image or an ROI (bounding box).
```bash
# General retrieval
python -m scripts.retrieve --dataset paris --feature_type lbp --query_img path/to/query.jpg

# Region of Interest (Bounding Box) retrieval
python -m scripts.retrieve --dataset paris --feature_type lbp --query_img path/to/query.jpg --use_roi
```

### 5. Evaluation
Test the performance of your features using Precision@10 on a subset of test queries.
```bash
python -m scripts.evaluate --dataset paris --feature_type lbp
```

## GUI Interface

For interactive Bounding Box object retrieval (Part B constraint) or whole image retrieval, an interface is provided. You can select ROIs via custom GUI selections.

```bash
python -m scripts.gui
```

To visualize SIFT keypoints specifically:
```bash
python -m scripts.gui_sift
```
