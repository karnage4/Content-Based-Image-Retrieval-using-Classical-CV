# Assignment 4: Content-Based Image Retrieval (CBIR) Using Classical Computer Vision Features

## 1. Introduction & Core Constraints
Content-Based Image Retrieval (CBIR) focuses on searching an image database using visual content (low-level features like color, texture, and shape) rather than text labels.

* **Strict Rule:** Deep learning (CNNs, ViTs, Autoencoders, etc.) is **strictly prohibited**.
* **Consequence:** Using prohibited methods will result in zero marks for that component.
* **Requirement:** All features must be hand-crafted classical descriptors.

---

## 2. Dataset Specifications
You are required to implement and submit your solution for two distinct datasets independently[cite: 11]:

| Dataset | Size | Characteristics |
| :--- | :--- | :--- |
| **Food-101** | 101,000 images (101 classes) | High intra-class variation, natural lighting, background clutter |
| **Paris Buildings** | 6,412 images (11 landmarks) | Repetitive structural patterns, viewpoint/scale changes, rich edge content |

> **Efficiency Note for Food-101:** Given the 101k image volume, focus on "first principles" to simplify and remove redundancy. This includes using efficient data structures (KD-Trees/FLANN) and dimensionality reduction (PCA) to ensure the system runs on standard hardware.

---

## 3. Part A: Whole-Image Similarity Search
**Objective:** Retrieve and display the top 10 most visually similar images for a given query.

### 3.1 Feature Extraction (Implement at least 2)
* **SIFT + Bag of Visual Words (BoVW):** Cluster SIFT descriptors via K-Means to create a visual vocabulary, then represent images as histograms of visual word frequencies.
* **Color SIFT:** **(Recommended)** Explore "Color SIFT" (e.g., Opponent SIFT) to incorporate chromatic information alongside local geometry.
* **HOG (Histogram of Oriented Gradients):** Extract gradients from resized images (e.g., $128 \times 128$) for shape-based comparison.
* **LBP (Local Binary Patterns):** Excellent for micro-textures, specifically for stone facades in the Paris dataset.
* **Color Spaces:** Utilize different color spaces (HSV, LAB, or YCrCb) to extract color histograms and analyze their impact on retrieval accuracy.

### 3.2 Similarity & Indexing (Implement at least 2)
* **Metrics:** Euclidean (L2), Cosine Similarity, Chi-squared distance (ideal for histograms), or Manhattan (L1).
* **Efficient Search:** Use `sklearn.neighbors` (KD-Tree, Ball Tree) or OpenCV's FLANN-based search to handle large-scale retrieval.

---

## 4. Part B: Bounding Box Object Retrieval
**Objective:** Retrieve images based on a user-selected Region of Interest (ROI).

### 4.1 Interactive Interface
Use one of the following to select a bounding box:
* `cv2.selectROI()`: The simplest built-in OpenCV tool.
* **Tkinter/Matplotlib:** Custom canvas or rectangle selectors.

### 4.2 Retrieval Pipeline
1.  **Crop & Resize:** Isolate the selected region and resize it to a fixed resolution.
2.  **Feature Extraction:** Extract features from the crop using the same methods as Part A.
3.  **Comparison:** Compare the crop's feature vector against the pre-computed database of full-image vectors.
4.  **Results:** Return the top 10 matches.

---

## 5. Part C: Analysis & LaTeX Report
You must produce a structured report including:

* **EDA & Implementation:** A step-by-step breakdown of your implementation logic and an Exploratory Data Analysis (EDA) of the datasets.
* **Feature Comparison:** Quantitative evidence (Precision@10 table) showing which method worked best for specific content (e.g., LBP for architecture vs. Color Histograms for food).
* **Performance Metrics:** Report time taken for feature extraction and single-query execution. Mention optimizations like PCA or approximate indexing.
* **Bounding Box Analysis:** Discuss the impact of box size and position (central vs. background) with at least two contrasting examples.
* **Dataset Challenges:** Discuss how you handled background clutter in Food-101 and viewpoint/scale variations in Paris Buildings.
* **Proposed Improvements:** Describe two classical (non-DL) techniques to improve results, such as Spatial Pyramid Matching, RANSAC verification, or Query Expansion.

---

## 6. Summary of Deliverables
1.  **Code:** Functional retrieval scripts for both datasets (Whole-Image and Bounding Box).
2.  **Feature Files:** Pre-extracted feature vectors saved to disk.
3.  **Visual Evidence:** * 3–5 Whole-image query results (Top 10).
    * Screenshot of the ROI selection interface.
    * 5 Bounding box query results per dataset.
4.  **Formal Report:** The structured LaTeX document covering all analysis points.# Assignment 4: Content-Based Image Retrieval (CBIR) Using Classical Computer Vision Features

## 1. Introduction & Core Constraints
Content-Based Image Retrieval (CBIR) focuses on searching an image database using visual content (low-level features like color, texture, and shape) rather than text labels.

* **Strict Rule:** Deep learning (CNNs, ViTs, Autoencoders, etc.) is **strictly prohibited**. 
* **Consequence:** Using prohibited methods will result in zero marks for that component.
* **Requirement:** All features must be hand-crafted classical descriptors.

---

## 2. Dataset Specifications
You are required to implement and submit your solution for two distinct datasets independently[cite: 11]:

| Dataset | Size | Characteristics |
| :--- | :--- | :--- |
| **Food-101** | 101,000 images (101 classes) | High intra-class variation, natural lighting, background clutter |
| **Paris Buildings** | 6,412 images (11 landmarks) | Repetitive structural patterns, viewpoint/scale changes, rich edge content |

> **Efficiency Note for Food-101:** Given the 101k image volume, focus on "first principles" to simplify and remove redundancy. This includes using efficient data structures (KD-Trees/FLANN) and dimensionality reduction (PCA) to ensure the system runs on standard hardware.

---

## 3. Part A: Whole-Image Similarity Search
**Objective:** Retrieve and display the top 10 most visually similar images for a given query.

### 3.1 Feature Extraction (Implement at least 2)
* **SIFT + Bag of Visual Words (BoVW):** Cluster SIFT descriptors via K-Means to create a visual vocabulary, then represent images as histograms of visual word frequencies.
* **Color SIFT:** **(Recommended)** Explore "Color SIFT" (e.g., Opponent SIFT) to incorporate chromatic information alongside local geometry.
* **HOG (Histogram of Oriented Gradients):** Extract gradients from resized images (e.g., $128 \times 128$) for shape-based comparison.
* **LBP (Local Binary Patterns):** Excellent for micro-textures, specifically for stone facades in the Paris dataset.
* **Color Spaces:** Utilize different color spaces (HSV, LAB, or YCrCb) to extract color histograms and analyze their impact on retrieval accuracy.

### 3.2 Similarity & Indexing (Implement at least 2)
* **Metrics:** Euclidean (L2), Cosine Similarity, Chi-squared distance (ideal for histograms), or Manhattan (L1).
* **Efficient Search:** Use `sklearn.neighbors` (KD-Tree, Ball Tree) or OpenCV's FLANN-based search to handle large-scale retrieval.

---

## 4. Part B: Bounding Box Object Retrieval
**Objective:** Retrieve images based on a user-selected Region of Interest (ROI).

### 4.1 Interactive Interface
Use one of the following to select a bounding box:
* `cv2.selectROI()`: The simplest built-in OpenCV tool.
* **Tkinter/Matplotlib:** Custom canvas or rectangle selectors.

### 4.2 Retrieval Pipeline
1.  **Crop & Resize:** Isolate the selected region and resize it to a fixed resolution.
2.  **Feature Extraction:** Extract features from the crop using the same methods as Part A.
3.  **Comparison:** Compare the crop's feature vector against the pre-computed database of full-image vectors.
4.  **Results:** Return the top 10 matches.

---

## 5. Part C: Analysis & LaTeX Report
You must produce a structured report including:

* **EDA & Implementation:** A step-by-step breakdown of your implementation logic and an Exploratory Data Analysis (EDA) of the datasets.
* **Feature Comparison:** Quantitative evidence (Precision@10 table) showing which method worked best for specific content (e.g., LBP for architecture vs. Color Histograms for food).
* **Performance Metrics:** Report time taken for feature extraction and single-query execution. Mention optimizations like PCA or approximate indexing.
* **Bounding Box Analysis:** Discuss the impact of box size and position (central vs. background) with at least two contrasting examples.
* **Dataset Challenges:** Discuss how you handled background clutter in Food-101 and viewpoint/scale variations in Paris Buildings.
* **Proposed Improvements:** Describe two classical (non-DL) techniques to improve results, such as Spatial Pyramid Matching, RANSAC verification, or Query Expansion.

---

## 6. Summary of Deliverables
1.  **Code:** Functional retrieval scripts for both datasets (Whole-Image and Bounding Box).
2.  **Feature Files:** Pre-extracted feature vectors saved to disk.
3.  **Visual Evidence:** * 3–5 Whole-image query results (Top 10).
    * Screenshot of the ROI selection interface.
    * 5 Bounding box query results per dataset.
4.  **Formal Report:** The structured LaTeX document covering all analysis points.


------------

### important instructions

also firstly as the size of the first dataset is 101,000, we are advised to "Use power of conventional CV/Image processing, if you are good at first principles you know how to simplify / remove redundancy and run efficiently on small devices" and secondly "you are recommended to use color spaces and their impacts and also explore the concept of color SIFT" for this assignment.

---------------
### overall pipeline (high level)
1. Load dataset
2. Extract features
3. Save features
4. Build index (optional)
5. Query → extract feature
6. Compute similarity
7. Retrieve top-10
8. Evaluate (Precision@10)

---------------------
the two datasets are copied in its folder with the subfolders 'food-101' for the first one and 'paris' as the second one and the rest of the file structure is already created, we just need to fill in the code

---------------------