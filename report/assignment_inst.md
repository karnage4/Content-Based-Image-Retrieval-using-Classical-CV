Assignment 4: Content-Based Image Retrieval
Using Classical Computer Vision Features

1. Introduction & Background
Content-Based Image Retrieval (CBIR) is the task of searching a database of images using visual content rather than metadata or text labels. Given a query image, a CBIR system extracts low-level visual features such as color, texture, shape, or local keypoints and retrieves the most visually similar images from a pre-indexed dataset.
In this assignment, you will implement a complete CBIR pipeline using classical computer vision and machine learning techniques only. You will work with two distinct real-world datasets and implement two retrieval modes: whole-image retrieval and region-based retrieval using a bounding box.
This assignment will develop your understanding of how hand-crafted feature descriptors capture visual information, and how similarity metrics and indexing structures enable efficient image search at scale.
Important: Deep learning methods (CNNs, ViT, autoencoders, pretrainedembeddings, etc.) are strictly prohibited. All feature extraction must use hand-crafted classical descriptors. Violation of this rule will result in zero marks for the affected component.
2. Datasets
You must use both of the following datasets. Your complete solution for both Part A and Part B must be implemented and submitted for each dataset independently.
Dataset 1 — Food-101
•	Size: 101,000 images across 101 food categories (1,000 images per class)
•	Content: Photographs of dishes such as pizza, sushi, steak, waffles, etc., captured in real-world settings with natural lighting variation
•	Characteristics: High intra-class variation, rich color and texture information, some background clutter
•	Download: https://www.tensorflow.org/datasets/catalog/food101

Dataset 2 — Paris Buildings (Paris 6K)
•	Size: 6,412 images of 11 Paris landmarks (Eiffel Tower, Notre Dame, Louvre, Moulin Rouge, etc.)
•	Content: Photographs of Paris architecture collected from Flickr, taken from varying viewpoints, distances, and lighting conditions
•	Characteristics: Repetitive structural patterns, viewpoint and scale variation, rich edge and texture content
•	Download: https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/

PART A — Whole-Image Similarity Search 
3. Part A: Whole-Image Similarity Search
3.1 Objective
Given a query image from a dataset, your program must retrieve and display the top 10 most visually similar images from the same dataset using classical feature extraction and distance-based ranking.
3.2 Feature Extraction
You must implement at least two different feature extraction methods and compare their retrieval performance. Each method must produce a fixed-length feature vector for every image in the dataset. The following methods are suggested, but you are encouraged to explore others:
Suggested Feature Descriptors
•	SIFT (Scale-Invariant Feature Transform) — Extract keypoint descriptors and aggregate using Bag of Visual Words (BoVW). Build a visual vocabulary via K-Means clustering on SIFT descriptors, then represent each image as a histogram of visual word frequencies.
•	HOG (Histogram of Oriented Gradients) — Resize images to a fixed resolution (e.g., 128×128) and extract HOG descriptors. Produces a fixed-length vector suitable for direct comparison.
•	LBP (Local Binary Patterns) — Effective for capturing micro-texture patterns. Particularly suited to Paris Buildings (stone facades, carved surfaces).
•	You can use any other feature extraction models as well.
3.3 Similarity Search
Once features are extracted and stored, implement a similarity or distance function to rank all database images against a query. You must implement and compare at least two distance/similarity measures. Options include:
•	Euclidean distance (L2 norm)
•	Cosine similarity
•	Chi-squared distance (recommended for histogram-based features)
•	Manhattan distance (L1 norm)
•	KD-Tree or Ball Tree search (via sklearn.neighbors)
•	FLANN-based approximate nearest neighbor search (via OpenCV)
•	You are allowed to use any other machine learning models or similarity measure as well. Any model/measure can be used without incorporating cnn or deep learning.
3.4 Program Interface
The program must accept a user-supplied query image (from the same dataset) and display the top 10 most similar images with their similarity scores. 
3.5 Deliverables — Part A
•	Working retrieval code for both datasets
•	Pre-extracted feature files saved to disk
•	Comparison table: Precision 10 for each feature method × each dataset
•	Visual results: 3–5 example queries showing the query image and top-10 retrieved images
PART B — Bounding Box Object Retrieval  
4. Part B: Bounding Box Object Retrieval
4.1 Objective
Given a query image and a user-drawn bounding box around a specific region of interest, your program must extract features from the cropped region and retrieve the top 10 most visually similar images from the full dataset based on that region's content.
4.2 Bounding Box Interface
Implement an interactive interface allowing the user to select a region of interest on a query image. Recommended approaches include:
•	cv2.selectROI() — Built-in OpenCV region selector (simplest option)
•	A Tkinter canvas with mouse-click bounding box drawing
•	A simple matplotlib interface with rectangle selectors
Once the region is selected, the bounding box coordinates (x, y, width, height) are passed to the feature extraction and retrieval pipeline.
4.3 Crop Feature Extraction & Retrieval
Crop the selected region from the query image and resize it to a fixed. Extract features from the crop using the same pipeline as Part A, then compare the crop's feature vector against the pre-computed vectors of all full images in the dataset. Return the top 10 matches by distance.
Dataset-specific guidance:
Food-101: The bounding box should isolate a specific food item within the image — for example, a slice of cake on a plate, or a sushi roll.
Paris Buildings: The bounding box should isolate an architectural element — for example, the Eiffel Tower's lattice structure, an arch of Notre Dame, or a window pattern on the Louvre facade.
4.4 Evaluation
Since ground truth for arbitrary crops is not formally available, you must perform a manual qualitative evaluation. For each dataset, demonstrate at least 5 different bounding box queries and provide a brief qualitative discussion on whether the retrieved results are semantically or visually relevant to the queried region.
4.5 Deliverables — Part B
•	Working bounding box retrieval code for both datasets
•	Screenshot of the bounding box selection interface in action
•	Visual results: top-10 retrieved images for at least 5 bounding box queries per dataset
•	Brief qualitative analysis (1–2 paragraphs per dataset)



PART C — Analysis & Written Report  
5. Part C: Analysis & Written Report
Write a structured latex report. The report must address all of the following sections:

5.1 Feature Descriptor Comparison
Compare the two or more feature methods you implemented across both datasets. Which method performed better on Food-101 vs. Paris Buildings? Provide quantitative evidence (Precision 10 table) and explain why certain features are better suited to certain types of visual content.
5.2 Computational Performance
Report the approximate time taken to: (a) extract and save features for the full dataset, and (b) execute a single query and retrieve the top-10 results. Describe any optimizations you applied, such as PCA for dimensionality reduction, approximate nearest neighbor indexing (FLANN, KD-Tree), or pre-computed distance matrices.
5.3 Effect of Bounding Box Size and Position
In Part B, what happened when the bounding box covered a very small region versus a large region? Did retrieval quality change depending on where in the image the box was placed (e.g., centered on the main object versus background)? Provide at least two contrasting examples.
5.4 Dataset-Specific Challenges
Discuss the specific challenges encountered with each dataset and how your chosen feature(s) addressed or failed to address them:
•	Food-101: Color similarity across different dishes, background clutter, varying portion sizes
•	Paris Buildings: Repetitive structures (e.g., many arches look similar), viewpoint and scale changes, illumination variation
5.5 Suggested Classical Improvements
Without using deep learning, describe at least two concrete techniques that could improve your retrieval results. Examples: spatial pyramid matching, VLAD (Vector of Locally Aggregated Descriptors) encoding, query expansion, spatial verification with RANSAC, or re-ranking using a second-stage classifier.
The EDA/understanding of the dataset and the step by step process and you own understanding of the implementation is also mandatory to add in the report.



