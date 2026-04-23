import os
import sys
import argparse
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt

# Add parent dir to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.index import FeatureIndex
from src.features import FeatureExtractor

def plot_results(query_img, results, out_path=None):
    fig = plt.figure(figsize=(15, 8))
    
    # Plot query
    ax = fig.add_subplot(3, 4, 1)
    ax.imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Query Image")
    ax.axis('off')
    
    # Plot results
    for i, (path, dist) in enumerate(results):
        img_res = cv2.imread(path)
        if img_res is not None:
            ax = fig.add_subplot(3, 4, i + 2)
            ax.imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Rank {i+1}\nDist: {dist:.3f}")
            ax.axis('off')
            
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        print(f"Saved results plot to {out_path}")
    
    # Always display the plot
    plt.show()

def retrieve(dataset, method, query_path, use_roi=False):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(base_dir, 'features', dataset)
    
    index_path = os.path.join(features_dir, f"{method}_index.pkl")
    
    if not os.path.exists(index_path):
        print(f"Index for {method} on dataset {dataset} not found. Build it first.")
        return
        
    print(f"Loading index from {index_path}...")
    index = FeatureIndex()
    index.load_index(index_path)
    
    # Load feature extractor and vocab if needed
    kwargs = {}
    if method in ['sift_bovw', 'color_sift_bovw']:
        vocab_path = os.path.join(features_dir, f"{method}_vocab.pkl")
        with open(vocab_path, 'rb') as f:
            kmeans = pickle.load(f)
        kwargs['vocab_size'] = kmeans.n_clusters
        
    extractor = FeatureExtractor(method=method, **kwargs)
    if method in ['sift_bovw', 'color_sift_bovw']:
        extractor.kmeans = kmeans
        
    # Read query image
    query_img = cv2.imread(query_path)
    if query_img is None:
        print(f"Could not read query image {query_path}")
        return
        
    display_img = query_img.copy()
        
    if use_roi:
        print("Select a bounding box and press SPACE or ENTER. Press c to cancel.")
        # Create a resizable window to handle images larger than the monitor natively
        cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select ROI", min(query_img.shape[1], 1000), min(query_img.shape[0], 700))
        
        # cv2.selectROI opens a window and returns (x, y, w, h)
        roi = cv2.selectROI("Select ROI", query_img, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        x, y, w, h = roi
        if w > 0 and h > 0:
            query_img = query_img[y:y+h, x:x+w]
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # requirement to resize to fixed resolution
            query_img = cv2.resize(query_img, (128, 128))
            print(f"Cropped to ({x}, {y}, {w}, {h}) and resized.")
        else:
            print("Invalid ROI selected or cancelled. Exiting.")
            return

    # Extract feature
    feat = extractor.compute(query_img)
    if feat is None:
        print("Failed to extract features from query image.")
        return
        
    # Query index
    print("Querying index for top 10 matches...")
    # Ask for 15 to ensure we have at least 10 after removing the query itself
    results_raw = index.query(feat, k=15)
    
    query_abs = os.path.normcase(os.path.abspath(query_path))
    results = []
    for path, dist in results_raw:
        if os.path.normcase(os.path.abspath(path)) != query_abs:
            results.append((path, dist))
        if len(results) == 10:
            break
            
    print("\nTop 10 matches:")
    for i, (path, dist) in enumerate(results):
        print(f"{i+1}: {path} (Distance: {dist:.4f})")
        
    out_dir = os.path.join(base_dir, 'results', dataset)
    os.makedirs(out_dir, exist_ok=True)
    query_name = os.path.basename(query_path).split('.')[0]
    out_file = os.path.join(out_dir, f"{query_name}_{method}_{'roi' if use_roi else 'whole'}.png")
    
    plot_results(display_img, results, out_path=out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['food-101', 'paris'])
    parser.add_argument('--method', required=True, choices=['color_hist', 'lbp', 'hog', 'sift_bovw', 'color_sift_bovw'])
    parser.add_argument('--query', required=True, help='Path to query image')
    parser.add_argument('--roi', action='store_true', help='Use Bounding Box region selection')
    args = parser.parse_args()
    
    retrieve(args.dataset, args.method, args.query, args.roi)
