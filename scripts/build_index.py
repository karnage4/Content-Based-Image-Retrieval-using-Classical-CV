import os
import sys
import argparse
import numpy as np
import pickle

# Add parent dir to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.index import FeatureIndex

def build_and_save_index(dataset, method, metric='euclidean', n_neighbors=10):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(base_dir, 'features', dataset)
    
    feat_path = os.path.join(features_dir, f"{method}_features.npy")
    paths_path = os.path.join(features_dir, f"{method}_paths.pkl")
    
    if not os.path.exists(feat_path) or not os.path.exists(paths_path):
        print(f"Features for {method} on dataset {dataset} not found.")
        return
        
    print(f"Loading features from {feat_path}...")
    features = np.load(feat_path)
    with open(paths_path, 'rb') as f:
        paths = pickle.load(f)
        
    # If the user selects chi2 or cosine for histogram, we pass it
    # We will use manhattan or euclidean as starting point based on method
    if metric == 'auto':
        if 'hist' in method or 'lbp' in method or 'bovw' in method:
            # Chi-Squared is significantly better for comparing histograms
            metric = 'chi2'
        else:
            metric = 'euclidean'
            
    print(f"Building index using metric: {metric}")
    index = FeatureIndex(metric=metric, n_neighbors=n_neighbors)
    index.build_index(features, paths)
    
    index_path = os.path.join(features_dir, f"{method}_index.pkl")
    index.save_index(index_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['food-101', 'paris'])
    parser.add_argument('--method', required=True, choices=['color_hist', 'lbp', 'hog', 'sift_bovw', 'color_sift_bovw'])
    parser.add_argument('--metric', default='auto')
    args = parser.parse_args()
    
    build_and_save_index(args.dataset, args.method, args.metric)
