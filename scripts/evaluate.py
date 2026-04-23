import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm

# Add parent dir to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.index import FeatureIndex

def get_class_name(filepath):
    """
    Given a dataset filepath like data/paris/eiffel/eiffel_1.jpg
    returns 'eiffel'
    """
    return os.path.basename(os.path.dirname(filepath))

def evaluate(dataset, method):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(base_dir, 'features', dataset)
    
    index_path = os.path.join(features_dir, f"{method}_index.pkl")
    feat_path = os.path.join(features_dir, f"{method}_features.npy")
    paths_path = os.path.join(features_dir, f"{method}_paths.pkl")
    
    if not os.path.exists(index_path) or not os.path.exists(feat_path):
        print(f"Index or features for {method} on dataset {dataset} not found. Build it first.")
        return
        
    print(f"Loading index from {index_path}...")
    index = FeatureIndex()
    index.load_index(index_path)
    
    print(f"Loading features to use as queries...")
    features = np.load(feat_path)
    with open(paths_path, 'rb') as f:
        paths = pickle.load(f)
        
    num_queries = min(1000, len(paths)) # evaluate on 1000 random queries
    np.random.seed(42)
    query_indices = np.random.choice(len(paths), num_queries, replace=False)
    
    precisions = []
    
    print(f"Evaluating Precision@10 for {method} on {dataset} using {num_queries} queries...")
    for idx in tqdm(query_indices):
        q_feat = features[idx]
        q_path = paths[idx]
        q_class = get_class_name(q_path)
        
        # we ask for 11 so we can ignore the query itself (distance 0)
        results = index.query(q_feat, k=11)
        
        relevant_retrieved = 0
        total_eval = 0
        
        for p, dist in results:
            if p == q_path:
                continue
            r_class = get_class_name(p)
            if r_class == q_class:
                relevant_retrieved += 1
            total_eval += 1
            if total_eval == 10:
                break
                
        prec_at_10 = relevant_retrieved / 10.0
        precisions.append(prec_at_10)
        
    mean_ap = np.mean(precisions)
    print(f"Mean Precision@10 for {method}: {mean_ap:.4f}")
    return mean_ap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['food-101', 'paris'])
    parser.add_argument('--method', required=True, choices=['color_hist', 'lbp', 'hog', 'sift_bovw', 'color_sift_bovw'])
    args = parser.parse_args()
    
    evaluate(args.dataset, args.method)
