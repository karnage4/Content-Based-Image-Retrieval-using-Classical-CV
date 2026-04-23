import os
import sys
import argparse
import numpy as np
import pickle
from tqdm import tqdm

# Add parent dir to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import ImageDataset
from src.features import FeatureExtractor

def extract_for_dataset(dataset_path, feature_method, out_dir, kwargs={}):
    print(f"Loading dataset from: {dataset_path}")
    
    dataset_kwargs = {}
    if 'limit_per_class' in kwargs:
        dataset_kwargs['limit_per_class'] = kwargs['limit_per_class']
        
    dataset = ImageDataset(dataset_path, **dataset_kwargs)
    
    # We may need to limit for testing if it's too large, but let's process all.
    # If BoVW, we first need to build a vocabulary.
    extractor = FeatureExtractor(method=feature_method, **kwargs)
    
    # Check if we need to build vocabulary
    if feature_method in ['sift_bovw', 'color_sift_bovw']:
        print("Collecting random descriptors for vocabulary...")
        # sample randomly ~1000 images for vocab building
        sample_indices = np.random.choice(len(dataset), size=min(1000, len(dataset)), replace=False)
        all_descriptors = []
        for idx in tqdm(sample_indices, desc="Extracting descriptors for vocab"):
            _, img = dataset[idx]
            if img is not None:
                if feature_method == 'sift_bovw':
                    des = extractor.extract_sift_descriptors(img)
                else:
                    des = extractor.extract_color_sift_descriptors(img)
                if des is not None:
                    # Randomly sample some descriptors from each image to avoid memory limits
                    if len(des) > 100:
                        des_sampled = des[np.random.choice(len(des), 100, replace=False)]
                    else:
                        des_sampled = des
                    all_descriptors.append(des_sampled)
                    
        if len(all_descriptors) > 0:
            descriptor_list = np.vstack(all_descriptors)
            extractor.build_vocabulary(descriptor_list)
            # save the kmeans model
            os.makedirs(out_dir, exist_ok=True)
            vocab_path = os.path.join(out_dir, f"{feature_method}_vocab.pkl")
            with open(vocab_path, 'wb') as f:
                pickle.dump(extractor.kmeans, f)
            print(f"Saved vocabulary to {vocab_path}")
        else:
            print("Failed to collect descriptors for vocabulary.")
            return

    # Extract features for all images
    features = []
    valid_paths = []
    
    print(f"Extracting features using method: {feature_method}")
    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        path, img = dataset[idx]
        if img is None:
            continue
            
        feat = extractor.compute(img)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)
            
    features = np.array(features, dtype=np.float32)
    os.makedirs(out_dir, exist_ok=True)
    
    feat_path = os.path.join(out_dir, f"{feature_method}_features.npy")
    paths_path = os.path.join(out_dir, f"{feature_method}_paths.pkl")
    
    np.save(feat_path, features)
    with open(paths_path, 'wb') as f:
        pickle.dump(valid_paths, f)
        
    print(f"Saved features shape {features.shape} to {feat_path}")
    print(f"Saved paths to {paths_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['food-101', 'paris'])
    parser.add_argument('--method', required=True, choices=['color_hist', 'lbp', 'hog', 'sift_bovw', 'color_sift_bovw'])
    parser.add_argument('--vocab_size', type=int, default=500)
    parser.add_argument('--limit_per_class', type=int, default=None, help='Limit number of images per class')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'data', args.dataset)
    
    if args.dataset == 'food-101':
        dataset_path = os.path.join(dataset_path, 'images')

    out_dir = os.path.join(base_dir, 'features', args.dataset)
    
    kwargs = {}
    if args.method in ['sift_bovw', 'color_sift_bovw']:
        kwargs['vocab_size'] = args.vocab_size
    if args.limit_per_class is not None:
        kwargs['limit_per_class'] = args.limit_per_class
        
    extract_for_dataset(dataset_path, args.method, out_dir, kwargs)
