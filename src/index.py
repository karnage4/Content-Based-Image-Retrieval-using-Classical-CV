import numpy as np
import pickle
import os
from sklearn.neighbors import NearestNeighbors

class FeatureIndex:
    def __init__(self, metric='euclidean', n_neighbors=10):
        self.metric = metric
        self.n_neighbors = n_neighbors
        
        # We manually process chi2 without sklearn for ultra-fast vectorized performance
        algorithm = 'auto'
        if metric not in ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'chi2']:
            algorithm = 'brute'
            
        if self.metric != 'chi2':
            self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, algorithm=algorithm, n_jobs=-1)
        else:
            self.nn = None
            
        self.features = None
        self.image_paths = None

    def build_index(self, features, image_paths):
        self.features = np.array(features)
        self.image_paths = list(image_paths)
        if self.nn is not None:
            print("Fitting NearestNeighbors index...")
            self.nn.fit(self.features)
        else:
            print(f"Index mapped directly into memory for {self.metric} metric.")
        print("Index successfully built.")

    def save_index(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            'features': self.features,
            'image_paths': self.image_paths,
            'metric': self.metric,
            'n_neighbors': self.n_neighbors
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.metric = data['metric']
        self.n_neighbors = data['n_neighbors']
        algorithm = 'auto'
        if self.metric not in ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'chi2']:
            algorithm = 'brute'
            
        if self.metric != 'chi2':
            self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, algorithm=algorithm, n_jobs=-1)
        else:
            self.nn = None
            
        self.build_index(data['features'], data['image_paths'])

    def query(self, feature_vector, k=None):
        if k is None:
            k = self.n_neighbors
            
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        if self.metric == 'chi2':
            eps = 1e-10
            # Broadcast chi2 calculation across all features natively
            dists = 0.5 * np.sum(((self.features - feature_vector)**2) / (self.features + feature_vector + eps), axis=1)
            indices = np.argsort(dists)[:k]
            distances = dists[indices]
            
            results = []
            for i in range(len(indices)):
                idx = indices[i]
                results.append((self.image_paths[idx], distances[i]))
            return results
        else:
            distances, indices = self.nn.kneighbors(feature_vector, n_neighbors=k)
            
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                results.append((self.image_paths[idx], dist))
                
            return results
