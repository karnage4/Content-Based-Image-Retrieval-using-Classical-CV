import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.cluster import MiniBatchKMeans

class FeatureExtractor:
    def __init__(self, method='color_hist', **kwargs):
        """
        method: 'color_hist', 'lbp', 'hog', 'sift_bovw', 'color_sift_bovw'
        """
        self.method = method
        self.kwargs = kwargs
        
        if method in ['sift_bovw', 'color_sift_bovw']:
            self.sift = cv2.SIFT_create()
            # vocab will be fitted later
            self.kmeans = None
            self.vocab_size = kwargs.get('vocab_size', 500)
            
    def compute_color_hist(self, image):
        """ Computes a 3D HSV color histogram. """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256]).flatten()
        # Explicit L1 Normalization for histograms
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist
        
    def compute_lbp(self, image):
        """ Computes Spatial LBP histogram to encode layout. """
        gray = cv2.cvtColor(cv2.resize(image, (256, 256)), cv2.COLOR_BGR2GRAY)
        # Using radius 2 and 16 points for richer texture
        radius = self.kwargs.get('lbp_radius', 2)
        n_points = self.kwargs.get('lbp_points', 8 * radius)
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = int(lbp.max() + 1)
        
        grid_y, grid_x = 4, 4
        h, w = lbp.shape
        h_step, w_step = h // grid_y, w // grid_x
        
        hists = []
        for i in range(grid_y):
            for j in range(grid_x):
                block = lbp[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                hist, _ = np.histogram(block.ravel(), bins=n_bins, range=(0, n_bins), density=True)
                hists.append(hist)
                
        # Normalize concatenated vector
        final_hist = np.concatenate(hists)
        if final_hist.sum() > 0:
            final_hist /= final_hist.sum()
            
        return final_hist

    def compute_hog(self, image):
        """ Computes HOG feature descriptor. """
        # Resize to fixed size to have consistent HOG features
        resized = cv2.resize(image, (128, 128))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        features = hog(gray, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
                       
        # Global L2 Normalization for dense vectors
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
            
        return features
        
    def extract_sift_descriptors(self, image):
        image = cv2.resize(image, (256, 256))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Max descriptor sampling to prevent explosion
        max_desc = 200
        if descriptors is not None and len(descriptors) > max_desc:
            sampled_indices = np.random.choice(len(descriptors), max_desc, replace=False)
            descriptors = descriptors[sampled_indices]
            
        return descriptors

    def extract_color_sift_descriptors(self, image):
        """ Smart HS Color SIFT descriptors - Skips V channel for lighting invariance """
        image = cv2.resize(image, (256, 256))
        
        # Convert to HSV and extract Hue and Saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_chan, s_chan, _ = cv2.split(hsv)
        
        # Detect keypoints on Grayson logic for structural layout
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints = self.sift.detect(gray, None)
        
        if not keypoints:
            return None
            
        # Compute descriptors solely on H and S
        _, d1 = self.sift.compute(h_chan, keypoints)
        _, d2 = self.sift.compute(s_chan, keypoints)
        
        if d1 is None or d2 is None:
            return None
            
        descriptors = np.hstack((d1, d2))
        
        # Max descriptor sampling to prevent explosion
        max_desc = 200
        if len(descriptors) > max_desc:
            sampled_indices = np.random.choice(len(descriptors), max_desc, replace=False)
            descriptors = descriptors[sampled_indices]
            
        return descriptors

    def compute_bovw_hist(self, descriptors):
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.vocab_size)
        words = self.kmeans.predict(descriptors)
        hist, _ = np.histogram(words, bins=self.vocab_size, range=(0, self.vocab_size))
        # Normalize
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist

    def compute(self, image):
        if self.method == 'color_hist':
            return self.compute_color_hist(image)
        elif self.method == 'lbp':
            return self.compute_lbp(image)
        elif self.method == 'hog':
            return self.compute_hog(image)
        elif self.method == 'sift_bovw':
            if self.kmeans is None:
                raise ValueError("KMeans vocabulary not initialized for BOVW.")
            des = self.extract_sift_descriptors(image)
            return self.compute_bovw_hist(des)
        elif self.method == 'color_sift_bovw':
            if self.kmeans is None:
                raise ValueError("KMeans vocabulary not initialized for BOVW.")
            des = self.extract_color_sift_descriptors(image)
            return self.compute_bovw_hist(des)
        else:
            raise ValueError(f"Unknown method {self.method}")

    def build_vocabulary(self, descriptor_list):
        print(f"Building vocabulary of size {self.vocab_size} using MiniBatchKMeans...")
        self.kmeans = MiniBatchKMeans(n_clusters=self.vocab_size, batch_size=1000, random_state=42, n_init=3)
        self.kmeans.fit(descriptor_list)
        print("Vocabulary built.")
