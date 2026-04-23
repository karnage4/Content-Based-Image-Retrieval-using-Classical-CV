import os
import cv2
import glob

class ImageDataset:
    def __init__(self, dataset_path, **kwargs):
        self.dataset_path = dataset_path
        self.limit_per_class = kwargs.get('limit_per_class', None)
        self.image_paths = []
        self._load_image_paths()

    def _load_image_paths(self):
        # We will look for common image formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        all_paths = []
        for ext in extensions:
            # Check if dataset has subfolders or flat structure
            path_glob = os.path.join(self.dataset_path, "**", ext)
            all_paths.extend(glob.glob(path_glob, recursive=True))
        
        # Deduplicate paths (Windows glob is case-insensitive)
        all_paths = list(set([os.path.normpath(p) for p in all_paths]))
        
        # Sort to ensure consistent order
        all_paths = sorted(all_paths)
        
        if self.limit_per_class is not None:
            from collections import defaultdict
            class_groups = defaultdict(list)
            for p in all_paths:
                cls_name = os.path.basename(os.path.dirname(p))
                class_groups[cls_name].append(p)
                
            for cls_name, paths in class_groups.items():
                if len(paths) > self.limit_per_class:
                    step = max(1, len(paths) // self.limit_per_class)
                    sampled = paths[::step][:self.limit_per_class]
                    self.image_paths.extend(sampled)
                else:
                    self.image_paths.extend(paths)
        else:
            self.image_paths = all_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Failed to load image {img_path}")
        return img_path, image
    
    def get_path(self, idx):
        return self.image_paths[idx]
