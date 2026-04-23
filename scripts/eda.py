import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import argparse

def perform_eda(dataset_path, dataset_name, out_dir):
    print(f"--- Starting EDA for {dataset_name} ---")
    
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    total_images = 0
    class_counts = {}
    resolutions = []
    sample_images = {}
    
    print(f"Scanning {len(classes)} classes. This might take a few moments...")
    
    for cls in classes:
        cls_dir = os.path.join(dataset_path, cls)
        images = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        class_counts[cls] = len(images)
        total_images += len(images)
        
        if images:
            # Pick a random sample image for the grid
            sample_img = random.choice(images)
            sample_images[cls] = os.path.join(cls_dir, sample_img)
            
            # Analyze a subset of images for resolution to save time
            subset_for_res = random.sample(images, min(20, len(images)))
            for img_name in subset_for_res:
                img_path = os.path.join(cls_dir, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    h, w, _ = img.shape
                    resolutions.append((w, h))

    print(f"Total Images: {total_images}")
    print(f"Average Images per Class: {total_images / len(classes):.2f}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Bar Chart: Class Distribution
    # If 101 classes, just show top 20 and bottom 5 for readability
    plt.figure(figsize=(15, 6))
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_classes) > 30:
        display_classes = sorted_classes[:15] + sorted_classes[-10:]
        title = f"{dataset_name.capitalize()} - Class Distribution (Top 15 & Bottom 10)"
    else:
        display_classes = sorted_classes
        title = f"{dataset_name.capitalize()} - Class Distribution"
        
    labels = [c[0] for c in display_classes]
    values = [c[1] for c in display_classes]
    
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_class_distribution.png"))
    plt.close()
    
    # 2. Scatter Plot: Image Resolutions
    if resolutions:
        widths, heights = zip(*resolutions)
        plt.figure(figsize=(8, 6))
        plt.scatter(widths, heights, alpha=0.5, color='coral')
        plt.title(f"{dataset_name.capitalize()} - Image Resolution Spread (Sampled)")
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset_name}_resolutions.png"))
        plt.close()

    # 3. Sample Image Grid
    grid_size = min(11 if dataset_name == 'paris' else 25, len(classes))
    cols = 5
    rows = int(np.ceil(grid_size / cols))
    
    plt.figure(figsize=(15, 3 * rows))
    sampled_keys = list(sample_images.keys())
    random.shuffle(sampled_keys)
    
    for i, cls in enumerate(sampled_keys[:grid_size]):
        img_path = sample_images[cls]
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = plt.subplot(rows, cols, i + 1)
            ax.imshow(img)
            ax.set_title(cls)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{dataset_name}_sample_grid.png"))
    plt.close()
    
    print(f"Generated EDA plots saved to: {out_dir}")
    print("-" * 40)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, 'results', 'eda')
    
    paris_path = os.path.join(base_dir, 'data', 'paris')
    if os.path.exists(paris_path):
        perform_eda(paris_path, 'paris', out_dir)
        
    food_path = os.path.join(base_dir, 'data', 'food-101', 'images')
    if os.path.exists(food_path):
        perform_eda(food_path, 'food-101', out_dir)
