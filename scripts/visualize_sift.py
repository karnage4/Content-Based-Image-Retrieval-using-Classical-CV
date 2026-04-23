import os
import cv2
import argparse
import matplotlib.pyplot as plt

def draw_single_image_keypoints(img_path, out_path):
    print(f"Extracting SIFT keypoints for {img_path}...")
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image.")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, _ = sift.detectAndCompute(gray, None)
    
    print(f"Found {len(keypoints)} keypoints.")
    
    # Draw keypoints (flags=4 draws rich keypoints with circle showing scale and orientation)
    img_with_kp = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT Keypoints ({len(keypoints)} found)")
    plt.axis('off')
    
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved keypoint visualization to {out_path}")
    plt.show()

def draw_image_matches(img1_path, img2_path, out_path):
    print(f"Matching SIFT descriptors between {img1_path} and {img2_path}...")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Failed to load one or both images.")
        return
        
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Brute Force Matcher with default spatial params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Lowe's Ratio Test to mathematically isolate pure matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])
            
    print(f"Found {len(good_matches)} high-quality structural matches!")
            
    # Draw the spatial connecting lines
    match_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, 
                                   matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"SIFT Descriptor Matching ({len(good_matches)} robust connections via Lowe's Ratio)")
    plt.axis('off')
    
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved matching visualization to {out_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SIFT Keypoints and Matches for Reports.")
    parser.add_argument('--img1', required=True, help="Path to first image (Query)")
    parser.add_argument('--img2', required=False, help="Path to second image (Database Match)")
    parser.add_argument('--out', required=False, help="Save path for the resulting plot (e.g. results/sift_plot.png)")
    
    args = parser.parse_args()
    
    if args.img2:
        draw_image_matches(args.img1, args.img2, args.out)
    else:
        draw_single_image_keypoints(args.img1, args.out)
