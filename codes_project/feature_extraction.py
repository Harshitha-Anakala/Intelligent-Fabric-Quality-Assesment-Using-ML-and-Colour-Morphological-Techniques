import os
import cv2
import random

# Input and output paths
input_folder = r'C:\Users\harsh\OneDrive\Desktop\project\shadow_removed_cnn'
output_base = r'C:\Users\harsh\OneDrive\Desktop\project\feture_new'

train_dir = os.path.join(output_base, 'train')
test_dir = os.path.join(output_base, 'test')
val_dir = os.path.join(output_base, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Collect image files
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Shuffle images randomly
random.shuffle(image_files)

# Split ratios
total = len(image_files)
train_split = int(0.7 * total)
test_split = int(0.9 * total)  # 70% train + 20% test

train_files = image_files[:train_split]
test_files = image_files[train_split:test_split]
val_files = image_files[test_split:]

def extract_features(file_list, save_dir):
    sift = cv2.SIFT_create()
    
    for filename in file_list:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # SIFT Feature Extraction
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        sift_img = cv2.drawKeypoints(img, keypoints, None)

        # Canny Edge Detection
        edges = cv2.Canny(gray, 100, 200)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Combine SIFT + Edges side by side
        combined = cv2.hconcat([sift_img, edges_bgr])

        # Save output
        out_path = os.path.join(save_dir, f'features_{filename}')
        cv2.imwrite(out_path, combined)

# Process splits
extract_features(train_files, train_dir)
extract_features(test_files, test_dir)
extract_features(val_files, val_dir)

print("✅ Feature extraction completed with dataset split!")
print(f"Training images: {len(train_files)}")
print(f"Testing images: {len(test_files)}")
print(f"Validation images: {len(val_files)}")
print(f"Total images: {total}")
print(f"Training images: {len(train_files)} ({len(train_files)/total*100:.2f}%)")
print(f"Testing images: {len(test_files)} ({len(test_files)/total*100:.2f}%)")
print(f"Validation images: {len(val_files)} ({len(val_files)/total*100:.2f}%)")

