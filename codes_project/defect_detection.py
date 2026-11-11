import os
import cv2
import numpy as np
import random
import shutil

# Input and output paths
input_folder = r'C:\Users\harsh\OneDrive\Desktop\project\shadow_removed_cnn'
output_base = r'C:\Users\harsh\OneDrive\Desktop\project\defect_new'

train_dir = os.path.join(output_base, 'train')
test_dir = os.path.join(output_base, 'test')
val_dir = os.path.join(output_base, 'val')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Collect all image files
image_files = [f for f in os.listdir(input_folder)
               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# Shuffle images
random.shuffle(image_files)

# Split ratios
total = len(image_files)
train_split = int(0.7 * total)
test_split = int(0.9 * total)  # 70% train + 20% test

train_files = image_files[:train_split]
test_files = image_files[train_split:test_split]
val_files = image_files[test_split:]

def process_and_save(file_list, save_dir):
    for filename in file_list:
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        out_path = os.path.join(save_dir, f"box_{filename}")
        cv2.imwrite(out_path, img)

# Process each split
process_and_save(train_files, train_dir)
process_and_save(test_files, test_dir)
process_and_save(val_files, val_dir)

print("✅ Dataset split and defect detection completed!")
print(f"Training images: {len(train_files)}")
print(f"Testing images: {len(test_files)}")
print(f"Validation images: {len(val_files)}")
