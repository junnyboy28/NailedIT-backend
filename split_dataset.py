import os
import random
import shutil

# Updated: Use current directory as base instead of nail_dataset
base_dir = "."  # Current directory
images_dir = os.path.join(base_dir, "synthetic_images")
labels_dir = os.path.join(base_dir, "labels")

# Create a directory structure for train/val split
train_img_out = os.path.join("nail_dataset", "images/train")
val_img_out = os.path.join("nail_dataset", "images/val")
train_lbl_out = os.path.join("nail_dataset", "labels/train")
val_lbl_out = os.path.join("nail_dataset", "labels/val")

os.makedirs(train_img_out, exist_ok=True)
os.makedirs(val_img_out, exist_ok=True)
os.makedirs(train_lbl_out, exist_ok=True)
os.makedirs(val_lbl_out, exist_ok=True)

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
random.shuffle(image_files)

split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

def move_split(files, img_out, lbl_out):
    for f in files:
        shutil.copy(os.path.join(images_dir, f), os.path.join(img_out, f))
        lbl_file = f.replace(".jpg", ".txt")
        shutil.copy(os.path.join(labels_dir, lbl_file), os.path.join(lbl_out, lbl_file))

move_split(train_files, train_img_out, train_lbl_out)
move_split(val_files, val_img_out, val_lbl_out)

print(f"✅ Split complete: {len(train_files)} train / {len(val_files)} val")
