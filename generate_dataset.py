import os
import random
import cv2
import numpy as np
from PIL import Image
import albumentations as A

# Set paths - updated to match your actual directory structure
BASE_DIR = "."  # Current directory, not "nail_dataset"
NAILS_DIR = os.path.join(BASE_DIR, "nails_cutouts")
BACKGROUNDS_DIR = os.path.join(BASE_DIR, "backgrounds")
OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, "synthetic_images")
OUTPUT_LABELS_DIR = os.path.join(BASE_DIR, "labels")

# Create output dirs if not exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# Load file names
nail_filenames = sorted([f for f in os.listdir(NAILS_DIR) if f.endswith('.png')])
background_filenames = sorted([f for f in os.listdir(BACKGROUNDS_DIR) if f.endswith('.jpg')])

# Settings
NUM_SYNTHETIC_IMAGES = 500
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
MIN_NAILS = 3
MAX_NAILS = 7

# Helper to paste a transparent PNG onto a background


def improved_paste_nail(bg, nail, position, scale=1.0, angle=0):
    """Create tighter bounding boxes by using a minimum area rectangle approach"""
    # Calculate original size
    original_w, original_h = nail.size
    scaled_w, scaled_h = int(original_w * scale), int(original_h * scale)
    
    # Resize first (to avoid quality loss)
    nail_resized = nail.resize((scaled_w, scaled_h))
    
    # Create a mask of the nail (assuming the fourth channel is alpha)
    nail_mask = np.array(nail_resized)[:, :, 3]
    
    # Rotate without expand to maintain size
    nail_rotated = nail_resized.rotate(angle, expand=False, resample=Image.BICUBIC)
    
    # Find non-zero pixels in alpha channel to calculate the actual occupied area
    nail_array = np.array(nail_rotated)
    alpha = nail_array[:, :, 3]
    non_zero_pixels = np.where(alpha > 50)  # Filter out very transparent pixels
    
    if len(non_zero_pixels[0]) > 0:
        # Find the actual bounds of the nail
        min_y, max_y = np.min(non_zero_pixels[0]), np.max(non_zero_pixels[0])
        min_x, max_x = np.min(non_zero_pixels[1]), np.max(non_zero_pixels[1])
        
        # Create a new tighter image with just the nail
        tight_nail = nail_rotated.crop((min_x, min_y, max_x+1, max_y+1))
        
        nx, ny = position
        
        # Paste nail on background
        bg.paste(tight_nail, (nx, ny), tight_nail)
        
        # Return actual dimensions for the bounding box
        return bg, (max_x-min_x+1, max_y-min_y+1), (nx, ny)
    else:
        return bg, (scaled_w, scaled_h), position

# Main generation loop
for i in range(NUM_SYNTHETIC_IMAGES):
    bg_file = random.choice(background_filenames)
    bg_path = os.path.join(BACKGROUNDS_DIR, bg_file)
    bg = Image.open(bg_path).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    # Add more augmentation
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(p=0.3),
    ])

    # Apply to background before pasting nails
    bg_array = np.array(bg)
    bg_array = transform(image=bg_array)['image']
    bg = Image.fromarray(bg_array)

    num_nails = random.randint(MIN_NAILS, MAX_NAILS)
    boxes = []

    for _ in range(num_nails):
        nail_file = random.choice(nail_filenames)
        nail_path = os.path.join(NAILS_DIR, nail_file)
        nail = Image.open(nail_path).convert("RGBA")

        # Calculate appropriate scale to ensure nail fits in background
        max_scale = min(0.8, 
                      min(IMAGE_WIDTH / nail.width, 
                          IMAGE_HEIGHT / nail.height))
        scale = random.uniform(0.1, max_scale)
        angle = random.uniform(-30, 30)
        
        # Pre-calculate the size after resize and rotation
        temp_nail = nail.resize((int(nail.width * scale), int(nail.height * scale)))
        temp_nail = temp_nail.rotate(angle, expand=True)
        
        # Ensure the nail will fit within the background
        if temp_nail.width >= IMAGE_WIDTH:
            temp_nail = temp_nail.resize((int(IMAGE_WIDTH * 0.8), int(temp_nail.height * 0.8 * (IMAGE_WIDTH / temp_nail.width))))
        if temp_nail.height >= IMAGE_HEIGHT:
            temp_nail = temp_nail.resize((int(temp_nail.width * 0.8 * (IMAGE_HEIGHT / temp_nail.height)), int(IMAGE_HEIGHT * 0.8)))
        
        # Calculate valid position ranges
        max_x = max(0, IMAGE_WIDTH - temp_nail.width)
        max_y = max(0, IMAGE_HEIGHT - temp_nail.height)
        
        # Handle edge case if max_x or max_y is 0
        nx = random.randint(0, max(1, max_x))
        ny = random.randint(0, max(1, max_y))

        # Paste nail and get final size
        bg, (bw, bh), (nx, ny) = improved_paste_nail(bg, nail, (nx, ny), scale=scale, angle=angle)

        # Calculate bounding box in YOLO format
        x_center = (nx + bw / 2) / IMAGE_WIDTH
        y_center = (ny + bh / 2) / IMAGE_HEIGHT
        width = min(bw / IMAGE_WIDTH, 1.0)  # Ensure width doesn't exceed 1
        height = min(bh / IMAGE_HEIGHT, 1.0)  # Ensure height doesn't exceed 1
        boxes.append([0, x_center, y_center, width, height])  # class_id = 0 for nail

    # Save image
    out_image_path = os.path.join(OUTPUT_IMAGES_DIR, f"{i}.jpg")
    bg.save(out_image_path)

    # Save labels
    out_label_path = os.path.join(OUTPUT_LABELS_DIR, f"{i}.txt")
    with open(out_label_path, "w") as f:
        for box in boxes:
            f.write(" ".join([str(round(b, 6)) for b in box]) + "\n")

    if i % 50 == 0:
        print(f"[{i}] images generated...")

print(f"\n✅ Done! {NUM_SYNTHETIC_IMAGES} images + labels generated.")  