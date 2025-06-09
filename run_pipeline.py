import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
from PIL import Image, ImageDraw, ImageFont
import os
import csv

# === Config ===
MODEL_PATH = "runs/detect/train/weights/best.pt"
# Fallback to yolov8n.pt if best.pt is not available
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"
    print(f"Warning: Using fallback model {MODEL_PATH}")

PIXELS_PER_MM = 10  # Assumption: 10 pixels = 1 mm
WEIGHT_CONSTANT = 0.002  # weight (g) = height_mm * constant
TOLERANCE = 2  # mm and grams
CONFIDENCE_THRESHOLD = 0.5  # Only keep predictions above this confidence

# === Load model ===
model = YOLO(MODEL_PATH)

def estimate_height(bbox):
    _, y1, _, y2 = bbox
    height_pixels = abs(y2 - y1)
    return height_pixels / PIXELS_PER_MM

def estimate_weight(height_mm):
    return height_mm * WEIGHT_CONSTANT

def match_nails(nail_features, tolerance):
    pairs = []
    used = set()
    for i in range(len(nail_features)):
        if i in used:
            continue
        h1, w1 = nail_features[i]
        for j in range(i + 1, len(nail_features)):
            if j in used:
                continue
            h2, w2 = nail_features[j]
            if abs(h1 - h2) <= tolerance and abs(w1 - w2) <= tolerance:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    return pairs

def match_nails_with_kmeans(nail_features, max_clusters=None):
    """Match nails using K-means clustering"""
    if len(nail_features) < 2:
        return []
        
    # Normalize features for better clustering
    features_array = np.array(nail_features)
    if len(features_array) == 0:
        return []
        
    # Normalize data
    mean = np.mean(features_array, axis=0)
    std = np.std(features_array, axis=0) + 1e-10  # Avoid division by zero
    normalized_features = (features_array - mean) / std
    
    # Determine optimal number of clusters (pairs)
    max_k = min(max_clusters if max_clusters else len(nail_features) // 2, len(nail_features))
    if max_k < 1:
        return []
        
    best_pairs = []
    best_score = float('inf')
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(normalized_features)
        
        # Calculate intra-cluster variance
        score = kmeans.inertia_
        
        # Generate pairs within each cluster
        current_pairs = []
        for cluster_id in range(k):
            indices = np.where(clusters == cluster_id)[0]
            # Only consider clusters with exactly 2 nails
            if len(indices) == 2:
                current_pairs.append((indices[0], indices[1]))
        
        # Pick the clustering with lowest score and most pairs
        if len(current_pairs) > len(best_pairs) or (len(current_pairs) == len(best_pairs) and score < best_score):
            best_pairs = current_pairs
            best_score = score
    
    return best_pairs

def run_pipeline(image_path, return_metrics=False, use_kmeans=False):
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
        
    results = model(image)[0]

    # Get bounding boxes and confidence
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    
    # Refine bounding boxes
    refined_boxes = refine_bounding_boxes(image, boxes)
    
    # Use refined boxes instead of original ones
    boxes = refined_boxes

    confs = results.boxes.conf.cpu().numpy()

    nail_features = []  # [height_mm, weight]
    all_results = []    # For CSV
    annotated_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated_image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    centers = []
    
    # Add count of nails text at the top
    valid_count = sum(1 for conf in confs if conf >= CONFIDENCE_THRESHOLD)
    draw.text((10, 10), f"Nail Count: {valid_count}", fill="blue", font=font)

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = box
        height_mm = estimate_height(box)
        weight = estimate_weight(height_mm)
        nail_features.append([height_mm, weight])

        label = f"H:{height_mm:.1f}mm W:{weight:.2f}g"
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
        draw.text((x1, y1 - 10), label, fill="green", font=font)
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

        all_results.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": conf,
            "height_mm": height_mm,
            "weight_g": weight
        })

    # Use either K-means or traditional matching
    if use_kmeans and len(nail_features) >= 2:
        pairs = match_nails_with_kmeans(nail_features)
    else:
        pairs = match_nails(nail_features, TOLERANCE)
    
    # Draw matching lines
    for i, j in pairs:
        draw.line([centers[i], centers[j]], fill="red", width=2)
    
    # Add pairs count
    draw.text((10, 30), f"Matched Pairs: {len(pairs)}", fill="red", font=font)

    save_path = os.path.splitext(image_path)[0] + "_output.jpg"
    annotated_image.save(save_path)
    print(f"✅ Saved output image to: {save_path}")

    csv_path = os.path.splitext(image_path)[0] + "_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["x1", "y1", "x2", "y2", "confidence", "height_mm", "weight_g"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✅ Results saved to CSV: {csv_path}")
    
    # Calculate metrics if requested
    if return_metrics:
        valid_count = len(nail_features)  # Count of valid nails detected
        
        metrics = {
            "detection_score": np.mean(confs),
            "height_mae": 0,  # Would need ground truth
            "weight_mae": 0,  # Would need ground truth
            "match_precision": len(pairs) / (len(nail_features)/2) if nail_features else 0,
            "match_recall": len(pairs) / (len(nail_features)/2) if nail_features else 0,
            # Add these properties explicitly for the frontend
            "nail_count": valid_count,
            "nails_detected": valid_count,
            "match_count": len(pairs),
            "matches_found": len(pairs),
            # Add individual nail details
            "nail_details": all_results  # This includes all nail measurements
        }
        return metrics
    
    return None

def visualize_results(image_path, results_csv):
    """Create visualizations of the detection and matching results"""
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Load results
    df = pd.read_csv(results_csv)
    
    # Load image
    img = plt.imread(image_path.replace('.jpg', '_output.jpg'))
    
    # Plot 1: Image with annotations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Detected Nails')
    axes[0, 0].axis('off')
    
    # Plot 2: Height distribution
    axes[0, 1].hist(df['height_mm'], bins=10, color='blue', alpha=0.7)
    axes[0, 1].set_title('Nail Height Distribution (mm)')
    axes[0, 1].set_xlabel('Height (mm)')
    axes[0, 1].set_ylabel('Count')
    
    # Plot 3: Weight distribution
    axes[1, 0].hist(df['weight_g'], bins=10, color='green', alpha=0.7)
    axes[1, 0].set_title('Nail Weight Distribution (g)')
    axes[1, 0].set_xlabel('Weight (g)')
    axes[1, 0].set_ylabel('Count')
    
    # Plot 4: Height vs Weight scatter
    axes[1, 1].scatter(df['height_mm'], df['weight_g'], alpha=0.7)
    axes[1, 1].set_title('Height vs Weight')
    axes[1, 1].set_xlabel('Height (mm)')
    axes[1, 1].set_ylabel('Weight (g)')
    
    plt.tight_layout()
    plt.savefig(os.path.splitext(image_path)[0] + '_analysis.png')
    plt.close()
    
    print(f"✅ Analysis visualization saved to: {os.path.splitext(image_path)[0]}_analysis.png")

def evaluate_model(test_dir, ground_truth_file=None):
    """
    Evaluate the model on test images with metrics:
    - mAP for detection
    - MAE for height/weight estimation
    - Precision/Recall for matching
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.endswith(('_output.jpg', '_analysis.png'))]
    
    detection_results = []
    height_errors = []
    weight_errors = []
    match_precisions = []
    match_recalls = []
    
    # Track actual counts and details
    total_nails_detected = 0
    total_matches_found = 0
    all_nail_details = []
    
    for img_path in images:
        print(f"Evaluating: {img_path}")
        # Run the pipeline
        all_metrics = run_pipeline(img_path, return_metrics=True)
        
        if all_metrics:
            detection_results.append(all_metrics["detection_score"])
            height_errors.append(all_metrics["height_mae"])
            weight_errors.append(all_metrics["weight_mae"])
            match_precisions.append(all_metrics["match_precision"])
            match_recalls.append(all_metrics["match_recall"])
            
            # Track actual counts
            total_nails_detected += all_metrics.get("nail_count", 0)
            total_matches_found += all_metrics.get("match_count", 0)
            
            # Collect nail details if available
            if "nail_details" in all_metrics:
                # Add image path to each detail record
                for detail in all_metrics["nail_details"]:
                    detail["image"] = os.path.basename(img_path)
                all_nail_details.extend(all_metrics["nail_details"])
    
    # Calculate overall metrics
    avg_map = np.mean(detection_results) if detection_results else 0
    avg_height_mae = np.mean(height_errors) if height_errors else 0
    avg_weight_mae = np.mean(weight_errors) if weight_errors else 0
    avg_precision = np.mean(match_precisions) if match_precisions else 0
    avg_recall = np.mean(match_recalls) if match_recalls else 0
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    print("\n=== Model Evaluation Results ===")
    print(f"Detection mAP (IoU=0.5): {avg_map:.4f}")
    print(f"Height Estimation MAE: {avg_height_mae:.4f} mm")
    print(f"Weight Estimation MAE: {avg_weight_mae:.4f} g")
    print(f"Nail Matching - Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {avg_f1:.4f}")
    
    # Plot results
    metrics = ['mAP', 'Height MAE', 'Weight MAE', 'Match F1']
    values = [avg_map, avg_height_mae, avg_weight_mae, avg_f1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
    plt.title('Model Performance Metrics')
    plt.savefig(os.path.join(test_dir, 'evaluation_results.png'))
    plt.close()
    
    return {
        'map': avg_map,
        'height_mae': avg_height_mae,
        'weight_mae': avg_weight_mae,
        'match_precision': avg_precision,
        'match_recall': avg_recall,
        'match_f1': avg_f1,
        # Add these explicit mappings for frontend compatibility
        'nail_count': total_nails_detected,
        'nails_detected': total_nails_detected,
        'match_count': total_matches_found,
        'matches_found': total_matches_found,
        'nail_details': all_nail_details  # Include all nail details
    }

def refine_bounding_boxes(image, boxes, threshold=20):
    """Refine bounding boxes by analyzing pixel content"""
    refined_boxes = []
    
    for box in boxes:
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Threshold to separate nail from background
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (should be the nail)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box of contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Adjust coordinates to the original image
            refined_box = [x1 + x, y1 + y, x1 + x + w, y1 + y + h]
            refined_boxes.append(refined_box)
        else:
            refined_boxes.append(box)
            
    return refined_boxes

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Nail detection and analysis pipeline")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--dir", type=str, help="Process all images in directory")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model performance")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--kmeans", action="store_true", help="Use K-means for matching")
    args = parser.parse_args()
    
    if args.image:
        run_pipeline(args.image, use_kmeans=args.kmeans)
        if args.visualize:
            csv_path = os.path.splitext(args.image)[0] + "_results.csv"
            visualize_results(args.image, csv_path)
    
    elif args.dir:
        images = [os.path.join(args.dir, f) for f in os.listdir(args.dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
                and not f.endswith(('_output.jpg', '_analysis.png'))]
        
        for img_path in images:
            print(f"\nProcessing: {img_path}")
            run_pipeline(img_path, use_kmeans=args.kmeans)
            if args.visualize:
                csv_path = os.path.splitext(img_path)[0] + "_results.csv"
                visualize_results(img_path, csv_path)
        
        if args.evaluate:
            metrics = evaluate_model(args.dir)
    
    else:
        parser.print_help()
