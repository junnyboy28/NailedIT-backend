import os
import shutil
import json
import numpy as np
from typing import Optional, Any
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

from run_pipeline import run_pipeline, visualize_results, evaluate_model

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Helper function to ensure metrics are JSON serializable
def make_serializable(metrics):
    """Convert any numpy values to Python standard types"""
    if isinstance(metrics, dict):
        return {k: make_serializable(v) for k, v in metrics.items()}
    elif isinstance(metrics, list):
        return [make_serializable(item) for item in metrics]
    elif isinstance(metrics, (np.integer, np.int32, np.int64)):
        return int(metrics)
    elif isinstance(metrics, (np.floating, np.float32, np.float64)):
        return float(metrics)
    elif isinstance(metrics, np.ndarray):
        return metrics.tolist()
    elif isinstance(metrics, np.bool_):
        return bool(metrics)
    return metrics

# Create the FastAPI app
app = FastAPI(title="Nail Detection API")

# Configure CORS - IMPORTANT: This must be before any route definitions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Changed to False since we're using "*" for origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up directory structure
os.makedirs("runs/detect/train/weights", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
# Make sure directories are writable
os.chmod("uploads", 0o777)

class EvaluationRequest(BaseModel):
    test_dir: str

@app.post("/api/detect")
async def detect_nails(
    file: UploadFile = File(...),
    use_kmeans: Optional[bool] = Form(False),
    background_tasks: BackgroundTasks = None
):
    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the image
    metrics = run_pipeline(file_path, return_metrics=True, use_kmeans=use_kmeans)
    
    # Convert numpy values to standard Python types for JSON serialization
    metrics = make_serializable(metrics)
    
    # Ensure all nested data is also serializable
    nail_details = make_serializable(metrics.get('nail_details', []))
    
    # Create visualizations
    csv_path = os.path.splitext(file_path)[0] + "_results.csv"
    visualize_results(file_path, csv_path)
    
    # Get output image paths
    output_image = os.path.basename(os.path.splitext(file_path)[0] + "_output.jpg")
    analysis_image = os.path.basename(os.path.splitext(file_path)[0] + "_analysis.png")
    results_csv = os.path.basename(csv_path)
    
    # Clean up old files after response is sent
    if background_tasks:
        background_tasks.add_task(lambda: os.remove(file_path) if os.path.exists(file_path) else None)
    
    return {
        'metrics': metrics,
        'output_image': output_image,
        'analysis_image': analysis_image,
        'results_csv': results_csv,
        'nail_details': nail_details
    }

@app.get("/api/images/{filename}")
async def get_image(filename: str):
    # Check both in uploads folder and the main directory
    if os.path.exists(os.path.join("uploads", filename)):
        return FileResponse(os.path.join("uploads", filename))
    if os.path.exists(filename):
        return FileResponse(filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.get("/api/files/{filename}")
async def get_file(filename: str):
    # Check both in uploads folder and the main directory
    if os.path.exists(os.path.join("uploads", filename)):
        return FileResponse(os.path.join("uploads", filename))
    if os.path.exists(filename):
        return FileResponse(filename)
    return JSONResponse(status_code=404, content={"error": "File not found"})

@app.post("/api/evaluate")
async def evaluate(request: EvaluationRequest):
    metrics = evaluate_model(request.test_dir)
    
    # Convert numpy values to standard Python types
    metrics = make_serializable(metrics)
    
    # Ensure metrics contain the expected property names for frontend
    if 'nail_count' in metrics and 'nails_detected' not in metrics:
        metrics['nails_detected'] = metrics['nail_count']
        
    if 'match_count' in metrics and 'matches_found' not in metrics:
        metrics['matches_found'] = metrics['match_count']
    
    return {
        'metrics': metrics,
        'evaluation_results_image': "evaluation_results.png",
        'nail_details': metrics.get('nail_details', [])  # Include nail details in evaluation too
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")  # Add this for debugging
    uvicorn.run("api:app", host="0.0.0.0", port=port)  # Remove reload=True in production