# Nail Detection and Analysis API

This API provides nail detection, measurement, and matching capabilities using computer vision and machine learning.

## Features

- Detect nails in images using YOLOv8
- Estimate nail height and weight
- Match similar nails using either traditional methods or K-means clustering
- Visualize results with annotated images and analysis charts

## Application Screenshots

### Landing Page and Navigation
The NailedIT application features a modern, responsive interface designed for intuitive user interaction.

![Main dashboard with navigation menu, upload options, and quick start guide](ss/image.png)

### Image Upload Interface
Multiple upload methods provide flexibility for users to input nail images for analysis.

![Drag-and-drop upload area, file browser button, and sample image gallery](ss/image-1.png)

### Processing and Analysis
Real-time progress indicators keep users informed during the AI analysis process.

![Processing screen with progress bar, estimated time remaining, and current analysis step](ss/image-2.png)

### Results Dashboard
Comprehensive results display with both visual and numerical analysis outputs.

![Detected nails with bounding boxes, measurements overlay, and detection confidence scores](ss/image-3.png)

### Analytics and Metrics
Detailed performance metrics and distribution charts for in-depth analysis.

![Bar charts of nail distributions, scatter plots of height vs weight, and performance metrics cards](ss/image-4.png)

### Matching Results
Visual representation of matched nail pairs with similarity scores and matching criteria.

![Paired nails highlighted with connecting lines, matching scores, and algorithm selection options](ss/image-5.png)

### Mobile Responsive Design
Optimized mobile experience maintaining full functionality across devices.

![Mobile view of the application with collapsed navigation and touch-friendly interface](ss/image-6.png)  ![Mobile view of the application with collapsed navigation and touch-friendly interface](ss/image-7.png)

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone this repository:

git clone https://github.com/junnyboy28/NailedIT-backend.git

2. Install dependencies:

pip install -r requirements_backend.txt

3. Download pre-trained model weights (if not included in the repository)

### Running Locally

Start the API server:

python api.py

The API will be available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

## API Endpoints

- `POST /api/detect` - Upload and analyze an image
- `GET /api/images/{filename}` - Get processed images
- `GET /api/files/{filename}` - Get result files (CSV)
- `POST /api/evaluate` - Evaluate model on a test dataset

## CLI Usage

You can also use the pipeline directly from the command line:

python run_pipeline.py --image path/to/image.jpg --kmeans --visualize

## Deployment

This API can be deployed to Render using the included configuration.

## Contact

For access to the full training dataset or any questions, contact me: jimilmandani28@gmail.com

