# Object Detection and Selective Blur System

This project implements an intelligent object detection and selective blur system using YOLO (You Only Look Once) and OpenCV. The system can detect specified objects in real-time video or images, blur the background while keeping the detected objects clear, and provide various deployment options.

## Features

- Real-time object detection using YOLO
- Selective background blur with clear target objects
- Multiple deployment options:
  - Local camera feed processing
  - Web-based interface with adjustable blur intensity
  - Batch processing for image files
- Intelligent bounding box clustering for better object grouping
- Configurable confidence and distance thresholds

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/object-detection-blur.git
cd object-detection-blur
```

2. Install the required dependencies:
```bash
pip install ultralytics opencv-python numpy scipy flask
```

3. Download the YOLO model and place it in the `model` directory.

## 📝 Usage

### Local Camera Feed Processing

Run the basic version with local camera feed:
```bash
python obj_dct_and_blur.py
```

### Web Interface Version

Start the web server for browser-based access:
```bash
python other_device.py
```
Then open your browser and navigate to `http://localhost:8080`

### Batch Image Processing

Process multiple images from a directory:
```bash
python save_blur_image.py
```

## ⚙️ Configuration

You can adjust the following parameters in each script:

- `model_path`: Path to your YOLO model file
- `target_class`: Class name for detection (default: 'toothbrush')
- `confidence_threshold`: Detection confidence threshold (default: 0.1)
- `distance_threshold`: Distance threshold for clustering (default: 300)
- `blur_intensity`: Blur effect strength (web interface only)

## 🗂️ Project Structure

```
├── obj_dct_and_blur.py    # Basic camera feed processing
├── other_device.py        # Web interface version
├── save_blur_image.py     # Batch image processing
├── model/                 # YOLO model directory
├── images/               # Input images directory
└── output/              # Processed images directory
```

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementation
- [OpenCV](https://opencv.org/) for image processing capabilities
