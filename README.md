# Object Detection Model with Enhanced YOLOv8

This repository contains an enhanced implementation of YOLOv8 for object detection, specifically trained to detect FireExtinguisher, ToolBox, and OxygenTank objects. The implementation includes custom training and prediction scripts with improved monitoring and analysis capabilities.

> **Note**: The `data/` folder containing the training, validation, and test datasets is not included in this repository due to its large size. The dataset was provided by the organizers in a separate zip file.

## Project Overview

This project builds upon the pre-existing models provided by the organizers. We first evaluated the provided models and then developed our own enhanced versions trained on the same dataset to improve performance.

## Requirements

```bash
# Core dependencies
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
pyyaml>=5.4.0
```

## Project Structure

```
├── train_enhanced.py        # Enhanced training script with improved monitoring
├── predict_enhanced.py      # Prediction script with confidence-based detection
├── data/                   # Dataset directory (not included in repo)
│   ├── train/             # Training dataset
│   ├── val/               # Validation dataset
│   └── test/              # Test dataset
├── runs/
│   └── detect/
│       ├── train_personal_model/  # Our custom model training results
│       │   ├── weights/          # Trained model weights
│       │   ├── results.png       # Training metrics visualization
│       │   ├── confusion_matrix.png
│       │   ├── F1_curve.png
│       │   ├── PR_curve.png
│       │   └── val_batch*.jpg    # Validation predictions
│       └── val_personal_model/   # Validation results from our model
│           ├── confusion_matrix.png
│           ├── PR_curve.png
│           └── val_batch*.jpg    # Test set predictions
└── yolo_params.yaml        # YOLO configuration parameters

```

## Dataset

The dataset used in this project is organized in the YOLO format but is not included in this repository due to its substantial size. To use this project:

1. Obtain the dataset from the organizers
2. Place it in the `data/` directory with the following structure:
   ```
   data/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

## Setup Instructions

1. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install ultralytics opencv-python numpy pyyaml
   ```

3. Place the dataset in the `data/` directory as described above

## Training the Model

The training script (`train_enhanced.py`) includes several improvements over the basic YOLOv8 training:

1. Run the training:
   ```bash
   python train_enhanced.py
   ```

Key features:
- Automatic dataset analysis for challenging cases
- Enhanced monitoring and statistics tracking
- Conservative augmentation pipeline
- Detailed training statistics saved to 'training_stats.json'

### Training Results

The training results are stored in `runs/detect/train_personal_model/` and include:
- `results.png`: Training metrics plot
- `confusion_matrix.png`: Model's classification performance
- `F1_curve.png`, `P_curve.png`, `R_curve.png`: Performance curves
- `val_batch*_pred.jpg`: Validation predictions
- `weights/best.pt`: Best model weights

## Making Predictions

To run predictions using your trained model:

```bash
python predict_enhanced.py
```

The prediction script will:
1. Automatically locate the best model weights
2. Process test images with a confidence threshold of 0.5
3. Save annotated images and detection results

### Prediction Results

Results are stored in `runs/detect/val_personal_model/` and include:
- Annotated images with bounding boxes and confidence scores
- Text files containing detection coordinates
- Performance metrics on the test set

## Interpreting Results

### Training Metrics
- `results.png`: Shows training progress including loss and mAP metrics
- `confusion_matrix.png`: Displays model's classification accuracy across classes
- `PR_curve.png`: Precision-Recall curve indicating model's detection performance

### Validation Results
- `val_batch*_pred.jpg`: Shows model predictions on validation images
- `val_batch*_labels.jpg`: Shows ground truth labels for comparison

The model's performance can be evaluated using:
- mAP (mean Average Precision)
- Precision and Recall curves
- F1-score
- Confusion matrix

## Notes

- The model uses AdamW optimizer with conservative learning rates
- Augmentation includes moderate mosaic, rotation, and HSV adjustments
- Early stopping is implemented with a patience of 30 epochs
- The model automatically detects and logs challenging cases (overexposed, underexposed, low contrast)

## Reproducing Results

To reproduce the exact results:

1. Obtain the dataset from the organizers and set it up as specified
2. Run training with default parameters:
   ```bash
   python train_enhanced.py
   ```
3. For validation, run:
   ```bash
   python predict_enhanced.py
   ```

The results should match those in `runs/detect/train_personal_model/` and `runs/detect/val_personal_model/`.

---

## Team Information

Made by Team: OpenHivers at HackByte 3.0

### Team Members:
- Sujal Sakahre
- Garv Anand
- Harshit Chakravarti
- Arjun 
