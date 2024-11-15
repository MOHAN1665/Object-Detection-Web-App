# YOLOv5 Object Detection Web App

## Overview
The **YOLOv5 Object Detection Web App** is a web application built with Streamlit that uses the YOLOv5 model for object detection tasks. The app allows users to upload an image, performs object detection using a pre-trained YOLOv5 model, and displays the predicted class names and their confidence scores. It also draws bounding boxes around the detected objects in the image.

## Deployed Version

You can try the live version of the app on Hugging Face Spaces here:

[YOLOv5 Object Detection Web App on Hugging Face](https://huggingface.co/spaces/Cyherix/Object-Detector-Stream)


## Features
- Upload an image to perform object detection using YOLOv5.
- Displays bounding boxes around the detected objects.
- Shows predicted class names and confidence scores for each detection.
- Visualizes Precision, Recall, and F1-Score metrics for the detected classes.

## Requirements

- **Python 3.8+**
- **Streamlit**: To build the interactive web application.
- **OpenCV**: For image processing and visualization.
- **Ultralytics (YOLOv5)**: For using the pre-trained YOLOv5 model.
- **Scikit-learn**: For calculating Precision, Recall, and F1-Score metrics.
- **NumPy**: For numerical operations on images.

To install all dependencies, run:

```bash
pip install -r requirements.txt
```
# Installation
## Step 1: Clone the Repository
```bash
  git clone https://github.com/MOHAN1665/Object-Detection-Web-App.git
  cd Object-Detection-Web-App
```

## Step 2: Install Dependencies
Create and activate a virtual environment, then install the required Python packages:
## Create virtual environment
```
    python -m venv env
```
## Activate the environment
## On Windows
```
    env\Scripts\activate
```
## On MacOS/Linux
```
    source env/bin/activate
```
## Install dependencies
```
    pip install -r requirements.txt
```

## Step 3: Set YOLOv5 Weights
Download the trained YOLOv5 weights file (best.pt) from your training or source, and place it in the runs/detect/train5/weights/ directory. If the path is different, make sure to update the weights_path in the code accordingly.

## Step 4: Run the Streamlit App
To start the app, use the following command:

## Step 4: Run the Streamlit App
To start the app, use the following command:

```
    streamlit run app.py
```
This will open the app in your default web browser.

# Usage
- **Upload Image**: Click the "Upload Image" button to upload an image (JPEG, PNG, or JPG format).
- **Object Detection**: The app processes the image and detects objects using the YOLOv5 model.
- **Bounding Boxes & Predictions**: The app will display bounding boxes around detected objects and show the predicted class names with confidence scores.
- **Metrics**: The app also displays Precision, Recall, and F1-Score metrics for the detected classes.

# Example Output
Once an image is uploaded, the web app will display:
- **Bounding Boxes around detected objects.**
- **Class Names (e.g., RBC, WBC, Platelets).**
- **Confidence Scores for each detected class.**
- **Precision, Recall, and F1-Score for each detected class.**

# Metrics Calculation
The app calculates and displays the following classification metrics for the detected classes:
- **Precision**: The fraction of relevant instances retrieved by the model.
- **Recall**: The fraction of relevant instances that were retrieved by the model.
- **F1-Score**: The harmonic mean of Precision and Recall, providing a balance between them.

# Contributing
Feel free to fork this repository and submit pull requests for any improvements or bug fixes. Contributions are welcome!

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgements
- **YOLOv5**: An efficient object detection model developed by Ultralytics.
- **Streamlit**: An open-source app framework used to build the web app.
- **OpenCV**: For image processing and visualization.

# Note:
Ensure you have the best.pt model weights available in the correct directory for detection to work.
