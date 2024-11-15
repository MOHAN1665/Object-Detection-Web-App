import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score
import time

# Load YOLO model
weights_path = r'C:/Users/pmoha/runs/detect/train5/weights/best.pt'  # Path to YOLOv5 trained weights
model = YOLO(weights_path)  # Update the path to your trained model

def detect_objects(image):
    # Run inference on the uploaded image
    results = model(image)

    # Extract information
    boxes = results[0].boxes.xyxy.numpy()  # Bounding boxes
    classes = results[0].boxes.cls.numpy()  # Class IDs
    confidence_scores = results[0].boxes.conf.numpy()  # Confidence scores
    names = results[0].names  # Class names dictionary

    # Map class IDs to class names
    class_names = [names[int(cls)] for cls in classes]

    # Apply Non-Maximum Suppression (NMS)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidence_scores.tolist(), score_threshold=0.5, nms_threshold=0.4)
    
    # Filter out the non-maximal boxes (duplicates)
    if len(indices) > 0:
        indices = indices.flatten()  # Flatten indices array
        boxes = boxes[indices]
        class_names = [class_names[i] for i in indices]
        confidence_scores = confidence_scores[indices]

    return boxes, class_names, confidence_scores

def display_result(image, boxes, class_names, confidence_scores):
    # Draw bounding boxes and labels on the image
    for box, class_name, score in zip(boxes, class_names, confidence_scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image, f"{class_name} {score:.2f}", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return image

def calculate_metrics(predicted_classes, true_classes):
    if len(predicted_classes) == len(true_classes):
        precision = precision_score(true_classes, predicted_classes, average=None, zero_division=0)
        recall = recall_score(true_classes, predicted_classes, average=None, zero_division=0)
        f1 = f1_score(true_classes, predicted_classes, average=None, zero_division=0)
        return precision, recall, f1
    else:
        print("Length mismatch between true and predicted classes.")
        return None, None, None

def create_metrics_df(precision, recall, f1, class_names):
    # Debugging step: check lengths of all arrays
    print(f"Precision length: {len(precision)}")
    print(f"Recall length: {len(recall)}")
    print(f"F1 Score length: {len(f1)}")

    # Check if lengths match
    if len(precision) == len(recall) == len(f1):
        # Create DataFrame only if lengths match
        metrics_df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        return metrics_df
    else:
        raise ValueError("All arrays (precision, recall, f1) must have the same length")

def main():
    st.set_page_config(page_title="YOLOv5 Object Detection", page_icon=":guardsman:", layout="centered")
    st.title("YOLOv5 Object Detection Web App")
    
    # Add custom CSS for styling the page
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f2f6;
        }
        h1 {
            font-size: 3em;
            color: #0073e6;
        }
        .stButton>button {
            background-color: #0073e6;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
        .stTable {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # File uploader widget for image
    uploaded_image = st.file_uploader("Upload an Image for Object Detection", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image
        image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Display progress bar while detecting objects
        with st.spinner("Detecting objects..."):
            time.sleep(2)  # Simulate processing time (adjust this based on actual model inference time)
            boxes, class_names, confidence_scores = detect_objects(image)

            # Display detected result on the image
            result_image = display_result(image, boxes, class_names, confidence_scores)
            st.image(result_image, channels="BGR", use_container_width=True)

            # Display precision, recall, and F1-score table
            st.subheader("Precision, Recall, and F1-Score")
            # Example true classes (you can use a predefined set for this image or integrate a ground truth)
            true_classes = ['RBC', 'WBC', 'Platelets']  # Replace with actual true labels for the image

            # Align lengths of predicted and true classes
            min_length = min(len(class_names), len(true_classes))
            predicted_classes = class_names[:min_length]
            true_classes = true_classes[:min_length]

            # Calculate metrics
            precision, recall, f1 = calculate_metrics(predicted_classes, true_classes)

            metrics_df = pd.DataFrame({
                "Class": predicted_classes,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1
            })

            st.table(metrics_df)

    else:
        st.warning("Please upload an image to get started.")
    
    # Display additional information about the app
    st.sidebar.title("About")
    st.sidebar.info("""
        This web app uses YOLOv5 for object detection. Simply upload an image, and the app will detect objects, 
        draw bounding boxes around them, and display the predicted class with confidence scores.
    """)

if __name__ == "__main__":
    main()
