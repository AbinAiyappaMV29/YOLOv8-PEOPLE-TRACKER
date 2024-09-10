import cv2
from ultralytics import YOLO
import os

# Load your YOLOv8 model with custom weights
model = YOLO('People_tracker.pt')  # Adjust path if necessary

# Class names (assuming you've trained for Adult and Child)
class_names = ['Adult', 'Child']

# Open the video file
video_path = 'video4.mp4'  # Adjust path if necessary
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create the output directory if it doesn't exist
output_dir = 'runs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Prepare the output video file path
output_path = os.path.join(output_dir, 'output_video_fixed.mp4')

# Get the video writer initialized to save output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Set a confidence threshold to reduce unnecessary bounding boxes
confidence_threshold = 0.6  # Increase this to 0.6 or 0.7 if necessary to reduce clutter

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended or failed to grab a frame")
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Loop over the detections and filter based on confidence and class
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class ID
            confidence = box.conf[0]    # Confidence score

            # Apply confidence threshold
            if confidence >= confidence_threshold:
                if class_names[class_id] in ['Adult', 'Child']:  # Filter for specific classes
                    print(f"Detected {class_names[class_id]} with confidence {confidence:.2f}")

    # Plot and get the updated frame with bounding boxes
    annotated_frame = results[0].plot()

    # Write the frame to the output video file
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()

print(f"Processing complete. Output video saved at {output_path}")