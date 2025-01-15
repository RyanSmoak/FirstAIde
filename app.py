import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as ort
from PIL import Image as img
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from AIDE_GPT import main
yolo_model = "C:/Users/Administrator/Documents/Year 3/FirstAIde/best.onnx"
yolo = ort.InferenceSession(yolo_model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#define helper functions

#preprocess the image
def preprocess_image(image):
    image = image.resize((640,640)) #resize to the same size as the training data
    image = image.convert("RGB") #convert to RGB format
    image = np.asarray(image).astype(np.float32)/255.0 #normalize
    image = np.array(image).transpose(2, 0, 1) #HWC to CHW format
    return np.expand_dims(image, axis=0)

#run inference
def run_inference(image_array):
    ort_inputs = {yolo.get_inputs()[0].name: image_array}
    outputs = yolo.run(None, ort_inputs)
    return outputs

def decode_output(output, conf_threshold=0.5, nms_threshold=0.4):
    boxes = []
    confidences = []
    class_probs = []
    # Number of classes
    num_classes = 3  

    # Iterate through the predictions in the output
    for detection in output[0][0]:
        # Extract bounding box parameters (center x, center y, width, height, confidence)
        x, y, w, h, obj_confidence = detection[:5]
        class_scores = detection[5:]

        # Apply sigmoid to get probabilities
        obj_confidence = 1 / (1 + np.exp(-obj_confidence))  # Sigmoid
        class_scores = 1 / (1 + np.exp(-class_scores))  # Sigmoid

        if obj_confidence > conf_threshold:
            # Find the class with the highest score
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            if class_confidence > conf_threshold:
                x1 = int((x - w / 2) * 640)  
                y1 = int((y - h / 2) * 640)
                x2 = int((x + w / 2) * 640)
                y2 = int((y + h / 2) * 640)

                boxes.append([x1, y1, x2, y2])
                confidences.append(obj_confidence * class_confidence)
                class_probs.append(class_id)

    # Apply Non-Maximum Suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    final_boxes = []
    for i in indices.flatten():
        final_boxes.append({
            'box': boxes[i],
            'confidence': confidences[i],
            'class': class_probs[i]
        })

    return final_boxes


#annotate the image to show bounding boxes
def annotate_image(image, detections):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    burn_types = ["First Degree", "Second Degree", "Third Degree"]

    highest_confidence = 0
    burn_type = None

    for detection in detections:
        box = detection['box']
        class_id = detection['class']
        confidence = detection['confidence']
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label text
        label = f"Class: {class_id}, Confidence: {confidence:.2f}"

        # Check if the current detection has the highest confidence
        if confidence > highest_confidence:
            highest_confidence = confidence
            burn_type = burn_types[class_id]

        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    return burn_type, highest_confidence

app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    # Check if the uploaded file is valid
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Debug: Log the filename
        print(f"Uploaded file: {file.filename}")

        # Save the file temporarily
        save_path = os.path.join("temp", file.filename)
        os.makedirs("temp", exist_ok=True)  # Ensure the folder exists
        file.save(save_path)

        # Debug: Log the save path
        print(f"File saved to: {save_path}")

        # Load and preprocess the image
        image = img.open(file)  # Open the uploaded image file
        image_array = preprocess_image(image)

        # Run inference
        outputs = run_inference(image_array)

        # Decode the output
        detections = decode_output(outputs, conf_threshold=0.5, nms_threshold=0.4)

        # Extract results
        burn_type, _ = annotate_image(save_path, detections)
        
        # Generate first aid advice
        first_aid = main.generate_from_starting_text(burn_type)

        # Return the results
        return jsonify(
            {
                "burn_type": burn_type,
                "first_aid": first_aid
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
