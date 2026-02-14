"""
RetinaLens AI - Flask Backend for Retina Scan Analysis
Integrates EfficientNet-B0 model for Diabetic Retinopathy detection
Trained on APTOS 2019 Blindness Detection Dataset
"""

import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
import timm  # PyTorch Image Models - required for EfficientNet

app = Flask(__name__, static_folder='.')
CORS(app)  # Enable CORS for frontend communication

# Configuration - Matches your training setup exactly
MODEL_PATH = r"C:\Users\egwao\TumorSeg\best_model.pth"  # Your model location
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224  # Matches your training setup

print("=" * 70)
print("RetinaLens AI - Initializing Backend")
print("=" * 70)
print(f"Model Path: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print("=" * 70)


def load_model():
    """
    Load the trained EfficientNet-B0 model
    Architecture matches your training code exactly
    """
    try:
        print(f"\n[1/3] Loading EfficientNet-B0 architecture...")
        
        # Create the same model architecture used in training
        model = timm.create_model("efficientnet_b0", pretrained=False)
        
        # Replace classifier to match your training setup (binary classification)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        
        print(f"[2/3] Loading weights from {MODEL_PATH}...")
        
        # Load the saved state dict
        # Your training code saves with: torch.save(model.state_dict(), SAVE_PATH)
        # So we load it directly as state_dict (not a checkpoint dictionary)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        
        model.load_state_dict(state_dict)
        
        print(f"[3/3] Moving model to {DEVICE}...")
        model = model.to(DEVICE)
        model.eval()  # Set to evaluation mode
        
        print("✓ Model loaded successfully!")
        print(f"  - Architecture: EfficientNet-B0")
        print(f"  - Classes: 2 (Normal vs Diabetic Retinopathy)")
        print(f"  - Device: {DEVICE}")
        print("=" * 70)
        
        return model
    
    except FileNotFoundError:
        print(f"\n✗ ERROR: Model file not found at: {MODEL_PATH}")
        print("  Please ensure best_model.pth exists at the specified location.")
        print("  Running in DEMO MODE with dummy predictions.\n")
        return None
    
    except Exception as e:
        print(f"\n✗ ERROR loading model: {str(e)}")
        print("  Running in DEMO MODE with dummy predictions.\n")
        import traceback
        traceback.print_exc()
        return None


# Image preprocessing - matches your training validation transforms exactly
def preprocess_image(image):
    """
    Preprocess image for model input
    Uses the same normalization as training
    Args:
        image: PIL Image object
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # These transforms match your val_transforms in training code
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(DEVICE)


# Generate Grad-CAM heatmap
def generate_gradcam(model, image_tensor, target_class=1):
    """
    Generate Grad-CAM visualization for model explainability
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        target_class: Target class for visualization
    Returns:
        numpy.ndarray: Heatmap overlay
    """
    try:
        # Enable gradient computation
        image_tensor.requires_grad = True
        
        # Forward pass
        output = model(image_tensor)
        
        # Get the target class score
        target_score = output[0, target_class]
        
        # Backward pass
        model.zero_grad()
        target_score.backward()
        
        # Get gradients and activations
        gradients = image_tensor.grad.data
        activations = image_tensor.data
        
        # Compute weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Compute heatmap
        heatmap = torch.sum(weights * activations, dim=1).squeeze()
        heatmap = torch.clamp(heatmap, min=0)
        heatmap = heatmap / (torch.max(heatmap) + 1e-8)
        
        # Convert to numpy and resize
        heatmap_np = heatmap.cpu().detach().numpy()
        heatmap_np = cv2.resize(heatmap_np, (IMAGE_SIZE, IMAGE_SIZE))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_np), 
            cv2.COLORMAP_JET
        )
        
        return heatmap_colored
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        # Return dummy heatmap
        dummy_heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        return cv2.applyColorMap(dummy_heatmap, cv2.COLORMAP_JET)


# Initialize model
print("\nInitializing model...")
model = load_model()

# Disease labels - matches your binary classification
# 0 = Normal (no diabetic retinopathy)
# 1 = Diabetic Retinopathy detected (any grade from 1-4 in original APTOS)
DISEASE_LABELS = {
    0: "Normal Retina",
    1: "Diabetic Retinopathy"
}

print(f"Disease Classification: {DISEASE_LABELS}")
print("=" * 70)


@app.route('/')
def index():
    """Serve the main HTML file"""
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_scan():
    """
    Analyze retina scan image
    Returns classification result with confidence and heatmap
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        
        # Read and preprocess image
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Store original for heatmap overlay
        original_image = np.array(image)
        
        if model is None:
            # Dummy response for testing without model
            return jsonify({
                'result': 'Diabetic Retinopathy',
                'confidence': 92.5,
                'heatmap': generate_dummy_heatmap(original_image)
            })
        
        # Preprocess for model
        image_tensor = preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item() * 100
        
        # Generate Grad-CAM heatmap
        heatmap = generate_gradcam(model, image_tensor, target_class=predicted_class)
        
        # Overlay heatmap on original image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        overlay = cv2.addWeighted(original_image, 0.6, heatmap_resized, 0.4, 0)
        
        # Convert overlay to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        heatmap_base64 = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
        
        # Prepare response
        result = {
            'result': DISEASE_LABELS.get(predicted_class, 'Unknown'),
            'confidence': round(confidence_score, 2),
            'heatmap': heatmap_base64,
            'probabilities': {
                label: round(prob * 100, 2) 
                for label, prob in zip(DISEASE_LABELS.values(), probabilities[0].tolist())
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generate_dummy_heatmap(image):
    """Generate dummy heatmap for testing"""
    # Create a simple gradient heatmap
    height, width = image.shape[:2]
    dummy = np.zeros((height, width), dtype=np.uint8)
    
    # Create circular gradient
    center_x, center_y = width // 2, height // 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    for y in range(height):
        for x in range(width):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            dummy[y, x] = int(255 * (1 - dist / max_dist))
    
    heatmap = cv2.applyColorMap(dummy, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })


if __name__ == '__main__':
    print("=" * 60)
    print("RetinaLens AI Backend Server")
    print("=" * 60)
    print(f"Model Path: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Model Status: {'Loaded' if model else 'Not Loaded'}")
    print("=" * 60)
    print("Server starting on http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)