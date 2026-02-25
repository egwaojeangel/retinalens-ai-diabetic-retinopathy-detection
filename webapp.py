"""
RetinaLens AI - Flask Backend
EfficientNet-B4 | 5-Class Diabetic Retinopathy Detection
Trained on EyePACS + APTOS + MESSIDOR (143,669 images)
Grad-CAM: proper implementation using last conv layer activations
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
import timm

app = Flask(__name__, static_folder='.')
CORS(app)

# ════════════════════════════════════════════════════════════
#  CONFIG — must match your training exactly
# ════════════════════════════════════════════════════════════
MODEL_PATH  = r"C:\Users\egwao\TumorSeg\best_model.pth"
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE  = 260
NUM_CLASSES = 5

GRADE_LABELS = {
    0: "No Diabetic Retinopathy",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

GRADE_SEVERITY = {
    0: "None",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Critical"
}

GRADE_COLORS = {
    0: "#22c55e",
    1: "#84cc16",
    2: "#f59e0b",
    3: "#f97316",
    4: "#ef4444"
}

GRADE_RECOMMENDATIONS = {
    0: "No signs of diabetic retinopathy detected. Routine annual screening recommended.",
    1: "Mild non-proliferative DR detected. Optimise blood sugar and blood pressure control. Re-screen in 12 months.",
    2: "Moderate non-proliferative DR detected. Refer to ophthalmologist. Re-screen in 6 months.",
    3: "Severe non-proliferative DR detected. Urgent ophthalmology referral required within 4 weeks.",
    4: "Proliferative DR detected. Immediate ophthalmology referral required. Risk of vision loss."
}

print("=" * 70)
print("RetinaLens AI — EfficientNet-B4 | 5-Class DR Detection")
print(f"Device: {DEVICE}  |  Image Size: {IMAGE_SIZE}px")
print("=" * 70)


# ════════════════════════════════════════════════════════════
#  PROPER GRAD-CAM IMPLEMENTATION
#
#  How it works:
#  1. Hook into the LAST CONV LAYER of EfficientNet-B4 (blocks[-1])
#  2. During forward pass  → save the feature maps (activations)
#  3. During backward pass → save the gradients flowing into that layer
#  4. Weight each feature map channel by its average gradient
#  5. Sum weighted maps → ReLU → resize → overlay on image
#
#  This is the original Grad-CAM paper method (Selvaraju et al. 2017)
#  It highlights exactly WHICH retinal regions the model focused on
#  e.g. microaneurysms, haemorrhages, neovascularisation
# ════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.activations = None   # feature maps from last conv layer
        self.gradients   = None   # gradients from last conv layer

        # EfficientNet-B4 in timm: the last conv block is model.blocks[-1]
        # We hook into it to capture activations and gradients
        target_layer = model.blocks[-1]

        # Forward hook — runs during forward pass, saves feature maps
        self.forward_hook = target_layer.register_forward_hook(
            self._save_activations
        )

        # Backward hook — runs during backward pass, saves gradients
        self.backward_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        # output shape: [batch, channels, H, W]
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0] shape: [batch, channels, H, W]
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, target_class, original_bgr=None):
        """
        Generate Grad-CAM heatmap. If original_bgr is provided,
        masks the CAM to only show hotspots inside the eye region.
        """
        self.model.eval()

        image_tensor = image_tensor.to(DEVICE)
        output       = self.model(image_tensor)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)

        weights     = self.gradients[0].mean(dim=(1, 2))
        activations = self.activations[0]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam    = torch.clamp(cam, min=0)
        cam    = cam - cam.min()
        cam    = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()

        if original_bgr is not None:
            h_o, w_o = original_bgr.shape[:2]
            cam_np   = cv2.resize(cam_np, (w_o, h_o))
            # Build eye mask — threshold the dark background
            gray        = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
            blurred     = cv2.GaussianBlur(gray, (21, 21), 0)
            _, eye_mask = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
            k           = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            eye_mask    = cv2.morphologyEx(eye_mask, cv2.MORPH_CLOSE, k)
            eye_mask    = cv2.morphologyEx(eye_mask, cv2.MORPH_OPEN,  k)
            # Zero out CAM values outside the eye before colormap
            cam_np[eye_mask == 0] = 0
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
            heatmap[eye_mask == 0] = [0, 0, 0]
            return heatmap

        cam_np  = cv2.resize(cam_np, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
        return heatmap

    def remove_hooks(self):
        """Call this to clean up hooks when done."""
        self.forward_hook.remove()
        self.backward_hook.remove()


# ════════════════════════════════════════════════════════════
#  MODEL LOADING
# ════════════════════════════════════════════════════════════
def load_model():
    try:
        print("[1/3] Creating EfficientNet-B4 architecture...")
        model = timm.create_model(
            "efficientnet_b4",
            pretrained=False,
            num_classes=NUM_CLASSES,
            drop_rate=0.4,
            drop_path_rate=0.2,
        )

        print(f"[2/3] Loading weights from {MODEL_PATH}...")
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)

        print(f"[3/3] Moving model to {DEVICE}...")
        model = model.to(DEVICE)
        model.eval()

        print("✓ Model loaded successfully!")
        print(f"  Architecture : EfficientNet-B4")
        print(f"  Classes      : {NUM_CLASSES} (Grade 0–4)")
        print(f"  Device       : {DEVICE}")
        print("=" * 70)
        return model

    except FileNotFoundError:
        print(f"\n✗ Model file not found at: {MODEL_PATH}")
        print("  Download best_model.pth from Kaggle Output tab.")
        print("  Running in DEMO MODE.\n")
        return None

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        import traceback; traceback.print_exc()
        return None


# ════════════════════════════════════════════════════════════
#  PREPROCESSING — matches val_transforms in training exactly
# ════════════════════════════════════════════════════════════
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


# ════════════════════════════════════════════════════════════
#  RETINAL IMAGE VALIDATOR
# ════════════════════════════════════════════════════════════
def is_retinal_image(image: Image.Image):
    arr  = np.array(image.resize((256, 256)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # Dark corners check
    corner_size = 30
    corners = [
        gray[:corner_size, :corner_size], gray[:corner_size, -corner_size:],
        gray[-corner_size:, :corner_size], gray[-corner_size:, -corner_size:],
    ]
    corner_mean = np.mean([c.mean() for c in corners])
    cx, cy = w // 2, h // 2
    r = min(w, h) // 4
    centre_mean = gray[cy-r:cy+r, cx-r:cx+r].mean()

    if corner_mean > 60:
        return False, "Image does not appear to be a retinal fundus scan. Please upload a fundus photograph."
    if centre_mean < corner_mean + 10:
        return False, "Image does not appear to be a retinal fundus scan. Please upload a fundus photograph."

    # Circular bright region check
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, bw   = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False, "No retinal disc detected. Please upload a fundus photograph."
    largest   = max(cnts, key=cv2.contourArea)
    area      = cv2.contourArea(largest)
    hull_area = cv2.contourArea(cv2.convexHull(largest))
    if hull_area > 0 and area / hull_area < 0.6:
        return False, "Image does not appear to be a retinal fundus scan. Please upload a fundus photograph."
    if area < (h * w * 0.20):
        return False, "Image does not appear to be a retinal fundus scan. Please upload a fundus photograph."

    # Red/orange colour tone check
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()
    if not (r_mean > g_mean and r_mean > b_mean and r_mean > 30):
        return False, "Image colour profile does not match a retinal fundus scan. Please upload a fundus photograph."

    return True, "OK"


# ════════════════════════════════════════════════════════════
#  INITIALIZE MODEL + GRAD-CAM
# ════════════════════════════════════════════════════════════
model   = load_model()
gradcam = GradCAM(model) if model is not None else None


# ════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_scan():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file   = request.files['image']
        image_bytes  = image_file.read()
        image        = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        original_arr = np.array(image)

        # ── RETINAL IMAGE VALIDATION ────────────────────────────────────────
        is_valid, reason = is_retinal_image(image)
        if not is_valid:
            return jsonify({'error': reason}), 400

        # ── DEMO MODE ────────────────────────────────────────
        if model is None:
            import random
            demo_grade = random.randint(0, 4)
            return jsonify({
                'grade':          demo_grade,
                'result':         GRADE_LABELS[demo_grade],
                'severity':       GRADE_SEVERITY[demo_grade],
                'confidence':     round(random.uniform(70, 95), 2),
                'color':          GRADE_COLORS[demo_grade],
                'recommendation': GRADE_RECOMMENDATIONS[demo_grade],
                'probabilities':  {GRADE_LABELS[i]: round(100/5, 2) for i in range(5)},
                'heatmap':        generate_demo_heatmap(original_arr),
                'demo_mode':      True
            })

        # ── REAL INFERENCE ───────────────────────────────────
        image_tensor = preprocess_image(image)

        # Get prediction
        with torch.no_grad():
            outputs       = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predicted     = torch.argmax(probabilities).item()
            confidence    = probabilities[predicted].item() * 100

        # ── PROPER GRAD-CAM with eye masking ──────────────────
        original_bgr = cv2.cvtColor(original_arr, cv2.COLOR_RGB2BGR)
        heatmap      = gradcam.generate(image_tensor, predicted, original_bgr)
        overlay      = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)

        # Encode to base64 for frontend
        _, buf       = cv2.imencode('.png', overlay)
        heatmap_b64  = f"data:image/png;base64,{base64.b64encode(buf).decode()}"

        # All 5 class probabilities
        prob_dict = {
            GRADE_LABELS[i]: round(probabilities[i].item() * 100, 2)
            for i in range(NUM_CLASSES)
        }

        return jsonify({
            'grade':          predicted,
            'result':         GRADE_LABELS[predicted],
            'severity':       GRADE_SEVERITY[predicted],
            'confidence':     round(confidence, 2),
            'color':          GRADE_COLORS[predicted],
            'recommendation': GRADE_RECOMMENDATIONS[predicted],
            'probabilities':  prob_dict,
            'heatmap':        heatmap_b64,
            'demo_mode':      False
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generate_demo_heatmap(image):
    h, w    = image.shape[:2]
    cx, cy  = w // 2, h // 2
    Y, X    = np.ogrid[:h, :w]
    dist    = np.sqrt((X - cx)**2 + (Y - cy)**2)
    dummy   = np.uint8(255 * (1 - dist / dist.max()))
    heatmap = cv2.applyColorMap(dummy, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               0.6, heatmap, 0.4, 0)
    _, buf  = cv2.imencode('.png', overlay)
    return f"data:image/png;base64,{base64.b64encode(buf).decode()}"


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status':       'healthy',
        'model_loaded': model is not None,
        'model':        'EfficientNet-B4',
        'classes':      NUM_CLASSES,
        'device':       str(DEVICE),
        'image_size':   IMAGE_SIZE,
        'gradcam':      'Proper Grad-CAM (last conv layer)',
        'trained_on':   'EyePACS + APTOS + MESSIDOR (143,669 images)'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("RetinaLens AI Backend — http://localhost:5000")
    print(f"Model  : {'Loaded ✓' if model else 'Not found — DEMO MODE'}")
    print(f"GradCAM: {'Ready ✓' if gradcam else 'Not available'}")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)