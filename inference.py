#!/usr/bin/env python3
"""
Complete Deepfake Detection System - Inference Script

This script combines all three models:
1. Gating Classifier - Decides which model to use
2. ViT Model - Detects deepfakes in images
3. CNN Model - Detects deepfakes in videos (optional)

Usage:
    python3 inference.py --input video.mp4
    python3 inference.py --input image.jpg
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
from pathlib import Path
import os

# Import our modules
from feature_extraction import extract_video_features
from gating_model import predict_with_confidence as predict_gating
from rule_based import get_decision_explanation
from train_vit import ViTDeepfakeDetector

class DeepfakeInference:
    """Complete deepfake detection inference pipeline."""
    
    def __init__(self, 
                 gating_model_path='gating_rf.joblib',
                 vit_model_path='vit_deepfake.pth',
                 cnn_model_path=None,
                 device='auto'):
        """Initialize all models."""
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Initializing Deepfake Detection System")
        print(f"Device: {self.device}")
        
        # Load gating classifier
        self.gating_model_path = gating_model_path
        self.has_gating = os.path.exists(gating_model_path)
        if self.has_gating:
            print(f"✅ Loaded gating classifier: {gating_model_path}")
        else:
            print(f"⚠️  Gating classifier not found, using rule-based routing")
        
        # Load ViT model
        self.has_vit = False
        if vit_model_path and os.path.exists(vit_model_path):
            try:
                self.vit_model = ViTDeepfakeDetector(model_name='vit_tiny_patch16_224', 
                                                     pretrained=False, num_classes=2)
                checkpoint = torch.load(vit_model_path, map_location=self.device,weights_only=True)
                self.vit_model.load_state_dict(checkpoint['model_state_dict'])
                self.vit_model = self.vit_model.to(self.device)
                self.vit_model.eval()
                self.has_vit = True
                print(f"✅ Loaded ViT model: {vit_model_path}")
            except Exception as e:
                print(f"⚠️  Could not load ViT model: {e}")
        else:
            print(f"⚠️  ViT model not found: {vit_model_path}")
        
        # Load CNN model (optional)
        self.has_cnn = False
        if cnn_model_path and os.path.exists(cnn_model_path):
            print(f"✅ Loaded CNN model: {cnn_model_path}")
            self.has_cnn = True
        else:
            print(f"ℹ️  CNN model not available (optional)")
        
        # Image transform for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print()
    
    def predict_image(self, image_path):
        """Predict if an image is a deepfake."""
        if not self.has_vit:
            return {"error": "ViT model not loaded"}
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.vit_model(image_tensor)
                probs = torch.softmax(output, dim=1)[0]
                
            fake_prob = probs[1].item()
            real_prob = probs[0].item()
            
            prediction = "FAKE" if fake_prob > 0.5 else "REAL"
            confidence = max(fake_prob, real_prob)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "fake_probability": fake_prob,
                "real_probability": real_prob,
                "model_used": "ViT"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def predict_video(self, video_path):
        """Predict if a video is a deepfake."""
        
        # Step 1: Extract features for gating
        print("📊 Extracting video features...")
        features = extract_video_features(video_path)
        
        if features is None:
            return {"error": "Could not extract features from video"}
        
        # Step 2: Decide which model to use
        if self.has_gating:
            try:
                decision, confidence = predict_gating(features, self.gating_model_path)
                print(f"🎯 Gating decision: {decision}")
                print(f"   Confidence: {', '.join([f'{k}: {v:.1%}' for k, v in confidence.items()])}")
            except:
                decision, explanation = get_decision_explanation(features)
                print(f"🎯 Rule-based decision: {decision}")
                print(f"   {explanation}")
        else:
            decision, explanation = get_decision_explanation(features)
            print(f"🎯 Rule-based decision: {decision}")
            print(f"   {explanation}")
        
        # Step 3: Use the appropriate model
        if decision == "ViT" or decision == "ViT + CNN":
            if self.has_vit:
                print("\n🔍 Running ViT detection on video frames...")
                return self._predict_video_with_vit(video_path)
            else:
                return {"error": "ViT model recommended but not loaded"}
        
        elif decision == "CNN":
            if self.has_cnn:
                print("\n🔍 Running CNN detection...")
                return {"error": "CNN model not yet implemented"}
            else:
                # Fallback to ViT
                if self.has_vit:
                    print("\n🔍 CNN not available, using ViT instead...")
                    return self._predict_video_with_vit(video_path)
                else:
                    return {"error": "No detection model available"}
        
        return {"error": "Unknown routing decision"}
    
    def _predict_video_with_vit(self, video_path, num_frames=10):
        """Analyze video using ViT on sampled frames."""
        
        # Sample frames from video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return {"error": "Could not read video"}
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        predictions = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                output = self.vit_model(image_tensor)
                probs = torch.softmax(output, dim=1)[0]
                fake_prob = probs[1].item()
                predictions.append(fake_prob)
        
        cap.release()
        
        # Average predictions
        avg_fake_prob = np.mean(predictions)
        avg_real_prob = 1 - avg_fake_prob
        
        prediction = "FAKE" if avg_fake_prob > 0.5 else "REAL"
        confidence = max(avg_fake_prob, avg_real_prob)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "fake_probability": avg_fake_prob,
            "real_probability": avg_real_prob,
            "model_used": "ViT (frame-based)",
            "frames_analyzed": len(predictions)
        }
    
    def predict(self, input_path):
        """Auto-detect input type and predict."""
        path = Path(input_path)
        
        if not path.exists():
            return {"error": f"File not found: {input_path}"}
        
        # Check file type
        ext = path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            print(f"\n📷 Image detected: {path.name}")
            return self.predict_image(input_path)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"\n🎥 Video detected: {path.name}")
            return self.predict_video(input_path)
        else:
            return {"error": f"Unsupported file type: {ext}"}

def main():
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference')
    parser.add_argument('--input', type=str, required=True, help='Input image or video')
    parser.add_argument('--gating_model', type=str, default='gating_rf.joblib', help='Gating model path')
    parser.add_argument('--vit_model', type=str, default='vit_deepfake.pth', help='ViT model path')
    parser.add_argument('--cnn_model', type=str, default=None, help='CNN model path (optional)')
    parser.add_argument('--device', type=str, default='auto', help='Device: cuda, mps, cpu, or auto')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 Deepfake Detection System - Inference")
    print("=" * 70)
    print()
    
    # Initialize system
    detector = DeepfakeInference(
        gating_model_path=args.gating_model,
        vit_model_path=args.vit_model,
        cnn_model_path=args.cnn_model,
        device=args.device
    )
    
    # Run prediction
    result = detector.predict(args.input)
    
    # Display results
    print("=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Fake Probability: {result['fake_probability']:.1%}")
        print(f"Real Probability: {result['real_probability']:.1%}")
        print(f"Model Used: {result['model_used']}")
        if 'frames_analyzed' in result:
            print(f"Frames Analyzed: {result['frames_analyzed']}")
    
    print("=" * 70)
    print()

if __name__ == '__main__':
    main()
