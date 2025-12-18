#!/usr/bin/env python3
"""
Demo training script for gating classifier using synthetic data.
Once you have real videos, use: python main.py train --video_dir /path/to/videos
"""

import numpy as np
from gating_model import train_classifier, predict_with_confidence, CLASS_TO_ID

def generate_synthetic_data(n_samples=300):
    """
    Generate synthetic video features for demonstration.
    
    In reality, these would come from extract_video_features() on real videos.
    """
    np.random.seed(42)
    X = []
    y = []
    
    # Generate features for each class
    # Feature order: [area_log, aspect_std, motion, blur_inv, blockiness, fps_log, bitrate_log]
    
    # Class 0: ViT - High res, low motion, low compression
    for _ in range(n_samples // 3):
        features = np.array([
            np.random.normal(13.0, 0.5),   # High area_log (HD/4K videos)
            np.random.normal(0.05, 0.02),  # Low aspect variation
            np.random.normal(0.3, 0.1),    # Low motion
            np.random.normal(0.3, 0.1),    # Low blur (sharp)
            np.random.normal(1.5, 0.3),    # Low blockiness
            np.random.normal(3.5, 0.2),    # ~30 fps
            np.random.normal(14.0, 0.5),   # High bitrate
        ])
        X.append(features)
        y.append(CLASS_TO_ID["ViT"])
    
    # Class 1: CNN - High motion, compression artifacts
    for _ in range(n_samples // 3):
        features = np.array([
            np.random.normal(11.5, 0.5),   # Medium res
            np.random.normal(0.1, 0.03),   # Medium aspect variation
            np.random.normal(1.2, 0.3),    # High motion
            np.random.normal(1.0, 0.2),    # High blur/compression
            np.random.normal(4.0, 0.5),    # High blockiness
            np.random.normal(3.0, 0.3),    # ~20 fps
            np.random.normal(12.0, 0.5),   # Medium bitrate
        ])
        X.append(features)
        y.append(CLASS_TO_ID["CNN"])
    
    # Class 2: ViT + CNN - Mixed characteristics
    for _ in range(n_samples // 3):
        features = np.array([
            np.random.normal(12.0, 0.7),   # Medium-high res
            np.random.normal(0.08, 0.03),  # Medium aspect variation
            np.random.normal(0.7, 0.2),    # Medium motion
            np.random.normal(0.6, 0.15),   # Medium blur
            np.random.normal(2.5, 0.5),    # Medium blockiness
            np.random.normal(3.3, 0.2),    # ~25 fps
            np.random.normal(13.0, 0.5),   # Medium-high bitrate
        ])
        X.append(features)
        y.append(CLASS_TO_ID["ViT + CNN"])
    
    return np.array(X), np.array(y)

def main():
    print("=" * 70)
    print("🚀 DEMO: Training Gating Classifier with Synthetic Data")
    print("=" * 70)
    print()
    print("📝 Note: This is a demonstration using synthetic features.")
    print("   For real training, use: python main.py train --video_dir /path/to/videos")
    print()
    
    # Generate synthetic data
    print("📊 Generating synthetic video features...")
    X, y = generate_synthetic_data(n_samples=300)
    print(f"✅ Generated {len(X)} samples")
    print(f"   Feature shape: {X.shape}")
    print()
    
    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("📈 Class distribution:")
    for cls_id, count in zip(unique, counts):
        cls_name = [k for k, v in CLASS_TO_ID.items() if v == cls_id][0]
        print(f"   {cls_name}: {count} samples")
    print()
    
    # Train classifier
    print("🔧 Training Random Forest classifier...")
    print("-" * 70)
    train_classifier(X, y, model_path="gating_rf.joblib")
    print()
    
    # Test predictions on sample data
    print("=" * 70)
    print("🧪 Testing Predictions")
    print("=" * 70)
    print()
    
    # Test 1: High-res, low motion video (should prefer ViT)
    test_vit = np.array([13.5, 0.04, 0.2, 0.25, 1.2, 3.6, 14.5])
    pred, conf = predict_with_confidence(test_vit, "gating_rf.joblib")
    print("Test 1: High-res, static video")
    print(f"  Features: area_log=13.5, motion=0.2 (low)")
    print(f"  🎯 Prediction: {pred}")
    print(f"  Confidence: {', '.join([f'{k}: {v:.1%}' for k, v in conf.items()])}")
    print()
    
    # Test 2: High motion, compressed video (should prefer CNN)
    test_cnn = np.array([11.0, 0.12, 1.5, 1.2, 4.5, 2.8, 11.5])
    pred, conf = predict_with_confidence(test_cnn, "gating_rf.joblib")
    print("Test 2: Dynamic, compressed video")
    print(f"  Features: area_log=11.0, motion=1.5 (high)")
    print(f"  🎯 Prediction: {pred}")
    print(f"  Confidence: {', '.join([f'{k}: {v:.1%}' for k, v in conf.items()])}")
    print()
    
    # Test 3: Mixed characteristics (should use both)
    test_both = np.array([12.5, 0.08, 0.7, 0.6, 2.8, 3.2, 13.0])
    pred, conf = predict_with_confidence(test_both, "gating_rf.joblib")
    print("Test 3: Mixed characteristics")
    print(f"  Features: area_log=12.5, motion=0.7 (medium)")
    print(f"  🎯 Prediction: {pred}")
    print(f"  Confidence: {', '.join([f'{k}: {v:.1%}' for k, v in conf.items()])}")
    print()
    
    print("=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print()
    print("📁 Model saved to: gating_rf.joblib")
    print()
    print("🔜 Next Steps:")
    print("   1. Collect real deepfake videos")
    print("   2. Train on real data: python main.py train --video_dir /path/to/videos")
    print("   3. Test predictions: python main.py predict --video_path test.mp4 --model_path gating_rf.joblib")
    print()

if __name__ == "__main__":
    main()
