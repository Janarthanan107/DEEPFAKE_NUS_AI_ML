# main.py
import argparse
import numpy as np
from feature_extraction import extract_video_features
from gating_model import train_classifier, predict_model, predict_with_confidence, CLASS_TO_ID
from rule_based import rule_based_decision, get_decision_explanation
import os
import csv
from glob import glob

def load_dataset(video_dir, labels_csv=None):
    """
    Load dataset from video directory.
    
    Args:
        video_dir: Directory containing videos
        labels_csv: Optional CSV with video labels (format: video_path,label)
        
    Returns:
        tuple: (X, y) - features and labels
    """
    X, y = [], []
    label_map = {}
    
    # Load labels if provided
    if labels_csv and os.path.exists(labels_csv):
        with open(labels_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    label_map[row[0]] = row[1]
    
    # Process videos
    video_paths = glob(f"{video_dir}/**/*.mp4", recursive=True)
    print(f"Found {len(video_paths)} videos")
    
    for i, path in enumerate(video_paths):
        if i % 10 == 0:
            print(f"Processing {i}/{len(video_paths)}...")
        
        feats = extract_video_features(path)
        if feats is None:
            print(f"⚠️  Failed to extract features from {path}")
            continue
        
        # Get label from CSV or use rule-based
        label = label_map.get(os.path.basename(path))
        if not label:
            label = rule_based_decision(feats)
            print(f"Using rule-based label '{label}' for {os.path.basename(path)}")
        
        if label not in CLASS_TO_ID:
            print(f"⚠️  Invalid label '{label}' for {path}")
            continue
            
        X.append(feats)
        y.append(CLASS_TO_ID[label])
    
    if len(X) == 0:
        raise ValueError("No valid videos found!")
    
    return np.vstack(X), np.array(y)

def main():
    parser = argparse.ArgumentParser(description="Gating Classifier for Deepfake Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the gating classifier")
    train_parser.add_argument("--video_dir", required=True, help="Directory containing training videos")
    train_parser.add_argument("--labels_csv", default=None, help="CSV file with labels (optional)")
    train_parser.add_argument("--model_out", default="gating_rf.joblib", help="Output model path")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict model for a video")
    predict_parser.add_argument("--video_path", required=True, help="Path to video file")
    predict_parser.add_argument("--model_path", default=None, help="Path to trained model (optional)")
    predict_parser.add_argument("--verbose", action="store_true", help="Show detailed predictions")

    args = parser.parse_args()

    if args.command == "train":
        print("=" * 60)
        print("TRAINING GATING CLASSIFIER")
        print("=" * 60)
        
        X, y = load_dataset(args.video_dir, args.labels_csv)
        print(f"\n✅ Loaded {len(X)} samples")
        print(f"Feature shape: {X.shape}")
        
        # Show class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("\nClass distribution:")
        for cls_id, count in zip(unique, counts):
            cls_name = [k for k, v in CLASS_TO_ID.items() if v == cls_id][0]
            print(f"  {cls_name}: {count} samples")
        
        train_classifier(X, y, args.model_out)

    elif args.command == "predict":
        print("=" * 60)
        print("PREDICTING MODEL FOR VIDEO")
        print("=" * 60)
        print(f"Video: {args.video_path}\n")
        
        feats = extract_video_features(args.video_path)
        if feats is None:
            print("❌ Could not extract features from video.")
            return
        
        if args.verbose:
            print("Extracted features:")
            feature_names = ["area_log", "aspect_std", "motion", "blur_inv", "blockiness", "fps_log", "bitrate_log"]
            for name, val in zip(feature_names, feats):
                print(f"  {name}: {val:.4f}")
            print()
        
        # Predict using trained model or rule-based
        if args.model_path and os.path.exists(args.model_path):
            if args.verbose:
                result, confidence = predict_with_confidence(feats, args.model_path)
                print(f"🎯 Predicted model: {result}")
                print("\nConfidence scores:")
                for model, conf in confidence.items():
                    print(f"  {model}: {conf:.2%}")
            else:
                result = predict_model(feats, args.model_path)
                print(f"🎯 Predicted model: {result}")
        else:
            result, explanation = get_decision_explanation(feats)
            print(f"🎯 Predicted model (rule-based): {result}")
            if args.verbose:
                print(f"Explanation: {explanation}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
