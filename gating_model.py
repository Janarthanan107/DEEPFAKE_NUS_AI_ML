# gating_model.py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Class mappings for the gating classifier
CLASS_TO_ID = {"ViT": 0, "CNN": 1, "ViT + CNN": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}

def train_classifier(X, y, model_path="gating_rf.joblib"):
    """
    Train a Random Forest classifier for model selection.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        model_path: Path to save trained model
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = clf.predict(X_val)
    print("\n=== Validation Results ===")
    print(classification_report(y_val, y_pred, target_names=CLASS_TO_ID.keys()))
    
    # Save model
    joblib.dump(clf, model_path)
    print(f"\n✅ Model saved to {model_path}")

def predict_model(features, model_path="gating_rf.joblib"):
    """
    Predict which model(s) to use for a video.
    
    Args:
        features: Feature vector for a single video
        model_path: Path to trained model
        
    Returns:
        String indicating preferred model: "ViT", "CNN", or "ViT + CNN"
    """
    clf = joblib.load(model_path)
    pred = clf.predict(features.reshape(1, -1))[0]
    return ID_TO_CLASS[int(pred)]

def predict_with_confidence(features, model_path="gating_rf.joblib"):
    """
    Predict with confidence scores.
    
    Args:
        features: Feature vector for a single video
        model_path: Path to trained model
        
    Returns:
        tuple: (predicted_class, confidence_dict)
    """
    clf = joblib.load(model_path)
    pred = clf.predict(features.reshape(1, -1))[0]
    proba = clf.predict_proba(features.reshape(1, -1))[0]
    
    confidence = {ID_TO_CLASS[i]: float(proba[i]) for i in range(len(proba))}
    
    return ID_TO_CLASS[int(pred)], confidence
