# rule_based.py

def rule_based_decision(feats):
    """
    Rule-based fallback for model selection when no trained classifier is available.
    
    Args:
        feats: Feature vector [area_log, aspect_std, motion, blur_inv, blockiness, fps_log, bitrate_log]
        
    Returns:
        String indicating preferred model: "ViT", "CNN", or "ViT + CNN"
        
    Rules:
        - CNN: High motion + compression artifacts (good for dynamic/compressed videos)
        - ViT: High resolution + low motion + low compression (good for static/high-quality videos)
        - ViT + CNN: Everything else (mixed characteristics)
    """
    area_log, aspect_std, motion, blur_inv, blockiness, fps_log, bitrate_log = feats
    
    # Rule 1: High motion + compression artifacts → CNN
    if motion > 0.8 and (blur_inv > 0.8 or blockiness > 3.0):
        return "CNN"
    
    # Rule 2: High resolution + low motion + low blur → ViT
    if area_log > 12.0 and motion < 0.5 and blur_inv < 0.5:
        return "ViT"
    
    # Rule 3: Mixed characteristics → Use both
    return "ViT + CNN"

def get_decision_explanation(feats):
    """
    Get explanation for the rule-based decision.
    
    Args:
        feats: Feature vector
        
    Returns:
        tuple: (decision, explanation)
    """
    area_log, aspect_std, motion, blur_inv, blockiness, fps_log, bitrate_log = feats
    
    decision = rule_based_decision(feats)
    
    if decision == "CNN":
        explanation = f"High motion ({motion:.2f}) and compression artifacts detected. CNN is better for dynamic videos."
    elif decision == "ViT":
        explanation = f"High resolution (area_log={area_log:.2f}) with low motion ({motion:.2f}). ViT excels on static, high-quality videos."
    else:
        explanation = "Mixed characteristics detected. Using ensemble approach for best results."
    
    return decision, explanation
