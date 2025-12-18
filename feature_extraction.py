# feature_extraction.py
import cv2
import numpy as np
import math
import os

def sample_frames(cap, max_frames=48, stride=5):
    """Sample frames from video capture object."""
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        if len(frames) >= max_frames:
            break
        idx += 1
    return frames

def compute_resolution_stats(frames):
    """Compute resolution statistics from frames."""
    areas, ratios = [], []
    for f in frames:
        h, w = f.shape[:2]
        areas.append(w * h)
        ratios.append(w / max(h, 1))
    return np.mean(areas), np.std(ratios)

def compute_motion_intensity(frames):
    """Compute motion intensity using optical flow."""
    if len(frames) < 2:
        return 0.0
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    flow_mags = []
    for i in range(1, len(gray_frames)):
        prev = cv2.GaussianBlur(gray_frames[i - 1], (5, 5), 0)
        curr = cv2.GaussianBlur(gray_frames[i], (5, 5), 0)
        flow = cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_mags.append(np.mean(mag))
    return np.mean(flow_mags)

def variance_of_laplacian(image):
    """Compute variance of Laplacian for blur detection."""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def compute_blockiness(image, block=8):
    """Compute blockiness artifact measure."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grads = []
    for x in range(0, w, block):
        if 1 <= x < w - 1:
            grads.append(np.mean(np.abs(sobelx[:, x])))
    for y in range(0, h, block):
        if 1 <= y < h - 1:
            grads.append(np.mean(np.abs(sobely[y, :])))
    return np.mean(grads)

def compute_compression_proxy(frames):
    """Compute compression artifacts proxy metrics."""
    blurs, blocks = [], []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        blurs.append(variance_of_laplacian(gray))
        blocks.append(compute_blockiness(f))
    blur_inv = [1.0 / (b + 1e-6) for b in blurs]
    return np.mean(blur_inv), np.mean(blocks)

def extract_video_features(path, resize_to=(360, 640)):
    """
    Extract comprehensive features from video file.
    
    Args:
        path: Path to video file
        resize_to: Optional tuple (height, width) to resize frames
        
    Returns:
        numpy array of features or None if extraction fails
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    
    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    # Compute bitrate
    file_size = 0
    try:
        file_size = os.path.getsize(path)
    except:
        pass
    bitrate = (file_size * 8) / max(frames_total / max(fps, 1e-6), 1e-6) if frames_total > 0 else 0.0
    
    # Sample and optionally resize frames
    frames = sample_frames(cap)
    cap.release()
    
    if resize_to and frames:
        frames = [cv2.resize(f, (resize_to[1], resize_to[0])) for f in frames]
    
    # Compute features
    avg_area, aspect_std = compute_resolution_stats(frames)
    motion = compute_motion_intensity(frames)
    blur_inv, blockiness = compute_compression_proxy(frames)
    
    # Create feature vector
    features = np.array([
        math.log1p(avg_area),
        aspect_std,
        motion,
        blur_inv,
        blockiness,
        math.log1p(fps),
        math.log1p(bitrate)
    ], dtype=np.float32)
    
    return features
