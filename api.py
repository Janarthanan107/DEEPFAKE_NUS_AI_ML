from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from inference import DeepfakeDetector

app = FastAPI(title="Deepfake Detection API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Detector (Global to avoid reloading per request)
# Device will be auto-detected (MPS on Mac, CUDA if available, else CPU)
detector = DeepfakeDetector()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Deepfake Detection API is running"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video.")

        # Save uploaded file temporarily
        file_ext = file.filename.split('.')[-1]
        temp_filename = f"{uuid.uuid4()}.{file_ext}"
        temp_file_path = os.path.join(UPLOAD_DIR, temp_filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run Inference
        try:
            results = detector.analyze_video(temp_file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

        return results

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
