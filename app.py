import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Absolute imports from the inference module
from inference.image_pipeline import ImagePipeline
from inference.dicom_handler import convert_dicom_to_mp4

app = FastAPI(
    title="Angio AI Service",
    description="Medical AI service for stenosis detection.",
    version="1.0.0"
)

# CORS Configuration for external .NET Backend and Flutter App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use /tmp/uploads for HuggingFace Spaces to ensure write access and ephemerality
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define PROJECT_ROOT as the current directory (ai folder)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Initialize the ML Pipeline
# Loading models globally on startup
try:
    pipeline = ImagePipeline(PROJECT_ROOT)
except Exception as e:
    print(f"Warning: Failed to load ImagePipeline on startup: {e}")
    pipeline = None

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def is_safe_path(basedir: str, path: str) -> bool:
    # Resolve the absolute path and ensure it's within the basedir
    abs_path = os.path.abspath(path)
    return abs_path.startswith(os.path.abspath(basedir))

@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Angio AI service is running",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Secure filename extraction to prevent path injection
    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image format. Allowed formats: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")

    # Use a secure random UUID for the stored file
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    if not is_safe_path(UPLOAD_DIR, save_path):
        raise HTTPException(status_code=400, detail="Invalid file path")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="AI Pipeline is not loaded and unavailable.")

        output_prefix = os.path.splitext(unique_name)[0]

        # Process the image
        result = pipeline.analyze_image(
            image_path=save_path,
            output_prefix=output_prefix,
            vessel_threshold=float(os.getenv("VESSEL_THRESHOLD", "0.5")),
            stenosis_threshold=float(os.getenv("STENOSIS_THRESHOLD", "0.4")),
            roi_margin=int(os.getenv("ROI_MARGIN", "20")),
            save_outputs=True
        )

        measurement = result.get("measurement_results", [])
        measurement = measurement[0] if measurement else None
        
        percentage = measurement["stenosis_percentage"] if measurement else 0.0
        severity = measurement["severity"] if measurement else "Normal"
        
        diagnosis_details = f"{severity} stenosis detected at {percentage:.1f}%. "
        if percentage > 70:
            diagnosis_details += "Critical blockage found. Immediate clinical consultation required."
        elif percentage > 40:
            diagnosis_details += "Moderate blockage detected. Further clinical evaluation recommended."
        else:
            diagnosis_details += "Minimal blockage detected."

        # Clean JSON response structure
        return JSONResponse(status_code=200, content={
            "status": "success",
            "data": {
                "stenosis_percentage": percentage,
                "severity": severity,
                "artery_name": "Coronary Arterial Segment",
                "diagnosis_details": diagnosis_details,
                "analysis_metadata": {
                    "roi_bbox": result.get("roi_bbox"),
                    "saved_paths": result.get("saved_paths", {})
                }
            }
        })

    except Exception as e:
        # Avoid leaking internal paths or traceback details in production if possible,
        # but for debugging we include the error string.
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    # Placeholder implementation
    return JSONResponse(status_code=200, content={
        "status": "success",
        "data": {
            "stenosis_percentage": 0.0,
            "message": "Video analysis placeholder (implementation pending)",
            "report": "Analysis of medical video is currently being developed."
        }
    })

@app.post("/dicom-to-video")
async def dicom_to_video_endpoint(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    unique_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{unique_id}.dcm")
    output_video_path = os.path.join(UPLOAD_DIR, f"{unique_id}.mp4")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save DICOM file: {str(e)}")

    try:
        success = convert_dicom_to_mp4(input_path, output_video_path)
        
        if success and os.path.exists(output_video_path):
            return JSONResponse(status_code=200, content={
                "status": "success",
                "data": {
                    "video_path": output_video_path,
                    "filename": f"{unique_id}.mp4"
                }
            })
        else:
            raise HTTPException(status_code=500, detail="DICOM conversion failed or output video not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during DICOM conversion: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 and use PORT env variable for HuggingFace Spaces compatibility
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)