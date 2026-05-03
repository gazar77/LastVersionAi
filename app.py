import os
import uuid
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from inference.image_pipeline import ImagePipeline
from inference.dicom_handler import convert_dicom_to_mp4


# =========================
# FASTAPI APP CONFIG
# =========================
app = FastAPI(
    title="Angio AI Service",
    description="Medical AI service for stenosis detection.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# مهم جدًا لحل مشاكل Hugging Face routing
app.root_path = ""


# =========================
# CORS CONFIG
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# STORAGE CONFIG
# =========================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# =========================
# LOAD AI PIPELINE
# =========================
try:
    pipeline = ImagePipeline(PROJECT_ROOT)
except Exception as e:
    print(f"⚠️ Failed to load pipeline: {e}")
    pipeline = None


ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_safe_path(base_dir: str, path: str) -> bool:
    return os.path.abspath(path).startswith(os.path.abspath(base_dir))


# =========================
# ROOT
# =========================
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Angio AI service is running",
        "docs_url": "/docs"
    }


# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None
    }


# =========================
# IMAGE ANALYSIS
# =========================
@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = os.path.basename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {ext}"
        )

    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    if not is_safe_path(UPLOAD_DIR, save_path):
        raise HTTPException(status_code=400, detail="Invalid path")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if pipeline is None:
        raise HTTPException(status_code=503, detail="AI pipeline not loaded")

    try:
        result = pipeline.analyze_image(
            image_path=save_path,
            output_prefix=os.path.splitext(unique_name)[0],
            vessel_threshold=float(os.getenv("VESSEL_THRESHOLD", "0.5")),
            stenosis_threshold=float(os.getenv("STENOSIS_THRESHOLD", "0.4")),
            roi_margin=int(os.getenv("ROI_MARGIN", "20")),
            save_outputs=True
        )

        measurement = result.get("measurement_results", [])
        measurement = measurement[0] if measurement else None

        percentage = measurement["stenosis_percentage"] if measurement else 0.0
        severity = measurement["severity"] if measurement else "Normal"

        return JSONResponse({
            "status": "success",
            "data": {
                "stenosis_percentage": percentage,
                "severity": severity,
                "artery_name": "Coronary Arterial Segment",
                "diagnosis_details": f"{severity} stenosis detected at {percentage:.1f}%",
                "metadata": {
                    "roi_bbox": result.get("roi_bbox"),
                    "saved_paths": result.get("saved_paths", {})
                }
            }
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# VIDEO ANALYSIS (PLACEHOLDER)
# =========================
@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    return {
        "status": "success",
        "data": {
            "message": "Video analysis not implemented yet"
        }
    }


# =========================
# DICOM TO VIDEO
# =========================
@app.post("/dicom-to-video")
async def dicom_to_video(file: UploadFile = File(...)):

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    unique_id = uuid.uuid4().hex

    input_path = os.path.join(UPLOAD_DIR, f"{unique_id}.dcm")
    output_path = os.path.join(UPLOAD_DIR, f"{unique_id}.mp4")

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        success = convert_dicom_to_mp4(input_path, output_path)

        if success and os.path.exists(output_path):
            return {
                "status": "success",
                "data": {
                    "video_path": output_path,
                    "filename": f"{unique_id}.mp4"
                }
            }

        raise HTTPException(status_code=500, detail="Conversion failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# DEBUG ROUTE (optional)
# =========================
@app.get("/docs-test")
def docs_test():
    return {"message": "Docs routing working if you see this"}


# =========================
# MAIN RUN
# =========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 7860))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port
    )