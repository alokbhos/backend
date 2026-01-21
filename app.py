from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import os

from detect import detect_image

app = FastAPI()

# Ensure required folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")

# CORS (frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Backend running"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Save uploaded image
    filename = f"uploads/{uuid.uuid4()}_{file.filename}"

    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections, output_image = detect_image(filename)

    # 🚫 REJECT if no valid detections
    if not detections or len(detections) == 0:
        return {
            "status": "rejected",
            "reason": "No valid civic issue detected"
        }

    # ✅ Accept only if detection exists
    return {
        "status": "success",
        "detections": detections,
        "output_image": output_image
    }
