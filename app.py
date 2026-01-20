from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
import os
from fastapi.staticfiles import StaticFiles


from detect import detect_image

app = FastAPI()

# ✅ Ensure results directory exists (IMPORTANT for Render)
os.makedirs("results", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")

# Allow frontend access
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
    os.makedirs("uploads", exist_ok=True)

    filename = f"uploads/{uuid.uuid4()}_{file.filename}"

    with open(filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    detections, output_image = detect_image(filename)

    return {
        "detections": detections,
        "output_image": output_image
    }
