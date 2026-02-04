from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil

# 1. Initialize the App
app = FastAPI(title="AI Interview Coach API")

# 2. CORS Setup (Allows React to talk to Python)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a folder to save temporary videos
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "message": "Backend is online!"}

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    # Save the file locally
    file_location = f"{UPLOAD_DIR}/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"Received video file: {file.filename}")

    # Mock Response for now
    return {
        "transcript": "User speech text will appear here...",
        "feedback": "Good eye contact, but try to speak slower.",
        "metrics": {
            "eye_contact_score": 85,
            "nervousness": "Low",
            "emotion": "Neutral"
        }
    }