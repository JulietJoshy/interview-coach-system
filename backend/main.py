from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import whisper  # <--- NEW: The Listening AI
from moviepy import VideoFileClip # <--- NEW: Video Editor

# 1. Initialize the App
app = FastAPI(title="AI Interview Coach API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Folders
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 3. Load the Whisper Model (Do this once at startup)
print("Loading Whisper AI model... (this might take a moment)")
model = whisper.load_model("base") # 'base' is fast and accurate enough
print("Whisper AI model loaded!")

# --- ENDPOINTS ---

@app.get("/")
def health_check():
    return {"status": "active", "message": "Backend is online!"}

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    """
    1. Saves video.
    2. Extracts audio.
    3. Transcribes text using Whisper.
    4. Returns text + mock metrics.
    """
    # A. Save the video file
    video_filename = f"{UPLOAD_DIR}/{file.filename}"
    with open(video_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"Processing video: {video_filename}")

    # B. Extract Audio using MoviePy
    audio_filename = video_filename.replace(".webm", ".wav").replace(".mp4", ".wav")
    
    # Simple check to ensure we don't crash if file isn't video
    try:
        video_clip = VideoFileClip(video_filename)
        video_clip.audio.write_audiofile(audio_filename, logger=None)
        video_clip.close() # Close to release memory
    except Exception as e:
        return {"error": f"Failed to process video audio: {str(e)}"}

    # C. Run Whisper on the Audio (The "Listening" Part)
    result = model.transcribe(audio_filename)
    user_text = result["text"]
    
    print(f"Detected Text: {user_text}")

    # D. Return the Real Text + Mock Metrics
    return {
        "transcript": user_text,  # <--- THIS IS NOW REAL!
        "feedback": "You are doing great, but try to smile more.",
        "metrics": {
            "eye_contact_score": 85,
            "nervousness": "Low",
            "emotion": "Neutral"
        }
    }