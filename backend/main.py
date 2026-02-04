from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import whisper
from moviepy import VideoFileClip
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load Environment Variables (The Key)
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Check if the key is found
if not API_KEY:
    print("❌ ERROR: API Key not found! Check your .env file.")
else:
    print("✅ API Key found. Configuring Gemini...")
    genai.configure(api_key=API_KEY)

# 2. Initialize App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Setup Folders
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base")
print("Whisper ready!")

@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    video_path = f"{UPLOAD_DIR}/{file.filename}"
    audio_path = video_path.replace(".mp4", ".wav").replace(".webm", ".wav")
    
    try:
        # 1. Save Video
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Extract Audio
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path, logger=None)
        video_clip.close()

        # 3. Transcribe
        result = model.transcribe(audio_path)
        user_text = result["text"]

        # 4. Analyze
        prompt = f"""
        You are an expert interview coach. 
        The candidate said: "{user_text}"
        
        1. Give a rating out of 100 based on clarity and confidence.
        2. Provide one specific tip to improve.
        3. Generate a follow-up interview question.
        
        Return JSON: {{ "rating": 0, "feedback": "...", "follow_up_question": "..." }}
        """
        
        gemini_model = genai.GenerativeModel("gemini-flash-latest")
        response = gemini_model.generate_content(prompt)
        ai_feedback = response.text.replace("```json", "").replace("```", "")

        return {
            "transcript": user_text,
            "ai_analysis": ai_feedback
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # --- THE CLEANUP CREW ---
        # This runs even if the code crashes, keeping your server clean.
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        print("🧹 Cleaned up temporary files.")