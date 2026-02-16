import os
import json
import asyncio
import time
from typing import Optional 
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from io import BytesIO
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

# 1. Load Environment Variables
load_dotenv()

# 2. Setup App
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Memory for Context
user_context = {
    "job_role": "Software Engineer", 
    "resume_text": ""
}

# --- ROUTE 1: SETUP (Resume is OPTIONAL now) ---
@app.post("/upload_resume")
async def upload_resume(job_role: str = Form(...), file: Optional[UploadFile] = File(None)):
    print(f"📄 Receiving setup for role: {job_role}")
    
    try:
        user_context["job_role"] = job_role
        
        if file:
            # Case A: User uploaded a resume
            content = await file.read()
            pdf_reader = PdfReader(BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Limit text to avoid token limits
            user_context["resume_text"] = text[:2000]
            print(f"✅ Resume Parsed for {job_role}")
        else:
            # Case B: No resume provided
            user_context["resume_text"] = "No resume provided. Focus strictly on general competency for the role."
            print(f"⚠️ No Resume provided. Setting context for {job_role} only.")
            
        return {"status": "success", "message": "Context updated"}

    except Exception as e:
        print(f"❌ Error processing setup: {e}")
        return {"status": "error", "message": str(e)}

# --- ROUTE 2: VIDEO PROCESSING ---
@app.post("/process-video")
async def process_video(file: UploadFile = File(...), question: str = Form("Tell me about yourself")):
    print(f"🎥 Processing Real AI Analysis for: {question}")

    try:
        # 1. Save video temporarily
        temp_filename = "temp_video.webm"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "Missing API Key"}

        genai.configure(api_key=api_key)
        
        # Use the Stable Model
        model = genai.GenerativeModel(model_name="gemini-flash-latest")
        
        print("☁️ Uploading video...")
        video_file = genai.upload_file(temp_filename)
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(1)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed on Google's side.")

        # 3. Generate Content with RETRY LOGIC
        print("🧠 Asking AI...")
        
        prompt = f"""
        You are an Interview Coach. The user is answering: "{question}".
        Target Role: {user_context['job_role']}
        Resume Context: {user_context['resume_text']}
        
        Analyze the video answer. 
        CRITICAL: Return ONLY valid JSON. No markdown.
        Structure:
        {{
            "rating": (integer 1-100),
            "feedback": (string, 2-3 sentences),
            "improved_answer": (string, how to say it better),
            "follow_up_question": (string, a relevant next question based on the role)
        }}
        """

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content([video_file, prompt])
                break 
            except ResourceExhausted:
                if attempt < max_retries - 1:
                    print(f"⚠️ Quota hit. Waiting 5 seconds... (Attempt {attempt+1})")
                    time.sleep(5)
                else:
                    raise Exception("Daily Quota Exceeded. Please try again tomorrow.")

        # 4. Clean Response
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        print(f"✅ AI Success! Rating: {json.loads(text_response).get('rating')}")
        
        return {"ai_analysis": json.loads(text_response)}

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"status": "error", "message": str(e)}