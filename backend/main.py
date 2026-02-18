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
from emotion_recognition.inference import EmotionAnalyzer

# Initialize Emotion Analyzer
emotion_analyzer = EmotionAnalyzer()

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

# --- ROUTE 1.5: GENERATE RESUME QUESTIONS ---
@app.post("/generate_resume_questions")
async def generate_resume_questions():
    """Generate interview questions based on uploaded resume and target role."""
    print(f"📝 Generating resume-based questions for: {user_context['job_role']}")
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "questions": [], "message": "Missing API Key"}
        
        genai.configure(api_key=api_key)
        
        # Check if resume was provided
        resume_text = user_context.get("resume_text", "")
        if not resume_text or resume_text.startswith("No resume provided"):
            return {"status": "no_resume", "questions": []}
        
        model = genai.GenerativeModel(model_name="gemini-flash-latest")
        
        prompt = f"""
        You are an expert interviewer. Based on the following resume and target role, generate 5 specific interview questions.
        
        Target Role: {user_context['job_role']}
        Resume: {resume_text}
        
        Generate questions that:
        1. Are specific to the candidate's experience and skills mentioned in the resume
        2. Test technical knowledge relevant to both the resume and target role
        3. Explore projects or achievements mentioned in the resume
        4. Assess fit for the target role based on their background
        5. Challenge the candidate on potential gaps or transitions
        
        CRITICAL: Return ONLY a valid JSON array of exactly 5 question strings. No markdown, no explanation.
        Example: ["Question 1?", "Question 2?", "Question 3?", "Question 4?", "Question 5?"]
        """
        
        try:
            response = model.generate_content(prompt)
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            questions = json.loads(text_response)
            
            if isinstance(questions, list) and len(questions) > 0:
                print(f"✅ Generated {len(questions)} resume-based questions")
                return {"status": "success", "questions": questions[:5]}
            else:
                return {"status": "error", "questions": [], "message": "Invalid response format"}
        except ResourceExhausted:
            print("⚠️ Gemini quota exceeded for question generation")
            return {"status": "quota_exceeded", "questions": [], "message": "API quota exceeded"}
        except Exception as e:
            print(f"⚠️ Question generation failed: {e}")
            return {"status": "error", "questions": [], "message": str(e)}
    
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        return {"status": "error", "questions": [], "message": str(e)}

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
        
        # Define async helper functions for parallel processing
        async def run_gemini_analysis():
            """Upload video to Gemini and get AI analysis"""
            model = genai.GenerativeModel(model_name="gemini-flash-latest")
            
            print("☁️ Uploading video to Gemini...")
            video_file = genai.upload_file(temp_filename)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(1)
                video_file = genai.get_file(video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed on Google's side.")

            # Generate Content with RETRY LOGIC
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
                        await asyncio.sleep(5)
                    else:
                        raise Exception("Daily Quota Exceeded. Please try again tomorrow.")

            # Clean Response
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            ai_data = json.loads(text_response)
            print(f"✅ AI Success! Rating: {ai_data.get('rating')}")
            return ai_data
        
        async def run_emotion_analysis():
            """Analyze facial emotions and eye movements in the video"""
            print("😊 Analyzing Emotions and Eye Tracking...")
            # Run in thread pool since emotion_analyzer is sync
            loop = asyncio.get_event_loop()
            combined_results = await loop.run_in_executor(
                None, 
                emotion_analyzer.analyze_video_with_eye_tracking, 
                temp_filename
            )
            return combined_results
        
        # 3. Run both analyses in parallel for speed
        print("🚀 Starting parallel analysis (Gemini + Emotions + Eye Tracking + Drowsiness)...")
        ai_data, combined_data = await asyncio.gather(
            run_gemini_analysis(),
            run_emotion_analysis()
        )
        
        # Extract emotion, eye tracking, and drowsiness data
        emotion_data = combined_data.get("emotion_analysis", {})
        eye_tracking_data = combined_data.get("eye_tracking", {})
        drowsiness_data = combined_data.get("drowsiness", {})
        
        # Merge Results
        final_response = {
            "ai_analysis": ai_data,
            "emotion_analysis": emotion_data,
            "eye_tracking": eye_tracking_data,
            "drowsiness": drowsiness_data
        }
        
        return final_response

    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {"status": "error", "message": str(e)}