import os
import json
import asyncio
import time
from typing import Optional
from dotenv import load_dotenv

# Load environment variables FIRST before anything else
# override=True ensures a changed .env is picked up on uvicorn reload
load_dotenv(override=True)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from io import BytesIO
from google import genai
from google.genai import types as genai_types
from emotion_recognition.inference import EmotionAnalyzer
import atexit

# Initialize Emotion Analyzer
emotion_analyzer = EmotionAnalyzer()

def _cleanup():
    """Prevent the mediapipe 0.10.x FaceLandmarker __del__ shutdown crash.

    mediapipe's close() method starts with 'if not self._handle: return'.
    By setting _handle and _lib to None before Python GC runs, __del__ → close()
    hits the early-return guard without touching the broken serial_dispatcher path.
    """
    try:
        lm = getattr(
            getattr(emotion_analyzer, 'drowsiness_detector', None),
            'face_landmarker', None
        )
        if lm is not None:
            lm._handle = None   # triggers early-return in close()
            lm._lib    = None   # prevents any further C calls
    except Exception:
        pass

atexit.register(_cleanup)

# 1. Setup App
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
        
        client = genai.Client(api_key=api_key)
        
        # Check if resume was provided
        resume_text = user_context.get("resume_text", "")
        if not resume_text or resume_text.startswith("No resume provided"):
            return {"status": "no_resume", "questions": []}
        
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
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            questions = json.loads(text_response)
            
            if isinstance(questions, list) and len(questions) > 0:
                print(f"✅ Generated {len(questions)} resume-based questions")
                return {"status": "success", "questions": questions[:5]}
            else:
                return {"status": "error", "questions": [], "message": "Invalid response format"}
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                print("⚠️ Gemini quota exceeded for question generation")
                return {"status": "quota_exceeded", "questions": [], "message": "API quota exceeded"}
            print(f"⚠️ Question generation failed: {e}")
            return {"status": "error", "questions": [], "message": err}
    
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

        client = genai.Client(api_key=api_key)
        
        # Define async helper functions for parallel processing
        async def run_gemini_analysis():
            """Upload video to Gemini and get AI analysis"""
            print("☁️ Uploading video to Gemini...")
            video_file = client.files.upload(file=temp_filename)
            
            # Wait for processing
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(1)
                video_file = client.files.get(name=video_file.name)

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
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[video_file, prompt]
                    )
                    break
                except Exception as ex:
                    err_str = str(ex)
                    is_retryable = any(k in err_str.lower() for k in ["429", "quota", "resource_exhausted", "503", "unavailable"])
                    if is_retryable and attempt < max_retries - 1:
                        print(f"⚠️ API High Demand/Quota. Waiting 5 seconds... (Attempt {attempt+1})")
                        await asyncio.sleep(5)
                    elif attempt == max_retries - 1 and is_retryable:
                        raise Exception("API is currently experiencing high demand or quota limits. Please try again.")
                    else:
                        raise

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
        err_msg = str(e)
        print(f"❌ ERROR: {err_msg}")
        return {"status": "error", "message": err_msg}

# --- ROUTE 3: GENERATE BOOTCAMP PLAN ---
@app.post("/generate_bootcamp_plan")
async def generate_bootcamp_plan(
    job_role: str = Form(...),
    days: int = Form(...),
    resume_text: str = Form("")
):
    """Generate a personalized N-day interview preparation roadmap."""
    print(f"🗓️ Generating {days}-day bootcamp plan for: {job_role}")

    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "Missing API Key"}

        client = genai.Client(api_key=api_key)

        resume_context = f"\nCandidate Resume:\n{resume_text[:1500]}" if resume_text else ""

        day_templates = {
            1: "HR & Behavioral questions (Tell me about yourself, strengths/weaknesses, teamwork, conflict resolution)",
            2: "Core Technical Concepts relevant to the role (fundamentals, theory, commonly tested topics)",
            3: "Problem Solving & Scenario-Based questions (real-world challenges, how they'd handle situations)",
            4: "Advanced Technical & Project-Based questions (deep dives, architecture, past projects, edge cases)",
            5: "Full Mock Interview — mix of all above, simulating a real interview end-to-end"
        }

        actual_days = min(days, 5)
        days_plan = []
        for d in range(1, actual_days + 1):
            days_plan.append(f"Day {d}: {day_templates.get(d, day_templates[5])}")

        prompt = f"""
You are an expert interview coach creating a personalized preparation plan.

Target Role: {job_role}
Days Until Interview: {actual_days}
{resume_context}

Generate a {actual_days}-day interview preparation roadmap. For each day, provide:
- A short title (5 words max)
- A focus description (1 sentence)
- Exactly 5 interview questions tailored to that day's theme and the target role

Day themes to follow:
{chr(10).join(days_plan)}

CRITICAL: Return ONLY valid JSON in this exact structure, no markdown:
{{
  "role": "{job_role}",
  "total_days": {actual_days},
  "plan": [
    {{
      "day": 1,
      "title": "Day Title Here",
      "focus": "What this day focuses on",
      "questions": ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]
    }}
  ]
}}
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text = response.text.replace("```json", "").replace("```", "").strip()
            plan_data = json.loads(text)
            print(f"✅ Bootcamp plan generated for {actual_days} days")
            return {"status": "success", "plan": plan_data}
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                return {"status": "error", "message": "API quota exceeded. Try again later."}
            print(f"⚠️ Plan generation error: {e}")
            return {"status": "error", "message": err}

    except Exception as e:
        print(f"❌ Bootcamp plan error: {e}")
        return {"status": "error", "message": str(e)}


# --- ROUTE 4: PROCESS BOOTCAMP ANSWER ---
@app.post("/process_bootcamp_answer")
async def process_bootcamp_answer(
    file: UploadFile = File(...),
    question: str = Form("Tell me about yourself"),
    day: int = Form(1),
    job_role: str = Form("Software Engineer")
):
    """Process a bootcamp day answer — same AI analysis but with day context."""
    print(f"🎓 Processing bootcamp Day {day} answer for: {question}")

    try:
        temp_filename = f"temp_bootcamp_day{day}.webm"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "Missing API Key"}

        client = genai.Client(api_key=api_key)

        async def run_gemini_bootcamp():
            print("☁️ Uploading bootcamp video to Gemini...")
            video_file = client.files.upload(file=temp_filename)

            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(1)
                video_file = client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise ValueError("Video processing failed.")

            prompt = f"""
You are an Interview Coach evaluating a Day {day} bootcamp practice session.
Role: {job_role}
Question: "{question}"

Analyze the video response. Return ONLY valid JSON, no markdown:
{{
  "rating": (integer 1-100),
  "feedback": (2-3 sentences of specific feedback),
  "improved_answer": (how to say it better),
  "follow_up_question": (a relevant next question for this role)
}}
"""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[video_file, prompt]
                    )
                    break
                except Exception as ex:
                    err_str = str(ex)
                    is_retryable = any(k in err_str.lower() for k in ["429", "quota", "resource_exhausted", "503", "unavailable"])
                    if is_retryable and attempt < max_retries - 1:
                        await asyncio.sleep(5)
                    elif attempt == max_retries - 1 and is_retryable:
                        raise Exception("API is currently experiencing high demand or quota limits. Please try again.")
                    else:
                        raise

            text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(text)

        async def run_emotion_bootcamp():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                emotion_analyzer.analyze_video_with_eye_tracking,
                temp_filename
            )

        ai_data, combined_data = await asyncio.gather(
            run_gemini_bootcamp(),
            run_emotion_bootcamp()
        )

        return {
            "ai_analysis": ai_data,
            "emotion_analysis": combined_data.get("emotion_analysis", {}),
            "eye_tracking": combined_data.get("eye_tracking", {}),
            "drowsiness": combined_data.get("drowsiness", {})
        }

    except Exception as e:
        print(f"❌ Bootcamp answer error: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)