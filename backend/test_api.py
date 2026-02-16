import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

print("🔍 Checking available models for your API key...")

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"✅ Found Model: {m.name}")
except Exception as e:
    print(f"❌ Error: {e}")