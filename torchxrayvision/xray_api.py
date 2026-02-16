from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from uuid import uuid4
from pydantic import BaseModel
from typing import Dict, Any
import requests 
import json

# Import analyze_xray from your custom script
from probability_to_text_converter import analyze_xray

# --- CONFIGURATION ---
# PASTE YOUR REAL API KEY INSIDE THE QUOTES BELOW
GEMINI_API_KEY = "AIzaSyAnG3ejRrGBFTBNWdXuB0LL_nOFrQIB-bc" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatRequest(BaseModel):
    question: str
    context: Dict[str, Any]

# --- EXISTING ENDPOINTS ---

@app.post("/upload")
def upload_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, and .png files are allowed.")
    temp_filename = os.path.join(UPLOAD_DIR, f"uploaded_{uuid4().hex}{ext}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": temp_filename}

@app.post("/analyze")
def analyze_uploaded_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1]
    if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        raise HTTPException(status_code=400, detail="Only .jpg, .jpeg, and .png files are allowed.")
    temp_filename = os.path.join(UPLOAD_DIR, f"analyze_{uuid4().hex}{ext}")
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        results = analyze_xray(temp_filename)
        return JSONResponse(content=results["json_response"])
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.get("/")
def root():
    return {"message": "X-ray AI Analysis API."}

@app.post("/chat")
async def chat_agent(request: ChatRequest):
    # 1. Validation
    if "YOUR_REAL_API_KEY" in GEMINI_API_KEY or len(GEMINI_API_KEY) < 10:
        print("ERROR: You have not replaced the placeholder API key in main.py")
        raise HTTPException(status_code=500, detail="Server API Key not configured.")

    try:
        # DEBUG: Print what we received
        print("=" * 50)
        print("RECEIVED QUESTION:", request.question)
        print("RECEIVED CONTEXT:", request.context)
        print("=" * 50)
        
        # 2. Context & Prompt - IMPROVED FOR SIMPLICITY
        context_str = json.dumps(request.context, indent=2)
        prompt_text = f"""
        You are a friendly medical AI assistant helping patients understand their Chest X-Ray results.
        
        ANALYSIS DATA:
        {context_str}
        
        USER QUESTION: 
        {request.question}
        
        INSTRUCTIONS:
        - Answer in simple, easy-to-understand language (like explaining to a friend)
        - Be conversational and natural, not overly formal
        - Avoid bullet points unless specifically asked
        - Keep responses brief (2-4 sentences) unless asked for details
        - Use everyday words instead of complex medical terms
        - Be reassuring but honest
        - If asked "what did you find" or similar, mention only the TOP 2-3 most significant findings briefly
        - Don't repeat the entire analysis unless specifically asked
        
        Examples of good responses:
        - "The X-ray shows a hiatal hernia, which means part of your stomach is pushing up through your diaphragm. It's quite common and manageable. I'd recommend seeing a gastroenterologist to discuss treatment options."
        - "I can see a few things that need attention. The main concern is the hiatal hernia. There are also some lung findings that your doctor should review with you."
        
        Answer naturally and directly.
        """

        # 3. Call Google API - Using gemini-2.5-flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt_text}]
            }]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        
        # 4. Error Handling
        if response.status_code != 200:
            print("------------------------------------------------")
            print(f"GOOGLE API ERROR: {response.text}")
            print("------------------------------------------------")
            raise HTTPException(status_code=response.status_code, detail=f"Google API Error: {response.text}")

        # 5. Success
        data = response.json()
        ai_text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        
        if not ai_text:
            ai_text = "I'm sorry, I couldn't generate a response."

        return {"answer": ai_text}

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))