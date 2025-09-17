# mood_api_dummy.py - Ultra minimal API without model loading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI(title="Mood Detection API", version="1.0.0")

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_mood: str
    confidence: float

# Available moods
MOODS = ['Bahagia', 'Lelah', 'Marah', 'Netral', 'Sedih', 'Stress', 'Tenang']

@app.get("/")
async def root():
    return {"message": "Mood Detection API", "status": "running", "note": "Demo version"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": False, "demo": True}

@app.post("/predict_mood", response_model=PredictionResponse)
async def predict_mood(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Simple keyword-based prediction for demo
    text_lower = input_data.text.lower()
    
    if any(word in text_lower for word in ['senang', 'bahagia', 'gembira', 'suka']):
        mood = 'Bahagia'
        confidence = 0.85
    elif any(word in text_lower for word in ['capek', 'lelah', 'tired']):
        mood = 'Lelah'
        confidence = 0.80
    elif any(word in text_lower for word in ['marah', 'kesal', 'jengkel']):
        mood = 'Marah'
        confidence = 0.82
    elif any(word in text_lower for word in ['sedih', 'galau', 'down']):
        mood = 'Sedih'
        confidence = 0.78
    elif any(word in text_lower for word in ['stress', 'tertekan', 'pusing']):
        mood = 'Stress'
        confidence = 0.83
    elif any(word in text_lower for word in ['tenang', 'damai', 'rileks']):
        mood = 'Tenang'
        confidence = 0.79
    else:
        mood = 'Netral'
        confidence = 0.70
    
    return PredictionResponse(
        text=input_data.text,
        predicted_mood=mood,
        confidence=confidence
    )

@app.get("/models/info")
async def model_info():
    return {
        "model_name": "demo-keyword-classifier",
        "type": "keyword-based",
        "note": "This is a demo version using keyword matching"
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)