# mood_api_minimal.py - Ultra minimal version for Railway

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "model/mood_model_7class_final2")
DEVICE = "cpu"

# FastAPI App
app = FastAPI(title="Mood Detection API", version="1.0.0")

# Pydantic Models
class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_mood: str
    confidence: float

# Global Model Variables
model = None
tokenizer = None
id2label = None

def load_model():
    """Load model and tokenizer."""
    global model, tokenizer, id2label
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    
    # Load label mapping
    mapping_file = os.path.join(MODEL_PATH, "label_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            id2label = {int(k): v for k, v in mapping["id2label"].items()}
    else:
        id2label = {
            0: 'Bahagia', 1: 'Lelah', 2: 'Marah',
            3: 'Netral', 4: 'Sedih', 5: 'Stress', 6: 'Tenang'
        }

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def root():
    return {"message": "Mood Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict_mood", response_model=PredictionResponse)
async def predict_mood(input_data: TextInput):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Tokenize and predict
    inputs = tokenizer(input_data.text, return_tensors="pt", padding=True, 
                      truncation=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        confidence = probs[0][pred_id].item()
    
    return PredictionResponse(
        text=input_data.text,
        predicted_mood=id2label[pred_id],
        confidence=confidence
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)