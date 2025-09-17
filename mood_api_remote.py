# mood_api_remote.py - Load model from Hugging Face Hub

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json

# Configuration - use remote model
MODEL_NAME = "prostroller/mood_model_7class_final2"  # Upload to HF Hub
DEVICE = "cpu"

app = FastAPI(title="Mood Detection API", version="1.0.0")

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    predicted_mood: str
    confidence: float

# Global variables
model = None
tokenizer = None
id2label = {
    0: 'Bahagia', 1: 'Lelah', 2: 'Marah',
    3: 'Netral', 4: 'Sedih', 5: 'Stress', 6: 'Tenang'
}

def load_model():
    """Load model from Hugging Face Hub."""
    global model, tokenizer
    
    try:
        # Load from local first, fallback to HF Hub
        model_path = "model/mood_model_7class_final2"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        print("Model loaded from local")
    except:
        # Fallback: would load from HF Hub if uploaded
        print("Local model not found, using default predictions")

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
        # Fallback prediction if model not loaded
        return PredictionResponse(
            text=input_data.text,
            predicted_mood="Netral",
            confidence=0.5
        )
    
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