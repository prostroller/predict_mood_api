# mood_api.py - Production-ready API for Railway deployment

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from typing import List, Dict, Any
import time

# ======================
# Configuration
# ======================
MODEL_PATH = os.getenv("MODEL_PATH", "model/mood_model_7class_final2")
MAX_LENGTH = 128
DEVICE = "cpu"  # Railway typically uses CPU

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# FastAPI App
# ======================
app = FastAPI(
    title="Mood Detection API",
    description="Indonesian Text Mood Classification using IndoBERT",
    version="1.0.0"
)

# ======================
# Pydantic Models
# ======================
class TextInput(BaseModel):
    text: str
    
class BatchTextInput(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    text: str
    predicted_mood: str
    confidence: float
    top_predictions: List[Dict[str, Any]]
    processing_time: float

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_processing_time: float

# ======================
# Global Model Variables
# ======================
model = None
tokenizer = None
id2label = None
label2id = None

# ======================
# Model Loading
# ======================
def load_model():
    """Load model, tokenizer, and label mappings."""
    global model, tokenizer, id2label, label2id
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map=None  # Handle device manually
        )
        
        # Move to device
        model.to(DEVICE)
        model.eval()
        
        # Load label mapping
        mapping_file = os.path.join(MODEL_PATH, "label_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, "r", encoding="utf-8") as f:
                mapping = json.load(f)
                id2label = {int(k): v for k, v in mapping["id2label"].items()}
                label2id = mapping["label2id"]
        else:
            # Fallback mapping
            id2label = {
                0: 'Bahagia',
                1: 'Lelah', 
                2: 'Marah',
                3: 'Netral',
                4: 'Sedih',
                5: 'Stress',
                6: 'Tenang'
            }
            label2id = {v: k for k, v in id2label.items()}
        
        logger.info(f"Model loaded successfully on {DEVICE}")
        logger.info(f"Available labels: {list(id2label.values())}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# ======================
# Prediction Functions
# ======================
def predict_single_text(text: str, top_k: int = 3) -> Dict[str, Any]:
    """Predict mood for a single text."""
    start_time = time.time()
    
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        
        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(id2label)))
        
        # Format results
        top_predictions = []
        for prob, idx in zip(top_probs, top_indices):
            top_predictions.append({
                "mood": id2label[idx.item()],
                "confidence": float(prob.item())
            })
        
        # Main prediction
        predicted_mood = top_predictions[0]["mood"]
        confidence = top_predictions[0]["confidence"]
        
        processing_time = time.time() - start_time
        
        return {
            "text": text,
            "predicted_mood": predicted_mood,
            "confidence": confidence,
            "top_predictions": top_predictions,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error predicting text: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# ======================
# API Endpoints
# ======================
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Mood Detection API is running",
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "available_moods": list(id2label.values()) if id2label else []
    }

@app.get("/health")
async def health_check():
    """Health check with model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": DEVICE
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_mood(input_data: TextInput):
    """Predict mood for a single text."""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = predict_single_text(input_data.text)
    return PredictionResponse(**result)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_mood_batch(input_data: BatchTextInput):
    """Predict mood for multiple texts."""
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not input_data.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    if len(input_data.texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    start_time = time.time()
    results = []
    
    for text in input_data.texts:
        if text.strip():  # Skip empty texts
            result = predict_single_text(text)
            results.append(PredictionResponse(**result))
    
    total_time = time.time() - start_time
    
    return BatchPredictionResponse(
        results=results,
        total_processing_time=total_time
    )

@app.get("/models/info")
async def model_info():
    """Get model information."""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get model config
    config = model.config
    
    return {
        "model_name": getattr(config, 'name_or_path', 'unknown'),
        "num_labels": config.num_labels,
        "max_position_embeddings": getattr(config, 'max_position_embeddings', MAX_LENGTH),
        "vocab_size": getattr(config, 'vocab_size', 'unknown'),
        "labels": id2label,
        "model_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    }

# ======================
# Error Handlers
# ======================
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)