# app.py - For Hugging Face Spaces deployment
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os

# Load model
MODEL_PATH = "model/mood_model_7class_final2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Load label mapping
with open(f"{MODEL_PATH}/label_mapping.json", "r") as f:
    mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}

def predict_mood(text):
    """Predict mood from text."""
    if not text.strip():
        return "Error: Text cannot be empty", 0.0
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, 
                      truncation=True, max_length=128)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(outputs.logits, dim=-1).item()
        confidence = probs[0][pred_id].item()
    
    return id2label[pred_id], confidence

# Gradio interface
interface = gr.Interface(
    fn=predict_mood,
    inputs=gr.Textbox(lines=3, placeholder="Masukkan teks bahasa Indonesia..."),
    outputs=[
        gr.Textbox(label="Prediksi Mood"),
        gr.Number(label="Confidence")
    ],
    title="🎭 Indonesian Mood Detection",
    description="Deteksi mood dari teks bahasa Indonesia menggunakan IndoBERT",
    examples=[
        ["hari ini aku sangat bahagia sekali"],
        ["capek banget hari ini"],
        ["stress mikirin deadline"],
        ["tenang dan damai rasanya"]
    ]
)

if __name__ == "__main__":
    interface.launch()