# test_api.py - Test the deployed API

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"  # Change to your Railway URL when deployed

def test_api():
    """Test all API endpoints."""
    
    print("=== TESTING MOOD DETECTION API ===\n")
    
    # 1. Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 2. Test root endpoint
    print("2. Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 3. Test single prediction
    print("3. Testing single prediction...")
    test_text = "hari ini aku sangat bahagia sekali"
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": test_text}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Text: {result['text']}")
        print(f"Predicted Mood: {result['predicted_mood']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Processing Time: {result['processing_time']:.3f}s")
        print("Top Predictions:")
        for pred in result['top_predictions']:
            print(f"  - {pred['mood']}: {pred['confidence']:.2%}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 4. Test batch prediction
    print("4. Testing batch prediction...")
    test_texts = [
        "aku sangat senang hari ini",
        "capek banget hari ini",
        "biasa aja sih",
        "stress mikirin deadline",
        "tenang dan damai rasanya"
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={"texts": test_texts}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Total Processing Time: {result['total_processing_time']:.3f}s")
        print("Results:")
        for i, res in enumerate(result['results']):
            print(f"  {i+1}. '{res['text']}'")
            print(f"     → {res['predicted_mood']} ({res['confidence']:.2%})")
        print()
    except Exception as e:
        print(f"Error: {e}\n")
    
    # 5. Test model info
    print("5. Testing model info...")
    try:
        response = requests.get(f"{BASE_URL}/models/info")
        print(f"Status: {response.status_code}")
        info = response.json()
        print(f"Model: {info.get('model_name', 'unknown')}")
        print(f"Labels: {info.get('num_labels', 'unknown')}")
        print(f"Model Size: {info.get('model_size_mb', 0):.1f} MB")
        print(f"Available Moods: {list(info.get('labels', {}).values())}")
        print()
    except Exception as e:
        print(f"Error: {e}\n")

if __name__ == "__main__":
    test_api()