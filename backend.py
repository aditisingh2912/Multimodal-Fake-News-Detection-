import torch
import time
import shutil
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from Brain import load_model, get_metrics
from processor import get_multimodal_features

# 1. Initialize FastAP
app = FastAPI(title="REDDOT Multimodal Detection API")

# 2. Setup Device and Model Singleton
# This runs ONCE when you start the server, keeping the model "warm" in memory.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "checkpoints_pt/best_model.pt"

print("--- System Startup: Loading AI Engine ---")
MODEL = load_model(CHECKPOINT, device)

if MODEL is None:
    print("CRITICAL ERROR: Model failed to load. Check checkpoint path.")


@app.get("/health")
def health_check():
    return {"status": "online", "device": str(device)}


@app.post("/predict")
async def predict(image: UploadFile = File(...), caption: str = Form(...)):
    """
    Handles real-time multimodal inference requests.
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    start_time = time.time()

    try:
        # 1. Temporary storage for the uploaded file
        # This is necessary because get_multimodal_features expects a file path
        temp_filename = f"temp_{int(time.time())}_{image.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # 2. Feature Extraction (The "IO-Heavy" Part)
        features = get_multimodal_features(temp_filename, caption)

        # 3. Model Inference (The "Compute-Heavy" Part)
        with torch.no_grad():
            logits = MODEL(features.to(device))
            prob = torch.sigmoid(logits).item()

        # 4. Calculate Final Metrics via Engine
        prediction = "FAKE" if prob > 0.5 else "TRUE"
        confidence, entropy = get_metrics(prob)

        # 5. Calculate Latency
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)

        # 6. Cleanup temp file
        os.remove(temp_filename)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "entropy": entropy,
            "latency_ms": latency_ms,
            "status": "success"
        }

    except Exception as e:
        # In production, you'd log this error properly
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    # Run the server on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)