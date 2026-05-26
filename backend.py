import torch
import gc
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from Orchestrator import REDDOTRagEngine

app = FastAPI(
    title="REDDOT RAG Detection API",
    description="Scalable Inference and Multimodal Fusion Verification Engine",
    version="1.0.0"
)

print("=" * 60)
print("Startup: Initialising Global REDDOTRagEngine Context Cache...")
print("=" * 60)

try:
    ENGINE = REDDOTRagEngine(
        checkpoint_path = "checkpoints_pt/best_model.pt",
        store_path      = "faiss_joint_store",
        k_neighbours    = 5,
        entropy_gate    = 0.25,
    )
except Exception as e:
    print(f"CRITICAL ERROR: Engine failed to boot memory maps — {e}")
    ENGINE = None


@app.get("/health")
def health_check():
    """
    Diagnostic Endpoint to monitor hardware context states.
    """
    return {
        "status"       : "online" if ENGINE is not None else "degraded",
        "device"       : "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded" : ENGINE is not None and ENGINE.model is not None,
        "faiss_entries": len(ENGINE.metadata) if ENGINE else 0,
        "entropy_gate" : ENGINE.entropy_gate if ENGINE else None,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...), caption: str = Form(...)):
    """
    Dynamic Endpoint: Consumes raw bytes buffer directly from browser stream
    and delivers verification consensus rating objects.
    """
    if ENGINE is None:
        raise HTTPException(status_code=500, detail="Neural Vector Engine context is uninitialised.")

    try:
        # FIX 2: Stream raw bytes buffer into high-speed memory arrays
        image_bytes = await image.read()

        # Invoke the LangChain polymorphic pipeline layer dynamically
        result = ENGINE.run(
            image_input = image_bytes,
            caption     = caption,
            sample_id   = image.filename,
        )

        # Explicitly release connection streams for runtime efficiency
        await image.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"status": "success", **result}

    except Exception as pipeline_fault:
        print(f"❌ Predict Runtime Failure: {pipeline_fault}")
        # Always safe close connection buffers on unexpected crash gates
        try:
            await image.close()
        except:
            pass

        # FIX 3: Elevate standard HTTP code 500 for strict pipeline audit tracing
        raise HTTPException(
            status_code=500,
            detail=f"Internal inference failure encountered: {str(pipeline_fault)}"
        )


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 exposes the server to local networks, perfect for local Streamlit integration
    uvicorn.run(app, host="0.0.0.0", port=8000)