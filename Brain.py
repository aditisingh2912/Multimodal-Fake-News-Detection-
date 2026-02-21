import torch
import numpy as np
from model_def import RED_DOT

def load_model(checkpoint_path, device):
    print(f"Engine: Initializing REDDOT Architecture...")
    model = RED_DOT(tf_layers=4, tf_head=8, tf_dim=128, emb_dim=768).to(device)

    try:
        # 2. Load the checkpoint
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

        state_dict = checkpoint_data["model_state_dict"] if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data else checkpoint_data

        # 3. Fix the CLS Token Size Mismatch [768] -> [1, 1, 768]
        if "cls_token" in state_dict:
            if state_dict["cls_token"].dim() == 1:
                state_dict["cls_token"] = state_dict["cls_token"].view(1, 1, 768)
                print("Engine: Fixed cls_token shape alignment.")

        # 4. Load weights into the model
        model.load_state_dict(state_dict, strict=True)
        model.eval() # Set to evaluation mode (turns off Dropout)
        print("Engine: Weights loaded and model set to EVAL mode.")
        return model

    except Exception as e:
        print(f"Engine Error: Failed to load model weights: {e}")
        return None

def get_metrics(prob):
    """
    Mathematical core for uncertainty and confidence.
    Shared across Batch and API modes.
    """
    # 1. Confidence (Distance from the 0.5 boundary)
    confidence = prob if prob > 0.5 else (1 - prob)
    p_safe = np.clip(prob, 1e-9, 1 - 1e-9)
    entropy = -(p_safe * np.log2(p_safe) + (1 - p_safe) * np.log2(1 - p_safe))

    return round(confidence, 4), round(entropy, 4)