import torch
import pandas as pd
import numpy as np
from pathlib import Path

from lavis import load_model
from Brain import load_model, get_metrics
from processor import get_multimodal_features  # Ensure this file is named processor.py

# Set device based on your environment verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text(file_path):
    """Reads caption and ground truth files."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def run_custom_inference(test_data_root, checkpoint_path, output_csv):
    # 1. Initialize Model with 4 layers, 8 heads, 128 dim
    model = load_model(tf_layers=4, tf_head=8, tf_dim=128, emb_dim=768)

    # 2. Robust Weight Loading
    # 2. Robust Weight Loading with Shape Alignment
    print(f"Loading weights from: {checkpoint_path}")
    try:
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

        # Handle dictionary wrapper
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        else:
            state_dict = checkpoint_data

        # --- FIX SIZE MISMATCH FOR CLS_TOKEN ---
        if "cls_token" in state_dict:
            # Reshape from [768] to [1, 1, 768] to match your model definition
            if state_dict["cls_token"].dim() == 1:
                state_dict["cls_token"] = state_dict["cls_token"].view(1, 1, 768)
                print("Aligned cls_token shape to [1, 1, 768].")

        model.load_state_dict(state_dict, strict=True)
        print("SUCCESS: Weights loaded perfectly.")
    except Exception as e:
        print(f"FAILED to load weights: {e}")
        return

    results = []
    test_path = Path(test_data_root)

    # 3. Process 'sample' folders
    sample_folders = [f for f in test_path.iterdir() if f.is_dir()]
    print(f"Found {len(sample_folders)} sample folders for inference.")

    for folder in sample_folders:
        try:
            # Construct paths
            img_path = str(folder / "image.jpg")
            caption = load_text(folder / "caption.txt")
            ground_truth = load_text(folder / "GT.txt")

            # 4. Feature Extraction & Forward Pass
            features = get_multimodal_features(img_path, caption)

            with torch.no_grad():
                logits = model(features)
                prob = torch.sigmoid(logits).item()  # Convert logit to probability

            # 5. Calculate Required Metrics
            prediction = "FAKE" if prob > 0.5 else "TRUE"
            confidence, entropy = calculate_metrics(prob)

            results.append({
                "Sample_ID": folder.name,
                "Prediction": prediction,
                "Ground_Truth": ground_truth,
                "Confidence": round(confidence, 4),
                "Entropy": round(entropy, 4)
            })

        except Exception as e:
            print(f"Error in {folder.name}: {e}")

    # 6. Save Final Report
    report_df = pd.DataFrame(results)
    report_df.to_csv(output_csv, index=False)
    print(f"Inference Complete! Report saved to {output_csv}")


if __name__ == "__main__":
    # Ensure these paths match your PyCharm structure exactly
    DATA_DIR = "Test data"
    CHECKPOINT = "checkpoints_pt/best_model.pt"

    run_custom_inference(DATA_DIR, CHECKPOINT, "Final_Inference_Report.csv")