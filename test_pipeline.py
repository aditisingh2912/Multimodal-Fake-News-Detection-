import torch
import os
from model_def import RED_DOT
from processor import get_multimodal_features
from Brain import get_metrics
import torch
from model_def import RED_DOT

# Load the file
checkpoint = torch.load("checkpoints_pt/best_model.pt", map_location='cpu')
weights = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

# Get the model's expected keys
model = RED_DOT(tf_layers=4, tf_head=8, tf_dim=128, emb_dim=768)
model_keys = model.state_dict().keys()

print("--- KEYS IN FILE ---")
print(list(weights.keys())[:5])  # Show first 5 keys

print("\n--- KEYS EXPECTED BY CODE ---")
print(list(model_keys)[:5])
print("--- Step 1: Testing model_def.py ---")
try:
    model = RED_DOT(tf_layers=4, tf_head=8, tf_dim=128, emb_dim=768)
    print("SUCCESS: RED_DOT model initialized correctly.")
except Exception as e:
    print(f"FAILURE in model_def: {e}")

# 2. Test Feature Extraction (process.py)
print("\n--- Step 2: Testing process.py ---")
# Use one of your actual sample paths
test_img = "Test data/sample1/image.jpg"
test_caption = "A test caption for a news statement."

if os.path.exists(test_img):
    try:
        features = get_multimodal_features(test_img, test_caption)
        print(f"SUCCESS: Features extracted. Shape: {features.shape}") # Should be [1, 2, 768]
    except Exception as e:
        print(f"FAILURE in process.py: {e}")
else:
    print(f"ERROR: Test image not found at {test_img}. Please check your path.")

# 3. Test Integration (Full Inference)
print("\n--- Step 3: Testing Full Integration ---")
try:
    with torch.no_grad():
        logits = model(features)
        prob = torch.sigmoid(logits).item()
        print(f"SUCCESS: Model produced a prediction probability: {prob:.4f}")
    print("\n--- ALL THREE PACKAGES ARE WORKING LOCALLY! ---")
except Exception as e:
    print(f"FAILURE in integration: {e}")