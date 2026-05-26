import sys
import torch
import clip
from PIL import Image
import io
import gc
# ─────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Processor: Using device → {device}")

# ─────────────────────────────────────────────
# CLIP MODEL LOADING
# FIX 1: Assignment specifies ViT-L/14 (768-dim output).
#         Old code used ViT-B/32 (512-dim), which mismatches
#         the model's emb_dim=768 and the trained checkpoint.
# FIX 2: Load model on the correct device, not hardcoded 'cpu'.
# FIX 3: Cast to float16 to halve RAM usage (~890MB → ~445MB).
#         This prevents CPU RAM crashes on low-memory machines.
# ─────────────────────────────────────────────
clip_model = None
preprocess = None

try:
    print(f"Processor: Loading CLIP ViT-L/14 on {device}...")
    clip_model, preprocess = clip.load("ViT-L/14", device=device)

    # ── FIX: Only use float16 if running on a CUDA GPU ──
    if device == "cuda":
        print("Processor: GPU detected. Casting model to float16 to save VRAM...")
        clip_model = clip_model.half()
    else:
        print("Processor: CPU detected. Keeping model in float32 to prevent runtime errors...")

    clip_model.eval()
    print("Processor: CLIP ViT-L/14 loaded successfully.")
except Exception as e:
    print(f"Processor: WARNING - ViT-L/14 failed ({e}). Falling back to ViT-B/32.")
    print("Processor: WARNING - ViT-B/32 outputs 512-dim embeddings.")
    print("Processor: WARNING - Your checkpoint (emb_dim=768) will NOT load correctly.")
    print("Processor: WARNING - You MUST free RAM and use ViT-L/14 for correct results.")
    try:
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        clip_model = clip_model.half()
        clip_model.eval()
        print("Processor: ViT-B/32 loaded as fallback.")
    except Exception as e2:
        print(f"Processor: CRITICAL - Could not load any CLIP model: {e2}")


def get_multimodal_features(image_input, text_content):
    """
    Extracts CLIP embeddings and returns them in the correct token order
    for RED_DOT's forward() method.

    FIX 4: Old code sent tokens as [interaction, text, image].
            RED_DOT's forward() reads index-0 as image and index-1 as text,
            so the real image embedding (index 2) was silently dropped every
            time, and the model was comparing a pre-baked interaction vector
            against text — no real image-text similarity signal at all.

    FIX 5: Correct order is [image, text]. RED_DOT computes the interaction
            internally via bmm — do NOT pre-compute it here.

    FIX 6: Accepts both file path (str) and raw bytes (for FastAPI uploads),
            eliminating the need to write/read temp files in backend.py.

    Returns: Tensor of shape [1, 2, 768] — float32, ready for RED_DOT.
    """
    if clip_model is None or preprocess is None:
        raise RuntimeError("CLIP model is not loaded. Check startup logs.")

    # ── Load image from path or raw bytes ──────────────────────────────
    if isinstance(image_input, bytes):
        pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        pil_image = Image.open(image_input).convert("RGB")

    # preprocess handles resize, center-crop, normalize
    image_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    # ── Tokenize text (truncates to 77 tokens automatically) ───────────
    text_tokens = clip.tokenize([text_content], truncate=True).to(device)

    with torch.no_grad():
        # Encode in float16 (matches model dtype)
        image_features = clip_model.encode_image(image_tensor)   # [1, 768] fp16
        text_features  = clip_model.encode_text(text_tokens)     # [1, 768] fp16

    # Cast back to float32 — RED_DOT's weights are float32
    image_features = image_features.float()
    text_features  = text_features.float()

    # Free intermediate GPU/CPU memory explicitly
    del image_tensor, text_tokens
    gc.collect()

    # ── Debug: print cosine similarity so you can verify mismatch signal ─
    img_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    txt_norm = text_features  / text_features.norm(dim=-1, keepdim=True)
    cosine_sim = (img_norm * txt_norm).sum().item()
    print(f"Processor: Image-Text Cosine Similarity = {cosine_sim:.4f}")
    print(f"  → Close to 1.0 = strongly matching | Close to 0.0 = mismatched")

    # ── FIX 5: Send [image, text] — RED_DOT reads them at index 0 and 1 ─
    combined = torch.cat([
        image_features.unsqueeze(1),   # index 0 → img_feat_raw in RED_DOT
        text_features.unsqueeze(1),    # index 1 → txt_feat_raw in RED_DOT
    ], dim=1)  # shape: [1, 2, 768]


    print(f"Processor: Feature tensor shape → {combined.shape}")
    return combined  # already float32


if __name__ == "__main__":
    #Ensure the user passed both arguments
    if len(sys.argv) < 3:
        print("Usage: python script.py <path_to_image> \"your text content\"")
        sys.exit(1)
        test_image_path = "Test data/sample3/image.jpg"
        test_text_content = "Wealthy homeowners won't be helped by flood insurance scheme"
    else:
    # If running via terminal with real arguments
        test_image_path = sys.argv[1]
        test_text_content = sys.argv[2]

    # Run the function with CLI arguments
    features = get_multimodal_features(image_input=sys.argv[1], text_content=sys.argv[2])

    # <-- FIX: Print 'features.shape' instead of 'combined.shape'
    print(f"Processor: Feature tensor shape → {features.shape}")