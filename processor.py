import torch
import clip
from PIL import Image

# Use the environment-specific device you verified earlier
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the CLIP model once globally to optimize memory
# ViT-L/14 is specifically required for its 768-dim output
clip_model, preprocess = clip.load("ViT-L/14", device=device)


def get_multimodal_features(image_path, text_content):
    """
    Turns an image and a statement into a single tensor for REDDOT.
    Output shape: [1, 2, 768]
    """
    # 2. Process Image: Resizes, crops, and normalizes
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 3. Process Text: Tokenizes and truncates to 77 tokens
    text_tokens = clip.tokenize([text_content], truncate=True).to(device)

    with torch.no_grad():
        # Encode into the shared latent space
        image_features = clip_model.encode_image(image)  # [1, 768]
        text_features = clip_model.encode_text(text_tokens)  # [1, 768]

    # 4. Concatenate along the sequence dimension
    # We create a sequence of 2 tokens (1 Image, 1 Text)
    # The Model's CLS token will then attend to these
    combined_features = torch.stack([image_features, text_features], dim=1)

    return combined_features.float()