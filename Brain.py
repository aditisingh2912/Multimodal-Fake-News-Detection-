import torch
import numpy as np
from model_def import RED_DOT


def load_model(checkpoint_path, device):
    """
    Initialises RED_DOT with the assignment parameters and loads
    the trained checkpoint weights.

    Parameters
    ----------
    checkpoint_path : str   Path to best_model.pt
    device          : torch.device

    Returns
    -------
    model (RED_DOT) in eval mode, or None on failure.
    """
    print("Engine: Initialising RED_DOT architecture...")
    print(f"  tf_layers=4 | tf_head=8 | tf_dim=128 | emb_dim=768")

    model = RED_DOT(tf_layers=4, tf_head=8, tf_dim=128, emb_dim=768).to(device)

    try:
        print(f"Engine: Loading checkpoint from → {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path, map_location=device)

        # Handle both raw state-dict and wrapped checkpoint formats
        if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
            state_dict = checkpoint_data["model_state_dict"]
        else:
            state_dict = checkpoint_data

        # ── CLS TOKEN SHAPE ALIGNMENT ──────────────────────────────────
        # Some checkpoints save cls_token as a flat [768] vector.
        # RED_DOT expects [1, 1, 768].  Fix it before loading.
        if "cls_token" in state_dict:
            ct = state_dict["cls_token"]
            if ct.dim() == 1:
                state_dict["cls_token"] = ct.view(1, 1, 768)
                print("Engine: Fixed cls_token shape [768] → [1, 1, 768].")
            elif ct.dim() == 2:
                state_dict["cls_token"] = ct.view(1, 1, 768)
                print("Engine: Fixed cls_token shape [1, 768] → [1, 1, 768].")

        model.load_state_dict(state_dict, strict=True)
        model.eval()   # turns off Dropout for inference
        print("Engine: Checkpoint loaded. Model in EVAL mode. ✓")
        return model

    except FileNotFoundError:
        print(f"Engine ERROR: Checkpoint file not found at '{checkpoint_path}'")
        print("  → Check that checkpoints_pt/best_model.pt exists.")
        return None
    except RuntimeError as e:
        print(f"Engine ERROR: State dict mismatch — {e}")
        print("  → This usually means the checkpoint was trained with ViT-L/14")
        print("    (emb_dim=768) but you are feeding 512-dim ViT-B/32 features.")
        return None
    except Exception as e:
        print(f"Engine ERROR: Unexpected failure — {e}")
        return None


def get_metrics(prob):
    """
    Derives confidence and Shannon entropy from a raw sigmoid probability.

    Parameters
    ----------
    prob : float   Output of torch.sigmoid(logit), range [0, 1].
                   High prob → model thinks TRUE.
                   Low prob  → model thinks FAKE.

    Returns
    -------
    confidence : float   Distance from the 0.5 decision boundary.
                         Range [0.5, 1.0].  Higher = more certain.
    entropy    : float   Shannon entropy in bits.
                         Range [0, 1].  Higher = more uncertain.
    """
    # Confidence = how far the probability is from 0.5 (maximum uncertainty)
    confidence = prob if prob > 0.5 else (1.0 - prob)

    # Shannon entropy for binary distribution
    p = np.clip(prob, 1e-9, 1.0 - 1e-9)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    return round(float(confidence), 4), round(float(entropy), 4)