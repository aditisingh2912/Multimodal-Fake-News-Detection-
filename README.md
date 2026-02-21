# REDDOT: Multimodal Fake News Detection System
## An Inference AI pipeline for identifying contextual inconsistencies between news images and captions
### Key Features
    Multimodal Fusion: Cross-references visual (CLIP) and textual features using a 4-layer Transformer Encoder.
    Uncertainty Quantification: Reports Shannon Entropy to flag when the model is "confused" by conflicting evidence
    Production Architecture: Decoupled FastAPI backend and Streamlit frontend for high-performance, low-latency inference.
    Scalable Design: Implements the Model Singleton Pattern to serve predictions in $<500$ms.
