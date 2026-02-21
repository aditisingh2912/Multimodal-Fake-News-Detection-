# REDDOT: Multimodal Fake News Detection System
### An Inference AI pipeline for identifying contextual inconsistencies between news images and captions

#### Key Features

1.Multimodal Fusion: Cross-references visual (CLIP) and textual features using a 4-layer Transformer Encoder.

2.Entropy calculation : Reports Shannon Entropy to flag when the model is "confused" by conflicting evidence

3.Architecture: Decoupled FastAPI backend and Streamlit frontend for high-performance, low-latency inference.

4.Scalable Design: Implements the Model Singleton Pattern to serve predictions in $<500$ms.

## Arcitecture Design
### Stage 1: The Ingestion Layer 
1.1 The journey begins at the Frontend.

1.2 Binary Packing: When you upload image.jpg and type a caption, Streamlit packs them into a Multipart Form-Data request.

1.3 The API Call: Streamlit uses the requests library to send this "package" over a local network (HTTP) to localhost:8000.

1.4 UI Blocking: The frontend enters a "Loading" state, waiting for a JSON response from the backend.

### Stage 2. Backend(FAST API)
The Backend acts as the traffic controller. 

2.1 Request Parsing: FastAPI receives the binary stream. It validates that the image is a valid file and the caption is a string.

2.2 Handoff: The backend calls the engine.py functions to begin the "Math" phase.

### Stage 3: Preprocessing Layer 
This is where raw data becomes Tensors.

3.1 CLIP Encoding: The processor.py uses the CLIP ViT-L/14 model

   3.1.1 Visual Branch: The image is resized to $224 \times 224$ pixels and converted into a $[1, 768]$ vector.
   3.1.2 Textual Branch: The caption is tokenized and converted into a $[1, 768]$ vector.
   
3.2 Output: You now have two 768-dimensional "summaries" of the data.

### Stage 4: The Inference Layer 

This is the most complex part of the flow where the REDDOT Architecture takes over.

4.1 Concatenation (The Fusion): The model takes the CLS Token $[1, 1, 768]$, the Image $[1, 1, 768]$, and the Text $[1, 1, 768]$. It glues them together to create a sequence of $[1, 3, 768]$.

4.2 The Transformer Pass: This $[1, 3, 768]$ block passes through 4 layers of self-attention.

4.3 Interaction: The CLS token "attends" to both features, looking for contradictions (e.g., An image of people in the market Vs The people fighting context )

4.4 The Extraction: After the final layer, the model throws away the Image and Text results and only keeps the first item: the updated CLS Token $[1, 768]$.

4.5 Classification: This single vector passes through the TokenClassifier (Linear layers + GELU). It outputs a single number.

### Stage 5: Metric calculation 

The logit is a probability (0.0 to 1.0) via a Sigmoid function.

5.1 Metrics: engine.get_metrics() calculates:Label: $Prob > 0.5$ ? "FAKE" : "TRUE"

5.2 Confidence: How far from 0.5 we are.

5.3 Entropy: The level of conflict between modalities.

5.4 JSON Return: FastAPI packs these numbers into a JSON object and sends it back to Streamlit.

5.5 Display: Streamlit receives the JSON and renders the green/red labels and the latency metric.
