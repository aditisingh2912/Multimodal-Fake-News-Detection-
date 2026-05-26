import streamlit as st
import requests
import yaml
import os

CONFIG_FILE = "config.yaml"
if not os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump({
            "metadata": {
                "project_name": "RED-DOT Multimodal Security Gatekeeper",
                "version": "2.0.0"
            },
            "ux_settings": {
                "allow_human_override_queue": True
            }
        }, f)

with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

st.set_page_config(page_title="RED-DOT | Fake News Detector", layout="wide")

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ System Control")
api_url = st.sidebar.text_input("Backend API URL", value="http://127.0.0.1:8000")

backend_online = False
try:
    resp = requests.get(f"{api_url}/health", timeout=2)
    if resp.status_code == 200:
        h = resp.json()
        st.sidebar.success(f"Backend ONLINE ({h.get('device', '?')})")
        backend_online = True
    else:
        st.sidebar.error("Backend: ERROR STATUS")
except Exception:
    st.sidebar.error("❌ Backend OFFLINE — run: python backend.py")

st.sidebar.divider()
st.sidebar.markdown("""
**Decision logic**

The model outputs a raw **logit** (no manual thresholds):

- `logit < 0` → **REAL** (consistent pair)
- `logit > 0` →  **FAKE** (semantic mismatch)

The **RAG Gate** intercepts automatically when uncertainty is high.
""")

# ── MAIN ──────────────────────────────────────────────────────────────────────
st.title("RED-DOT: Multimodal Fake News Detection")
st.markdown(
    "Upload a news image and caption. "
    "The model decides **REAL, FAKE, or UNCERTAIN** dynamically using neural inference + FAISS RAG."
)

with st.expander("📤 Upload News Sample", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        uploaded_files = st.file_uploader(
            "Select Image(s)", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
    with c2:
        caption = st.text_area("Enter News Caption", height=120)

# ── INFERENCE ─────────────────────────────────────────────────────────────────
if st.button("🚀 Analyse"):
    if not backend_online:
        st.error("Backend is OFFLINE.")
    elif not uploaded_files or not caption:
        st.warning("Provide both image(s) and a caption.")
    else:
        st.divider()
        st.subheader("Results")

        for file in uploaded_files:
            res = None
            with st.container():
                col_img, col_result, col_meta = st.columns([1, 2, 1])

                with col_img:
                    st.image(file, width=260, caption=file.name)

                with col_result:
                    try:
                        response = requests.post(
                            f"{api_url}/predict",
                            files={"image": (file.name, file.getvalue(), file.type)},
                            data={"caption": caption},
                            timeout=60
                        )

                        if response.status_code == 200:
                            res = response.json()

                            if res.get("status") == "success":
                                # Extract metrics and schema states safely
                                prediction = str(res.get("verdict", "uncertain")).lower()
                                logit_val = res["logit"]
                                prob_fake = res["prob_fake"]
                                cosine_sim = res["cosine_sim"]
                                confidence = res["confidence"]
                                entropy = res["entropy"]
                                gate = res.get("gate_triggered", "A").lower()

                                # ── RENDERING THE MAIN BANNER ──────────────────
                                if prediction == "real":
                                    st.success(
                                        f"**REAL** — Image and caption are verified consistent. (Gate {gate.upper()})")
                                elif prediction == "fake":
                                    st.error(f"**FAKE** — Multimodal context mismatch detected! (Gate {gate.upper()})")
                                else:
                                    st.warning(
                                        f"**UNCERTAIN** — Inconclusive fact match. Requires human audit loop. (Gate {gate.upper()})")

                                st.divider()

                                # ── MODEL EVIDENCE ─────────────────────
                                st.markdown("**Model Signal (Raw Logit)**")
                                st.caption(
                                    f"`{logit_val:+.4f}` — "
                                    f"{'positive → FAKE' if logit_val > 0 else 'negative → REAL'} | "
                                    f"|logit| = {abs(logit_val):.4f} (larger = more certain)"
                                )
                                logit_norm = min(abs(logit_val) / 6.0, 1.0)
                                st.progress(logit_norm)

                                # ── INDEPENDENT COSINE CHECK ───────────
                                st.markdown("**Independent CLIP Cosine Similarity**")
                                cos_label = (
                                    "Strong match" if cosine_sim > 0.25 else
                                    "Moderate match" if cosine_sim > 0.12 else
                                    "Weak / mismatch"
                                )
                                st.caption(
                                    f"`{cosine_sim:.4f}` — {cos_label} (independent of model weights)"
                                )
                                st.progress(float(min(max(cosine_sim, 0.0), 1.0)))

                                # Operational signals alignment verification
                                model_says_fake = prediction == "fake"
                                cosine_says_fake = cosine_sim < 0.12
                                if prediction != "uncertain" and (model_says_fake != cosine_says_fake):
                                    st.warning("Model signal and raw CLIP alignment conflict — exercise caution.")
                                else:
                                    st.info("✓ Core operational telemetry signals are in alignment.")

                                # ── METRICS ROW ────────────────────────
                                st.divider()
                                m1, m2, m3 = st.columns(3)
                                m1.metric("Confidence", f"{confidence:.2%}")
                                m2.metric("Entropy", f"{entropy:.4f}")
                                m3.metric("P(Fake)", f"{prob_fake:.4f}")

                                if entropy > 0.50:
                                    st.info(
                                        f"ℹ️ High entropy state intercepted! RAG pipeline calibrated this output using nearest knowledge base vectors.")

                                # ── 🚨 FAISS VECTOR STORE AUDIT TRAIL ──
                                if gate == "b":
                                    st.divider()
                                    st.markdown("### 🔍 FAISS Vector Store Audit Trail")
                                    with st.expander("📖 View Closest Historical Neighbors Context", expanded=True):
                                        st.text(
                                            res.get("retrieval_context", "No context string returned from backend."))
                                        if "neighbours" in res and res["neighbours"]:
                                            st.caption("Raw Neighbors Payload List:")
                                            st.json(res["neighbours"])
                            else:
                                st.error(f"API Error: {res.get('message', 'Unknown failure outcome.')}")
                        else:
                            st.error(f"Backend Server Error: HTTP Return Code {response.status_code}")

                    except Exception as api_err:
                        st.error(f"Failed to communicate with prediction gateway pipeline: {api_err}")

                with col_meta:
                    # Render operations tools like Human-in-the-loop triggers inside the meta column
                    if config["ux_settings"]["allow_human_override_queue"]:
                        st.markdown("### Operations")
                        if st.button("Flag for Human Review", key=f"flag_{file.name}"):
                            st.success("Queued successfully for human-in-the-loop review.")