import streamlit as st
import requests
import pandas as pd

# 1. Page Configuration
st.set_page_config(
    page_title="REDDOT | Multimodal Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# 2. Sidebar - System Status & Settings
st.sidebar.title("⚙️ System Control")
# Use 127.0.0.1 for better local compatibility
api_url = st.sidebar.text_input("Backend API URL", value="http://127.0.0.1:8000")

# --- FIX: Robust Health Check ---
backend_online = False
try:
    health_resp = requests.get(f"{api_url}/health", timeout=2)
    if health_resp.status_code == 200:
        health = health_resp.json()
        st.sidebar.success(f"Backend: ONLINE ({health.get('device', 'Unknown')})")
        backend_online = True
    else:
        st.sidebar.error("Backend: ERROR STATUS")
except Exception:
    st.sidebar.error("Backend: OFFLINE (Run your FastAPI server!)")

st.sidebar.divider()
st.sidebar.info(
    "AI Principles Applied:\n"
    "- Model Singletoning\n"
    "- Uncertainty Quantification\n"
    "- Latency Monitoring"
)

# 3. Main UI Header
st.title("🛡️ REDDOT: Multimodal Fake News Detection")
st.markdown("""
    Upload news samples (Image + Caption) to verify authenticity. 
    The system analyzes **contextual consistency** between visual and textual data.
""")

# 4. Input Section
with st.expander("📥 Upload News Samples", expanded=True):
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_files = st.file_uploader(
            "Select Images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
    with col2:
        caption = st.text_area("Enter News Caption", height=100)

# 5. Inference Logic
if st.button("🚀 Run Multimodal Analysis"):
    if not backend_online:
        st.error("Cannot run analysis while backend is OFFLINE.")
    elif not uploaded_files or not caption:
        st.warning("Please provide both image(s) and a caption.")
    else:
        st.divider()
        st.subheader("📊 Analysis Results")

        for file in uploaded_files:
            # --- FIX: Initialize 'res' inside the loop to prevent NameError ---
            res = None

            with st.container():
                c1, c2, c3 = st.columns([1, 2, 1])

                with c1:
                    # Updated for 2026 Streamlit standards
                    st.image(file, width=300)

                with c2:
                    files = {"image": (file.name, file.getvalue(), file.type)}
                    data = {"caption": caption}

                    try:
                        response = requests.post(f"{api_url}/predict", files=files, data=data, timeout=10)
                        if response.status_code == 200:
                            res = response.json()

                            if res.get("status") == "success":
                                label = res["prediction"]
                                color = "green" if label == "TRUE" else "red"
                                st.markdown(f"### Label: :{color}[{label}]")

                                st.write(f"**Confidence:** {res['confidence']:.2%}")
                                st.progress(res['confidence'])

                                st.write(f"**Shannon Entropy:** {res['entropy']}")
                                if res['entropy'] > 0.7:
                                    st.warning("⚠️ High Uncertainty: Possible conflict.")
                            else:
                                st.error(f"Backend Logic Error: {res.get('message')}")
                        else:
                            st.error(f"Server Error: Status Code {response.status_code}")

                    except Exception as e:
                        st.error(f"Connection Failed: {e}")

                with c3:
                    # --- FIX: Safe access to 'res' using None-check ---
                    if res and res.get("status") == "success":
                        st.metric("Latency", f"{res.get('latency_ms', 'N/A')} ms")
                        st.caption("Target: < 500ms")

                st.divider()

# 6. Footer
st.caption("REDDOT Inference Pipeline | Developed for AI Engineering Portfolio")