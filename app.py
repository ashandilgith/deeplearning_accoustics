import streamlit as st
import os
from src.processing import train_mode, predict_health

# --- CONFIGURATION ---
st.set_page_config(page_title="Piranaware V2", page_icon="🚤", layout="centered")

# Temporary folder to hold the recorded wav file for processing
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_and_process(audio_value):
    """
    Helper to save the raw audio bytes from Streamlit to a file
    that Librosa can read.
    """
    if audio_value is None:
        return None
    
    # Save to disk
    save_path = os.path.join(TEMP_DIR, "input.wav")
    with open(save_path, "wb") as f:
        f.write(audio_value.read())
    
    return save_path

# --- UI HEADER ---
st.title("🚤 Piranaware V2")
st.caption("Acoustic Anomaly Detection | Autoencoder Digital Twin")

# --- TABS ---
tab_train, tab_test = st.tabs(["🛠️ 1. Calibration (Train)", "🩺 2. Diagnostics (Test)"])

# === TAB 1: TRAINING ===
with tab_train:
    st.info("Instructions: Record ~30s of HEALTHY audio for each speed to teach the AI.")
    
    # We use columns to organize the 3 modes
    col1, col2, col3 = st.columns(3)

    # --- IDLE ---
    with col1:
        st.subheader("Idle")
        # st.audio_input is the new native recorder
        audio_idle = st.audio_input("Record Idle", key="rec_idle")
        
        if st.button("Train IDLE", key="btn_train_idle"):
            if audio_idle:
                path = save_and_process(audio_idle)
                with st.spinner("Training Idle Brain..."):
                    result = train_mode(path, "idle")
                st.success("Done!")
                st.write(result)
            else:
                st.error("Record audio first.")

    # --- SLOW ---
    with col2:
        st.subheader("Slow")
        audio_slow = st.audio_input("Record Slow", key="rec_slow")
        
        if st.button("Train SLOW", key="btn_train_slow"):
            if audio_slow:
                path = save_and_process(audio_slow)
                with st.spinner("Training Slow Brain..."):
                    result = train_mode(path, "slow")
                st.success("Done!")
                st.write(result)
            else:
                st.error("Record audio first.")

    # --- FAST ---
    with col3:
        st.subheader("Fast")
        audio_fast = st.audio_input("Record Fast", key="rec_fast")
        
        if st.button("Train FAST", key="btn_train_fast"):
            if audio_fast:
                path = save_and_process(audio_fast)
                with st.spinner("Training Fast Brain..."):
                    result = train_mode(path, "fast")
                st.success("Done!")
                st.write(result)
            else:
                st.error("Record audio first.")

# === TAB 2: DIAGNOSTICS ===
with tab_test:
    st.divider()
    st.header("Check Engine Health")
    
    # 1. Select Mode
    mode = st.selectbox("Which speed are you testing?", ["idle", "slow", "fast"])
    
    # 2. Record
    audio_test = st.audio_input("Record Engine Sound", key="rec_test")
    
    # 3. Analyze
    if st.button("Analyze Sound", key="btn_analyze"):
        if audio_test:
            path = save_and_process(audio_test)
            
            with st.spinner(f"Running Autoencoder against {mode.upper()} profile..."):
                report = predict_health(path, mode)
            
            # Visual Feedback
            if "HEALTHY" in report:
                st.balloons()
                st.success("Engine is Healthy")
                st.code(report)
            elif "ANOMALY" in report:
                st.error("⚠️ Anomaly Detected")
                st.warning("The engine sound deviates significantly from the calibration.")
                st.code(report)
            else:
                # Handle "Model not found" errors
                st.warning(report)
        else:
            st.error("Please record audio first.")