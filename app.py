import streamlit as st
import os
from src.processing import train_mode, predict_health

# --- CONFIGURATION ---
st.set_page_config(page_title="Piranaware V2", page_icon="🚤", layout="centered")

TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_and_process(audio_value):
    # DEBUG: Print what we received
    if audio_value is None:
        st.warning("⚠️ Debug: Audio variable is NONE. Widget is empty.")
        return None
    
    # SAFETY: Rewind the file pointer to the beginning before reading
    audio_value.seek(0)
    
    st.info(f"✅ Debug: Received audio bytes: {audio_value.size} bytes")
    
    # Save to disk
    save_path = os.path.join(TEMP_DIR, "input.wav")
    with open(save_path, "wb") as f:
        f.write(audio_value.read())
    
    return save_path

# --- HELPER: Safe Audio Input ---
def get_audio_input(label, key):
    """
    Tries to use the new native recorder. 
    Falls back to file uploader if Streamlit is too old.
    """
    try:
        # Try the new feature
        return st.audio_input(label, key=key)
    except AttributeError:
        # Fallback for older versions
        return st.file_uploader(f"{label} (Upload .wav)", type=["wav"], key=key)

# --- UI HEADER ---
st.title("🚤 Piranaware V2")
st.caption("Acoustic Anomaly Detection | Autoencoder Digital Twin")

# --- TABS ---
tab_train, tab_test = st.tabs(["🛠️ 1. Calibration (Train)", "🩺 2. Diagnostics (Test)"])

# === TAB 1: TRAINING ===
with tab_train:
    st.info("Record ~30s of HEALTHY audio for each speed.")
    col1, col2, col3 = st.columns(3)

    # --- IDLE ---
    with col1:
        st.subheader("Idle")
        # USE THE HELPER FUNCTION HERE
        audio_idle = get_audio_input("Record Idle", key="rec_idle")
        
        if st.button("Train IDLE", key="btn_train_idle"):
            if audio_idle:
                path = save_and_process(audio_idle)
                with st.spinner("Training Idle Brain..."):
                    result = train_mode(path, "idle")
                st.success("Done!")
                st.write(result)
            else:
                st.error("No audio found.")

    # --- SLOW ---
    with col2:
        st.subheader("Slow")
        audio_slow = get_audio_input("Record Slow", key="rec_slow")
        
        if st.button("Train SLOW", key="btn_train_slow"):
            if audio_slow:
                path = save_and_process(audio_slow)
                with st.spinner("Training Slow Brain..."):
                    result = train_mode(path, "slow")
                st.success("Done!")
                st.write(result)
            else:
                st.error("No audio found.")

    # --- FAST ---
    with col3:
        st.subheader("Fast")
        audio_fast = get_audio_input("Record Fast", key="rec_fast")
        
        if st.button("Train FAST", key="btn_train_fast"):
            if audio_fast:
                path = save_and_process(audio_fast)
                with st.spinner("Training Fast Brain..."):
                    result = train_mode(path, "fast")
                st.success("Done!")
                st.write(result)
            else:
                st.error("No audio found.")

# === TAB 2: DIAGNOSTICS ===
with tab_test:
    st.divider()
    st.header("Check Engine Health")
    mode = st.selectbox("Which speed are you testing?", ["idle", "slow", "fast"])
    
    # USE THE HELPER FUNCTION HERE
    audio_test = get_audio_input("Record Engine Sound", key="rec_test")
    
    if st.button("Analyze Sound", key="btn_analyze"):
        if audio_test:
            path = save_and_process(audio_test)
            with st.spinner(f"Analyzing against {mode.upper()} profile..."):
                report = predict_health(path, mode)
            
            if "HEALTHY" in report:
                st.balloons()
                st.success("Engine is Healthy")
                st.code(report)
            elif "ANOMALY" in report:
                st.error("⚠️ Anomaly Detected")
                st.code(report)
            else:
                st.warning(report)
        else:
            st.error("No audio found.")