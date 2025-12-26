import streamlit as st
import os
from src.processing import train_mode, predict_health

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Piranaware V2",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PIRANAWARE COASTAL THEME (CSS) ---
st.markdown("""
    <style>
    /* Main Background - Light coastal atmosphere */
    .stApp {
        background-color: #F4F8FB; /* Very pale cool blue/grey */
        color: #1A202C; /* Dark grey text for readability */
    }
    
    /* Headers - Vibrant Sea Teal */
    h1, h2, h3 {
        color: #00838F !important; /* Deep cyan accent */
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    /* Bold text accent */
    strong { font-weight: 600; color: #006064; }
    
    /* Buttons - Coastal Glass Style */
    div.stButton > button {
        background-color: #E0F7FA; /* Lightest cyan */
        color: #00838F; /* Teal text */
        border: 1px solid #4DD0E1;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #26C6DA; /* Brighter cyan on hover */
        color: #FFFFFF; /* White text */
        border-color: #26C6DA;
        box-shadow: 0 4px 10px rgba(38, 198, 218, 0.2);
    }
    
    /* Styling the Tabs to match theme */
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #546E7A !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00838F !important;
        border-bottom: 3px solid #00838F !important;
    }

    /* Custom Result Boxes */
    .result-box-healthy {
        background-color: #E8F5E9; /* Light mint */
        border-left: 5px solid #2E7D32; /* Dark green */
        padding: 15px; border-radius: 5px;
    }
    .result-box-anomaly {
        background-color: #FFEBEE; /* Light red */
        border-left: 5px solid #C62828; /* Dark red */
        padding: 15px; border-radius: 5px;
    }
    .result-title { margin: 0 !important; font-size: 1.2rem; }
    </style>
""", unsafe_allow_html=True)

# --- SETUP ---
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

def save_and_process(audio_value):
    if audio_value is None: return None
    audio_value.seek(0)
    save_path = os.path.join(TEMP_DIR, "input.wav")
    with open(save_path, "wb") as f:
        f.write(audio_value.read())
    return save_path

# --- HELPER: Safe Audio Input ---
def get_audio_input(label, key):
    try:
        return st.audio_input(label, key=key)
    except AttributeError:
        return st.file_uploader(f"{label} (Upload .wav)", type=["wav"], key=key)

# --- UI HEADER ---
col_logo, col_title = st.columns([1, 7])
with col_logo:
    # NOTE: To use your actual logo, upload it to a public URL (ending in .png or .jpg)
    # and uncomment the following line, replacing the URL:
    # st.image("https://your-website.com/piranaware-logo.png", width=80)
    
    # Placeholder until direct image URL is available:
    st.markdown("<h1 style='font-size: 3.5rem; margin:0;'>🦈</h1>", unsafe_allow_html=True)

with col_title:
    st.title("PIRANAWARE")
    st.markdown("**Acoustic Digital Twin // V2.0**")
    st.caption("Marine Engine Anomaly Detection System")

st.markdown("---")

# --- TABS ---
tab_train, tab_test = st.tabs(["🛠️ CALIBRATION PROTOCOL", "🩺 DIAGNOSTIC SUITE"])

# === TAB 1: TRAINING ===
with tab_train:
    st.markdown("### 📡 Step 1: Establish Baseline")
    st.info("Instructions: Record ~30s of **HEALTHY** audio for each operational mode to train the AI baseline.")
    st.divider()
    
    col1, col2, col3 = st.columns(3)

    # --- IDLE ---
    with col1:
        st.markdown("#### 1. IDLE (Neutral)")
        audio_idle = get_audio_input("Source: Idle", key="rec_idle")
        if st.button("INITIATE TRAINING [IDLE]", key="btn_train_idle", use_container_width=True):
            if audio_idle:
                path = save_and_process(audio_idle)
                with st.spinner("Processing acoustic signature..."):
                    result = train_mode(path, "idle")
                st.success("✅ BASELINE UPDATED")
                st.caption(result)
            else:
                st.error("Awaiting Signal")

    # --- SLOW ---
    with col2:
        st.markdown("#### 2. SLOW (Cruising)")
        audio_slow = get_audio_input("Source: Slow", key="rec_slow")
        if st.button("INITIATE TRAINING [SLOW]", key="btn_train_slow", use_container_width=True):
            if audio_slow:
                path = save_and_process(audio_slow)
                with st.spinner("Processing acoustic signature..."):
                    result = train_mode(path, "slow")
                st.success("✅ BASELINE UPDATED")
                st.caption(result)
            else:
                st.error("Awaiting Signal")

    # --- FAST ---
    with col3:
        st.markdown("#### 3. FAST (Planing)")
        audio_fast = get_audio_input("Source: Fast", key="rec_fast")
        if st.button("INITIATE TRAINING [FAST]", key="btn_train_fast", use_container_width=True):
            if audio_fast:
                path = save_and_process(audio_fast)
                with st.spinner("Processing acoustic signature..."):
                    result = train_mode(path, "fast")
                st.success("✅ BASELINE UPDATED")
                st.caption(result)
            else:
                st.error("Awaiting Signal")

# === TAB 2: DIAGNOSTICS ===
with tab_test:
    st.markdown("### 🩺 Step 2: Health Check")
    st.divider()
    
    col_input, col_spacer, col_result = st.columns([2, 0.5, 3])
    
    with col_input:
        st.markdown("#### Signal Acquisition")
        mode = st.selectbox("Select Operational Mode Target", ["idle", "slow", "fast"])
        audio_test = get_audio_input("Record Live Engine Sound", key="rec_test")
        st.write("") # Spacer
        analyze_btn = st.button("RUN DIAGNOSTICS ⚡", key="btn_analyze", use_container_width=True)

    with col_result:
        st.markdown("#### Analysis Report")
        if analyze_btn:
            if audio_test:
                path = save_and_process(audio_test)
                with st.spinner(f"Comparing against {mode.upper()} Digital Twin..."):
                    report = predict_health(path, mode)
                
                # Dynamic Status Display using new CSS classes
                if "HEALTHY" in report:
                    st.markdown(f"""
                        <div class="result-box-healthy">
                            <h3 class="result-title" style="color: #2E7D32 !important;">🟢 SYSTEM NOMINAL</h3>
                            <p style="margin-top: 5px;">Acoustic signature aligns with calibrated baseline.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.code(report)
                elif "ANOMALY" in report:
                    st.markdown(f"""
                        <div class="result-box-anomaly">
                            <h3 class="result-title" style="color: #C62828 !important;">🔴 ANOMALY DETECTED</h3>
                            <p style="margin-top: 5px;">Significant deviation from baseline detected. Inspection recommended.</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.code(report)
                else:
                    st.warning(report)
            else:
                st.info("Ready for input signal...")
        else:
             st.markdown("""
                <div style="background-color: #E0F7FA; padding: 15px; border-radius: 5px; border: 1px solid #B2EBF2; color: #00838F;">
                    Waiting to analyze...
                </div>
            """, unsafe_allow_html=True)