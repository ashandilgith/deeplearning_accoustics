import gradio as gr
from src.processing import train_mode, predict_health

# --- UI Functions ---
def run_training(audio, mode):
    if audio is None:
        return "Please record audio first."
    return train_mode(audio, mode)

def run_analysis(audio, mode):
    if audio is None:
        return "Please record audio first."
    return predict_health(audio, mode)

# --- Layout ---
with gr.Blocks(title="Piranaware: Engine Health") as app:
    gr.Markdown("# 🚤 Piranaware: Autoencoder Anomaly Detection")
    gr.Markdown("Step 1: Train the system on HEALTHY sounds for each speed.\nStep 2: Test new sounds to detect faults.")
    
    with gr.Tab("Step 1: Training (Calibration)"):
        with gr.Row():
            # IDLE
            with gr.Column():
                gr.Markdown("### 1. Idle (Neutral)")
                train_idle_audio = gr.Audio(sources=["microphone", "upload"], type="filepath")
                btn_idle = gr.Button("Train IDLE Model")
                out_idle = gr.Textbox(label="Status")
                btn_idle.click(lambda x: run_training(x, "idle"), inputs=train_idle_audio, outputs=out_idle)
            
            # SLOW
            with gr.Column():
                gr.Markdown("### 2. Slow (15-20 km/h)")
                train_slow_audio = gr.Audio(sources=["microphone", "upload"], type="filepath")
                btn_slow = gr.Button("Train SLOW Model")
                out_slow = gr.Textbox(label="Status")
                btn_slow.click(lambda x: run_training(x, "slow"), inputs=train_slow_audio, outputs=out_slow)

            # FAST
            with gr.Column():
                gr.Markdown("### 3. Fast (40-50 km/h)")
                train_fast_audio = gr.Audio(sources=["microphone", "upload"], type="filepath")
                btn_fast = gr.Button("Train FAST Model")
                out_fast = gr.Textbox(label="Status")
                btn_fast.click(lambda x: run_training(x, "fast"), inputs=train_fast_audio, outputs=out_fast)

    with gr.Tab("Step 2: Diagnostics"):
        gr.Markdown("### Test Engine Health")
        # Dropdown to select which "Brain" to use
        mode_selector = gr.Radio(["idle", "slow", "fast"], label="Current Speed", value="idle")
        test_audio = gr.Audio(sources=["microphone", "upload"], type="filepath")
        test_btn = gr.Button("Analyze Sound")
        test_out = gr.Textbox(label="Diagnostic Report", lines=5)
        
        test_btn.click(run_analysis, inputs=[test_audio, mode_selector], outputs=test_out)

if __name__ == "__main__":
    app.launch()