import os
import gradio as gr
from fastapi.middleware.wsgi import WSGIMiddleware

# Import spaces for Hugging Face ZeroGPU compatibility
try:
    import spaces
    @spaces.GPU
    def dummy_gpu_fn():
        return "ZeroGPU Initialized"
except ImportError:
    pass

# Import the Flask application from the dashboard folder
from dashboard.dashboard import app as flask_app

# 1. Define a minimal Gradio interface (required by Hugging Face to detect Gradio SDK)
with gr.Blocks() as demo:
    gr.Markdown("# Live Deal Drop Dashboard")
    gr.HTML("<script>window.location.href = '/';</script>")

# 2. Mount the Flask app directly onto Gradio's underlying FastAPI app
# We mount it at the root "/" so that the Flask dashboard is served directly at the main URL.
demo.app.mount("/", WSGIMiddleware(flask_app))

# 3. Launch Gradio (Gradio will automatically bind to the correct port 7860)
if __name__ == "__main__":
    demo.launch()
