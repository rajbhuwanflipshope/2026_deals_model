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

# Define custom CSS to make the iframe occupy 100% of the viewport and hide Gradio's default styling/footer
custom_css = """
footer {visibility: hidden !important; display: none !important;}
.gradio-container {max-width: 100% !important; padding: 0 !important; margin: 0 !important; height: 100vh !important;}
iframe {width: 100%; height: 100vh; border: none; margin: 0; padding: 0;}
"""

with gr.Blocks(css=custom_css, title="2026 Deal Model Dashboard") as demo:
    # Embed the Flask app via a full-screen iframe
    gr.HTML("<iframe src='/app'></iframe>")

# Mount the Flask app onto Gradio's underlying FastAPI app under the '/app' subpath
demo.app.mount("/app", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    demo.launch()
