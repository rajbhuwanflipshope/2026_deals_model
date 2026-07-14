import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.wsgi import WSGIMiddleware
import gradio as gr

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

# 1. Initialize FastAPI
app = FastAPI()

# 2. Define a minimal Gradio Interface (required to keep the Hugging Face Gradio SDK happy)
def health_check(name):
    return f"Status: Active, Hello {name}!"

demo = gr.Interface(
    fn=health_check,
    inputs="text",
    outputs="text",
    title="Dashboard Health Check Interface"
)

# 3. Mount Gradio onto FastAPI under the /gradio path
app = gr.mount_gradio_app(app, demo, path="/gradio")

# 4. Mount the Flask app onto FastAPI at the root path "/"
app.mount("/", WSGIMiddleware(flask_app))

if __name__ == "__main__":
    # Hugging Face Spaces exposes port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
