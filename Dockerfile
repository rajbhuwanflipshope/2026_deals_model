FROM python:3.10-slim

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
WORKDIR /app

# Install system dependencies if any are needed (e.g., git, libgomp1 for xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY --chown=user:user . /app

# Switch to the non-root user
USER user

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the Flask app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "dashboard.dashboard:app"]
