# Use Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# # Install system-level dependencies required for PyMuPDF and scikit-learn
# RUN apt-get update && apt-get install -y \
#     libglib2.0-0 \
#     libgl1-mesa-glx \
#     build-essential \
#     && rm -rf /var/lib/apt/lists/*

# Copy only requirement file and install dependencies
COPY requirements.txt .
COPY main.py ./
COPY finalmodel.pkl ./
RUN pip install --no-cache-dir -r requirements.txt


# Define volumes for input and output directories
VOLUME ["/app/input", "/app/output"]

# Set environment variables with optional defaults
ENV PERSONA="Traveller"
ENV JOB_TO_BE_DONE="visit places and eat french cuisine"
ENV MIN_CONFIDENCE=0.2

# Run the application
CMD ["python", "main.py"]
