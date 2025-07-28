FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Copy required files into the container
COPY requirements.txt ./
COPY main.py ./
COPY finalmodel.pkl ./

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Create input/output folders inside container
RUN mkdir -p /app/input /app/output

# Entry point: run the script
CMD ["python", "main.py"]