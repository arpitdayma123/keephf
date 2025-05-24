# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by OpenCV or other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app
COPY . .

# Download and cache models during the build process.
# The initialize_models function in rp_handler should handle the downloading
# and correct placement of models into /app/weights/ if they are not found.
# This command assumes rp_handler.py is in the root and initialize_models is accessible.
# If rp_handler.py is elsewhere, adjust the import path.
# It also assumes that initialize_models will create necessary directories like /app/weights and its subdirectories.
RUN python -c "import rp_handler; rp_handler.initialize_models()"

# Make port 8080 available to the world outside this container (if needed, though RunPod usually handles this)
# EXPOSE 8080

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run rp_handler.py when the container launches
CMD ["python3", "-u", "rp_handler.py"]
