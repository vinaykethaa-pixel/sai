# Use the full Python image (non-slim) for better build reliability
FROM python:3.11-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies for OpenCV and dlib
# Note: full 'bookworm' already has build-essential and some others,
# but we explicitly add them for completeness.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    pkg-config \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libjpeg-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and set up build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel cmake

# Install dlib with verbose logging and memory limit
# Using -v lets us see the actual cmake error if it fails
RUN CMAKE_BUILD_PARALLEL_LEVEL=1 pip install -v --no-cache-dir dlib==19.24.2

# Copy the rest of the project files
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . /app/

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "face_detection_system.wsgi:application"]
