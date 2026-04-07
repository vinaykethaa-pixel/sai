# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies for OpenCV and dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libboost-all-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Install python build dependencies first
RUN pip install --no-cache-dir --upgrade pip wheel setuptools cmake

# Install dlib separately to isolate the build and limit memory usage
# Parallel level 1 is critical for 512MB RAM machines
RUN CMAKE_BUILD_PARALLEL_LEVEL=1 pip install --no-cache-dir dlib==19.24.2

# Copy requirements file (dlib and cmake removed from here in next step)
COPY requirements.txt /app/

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app/

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "face_detection_system.wsgi:application"]
