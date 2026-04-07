# Use the FULL Python image instead of slim (heavier but has complete build tools)
FROM python:3.11-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install additional system dependencies needed for dlib/opencv
# Full image already has most build-essential tools, but we add specific ones
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Use ONLY ONE CORE for compilation to avoid Render's 512MB RAM limit
# Verbose mode enabled for easier debugging if it fails
RUN CMAKE_BUILD_PARALLEL_LEVEL=1 pip install -v --no-cache-dir dlib==19.24.2

# Copy requirements file (dlib and cmake removed from here in previous step)
COPY requirements.txt /app/

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app/

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "face_detection_system.wsgi:application"]
