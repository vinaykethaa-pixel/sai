# Use the Bullseye version of Python 3.11 for better dlib compatibility
FROM python:3.11-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install building tools and common headers
# Added zlib1g-dev and libjpeg-dev which are critical for dlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-all-dev \
    libjpeg-dev \
    zlib1g-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and install core build tools first
RUN pip install --no-cache-dir --upgrade pip wheel setuptools cmake

# Install dlib separately to limit memory usage (ONE CORE ONLY)
# If this fails, we will try an older stable version of dlib
RUN CMAKE_BUILD_PARALLEL_LEVEL=1 pip install -v --no-cache-dir dlib==19.22.1

# Copy requirements file (dlib and cmake removed from here in previous step)
COPY requirements.txt /app/

# Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app/

# Run the app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "face_detection_system.wsgi:application"]
