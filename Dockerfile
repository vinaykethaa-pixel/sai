# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 10000

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements file (dlib and face-recognition removed)
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app/

# Run collectstatic during build instead of startup to save time
# Mock a SECRET_KEY for collectstatic to work if needed
RUN python manage.py collectstatic --noinput

# Create a start script to run migrations and then start Gunicorn
# Use $PORT variable for Render compatibility
RUN echo "#!/bin/sh\npython manage.py migrate --noinput\ngunicorn --bind 0.0.0.0:\$PORT --workers 1 --timeout 300 face_detection_system.wsgi:application" > /app/start.sh
RUN chmod +x /app/start.sh

# Run the app using the start script
# Render will provide the $PORT environment variable
CMD ["/app/start.sh"]
