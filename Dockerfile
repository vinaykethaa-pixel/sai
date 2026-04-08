# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy requirements file 
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app/

# Create a non-root user and set permissions
RUN useradd -m -u 1000 user && \
    chown -R user:user /app

# Switch to the non-root user
USER user

# Create a start script to run migrations, collect static files, and then start Gunicorn
RUN echo "#!/bin/sh\npython manage.py migrate --noinput\npython manage.py collectstatic --noinput --clear\ngunicorn --bind 0.0.0.0:7860 --workers 1 --timeout 300 face_detection_system.wsgi:application" > /app/start.sh
RUN chmod +x /app/start.sh

# Run the app using the start script
CMD ["/app/start.sh"]
