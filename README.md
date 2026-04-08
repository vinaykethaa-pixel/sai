---
title: Restricted Area Monitor
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 7860
---

# Restricted Area Monitor & Face Attendance System

This project is a Django-based surveillance system that uses YOLOv8 for mobile phone detection and OpenCV LBPH for face recognition.

## Deployment on Hugging Face Spaces

This application is configured to run on Hugging Face Spaces using Docker.

- **Port**: 7860
- **SDK**: Docker
- **RAM**: 16GB (CPU Basic)

## Local Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run migrations:
   ```bash
   python manage.py migrate
   ```
3. Start server:
   ```bash
   python manage.py runserver
   ```
