# Quick Start Guide

## Run the Application

1. Start the Django development server:
```bash
cd "C:\Users\Welcome\Desktop\my project ai"
python manage.py runserver
```

2. Open your browser and go to: http://127.0.0.1:8000/

## Usage Steps

### Step 1: Capture Faces
- Click "Capture Faces"
- Enter a username (e.g., "John")
- Click "Start Camera"
- Click "Capture Image" multiple times (10-20 images recommended)
- Try different angles and expressions

### Step 2: Train the Model
- Click "Training"
- Click "Start Training" button
- Wait for training to complete

### Step 3: Start Detection
- Click "Detection"
- The system will show live video feed
- It will detect faces and mobile phones
- When someone uses a phone, it saves the image automatically

### Step 4: View Admin Dashboard
- Click "Admin"
- Login: username = `admin`, password = `admin`
- View all detected incidents with timestamps and images

## System Features

✅ Live face capture with webcam
✅ Face detection using OpenCV
✅ Face recognition using LBPH algorithm
✅ Mobile phone detection using YOLOv8
✅ Automatic image saving when phone usage detected
✅ Admin dashboard with authentication
✅ Modern responsive UI

## Notes

- Make sure your webcam is connected
- Allow browser camera permissions
- Train the model before detection
- YOLOv8 model downloads automatically on first run
