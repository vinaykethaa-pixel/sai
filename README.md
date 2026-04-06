# Face Recognition & Mobile Phone Detection System

A Django-based AI-powered surveillance system that captures faces, trains recognition models, and detects mobile phone usage in real-time using YOLOv8.

## Features

- **Live Face Capture**: Capture and store face images organized by username
- **Model Training**: Train face recognition models on captured images
- **Real-time Detection**: Detect faces and mobile phone usage simultaneously
- **Admin Dashboard**: View all detected incidents with timestamps and images
- **Modern UI**: Clean, responsive interface with Bootstrap 5

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Note: If `face-recognition` or `dlib` installation fails on Windows, you may need to:
- Install Visual Studio Build Tools
- Or use pre-built wheels from: https://github.com/z-mahmud22/Dlib_Windows_Python3.x

2. Run database migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

3. Start the development server:
```bash
python manage.py runserver
```

4. Open your browser and navigate to: http://127.0.0.1:8000/

## Usage

### 1. Capture Faces
- Go to "Capture Faces" page
- Enter a username
- Start camera and capture multiple images (recommended: 10-20 images)
- Images are stored in `media/faces/username/`

### 2. Train Model
- Go to "Training" page
- Click "Start Training" button
- Wait for training to complete
- Model is saved in `media/trained_models/`

### 3. Live Detection
- Go to "Detection" page
- The system will:
  - Recognize trained faces
  - Detect mobile phone usage using YOLOv8
  - Automatically save images when a person is detected using a phone

### 4. Admin Dashboard
- Go to "Admin" page
- Login with credentials: `admin` / `admin`
- View all detected incidents with:
  - Person name
  - Timestamp
  - Captured image
  - Delete option

## Project Structure

```
face_detection_system/
├── face_capture/          # Face capture and training
├── detection/             # Real-time detection
├── admin_dashboard/       # Admin panel
├── media/
│   ├── faces/            # Captured face images
│   ├── detections/       # Detection screenshots
│   └── trained_models/   # Trained models
├── static/               # Static files
└── templates/            # HTML templates
```

## Technologies Used

- **Backend**: Django 5.0
- **Face Recognition**: face_recognition library
- **Object Detection**: YOLOv8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Frontend**: Bootstrap 5, JavaScript
- **Database**: SQLite

## Requirements

- Python 3.8+
- Webcam
- Windows/Linux/Mac

## Admin Credentials

- Username: `admin`
- Password: `admin`

## Notes

- Make sure your webcam is connected and accessible
- Train the model before starting detection
- Capture at least 10 images per person for better accuracy
- YOLOv8 model will be downloaded automatically on first run

## Troubleshooting

**Camera not working:**
- Check browser permissions for camera access
- Ensure no other application is using the camera

**Face not detected:**
- Ensure good lighting
- Face the camera directly
- Remove glasses/masks if possible

**Installation issues:**
- For dlib installation issues on Windows, use pre-built wheels
- Ensure you have Visual Studio Build Tools installed

## License

This project is for educational purposes.
