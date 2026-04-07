from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import PhoneDetection
from face_capture.models import Person
import cv2
import pickle
import os
import base64
import json
from django.conf import settings
from ultralytics import YOLO
import numpy as np
from datetime import datetime
from django.utils import timezone

# Load YOLO and Face models lazily to save memory
yolo_model = None
face_recognizer = None
label_map = None

def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO('yolov8n.pt')
    return yolo_model

def get_face_models():
    global face_recognizer, label_map
    if face_recognizer is None:
        model_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', 'face_recognizer.yml')
        label_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', 'label_map.pkl')

        if os.path.exists(model_path) and os.path.exists(label_path):
            face_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=80.0
            )
            face_recognizer.read(model_path)
            with open(label_path, 'rb') as f:
                label_map = pickle.load(f)
    return face_recognizer, label_map

def detection_page(request):
    return render(request, 'detection/detection.html')

@csrf_exempt
def detect_frame(request):
    """AJAX endpoint for real-time detection from browser webcam"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')

            if not image_data:
                return JsonResponse({'success': False, 'error': 'No image data'})

            # Decode base64 image
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)

            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                return JsonResponse({'success': False, 'error': 'Invalid image'})

            # Get models (Lazy loading)
            yolo = get_yolo_model()
            recognizer, labels = get_face_models()

            # 1. Phone Detection (YOLO)
            # Lowered confidence threshold to 0.2 for better sensitivity as requested
            results = yolo(frame, verbose=False, conf=0.2)
            phone_detected = False
            detections = []
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    # We only care about cell phones (class 67) for the primary status,
                    # but we can return other detections if needed. 
                    if cls == 67:
                        phone_detected = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            'type': 'phone',
                            'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            'label': 'Phone'
                        })

            # 2. Face Recognition (OpenCV LBPH)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray) # Better lighting handling
            # Parameters tuned for better sensitivity (1.1 scale, 4 neighbors)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

            found_names = []
            for (x, y, w, h) in faces:
                name = "Unknown"
                if recognizer and labels:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200)) # Standard size
                    label, confidence = recognizer.predict(face_roi)
                    
                    # Slightly relaxed threshold (80 used to be 75) for better recognition in varied lighting
                    if confidence < 80:
                        name = labels.get(label, "Unknown")
                        if name != "Unknown":
                            found_names.append(name)

                detections.append({
                    'type': 'face',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'label': name
                })

                # If phone + known user, save record
                if phone_detected and name != "Unknown":
                    save_detection(name, frame)

            return JsonResponse({
                'success': True,
                'phone_detected': phone_detected,
                'name': found_names[0] if found_names else "Unknown",
                'detections': detections
            })

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request'})

def save_detection(name, frame):
    """Helper to save phone usage detection to database"""
    try:
        person = Person.objects.get(username=name)
        # throttle (10 seconds)
        last_save = PhoneDetection.objects.filter(person=person).order_by('-detection_time').first()
        if not last_save or (timezone.now() - last_save.detection_time).total_seconds() > 10:
            detection_dir = os.path.join(settings.MEDIA_ROOT, 'detections')
            os.makedirs(detection_dir, exist_ok=True)
            
            timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f'{name}_{timestamp}.jpg'
            cv2.imwrite(os.path.join(detection_dir, image_filename), frame)
            
            PhoneDetection.objects.create(
                person=person,
                image=f'detections/{image_filename}'
            )
    except:
        pass

def video_feed(request):
    """Deprecated: Original local-only video feed view"""
    return JsonResponse({'error': 'Local video feed not available on live server. Use Detection page instead.'})
