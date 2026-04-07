from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import PhoneDetection
from face_capture.models import Person
import cv2
import pickle
import os
from django.conf import settings
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import base64
import json

# Load YOLO model
yolo_model = None
face_recognizer = None
label_map = None

def load_models():
    global yolo_model, face_recognizer, label_map

    if yolo_model is None:
        yolo_model = YOLO('yolov8n.pt')

    if face_recognizer is None:
        model_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', 'face_recognizer.yml')
        label_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', 'label_map.pkl')

        if os.path.exists(model_path) and os.path.exists(label_path):
            face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            face_recognizer.read(model_path)

            with open(label_path, 'rb') as f:
                label_map = pickle.load(f)

def detection_page(request):
    return render(request, 'detection/detection.html')

def gen_frames():
    load_models()

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    last_save_time = {}

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Detect mobile phones using YOLO
        results = yolo_model(frame, verbose=False)
        phone_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                # Class 67 is cell phone in COCO dataset
                if class_id == 67:
                    phone_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, 'Phone Detected', (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Face recognition
        if face_recognizer and label_map and phone_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]

                try:
                    label, confidence = face_recognizer.predict(face_roi)

                    if confidence < 100:  # Lower is better
                        name = label_map.get(label, "Unknown")

                        # Save detection (limit to once every 5 seconds per person)
                        current_time = datetime.now()
                        if name not in last_save_time or (current_time - last_save_time[name]).seconds > 5:
                            try:
                                person = Person.objects.get(username=name)
                                detection_dir = os.path.join(settings.MEDIA_ROOT, 'detections')
                                os.makedirs(detection_dir, exist_ok=True)

                                timestamp = current_time.strftime('%Y%m%d_%H%M%S')
                                image_path = os.path.join(detection_dir, f'{name}_{timestamp}.jpg')
                                cv2.imwrite(image_path, frame)

                                relative_path = f'detections/{name}_{timestamp}.jpg'
                                PhoneDetection.objects.create(person=person, image=relative_path)

                                last_save_time[name] = current_time
                            except:
                                pass
                    else:
                        name = "Unknown"
                except:
                    name = "Unknown"

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()


@csrf_exempt
def detect_frame(request):
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

            load_models()

            # Phone detection with YOLO
            yolo_results = yolo_model(frame, verbose=False)
            phone_detected = False
            for result in yolo_results:
                for box in result.boxes:
                    if int(box.cls[0]) == 67: # Phone class
                        phone_detected = True

            # Face recognition
            name = "Unknown"
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                if face_recognizer and label_map:
                    face_roi = gray[y:y+h, x:x+w]
                    label, confidence = face_recognizer.predict(face_roi)
                    if confidence < 100:
                        name = label_map.get(label, "Unknown")

                # If phone detected and name found, save detection record
                if phone_detected and name != "Unknown":
                    # Save record to database in background or simple logic
                    # (Simplified for now, similar to your original logic)
                    try:
                        person = Person.objects.get(username=name)
                        # Avoid saving too often (limit to once every 10 seconds)
                        # Using simpler last_save logic here if possible
                        # For simplicity in this example:
                        from django.utils import timezone
                        last_save = PhoneDetection.objects.filter(person=person).order_by('-detection_time').first()
                        if not last_save or (timezone.now() - last_save.detection_time).seconds > 10:
                            # Save image to media folder
                            detection_dir = os.path.join(settings.MEDIA_ROOT, 'detections')
                            os.makedirs(detection_dir, exist_ok=True)
                            image_filename = f'{name}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.jpg'
                            image_path = os.path.join(detection_dir, image_filename)
                            cv2.imwrite(image_path, frame)
                            PhoneDetection.objects.create(person=person, image=f'detections/{image_filename}')
                    except Exception as e:
                        print("Save error:", e)

            return JsonResponse({
                'success': True,
                'phone_detected': phone_detected,
                'name': name
            })

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request'})

def gen_frames():
    # Leave original gen_frames as is for backward compatibility or local use
    # but it won't work on Render
    ...
