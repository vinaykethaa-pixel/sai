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
#from twilio.rest import Client

# Load YOLO model
yolo_model = None
face_recognizer = None
label_map = None

'''def send_sms_alert(person_name):
    """Send SMS alert to admin when phone is detected"""
    try:
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

        message_body = f"ALERT: Phone detected!\nUser: {person_name}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        message = client.messages.create(
            body=message_body,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=settings.ADMIN_PHONE_NUMBER
        )
        print(f"SMS sent successfully: {message.sid}")
        return True
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False'''

def load_models():
    global yolo_model, face_recognizer, label_map

    if yolo_model is None:
        yolo_model = YOLO('yolov8n.pt')

    if face_recognizer is None:
        model_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', 'face_recognizer.yml')
        label_path = os.path.join(settings.MEDIA_ROOT, 'trained_models', 'label_map.pkl')

        if os.path.exists(model_path) and os.path.exists(label_path):
            # Use same parameters as training
            face_recognizer = cv2.face.LBPHFaceRecognizer_create(
                radius=2,
                neighbors=8,
                grid_x=8,
                grid_y=8,
                threshold=80.0
            )
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

        # Face recognition and detection saving
        if phone_detected:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Apply histogram equalization for better lighting normalization
            gray = cv2.equalizeHist(gray)

            # Improved face detection parameters
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                # Resize to standard size matching training
                face_roi_resized = cv2.resize(face_roi, (200, 200))
                name = "Unknown"
                confidence_score = 0

                # Try face recognition if models are loaded
                if face_recognizer and label_map:
                    try:
                        label, confidence = face_recognizer.predict(face_roi_resized)
                        confidence_score = confidence
                        # Stricter threshold for better accuracy (lower is better match)
                        if confidence < 70:  # More strict threshold
                            name = label_map.get(label, "Unknown")
                        else:
                            name = "Unknown"
                    except:
                        name = "Unknown"

                # Save detection for both known and unknown users
                current_time = datetime.now()
                save_key = name  # Use only name, not position

                if save_key not in last_save_time or (current_time - last_save_time[save_key]).total_seconds() > 10:
                    try:
                        detection_dir = os.path.join(settings.MEDIA_ROOT, 'detections')
                        os.makedirs(detection_dir, exist_ok=True)

                        timestamp = current_time.strftime('%Y%m%d_%H%M%S')
                        image_path = os.path.join(detection_dir, f'{name}_{timestamp}.jpg')
                        cv2.imwrite(image_path, frame)

                        relative_path = f'detections/{name}_{timestamp}.jpg'

                        # Try to link to person if known, otherwise save with person=None
                        person = None
                        if name != "Unknown":
                            try:
                                person = Person.objects.get(username=name)
                            except:
                                pass

                        PhoneDetection.objects.create(person=person, image=relative_path)
                        last_save_time[save_key] = current_time

                        # Send SMS alert to admin
                        #send_sms_alert(name)
                    except Exception as e:
                        print(f"Error saving detection: {e}")

                # Display confidence score for debugging
                display_text = f"{name} ({confidence_score:.1f})" if confidence_score > 0 else name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, display_text, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                content_type='multipart/x-mixed-replace; boundary=frame')
