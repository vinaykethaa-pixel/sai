from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import PhoneDetection, RestrictedArea
from face_capture.models import Person
import cv2
import pickle
import os
import base64
import json
from django.conf import settings
from ultralytics import YOLO
import numpy as np
from django.utils import timezone
import logging

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)

# Load YOLO and Face models lazily to save memory
yolo_model = None
face_recognizer = None
label_map = None


def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        onnx_path = os.path.join(settings.BASE_DIR, 'yolov8n.onnx')
        model_pt_path = os.path.join(settings.BASE_DIR, 'yolov8n.pt')
        
        if ort and os.path.exists(onnx_path):
            try:
                # Load ONNX model for CPU via Ultralytics (handles memory efficiency and postprocessing)
                yolo_model = YOLO(onnx_path, task='detect')
                print("SUCCESS: Loaded YOLOv8 ONNX model via Ultralytics.")
            except Exception as e:
                print(f"WARNING: Failing back to PT model. ONNX error: {str(e)}")
                yolo_model = YOLO(model_pt_path)
        else:
            yolo_model = YOLO(model_pt_path)
            
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
    areas = RestrictedArea.objects.filter(is_active=True)
    return render(request, 'detection/detection.html', {'areas': areas})


def get_active_areas(request):
    """API: Return list of active restricted areas"""
    areas = list(RestrictedArea.objects.filter(is_active=True).values('id', 'name', 'description'))
    return JsonResponse({'areas': areas})


@csrf_exempt
def detect_frame(request):
    """AJAX endpoint for real-time detection from browser webcam"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            area_id = data.get('area_id')  # Restricted area selected by user

            if not image_data:
                return JsonResponse({'success': False, 'error': 'No image data'})

            # Validate restricted area
            restricted_area = None
            if area_id:
                try:
                    restricted_area = RestrictedArea.objects.get(id=area_id, is_active=True)
                except RestrictedArea.DoesNotExist:
                    pass

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

            # 1. Phone Detection (YOLO / ONNX)
            phone_detected = False
            detections = []

            # Ultralytics transparently handles ONNX sessions, NMS, resizing, and coordinates
            results = yolo(frame, verbose=False, conf=0.20)
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if cls == 67 or cls == 65:  # 67=cell phone, 65=remote
                        phone_detected = True
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        detections.append({
                            'type': 'phone',
                            'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                            'label': f'Phone Detected! ({int(conf*100)}%)'
                        })
                        print(f"LOG: Phone detected with {conf:.2f} confidence")

            # 2. Face Recognition (OpenCV LBPH)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
            )

            found_names = []
            alert_triggered = False

            for (x, y, w, h) in faces:
                name = "Unknown"
                if recognizer and labels:
                    face_roi = gray[y:y+h, x:x+w]
                    face_roi = cv2.resize(face_roi, (200, 200))
                    label, confidence = recognizer.predict(face_roi)
                    if confidence < 80:
                        name = labels.get(label, "Unknown")
                        if name != "Unknown":
                            found_names.append(name)

                detections.append({
                    'type': 'face',
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'label': name
                })

                # Save only when: phone detected + in restricted area
                if phone_detected and restricted_area and name != "Unknown":
                    saved = save_detection_with_alert(name, frame, restricted_area)
                    if saved:
                        alert_triggered = True

            return JsonResponse({
                'success': True,
                'phone_detected': phone_detected,
                'name': found_names[0] if found_names else "Unknown",
                'detections': detections,
                'area_active': restricted_area is not None,
                'alert_triggered': alert_triggered,
                'area_name': restricted_area.name if restricted_area else None,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request'})


def save_detection_with_alert(name, frame, area):
    """Save phone detection with alert flag — only for restricted areas"""
    try:
        person = Person.objects.get(username=name)

        # Throttle: 10 seconds cooldown per person per area
        last_save = PhoneDetection.objects.filter(
            person=person, area=area
        ).order_by('-detection_time').first()

        if last_save and (timezone.now() - last_save.detection_time).total_seconds() < 10:
            return False  # Too soon, skip

        detection_dir = os.path.join(settings.MEDIA_ROOT, 'detections')
        os.makedirs(detection_dir, exist_ok=True)

        timestamp = timezone.now().strftime('%Y%m%d_%H%M%S')
        image_filename = f'{name}_{timestamp}.jpg'
        cv2.imwrite(os.path.join(detection_dir, image_filename), frame)

        PhoneDetection.objects.create(
            person=person,
            area=area,
            image=f'detections/{image_filename}',
            is_read=False  # triggers admin notification
        )
        return True

    except Exception:
        return False


def video_feed(request):
    """Deprecated: Not available on live server"""
    return JsonResponse({
        'error': 'Local video feed not available on live server. Use Detection page instead.'
    })
