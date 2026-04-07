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
                # Load ONNX model for CPU (memory efficient)
                yolo_model = {
                    'type': 'onnx',
                    'session': ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']),
                }
                print("SUCCESS: Loaded YOLOv8 ONNX model for CPU inference.")
            except Exception as e:
                print(f"WARNING: Failing back to PT model. ONNX error: {str(e)}")
                yolo_model = {'type': 'pt', 'model': YOLO(model_pt_path)}
        else:
            yolo_model = {'type': 'pt', 'model': YOLO(model_pt_path)}
            
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
            yolo = get_yolo_model()
            phone_detected = False
            detections = []

            if yolo['type'] == 'onnx':
                # Preprocess for ONNX
                img = cv2.resize(frame, (640, 640))
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))  # HWC to CHW
                img = np.expand_dims(img, axis=0)   # CHW to NCHW
                
                # Run inference
                input_name = yolo['session'].get_inputs()[0].name
                outputs = yolo['session'].run(None, {input_name: img})
                
                # Postprocess with Numpy Vectorization (Much faster)
                output = outputs[0][0].T  # [8400, 84]
                boxes = output[:, :4]     # [x, y, w, h]
                scores = output[:, 4:]    # [8400, 80]
                
                # Get max scores and indices
                max_scores = np.max(scores, axis=1)
                class_ids = np.argmax(scores, axis=1)
                
                # Filter by confidence and class (67=phone, 65=remote)
                mask = (max_scores > 0.3) & ((class_ids == 67) | (class_ids == 65))
                filtered_boxes = boxes[mask]
                filtered_scores = max_scores[mask]
                filtered_class_ids = class_ids[mask]
                
                if len(filtered_scores) > 0:
                    # Convert [x, y, w, h] to [x1, y1, x2, y2]
                    # Original coordinates are scaled to 640x640
                    x1 = (filtered_boxes[:, 0] - filtered_boxes[:, 2]/2) * frame.shape[1] / 640
                    y1 = (filtered_boxes[:, 1] - filtered_boxes[:, 3]/2) * frame.shape[0] / 640
                    x2 = (filtered_boxes[:, 0] + filtered_boxes[:, 2]/2) * frame.shape[1] / 640
                    y2 = (filtered_boxes[:, 1] + filtered_boxes[:, 3]/2) * frame.shape[0] / 640
                    
                    # Manual NMS (Non-Maximum Suppression)
                    indices = cv2.dnn.NMSBoxes(
                        [[int(x1[i]), int(y1[i]), int(x2[i]-x1[i]), int(y2[i]-y1[i])] for i in range(len(x1))],
                        filtered_scores.tolist(), 0.3, 0.45
                    )
                    
                    if len(indices) > 0:
                        phone_detected = True
                        for i in indices:
                            # If it's a list (older cv2) or numpy array
                            idx = i[0] if isinstance(i, (list, np.ndarray)) else i
                            
                            detections.append({
                                'type': 'phone',
                                'bbox': [int(x1[idx]), int(y1[idx]), int(x2[idx]-x1[idx]), int(y2[idx]-y1[idx])],
                                'label': f'Phone Detected! ({int(filtered_scores[idx]*100)}%)'
                            })
                            print(f"DEBUG: Found Phone confidence {filtered_scores[idx]:.2f}")
                else:
                    # Log high confidence objects for debugging if nothing was detected
                    # This helps see what the model is actually seeing
                    top_idx = np.argmax(max_scores)
                    if max_scores[top_idx] > 0.2:
                        print(f"DEBUG: No phone, but top object is {class_ids[top_idx]} with conf {max_scores[top_idx]:.2f}")
            else:
                # Fallback to Ultralytics PT
                results = yolo['model'](frame, verbose=False, conf=0.35)
                for result in results:
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        if cls == 67 or cls == 65:
                            phone_detected = True
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = float(box.conf[0])
                            detections.append({
                                'type': 'phone',
                                'bbox': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                                'label': f'Phone Detected! ({int(conf*100)}%)'
                            })
                            print(f"LOG: Phone (PT) detected with {conf:.2f} confidence")

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
