from django.shortcuts import render, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Person, FaceImage
import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from django.conf import settings
import base64
import json

def index(request):
    persons = Person.objects.all()
    return render(request, 'face_capture/index.html', {'persons': persons})

def capture_page(request):
    return render(request, 'face_capture/capture.html')

@csrf_exempt
def save_face(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            image_data = data.get('image')

            if not username or not image_data:
                return JsonResponse({'success': False, 'error': 'Missing data'})

            person, created = Person.objects.get_or_create(username=username)

            # Decode base64 image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)

            # Save to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Detect face using OpenCV
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return JsonResponse({'success': False, 'error': 'No face detected'})

            # Save image
            person_dir = os.path.join(settings.MEDIA_ROOT, 'faces', username)
            os.makedirs(person_dir, exist_ok=True)

            image_count = FaceImage.objects.filter(person=person).count()
            image_path = os.path.join(person_dir, f'face_{image_count + 1}.jpg')
            cv2.imwrite(image_path, img)

            # Save to database
            relative_path = f'faces/{username}/face_{image_count + 1}.jpg'
            FaceImage.objects.create(person=person, image=relative_path)

            return JsonResponse({'success': True, 'count': image_count + 1})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request'})

def training_page(request):
    persons = Person.objects.all()
    return render(request, 'face_capture/training.html', {'persons': persons})

@csrf_exempt
def train_model(request):
    if request.method == 'POST':
        try:
            # Using OpenCV LBPH Face Recognizer
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            recognizer = cv2.face.LBPHFaceRecognizer_create()

            faces_data = []
            labels = []
            label_map = {}

            persons = Person.objects.all()

            for idx, person in enumerate(persons):
                label_map[idx] = person.username
                face_images = FaceImage.objects.filter(person=person)

                for face_img in face_images:
                    img_path = os.path.join(settings.MEDIA_ROOT, str(face_img.image))

                    if os.path.exists(img_path):
                        image = cv2.imread(img_path)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                        for (x, y, w, h) in faces:
                            face_roi = gray[y:y+h, x:x+w]
                            faces_data.append(face_roi)
                            labels.append(idx)

                person.is_trained = True
                person.save()

            if len(faces_data) > 0:
                # Train the recognizer
                recognizer.train(faces_data, np.array(labels))

                # Save trained model
                model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
                os.makedirs(model_dir, exist_ok=True)

                recognizer.write(os.path.join(model_dir, 'face_recognizer.yml'))

                with open(os.path.join(model_dir, 'label_map.pkl'), 'wb') as f:
                    pickle.dump(label_map, f)

                return JsonResponse({
                    'success': True,
                    'message': f'Training completed! {len(faces_data)} faces trained.'
                })
            else:
                return JsonResponse({'success': False, 'error': 'No faces found for training'})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Invalid request'})
