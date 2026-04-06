from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from detection.models import PhoneDetection
from face_capture.models import Person
from django.conf import settings
import os
import cv2
import numpy as np
import pickle

def admin_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if username == 'admin' and password == 'admin':
            request.session['admin_logged_in'] = True
            return redirect('admin_dashboard')
        else:
            return render(request, 'admin_dashboard/login.html', {'error': 'Invalid credentials'})

    return render(request, 'admin_dashboard/login.html')

def admin_logout(request):
    request.session.flush()
    return redirect('admin_login')

def admin_dashboard(request):
    if not request.session.get('admin_logged_in'):
        return redirect('admin_login')

    detections = PhoneDetection.objects.all().select_related('person')
    persons = Person.objects.all()
    return render(request, 'admin_dashboard/dashboard.html', {
        'detections': detections,
        'persons': persons
    })

@csrf_exempt
def delete_detection(request, detection_id):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})

    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'})

    try:
        detection = PhoneDetection.objects.get(id=detection_id)
        detection.delete()
        return JsonResponse({'success': True})
    except PhoneDetection.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Detection not found'})

@csrf_exempt
def delete_user(request, user_id):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})

    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'})

    try:
        person = Person.objects.get(id=user_id)
        username = person.username

        # Delete user's face images from filesystem
        import shutil
        person_dir = os.path.join(settings.MEDIA_ROOT, 'faces', username)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)

        # Delete user from database (cascade will delete related images and detections)
        person.delete()

        # Retrain the model to remove this user from recognition
        retrain_model_after_deletion()

        return JsonResponse({'success': True})
    except Person.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'User not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def retrain_model_after_deletion():
    """Retrain the face recognition model after user deletion"""
    try:
        import cv2
        import numpy as np
        import pickle

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2,
            neighbors=8,
            grid_x=8,
            grid_y=8,
            threshold=80.0
        )

        faces_data = []
        labels = []
        label_map = {}

        persons = Person.objects.all()

        # If no users left, delete the model files
        if not persons.exists():
            model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
            model_path = os.path.join(model_dir, 'face_recognizer.yml')
            label_path = os.path.join(model_dir, 'label_map.pkl')

            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            return

        # Retrain with remaining users
        from face_capture.models import FaceImage

        for idx, person in enumerate(persons):
            label_map[idx] = person.username
            face_images = FaceImage.objects.filter(person=person)

            for face_img in face_images:
                img_path = os.path.join(settings.MEDIA_ROOT, str(face_img.image))

                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)

                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_roi = cv2.resize(face_roi, (200, 200))
                        faces_data.append(face_roi)
                        labels.append(idx)

        if len(faces_data) > 0:
            recognizer.train(faces_data, np.array(labels))

            model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
            os.makedirs(model_dir, exist_ok=True)

            recognizer.write(os.path.join(model_dir, 'face_recognizer.yml'))

            with open(os.path.join(model_dir, 'label_map.pkl'), 'wb') as f:
                pickle.dump(label_map, f)

    except Exception as e:
        print(f"Error retraining model: {e}")
