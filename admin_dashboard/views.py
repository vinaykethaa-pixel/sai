from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from detection.models import PhoneDetection, RestrictedArea
from face_capture.models import Person
from django.conf import settings
import os
import cv2
import numpy as np
import pickle
import json


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

    detections = PhoneDetection.objects.all().select_related('person', 'area').order_by('-detection_time')
    persons = Person.objects.all()
    areas = RestrictedArea.objects.all()
    unread_count = PhoneDetection.objects.filter(is_read=False).count()

    return render(request, 'admin_dashboard/dashboard.html', {
        'detections': detections,
        'persons': persons,
        'areas': areas,
        'unread_count': unread_count,
    })


# ─── Alert APIs ─────────────────────────────────────────────

def unread_alert_count(request):
    """Polling endpoint for admin notification badge"""
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'count': 0})
    count = PhoneDetection.objects.filter(is_read=False).count()
    return JsonResponse({'count': count})


@csrf_exempt
def mark_alert_read(request, detection_id):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})
    try:
        detection = PhoneDetection.objects.get(id=detection_id)
        detection.is_read = True
        detection.save()
        return JsonResponse({'success': True})
    except PhoneDetection.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Not found'})


@csrf_exempt
def mark_all_read(request):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})
    PhoneDetection.objects.filter(is_read=False).update(is_read=True)
    return JsonResponse({'success': True})


# ─── Detection Management ────────────────────────────────────

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


# ─── Restricted Area Management ─────────────────────────────

@csrf_exempt
def add_area(request):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'})
    try:
        data = json.loads(request.body)
        name = data.get('name', '').strip()
        description = data.get('description', '').strip()
        if not name:
            return JsonResponse({'success': False, 'error': 'Area name is required'})
        area = RestrictedArea.objects.create(name=name, description=description, is_active=True)
        return JsonResponse({'success': True, 'area': {
            'id': area.id, 'name': area.name,
            'description': area.description, 'is_active': area.is_active
        }})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@csrf_exempt
def toggle_area(request, area_id):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'})
    try:
        area = RestrictedArea.objects.get(id=area_id)
        area.is_active = not area.is_active
        area.save()
        return JsonResponse({'success': True, 'is_active': area.is_active})
    except RestrictedArea.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Area not found'})


@csrf_exempt
def delete_area(request, area_id):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'})
    try:
        area = RestrictedArea.objects.get(id=area_id)
        area.delete()
        return JsonResponse({'success': True})
    except RestrictedArea.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Area not found'})


# ─── User Management ─────────────────────────────────────────

@csrf_exempt
def delete_user(request, user_id):
    if not request.session.get('admin_logged_in'):
        return JsonResponse({'success': False, 'error': 'Unauthorized'})
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Method not allowed'})
    try:
        person = Person.objects.get(id=user_id)
        username = person.username

        import shutil
        person_dir = os.path.join(settings.MEDIA_ROOT, 'faces', username)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)

        person.delete()
        retrain_model_after_deletion()
        return JsonResponse({'success': True})
    except Person.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'User not found'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


def retrain_model_after_deletion():
    """Retrain the face recognition model after user deletion"""
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8, threshold=80.0
        )

        faces_data = []
        labels = []
        label_map = {}
        persons = Person.objects.all()

        if not persons.exists():
            model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
            for fname in ['face_recognizer.yml', 'label_map.pkl']:
                fpath = os.path.join(model_dir, fname)
                if os.path.exists(fpath):
                    os.remove(fpath)
            return

        from face_capture.models import FaceImage
        for idx, person in enumerate(persons):
            label_map[idx] = person.username
            for face_img in FaceImage.objects.filter(person=person):
                img_path = os.path.join(settings.MEDIA_ROOT, str(face_img.image))
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    for (x, y, w, h) in faces:
                        face_roi = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
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
