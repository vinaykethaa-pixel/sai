from django.urls import path
from . import views

urlpatterns = [
    path('', views.detection_page, name='detection_page'),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('detect-frame/', views.detect_frame, name='detect_frame'),
    path('areas/', views.get_active_areas, name='get_active_areas'),
]
