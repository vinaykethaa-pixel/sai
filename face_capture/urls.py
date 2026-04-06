from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('capture/', views.capture_page, name='capture_page'),
    path('save-face/', views.save_face, name='save_face'),
    path('training/', views.training_page, name='training_page'),
    path('train-model/', views.train_model, name='train_model'),
]
