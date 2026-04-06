from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.admin_login, name='admin_login'),
    path('logout/', views.admin_logout, name='admin_logout'),
    path('', views.admin_dashboard, name='admin_dashboard'),
    path('delete/<int:detection_id>/', views.delete_detection, name='delete_detection'),
    path('delete-user/<int:user_id>/', views.delete_user, name='delete_user'),
]
