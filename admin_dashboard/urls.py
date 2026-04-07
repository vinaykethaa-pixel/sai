from django.urls import path
from . import views

urlpatterns = [
    path('login/', views.admin_login, name='admin_login'),
    path('logout/', views.admin_logout, name='admin_logout'),
    path('', views.admin_dashboard, name='admin_dashboard'),

    # Detection management
    path('delete/<int:detection_id>/', views.delete_detection, name='delete_detection'),

    # Alert APIs
    path('alerts/unread-count/', views.unread_alert_count, name='unread_alert_count'),
    path('alerts/mark-read/<int:detection_id>/', views.mark_alert_read, name='mark_alert_read'),
    path('alerts/mark-all-read/', views.mark_all_read, name='mark_all_read'),

    # Restricted area management
    path('areas/add/', views.add_area, name='add_area'),
    path('areas/toggle/<int:area_id>/', views.toggle_area, name='toggle_area'),
    path('areas/delete/<int:area_id>/', views.delete_area, name='delete_area'),

    # User management
    path('delete-user/<int:user_id>/', views.delete_user, name='delete_user'),
]
