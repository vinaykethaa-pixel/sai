from django.db import models
from face_capture.models import Person


class RestrictedArea(models.Model):
    name = models.CharField(max_length=100)
    description = models.CharField(max_length=200, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']


class PhoneDetection(models.Model):
    person = models.ForeignKey(
        Person, on_delete=models.CASCADE,
        related_name='phone_detections', null=True, blank=True
    )
    area = models.ForeignKey(
        RestrictedArea, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='detections'
    )
    image = models.ImageField(upload_to='detections/')
    detection_time = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        username = self.person.username if self.person else "Unknown"
        area_name = self.area.name if self.area else "Unknown Area"
        return f"{username} @ {area_name} - {self.detection_time}"

    class Meta:
        ordering = ['-detection_time']
