from django.db import models
from face_capture.models import Person

class PhoneDetection(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='phone_detections', null=True, blank=True)
    image = models.ImageField(upload_to='detections/')
    detected_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        username = self.person.username if self.person else "Unknown"
        return f"{username} - {self.detected_at}"

    class Meta:
        ordering = ['-detected_at']
