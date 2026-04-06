from django.db import models

class Person(models.Model):
    username = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_trained = models.BooleanField(default=False)

    def __str__(self):
        return self.username

    class Meta:
        ordering = ['-created_at']

class FaceImage(models.Model):
    person = models.ForeignKey(Person, on_delete=models.CASCADE, related_name='face_images')
    image = models.ImageField(upload_to='faces/')
    captured_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.person.username} - {self.captured_at}"

    class Meta:
        ordering = ['-captured_at']
