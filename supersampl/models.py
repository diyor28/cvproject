from django.db import models


class ImageModel(models.Model):
    image = models.ImageField(upload_to='images/')
    intermediate = models.ImageField(blank=True)
    processed = models.ImageField(blank=True)



