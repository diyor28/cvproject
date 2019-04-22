from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.


class EmbeddingsModel(models.Model):
    full_name = models.CharField(default='', max_length=100)
    embedding = ArrayField(ArrayField(models.FloatField(null=True, blank=True)))
