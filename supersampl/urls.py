from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from .views import GetImageView

urlpatterns = [
    url(r'^$', GetImageView.as_view(), name='get-image')
]
