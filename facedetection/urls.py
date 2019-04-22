from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from .views import DetectFaceView, AddNewFaceView, StreamFaceView, StreamRawFaceView

urlpatterns = [
    url(r'^$', DetectFaceView.as_view(), name='recognize'),
    url(r'^raw-stream/$', StreamRawFaceView.as_view(), name='raw-stream'),
    url(r'^stream/$', StreamFaceView.as_view(), name='stream'),
    url(r'^add-face/$', AddNewFaceView.as_view(), name='add-face')
]
