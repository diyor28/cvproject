import time
import cv2
import numpy as np

from django.shortcuts import render, reverse, redirect
from django.views.generic import CreateView
from django.http import HttpResponse, StreamingHttpResponse

from facedetection.tf_face.recognition import FaceCV, RecognizeFace

from .forms import AddFaceForm
from .models import EmbeddingsModel

from cvision import settings

face_cv = FaceCV()
face_recon = RecognizeFace()

video = cv2.VideoCapture(0)


def raw_face_feed():
    while True:
        ret, frame = video.read()
        raw_face_feed.frame = frame
        frame = cv2.imencode('.jpg', frame)[1].tostring()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def frame_feed():
    while True:
        ret, frame = video.read()
        frame = face_cv.analyze_frame(frame, include_body=settings.INCLUDE_BODY,
                                      skip_frames=settings.SKIP_FRAMES, include_identity=True)
        frame = cv2.imencode('.jpg', frame)[1].tostring()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # time.sleep(0.05)


def read_video():
    with open(settings.VIDEO_PATH, 'rb') as file:
        bytes_video = file.read()
    return bytes_video


class AddNewFaceView(CreateView):
    template_name = 'facedetection/add_face.html'
    form_class = AddFaceForm

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        context = {"form": form}
        return render(request, template_name=self.template_name, context=context)

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            model = form.save(commit=False)

            embedding = face_recon.create_embedding(raw_face_feed.frame)

            if embedding:
                model.embedding = embedding
                model.save()

            context = {}
            return render(request, template_name=self.template_name, context=context)
        return redirect(reverse('facedetection:add-face'))


class DetectFaceView(CreateView):
    template_name = 'facedetection/detect_face.html'

    def get(self, request, *args, **kwargs):
        context = {}
        return render(request, template_name=self.template_name, context=context)

    def post(self, request, *args, **kwargs):
        context = {}
        return render(request, template_name=self.template_name, context=context)


class StreamRawFaceView(CreateView):
    def get(self, request, *args, **kwargs):
        return StreamingHttpResponse(raw_face_feed(), 'multipart/x-mixed-replace; boundary=frame')


class StreamFaceView(CreateView):

    def get(self, request, *args, **kwargs):
        return StreamingHttpResponse(frame_feed(), 'multipart/x-mixed-replace; boundary=frame')
