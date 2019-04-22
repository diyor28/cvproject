import os
import cv2

from django.shortcuts import render, reverse, redirect
from django.views.generic import CreateView
from django.core.exceptions import ValidationError

from .forms import UploadImageForm
from .models import ImageModel
from cvision import settings
from .scale_cnn import DistilledResNetSR

keras_model = DistilledResNetSR()


class GetImageView(CreateView):
    template_name = 'supersampl/get_image.html'
    form_class = UploadImageForm

    def remove_files(self, path):
        for p in os.listdir(path):
            os.remove(os.path.join(path, p))

    def get(self, request, *args, **kwargs):
        self.remove_files(settings.ORIGINAL)
        self.remove_files(settings.INTERMEDIATE)
        self.remove_files(settings.PROCESSED)

        form = self.form_class()
        context = {"form": form}
        return render(request, template_name=self.template_name, context=context)

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            model = form.save(commit=True)

        full_image_path = model.image.path

        width, height = cv2.imread(full_image_path).shape[:2]

        if width > 1000 or height > 1000:
            context = {"form": form,
                       "error_message": "Image is too big. Image should be smaller than 1000x1000px"}

            return render(request, template_name=self.template_name, context=context)

        image_path = os.path.split(full_image_path)[1]
        intermediate = os.path.join(settings.INTERMEDIATE, image_path)
        end_path = os.path.join(settings.PROCESSED, image_path)

        keras_model.upscale(origin_path=full_image_path, intermediate=intermediate,
                            end_path=end_path, verbose=False)

        model.intermediate = os.path.join('intermediate', image_path)
        model.processed = os.path.join('processed', image_path)
        model.save()

        intermediate = model.intermediate.url
        processed = model.processed.url
        ImageModel.objects.all().delete()

        context = {"form": form,
                   "intermediate": intermediate,
                   "processed": processed}
        return render(request, template_name=self.template_name, context=context)

