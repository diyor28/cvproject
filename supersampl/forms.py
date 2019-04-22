from django import forms
from .models import ImageModel


class UploadImageForm(forms.ModelForm):
    image = forms.ImageField(label='Choose a file', required=True, widget=forms.FileInput(
            attrs={"class": 'custom-file-input'}))

    class Meta:
        model = ImageModel
        fields = ['image']



