from django import forms
from .models import EmbeddingsModel


class AddFaceForm(forms.ModelForm):
    full_name = forms.CharField(max_length=100, required=True)

    class Meta:
        model = EmbeddingsModel
        fields = ['full_name']



