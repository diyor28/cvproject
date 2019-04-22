from django.shortcuts import render, reverse, redirect
from django.views.generic import CreateView


class HomePageView(CreateView):
    template_name = 'cvision/home.html'

    def get(self, request, *args, **kwargs):
        context = {}
        return render(request, template_name=self.template_name, context=context)

    def post(self, request, *args, **kwargs):
        context = {}
        return render(request, template_name=self.template_name, context=context)

