from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from .views import HomePageView
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path(r'facedetection/', include(('facedetection.urls', 'facedetection'), namespace='facedetection')),
    path(r'supersampl/', include(('supersampl.urls', 'supersampl'), namespace='supersampl')),
    url(r'^$', HomePageView.as_view(), name='home')
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
