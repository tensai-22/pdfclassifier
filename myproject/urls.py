# myproject/urls.py
from django.contrib import admin
from django.urls import include, path
from django.http import HttpResponse

def root_view(request):
    return HttpResponse("Esta es la vista raíz")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('pdfclassifier/', include('pdfclassifier.urls')),
    path('project1/', include('project1.urls')),
    path('', root_view, name='root'),
]
