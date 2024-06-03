# project1/views.py
from django.http import HttpResponse

def index(request):
    return HttpResponse("¡Hola, mundo!")
