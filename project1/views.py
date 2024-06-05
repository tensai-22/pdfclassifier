from django.http import HttpResponse

def index(request):
    print("La vista 'index' ha sido llamada")
    return HttpResponse("Hello, world. You're at the project1 index.")
