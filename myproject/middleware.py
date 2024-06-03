# myproject/middleware.py

class DebugMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Preprocesar la solicitud aquí
        print(f"Request: {request.method} {request.path}")

        # Pasar la solicitud a la siguiente capa de middleware
        response = self.get_response(request)

        # Procesar la respuesta aquí
        print(f"Response status code: {response.status_code}")
        print(f"Response content: {response.content[:500]}")  # Limitar la cantidad de contenido impreso

        return response
