from django.contrib import admin
from django.urls import path, include
from django.http import JsonResponse

# Rota de captura para URLs não correspondentes
def handler404(request, exception):
    return JsonResponse({"error": "Endpoint not found"}, status=404)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),  # Include API URLs
]

# Atribua a função handler404 diretamente
handler404 = handler404