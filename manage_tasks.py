import os
import django
from django.core.management import call_command

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

# Ejecutar migraciones
call_command('migrate')

# Crear superusuario automáticamente (modifica según sea necesario)
try:
    from django.contrib.auth import get_user_model
    User = get_user_model()
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser('admin', 'admin@example.com', 'password')
        print("Superuser created.")
    else:
        print("Superuser already exists.")
except Exception as e:
    print(f"Error creating superuser: {e}")
