#!/bin/sh

# Criar migrações para quaisquer alterações no modelo
python manage.py makemigrations

# Aplicar migrações ao banco de dados
python manage.py migrate

# Iniciar o servidor Django
exec "$@"