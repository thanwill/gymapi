# Use a imagem oficial do Python como base
FROM python:3.11

# Defina o diretório de trabalho no contêiner
WORKDIR /app

# Copie os arquivos de requisitos para o contêiner
COPY requirements.txt .

# Instale as dependências do projeto
RUN apt-get update && apt-get install -y sqlite3 && pip install --no-cache-dir -r requirements.txt

# Copie o restante do código do projeto para o contêiner
COPY . .

# Copie o script de entrypoint
COPY entrypoint.sh /entrypoint.sh

# Torne o script executável
RUN chmod +x /entrypoint.sh

# Exponha a porta que o Django usará
EXPOSE 8000

# Defina o entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Defina o comando padrão
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]