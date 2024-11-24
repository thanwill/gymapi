from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import os
import io
import csv
from django.conf import settings
from api.models import Dataset
from .serializers import DatasetSerializer
import uuid
import json
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class UploadDatasetView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        # 1. Obter o arquivo enviado
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file provided"}, status=400)

        if not file.name.endswith(".csv"):
            return Response({"error": "Invalid file format. Only CSV files are allowed."}, status=400)
        
        file_name = f"{uuid.uuid4().hex}_{file.name}"
        file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        # 3. Extrair as colunas do arquivo CSV
        try:
            with open(file_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                headers = next(reader)  # Pega a primeira linha (colunas)
        except Exception as e:
            return Response({"error": f"Failed to read columns: {str(e)}"}, status=400)
        
        # trata headers para ser tudo minusculo e sem espaços
        headers = [header.lower().replace(' ', '_') for header in headers]

        # 4. Criar o registro no banco de dados
        dataset = Dataset.objects.create(
            filename=file_name,
            columns=headers,
            status='PENDING',  # Status inicial
            error_message='',  # Sem erro inicialmente
        )

        # 5. Retornar a resposta ao cliente
        serializer = DatasetSerializer(dataset)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
        
class ListDatasetsView(APIView):    
    def get(self, request):
        datasets = Dataset.objects.all()
        serializer = DatasetSerializer(datasets, many=True)
        return Response(serializer.data)  

class ListDatasetsViewByID(APIView):
    def get(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            serializer = DatasetSerializer(dataset)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        
class RemoveDatasetView(APIView):
    def delete(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            dataset.delete()
            return Response(
                {
                    "message": "Dataset deleted successfully"
                }, 
                status=status.HTTP_204_NO_CONTENT)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)

class CreateAnalysisView(APIView):
    def get(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset.filename)
            df = pd.read_csv(file_path)

            # Informações gerais sobre o dataset
            buffer = io.StringIO()
            df.info(buf=buffer)
            info = buffer.getvalue()

            # Estatísticas descritivas do dataset
            describe = df.describe().to_json()

            # Valores ausentes no dataset
            missing_values = df.isnull().sum().to_json()

            # Visualizar a distribuição das variáveis numéricas
            plt.figure(figsize=(15, 10))
            df.hist(bins=30, figsize=(15, 10))
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()

            # Convertendo as variáveis categóricas em numéricas
            df = pd.get_dummies(df, drop_first=True)
            head = json.loads(df.head().to_json())
            describe = json.loads(describe)
            missing_values = json.loads(missing_values)

            response_data = {
                "info": info,
                "describe": describe,
                "missing_values": missing_values,
                "histogram": img_str,
                "head": head
            }

            return Response(response_data, status=status.HTTP_200_OK)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
class AnalysisResultsView(APIView):
    def get(self, request, analysis_id):
        return Response({
            "analysis_id": analysis_id,
            "results": {
                "predictions": [100, 200, 300]
            }
        })