from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .utils import process_dataset, get_correlations
import os
import csv
from django.conf import settings
from api.models import Dataset
from .serializers import DatasetSerializer
import uuid
import pandas as pd

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
            if not dataset_id:
                return Response({"error": "Dataset ID is required"}, status=status.HTTP_400_BAD_REQUEST)
            
            
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
            
            # Obtém o target_column dos parâmetros da requisição
            target_column = request.query_params.get('target_column')
            if not target_column:
                return Response({"error": "target_column parameter is required"}, status=status.HTTP_400_BAD_REQUEST)                    
            
            # Processa o dataset
            response_data = process_dataset(file_path, target_column, dataset_id)

            return Response(response_data, status=status.HTTP_200_OK)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
class GetCorrelationsView(APIView):
    
    def get(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset.filename)
            
            # Carrega o dataset
            df = pd.read_csv(file_path)
            
            # Obtém as colunas dos parâmetros da requisição
            columns = request.query_params.getlist('columns')
            if not columns:
                return Response({"error": "columns parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Calcula as correlações
            correlations = get_correlations(df, columns)
            
            return Response(correlations, status=status.HTTP_200_OK)
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