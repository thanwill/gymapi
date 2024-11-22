import os
import csv
from rest_framework import status
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from api.models import Dataset
from .serializers import DatasetSerializer

class UploadDatasetView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        # 1. Obter o arquivo enviado
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "No file provided"}, status=400)

        if not file.name.endswith(".csv"):
            return Response({"error": "Invalid file format. Only CSV files are allowed."}, status=400)

        # 2. Salvar o arquivo no disco
        file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', file.name)
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

        # 4. Criar o registro no banco de dados
        dataset = Dataset.objects.create(
            filename=file.name,
            columns=headers,
            status='PENDING',  # Status inicial
            error_message='',  # Sem erro inicialmente
        )

        # 5. Retornar a resposta ao cliente
        serializer = DatasetSerializer(dataset)
        return Response(serializer.data, status=status.HTTP_201_CREATED)
        
class ListDatasetsView(APIView):    
    def get(self, request):
        return Response({
            "dataset_id": "abc123",
            "analysis_type": "PREDICTION",
            "parameters": {
                "target_column": "Sales",
                "model": "RandomForest",
                "features": [
                    "Age",
                    "Income"
                ]
            }
        })    

class CreateAnalysisView(APIView):
    def post(self, request):
        dataset_id = request.data.get("dataset_id")
        analysis_type = request.data.get("analysis_type")
        parameters = request.data.get("parameters")
        # Cria a análise no banco
        return Response({"analysis_id": "xyz456"})
    
# 	•	GET /results/{analysis_id}: Retorna o resultado de uma análise ou predição.

class AnalysisResultsView(APIView):
    def get(self, request, analysis_id):
        return Response({
            "analysis_id": analysis_id,
            "results": {
                "predictions": [100, 200, 300]
            }
        })