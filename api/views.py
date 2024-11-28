from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import JsonResponse
from django.views import View
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from .utils import process_dataset, get_correlations, get_column_description
from .serializers import DatasetSerializer
from .models import Dataset, ColumnMetadata, Analyses
from django.shortcuts import render

import os
import csv
import io
import base64
import json
import matplotlib.pyplot as plt
import seaborn as sns
import uuid
import pandas as pd
from django.conf import settings
from sklearn.linear_model import LinearRegression

def index(request):
    return render(request, 'index.html')

class UploadDatasetView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        # 1. Obter o arquivo enviado
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "Nenhum arquivo fornecido."}, status=status.HTTP_400_BAD_REQUEST)

        if not file.name.endswith(".csv"):
            return Response({"error": "Formato de arquivo inválido. Apenas arquivos CSV são permitidos."}, status=status.HTTP_400_BAD_REQUEST)

        # Gerar um nome único para o arquivo para evitar conflitos
        file_name = f"{uuid.uuid4().hex}_{file.name}"
        file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Salvar o arquivo no sistema de arquivos
        try:
            with open(file_path, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
        except Exception as e:
            return Response({"error": f"Falha ao salvar o arquivo: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 2. Extrair as colunas do arquivo CSV e padronizar os headers
        try:
            with open(file_path, 'r', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                headers = next(reader)  # Pega a primeira linha (colunas)
        except Exception as e:
            return Response({"error": f"Falha ao ler as colunas: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Padronizar headers para minúsculas e substituir espaços por underscores
        headers = [header.lower().replace(' ', '_') for header in headers]

        # Salvar os novos headers no arquivo CSV
        try:
            df = pd.read_csv(file_path)
            df.columns = headers
            df.to_csv(file_path, index=False)
        except Exception as e:
            return Response({"error": f"Falha ao salvar os novos headers: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # 3. Criar o registro no banco de dados
        try:
            dataset = Dataset.objects.create(
                filename=file_name,
                columns=headers,
                status='PENDING',  # Status inicial
                error_message='',  # Sem erro inicialmente
            )
        except Exception as e:
            return Response({"error": f"Falha ao criar registro do dataset: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 4. Extração e Armazenamento dos Metadados das Colunas
        try:
            # Recarregar o DataFrame com os novos headers
            df = pd.read_csv(file_path)

            for column in headers:
                # Determinar se a coluna é contínua ou categórica
                type = ''
                
                '''
                0   Age                            973 non-null    int64                
                2   Weight (kg)                    973 non-null    float64
                3   Height (m)                     973 non-null    float64
                4   Max_BPM                        973 non-null    int64  
                5   Avg_BPM                        973 non-null    int64  
                6   Resting_BPM                    973 non-null    int64  
                7   Session_Duration (hours)       973 non-null    float64
                8   Calories_Burned                973 non-null    float64
                9   Workout_Type                   973 non-null    object
                1   Gender                         973 non-null    object  
                10  Fat_Percentage                 973 non-null    float64
                11  Water_Intake (liters)          973 non-null    float64
                12  Workout_Frequency (days/week)  973 non-null    int64  
                13  Experience_Level               973 non-null    int64  
                14  BMI                            973 non-null    float64
                '''
                
                if is_numeric_dtype(df[column]):
                    type = 'continuous'
                elif is_categorical_dtype(df[column]) or df[column].dtype == 'object':
                    type = 'categorical'
                else:
                    raise ValueError("Tipo de dado desconhecido.")

                # Obter a descrição da coluna
                description = get_column_description(column_name=column)

                # Criar registro de metadados para a coluna
                ColumnMetadata.objects.create(
                    dataset=dataset,
                    column_name=column,
                    data_type=type,
                    description=description
                )
                
        except Exception as e:
            # Atualizar o status do dataset para 'ERROR' em caso de falha
            dataset.status = 'ERROR'
            dataset.error_message = f"Falha ao processar metadados das colunas: {str(e)}"
            dataset.save()
            
            return Response({"error": f"Falha ao processar metadados das colunas: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 5. Atualizar o status do dataset para 'PROCESSED'
        try:
            dataset.status = 'PENDING'
            dataset.save()
        except Exception as e:
            return Response({"error": f"Falha ao atualizar status do dataset: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # 6. Serializar e retornar a resposta ao cliente, incluindo os metadados das colunas
        try:
            serializer = DatasetSerializer(dataset)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": f"Falha ao serializar os dados do dataset: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ListDatasetsView(APIView):    
    def get(self, request):
        datasets = Dataset.objects.all()
        serializer = DatasetSerializer(datasets, many=True)
        return Response(serializer.data)  

class ListDatasetsViewByID(APIView):
    def get(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            
            # Carrega o dataset
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

            # Cria o gráfico para identificar os valores nulos
            plt.figure(figsize=(15, 10))
            df.isnull().sum().plot(kind='bar')
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            missing_values_img = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()

            # Cria o gráfico para identificar a distribuição dos valores
            plt.figure(figsize=(15, 10))
            df.hist(bins=30, figsize=(15, 10))
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            distribution_img = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()

            # Convertendo variáveis categóricas em variáveis dummy
            df = pd.get_dummies(df, drop_first=True)

            # Cria o gráfico para identificar a correlação entre as variáveis
            plt.figure(figsize=(15, 10))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            correlation_img = base64.b64encode(img_buffer.read()).decode('utf-8')
            plt.close()
            
            # Obtendo os metadados das colunas
            columns_metadata = ColumnMetadata.objects.filter(dataset=dataset).values()

            # retornando os dados do dataset também
            serializer = DatasetSerializer(dataset)
            
            images = [
                {"name": "distribution_img", "data": distribution_img},
                {"name": "correlation_img", "data": correlation_img}
            ]
            response_data = {
                "info": info,
                "describe": json.loads(describe),
                "missing_values": json.loads(missing_values),                
                "dataset": serializer.data,
                "columns_metadata": list(columns_metadata),
                "images": images
            }

            return Response(response_data, status=status.HTTP_200_OK)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
         
class RemoveDatasetView(APIView):
    def delete(self, request, dataset_id):            
        
        try:
            if not dataset_id:
                return Response({"error": "Dataset ID is required"}, status=status.HTTP_400_BAD_REQUEST)
                
            dataset = Dataset.objects.get(id=dataset_id)
            
            dataset.delete()
            return Response({"message": "Dataset deleted successfully"}, status=status.HTTP_204_NO_CONTENT)
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
            
            # Processa o dataset def process_dataset(file_path, target_column, id):
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
                # Se não forem fornecidas colunas, use todas as colunas do ColumnMetadata
                columns = list(ColumnMetadata.objects.filter(dataset=dataset).values_list('column_name', flat=True))
            
            # Calcula as correlações
            correlations = get_correlations(df, columns)
            
            return Response(correlations, status=status.HTTP_200_OK)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class RemoveAnalysesView(APIView):
    def delete(self, request, analysis_id, *args, **kwargs):
        try:
            analysis = Analyses.objects.get(id=analysis_id)
            analysis.delete()
            return JsonResponse({"message": "Analysis deleted successfully."}, status=status.HTTP_204_NO_CONTENT)
        except Analyses.DoesNotExist:
            return JsonResponse({"error": "Analysis not found."}, status=status.HTTP_404_NOT_FOUND)
        
class AnalysisResultsView(View):
    def get(self, request):
        try:
            analyses = Analyses.objects.all()
            return JsonResponse(list(analyses.values()), safe=False)
        except Analyses.DoesNotExist:
            return JsonResponse([], safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class FilteredAnalysisResultsView(View):
    def get(self, request):
        try:
            # Obtém os parâmetros da query string
            analysis_type = request.GET.get('analysis_type')
            status = request.GET.get('status')
            target_column = request.GET.get('target_column')
            database_id = request.GET.get('database_id')

            # Filtra as análises com base nos parâmetros fornecidos
            analyses = Analyses.objects.all()
            
            if analysis_type:
                analyses = analyses.filter(analysis_type=analysis_type)
            if status:
                analyses = analyses.filter(status=status)
            if target_column:
                analyses = analyses.filter(parameters__target_column=target_column)
            if database_id:
                analyses = analyses.filter(dataset_id=database_id)

            return JsonResponse(list(analyses.values()), safe=False)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class CreateMultipleAnalysesView(APIView):

    def get(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset.filename)
            
            # Obtém os target_columns dos parâmetros da requisição
            target_columns = request.query_params.get('target_columns')
            if not target_columns:
                return Response({"error": "target_columns parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            
            target_columns = target_columns.split(',')
            response_data = []

            for target_column in target_columns:
                # Processa o dataset para cada target_column
                analysis_result = process_dataset(file_path, target_column, dataset_id)
                response_data.append({
                    "target_column": target_column,
                    "result": analysis_result
                })

            return Response(response_data, status=status.HTTP_200_OK)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class PredictView(APIView):

    def post(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset.filename)
            
            # Obtém os dados do usuário e os target_columns dos parâmetros da requisição
            user_data = request.data.get('user_data')
            target_columns = request.data.get('target_columns')
            if not user_data or not target_columns:
                return Response({"error": "user_data and target_columns parameters are required"}, status=status.HTTP_400_BAD_REQUEST)
            
            target_columns = target_columns.split(',')
            response_data = {}

            # Carrega o dataset
            df = pd.read_csv(file_path)
            
            for target_column in target_columns:
                if target_column not in df.columns:
                    return Response({"error": f"Target column '{target_column}' not found in dataset"}, status=status.HTTP_400_BAD_REQUEST)

            return Response(response_data, status=status.HTTP_200_OK)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


