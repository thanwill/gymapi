from venv import logger
from rest_framework.views import APIView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import HttpResponse, JsonResponse
from django.views import View
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.exceptions import NotFittedError
from .util import process_dataset, get_correlations, get_column_description
from .serializers import AnalysesSerializer, DatasetSerializer
from .models import Dataset, ColumnMetadata, Analyses
from .graphics import gerar_graficos, get_insights_types
from .analise import realizar_analise
from django.shortcuts import render
import pickle

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
    def post(self, request):
        # Obter dataset_id, features e target_column da requisição
        dataset_id = request.data.get("dataset_id")
        features = request.data.get("features")
        target_column = request.data.get("target_column")

        if not dataset_id or not features or not target_column:
            return Response({"error": "dataset_id, features e target_column são obrigatórios."}, status=status.HTTP_400_BAD_REQUEST)

        # Obter o dataset do banco de dados
        dataset = Dataset.objects.get(id=dataset_id)

        # Carregar o dataset em um DataFrame
        file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset.filename)
        df = pd.read_csv(file_path)

        try:
            # Invocar a função realizar_analise
            pipeline, X_test, y_test, model_type = realizar_analise(df, features, target_column)
            
            # Salvar o modelo treinado no sistema de arquivos
            model_name = f"model_{model_type}_{dataset_id}.pkl"
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', model_name)
            with open(model_path, 'wb') as model_file:
                pickle.dump(pipeline, model_file)
                
            # Salvar os resultados da análise no banco de dados
            analysis = Analyses.objects.create(
                dataset_id=dataset,
                analysis_type='PREDICTION',
                parameters={
                    "features": features,
                    "target_column": target_column
                },
                model_reference=model_name,
                status='COMPLETED',
                error_message=''
            )
        
            analysis.save()
            
            serializer = AnalysesSerializer(analysis)            
            
            return Response({"message": "Análise realizada com sucesso.", "data": serializer.data}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class PredictResultView(APIView):
    def post(self, request):
        try:
            target_column = request.data.get('target_column')
            user_data = request.data.get('user_data')
            analysis_id = request.data.get('analysis_id')

            if not target_column or not user_data or not analysis_id:
                return Response(
                    {"error": "dataset_id, target_column, and user_data are required."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Procurar em analises se existe um modelo treinado para o dataset_id e para o target_column
            analyses = Analyses.objects.filter(id=analysis_id, parameters__target_column=target_column)
            if not analyses.exists():
                return Response(
                    {"error": "No analysis found for the specified target_column."},
                    status=status.HTTP_404_NOT_FOUND
                )
                
            # Map user_data keys to the expected dataset column names
            user_data_mapped = {
                "age": user_data.get("age"),
                "height_(m)": user_data.get("height_m"),
                "gender": user_data.get("gender"),
                "weight_(kg)": user_data.get("weight_kg")
            }

            # Carregar o modelo treinado usando o model_reference
            model_name = analyses.first().model_reference
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', model_name)

            if not os.path.exists(model_path):
                return Response(
                    {"error": "Model not found."},
                    status=status.HTTP_404_NOT_FOUND
                )

            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)

            # Convert user data to DataFrame
            user_df = pd.DataFrame([user_data_mapped])

            # Fazer a previsão
            prediction = model.predict(user_df)

            return Response({"prediction": prediction.tolist()}, status=status.HTTP_200_OK)
        except ValueError as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
        
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
            
            # se analysis_id for all, exclua todas as análises
            if analysis_id == 'all':
                Analyses.objects.all().delete()
                return JsonResponse({"message": "All analyses deleted successfully."}, status=status.HTTP_204_NO_CONTENT)                
            
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

class InsightsView(APIView):
    def get(self, request):
        try:
            # Get dataset_id from query parameters
            dataset_id = request.query_params.get('dataset_id')
            if not dataset_id:
                return Response({"error": "dataset_id parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Get columns from query parameters
            columns = request.query_params.get('insights_types')
            insights_types = request.query_params.get('insights_types')
            
            if not insights_types:
                return Response({"error": "insights_types parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            insights_types = insights_types.split(',')
            
            if not columns:
                return Response({"error": "columns parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
            columns = columns.split(',')

            # Get dataset
            dataset = Dataset.objects.get(id=dataset_id)
            
            # Verifica se o dataset existe
            file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset.filename)
            
            if not os.path.exists(file_path):
                return Response({"error": "Dataset file not found"}, status=status.HTTP_404_NOT_FOUND)
            
            # chama o método de gerar gráficos que retorna um html.Div com os gráficos
            graficos = gerar_graficos(file_path, insights_types)
            
            # Se não houver gráficos, retorna um erro
            if not graficos:
                return Response({"error": "No informations found for the specified insights_types"}, status=status.HTTP_404_NOT_FOUND)
            
            return Response(graficos, status=status.HTTP_200_OK)
        
        except Dataset.DoesNotExist:
            
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        
class GetInsightsTypesView(APIView):
    def get(self, request):
        try:
            
            insights_types = get_insights_types()
            
            if not insights_types:
                return Response({"error": "No insights types found"}, status=status.HTTP_404_NOT_FOUND)
            
            return Response({"insights_types": insights_types}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

class GetImageView(APIView):
    def get(self, request, image_name):

        try:                    
            
            # Define the path to the image
            image_path = os.path.join(settings.MEDIA_ROOT, 'images', f"{image_name}.png")
            
            # Check if the image exists
            if not os.path.exists(image_path):
                return Response({"error": "Image not found"}, status=status.HTTP_404_NOT_FOUND)            
            
            
            # Open and read the image file
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                return Response(image_base64, content_type="application/json")
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
class DownloadDatasetView(APIView):
    def get(self, request, dataset_id):
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            file_path = os.path.join(settings.MEDIA_ROOT, 'datasets', dataset.filename)
            if not os.path.exists(file_path):
                return Response({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)

            with open(file_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type='text/csv')
                response['Content-Disposition'] = f'attachment; filename="{dataset.filename}"'
                return response
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset not found"}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        
class PredictByAttributeView(APIView):
    def post(self, request):
        target_column = request.query_params.get('target_column')
        if not target_column:
            return Response({"error": "target_column parameter is required."}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            analyses = Analyses.objects.filter(parameters__target_column=target_column)
            if not analyses.exists():
                return Response({"error": "No analysis found for the specified target_column."}, status=status.HTTP_404_NOT_FOUND)
            
            analysis = analyses.first()
            model_name = f"model_classifier_{analysis.dataset_id_id}.pkl" if target_column == 'workout_type' else f"model_regressor_{analysis.dataset_id_id}.pkl"
            model_path = os.path.join(settings.MEDIA_ROOT, 'models', model_name)
            if not os.path.exists(model_path):
                return Response({"error": "Trained model not found."}, status=status.HTTP_404_NOT_FOUND)
            
            with open(model_path, 'rb') as model_file:
                model = pickle.load(model_file)
            
            # Verifique se o modelo carregado é uma tupla
            if isinstance(model, tuple):
                model = model[0]  # Desempacote o pipeline do modelo
            
            user_data = request.data.get('user_data')
            
            if not user_data:
                return Response({"error": "user_data parameter is required."}, status=status.HTTP_400_BAD_REQUEST)
            
            # Lista das features esperadas pelo modelo
            if target_column == 'workout_type':
                required_features = [
                    "age",
                    "gender",
                    "weight_(kg)",
                    "height_(m)",
                    "max_bpm",
                    "resting_bpm",
                    "session_duration_(hours)",
                    "bmi",
                    "water_intake_(liters)",
                    "workout_frequency_(days/week)",
                    "experience_level"
                ]
            else:
                # Adicione as features para outros modelos de regressão se houver
                required_features = [
                    # Exemplo:
                    "feature1",
                    "feature2",
                    # ...
                ]
            
            missing_features = [feature for feature in required_features if feature not in user_data]
            if missing_features:
                return Response({"error": f"Missing required features: {', '.join(missing_features)}"}, status=status.HTTP_400_BAD_REQUEST)
            
            # Criar DataFrame de entrada
            input_df = pd.DataFrame([user_data])
            
            # Log das colunas do input_df
            logger.debug(f'Colunas do input_df: {input_df.columns.tolist()}')
            
            # Realizar predição
            prediction = model.predict(input_df)
            
            if target_column == 'workout_type':
                prediction = prediction.tolist()
            else:
                prediction = prediction[0]                
                            
            return Response({"prediction": prediction}, status=status.HTTP_200_OK)
        
        except Analyses.DoesNotExist:
            return Response({"error": "Analysis not found."}, status=status.HTTP_404_NOT_FOUND)
        
        except NotFittedError:
            return Response({"error": "Model not trained."}, status=status.HTTP_400_BAD_REQUEST)
        
        except Exception as e:
            logger.exception("Erro durante a predição:")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)