from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score, accuracy_score
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.feature_selection import SelectKBest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import json
from api.models import Dataset
import missingno as msno
import pickle
import os
import numpy as np
import pandas as pd

from .models import Analyses

def is_continuous_or_categorical(series):
    """
    Verifica se uma série é contínua ou categórica.

    Args:
        series (pd.Series): Série do pandas.

    Returns:
        str: 'continuous' se a série for contínua, 'categorical' se a série for categórica.
    """
    if is_numeric_dtype(series):
        return 'continuous'
    elif is_categorical_dtype(series) or series.dtype == 'object':
        return 'categorical'
    else:
        raise ValueError("Tipo de dado desconhecido.")

def process_dataset(file_path, target_column, id):
    """
    Processa o dataset, treina o modelo e retorna os resultados previstos.

    Args:
        file_path (str): Caminho para o arquivo CSV.
        target_column (str): Nome da coluna alvo.

    Returns:
        dict: Informações sobre o dataset, relatório de classificação e imagem da matriz de confusão em base64.
    """
    dataset = Dataset.objects.get(id=id)    
    dataset.status = 'PROCESSING'
    dataset.save()
    
    df = pd.read_csv(file_path)

    # Informações gerais sobre o dataset
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()

    # Estatísticas descritivas do dataset
    describe = df.describe().to_json()

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

    # Balancear o dataset
    #X_resampled, y_resampled = balance_dataset(df, target_column)
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = split_data(X, y)
        

    try:
        # Verificar se a coluna alvo é categórica ou contínua
        if is_categorical_dtype(df[target_column]) or df[target_column].dtype == 'object':
            # Usar um classificador
            pipeline = create_and_train_pipeline(X_train, y_train, model_type='classifier')
            evaluation_results = evaluate_model(pipeline, X_test, y_test, model_type='classifier')            
            
        elif is_numeric_dtype(df[target_column]):
            # Usar um regressor
            pipeline = create_and_train_pipeline(X_train, y_train, model_type='regressor')
            evaluation_results = evaluate_model(pipeline, X_test, y_test, model_type='regressor')
                                    
        else:
            raise ValueError("Tipo de rótulo desconhecido. Talvez você esteja tentando ajustar um classificador em um alvo de regressão com valores contínuos.")
        
        dataset.status = 'COMPLETED'
        
        # Salvar o modelo treinado em disco    
        model_filename = f'model_{id}.pkl'
        media_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'media', 'models'))
        os.makedirs(media_dir, exist_ok=True)  # Garantir que a pasta existe
        model_path = os.path.join(media_dir, model_filename)  # Pasta onde os modelos serão salvos

        # Salvar o modelo treinado em disco
        with open(model_path, 'wb') as model_file:
            pickle.dump(pipeline, model_file)

        # Parâmetros da análise        
        parameters = {
            "target_column" : target_column,
            "model_reference": model_path            
        }                    
                
        # Salvar os resultados da análise no banco de dados
        analysis = Analyses(
            dataset_id = dataset,
            analysis_type='PREDICTION',
            parameters=parameters,
            status='COMPLETED'
        )
        
        analysis.save()

        # Verifica se analise foi salva
        if analysis.id is None:
            dataset.status = 'ERROR'
            dataset.error_message = "Erro ao salvar a análise no banco de dados."
            dataset.save()
            raise ValueError("Erro ao salvar a análise no banco de dados.")
        
        dataset.status = 'COMPLETED'
        dataset.save()
        
        return {
            "info": info,
            "describe": json.loads(describe),
            "histogram_image": img_str,
            "evaluation_results": evaluation_results,
            "analysis_id": analysis.id,
            "model_reference": model_path
        }
        
    except ValueError as e:
        if "Unknown label type: continuous" in str(e):
                        
            response_data = {
                "error": "Erro: Tipo de rótulo desconhecido. Talvez você esteja tentando ajustar um classificador em um alvo de regressão com valores contínuos."
            }
        else:
            
            response_data = { 
               "error": str(e)
            }
        
        dataset.status = 'ERROR'
        dataset.error_message = e
        dataset.save()
       
        
        return response_data 


def balance_dataset(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_and_train_pipeline(X_train, y_train, model_type):
    if model_type == 'classifier':
        # Escolher o modelo de classificação
        model = LogisticRegression(random_state=42)
    elif model_type == 'regressor':
        # Escolher o modelo de regressão
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'categorical':
        # Escolher o modelo para dados categóricos
        model = SVC(random_state=42)
    elif model_type == 'continuous':
        # Escolher o modelo para dados contínuos
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError("Tipo de modelo desconhecido. Use 'classifier', 'regressor', 'categorical' ou 'continuous'.")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test, model_type='classifier'):
    y_pred = pipeline.predict(X_test)
    images = []

    if model_type == 'classifier':
        
        classification_report_str = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Matriz de Confusão')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        # Matriz de Confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
        plt.title('Matriz de Confusão')
        plt.xlabel('Classe Predita')
        plt.ylabel('Classe Real')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        images.append(base64.b64encode(buffer.read()).decode('utf-8'))
        plt.close()

        accuracy = accuracy_score(y_test, y_pred)
        # Acurácia do Modelo
        plt.figure(figsize=(6, 4))
        plt.bar(['Acurácia'], [accuracy])
        plt.ylim(0, 1)
        plt.title('Acurácia do Modelo')
        plt.ylabel('Acurácia')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        images.append(base64.b64encode(buffer.read()).decode('utf-8'))
        plt.close()

        return {
            "classification_report": classification_report_str,
            "confusion_matrix_image": img_str
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            "mean_squared_error": mse,
            "r2_score": r2
        }
        
def get_column_description(column_name):
        """
        Retorna a descrição da coluna baseada no nome.
        """
        descriptions = {
            'age': 'Idade da pessoa em anos.',
            'gender': 'Gênero da pessoa (por exemplo, masculino, feminino, outro).',
            'weight_(kg)': 'Peso da pessoa em quilogramas.',
            'height_(m)': 'Altura da pessoa em metros.',
            'max_bpm': 'Frequência cardíaca máxima registrada durante o treino (batimentos por minuto).',
            'avg_bpm': 'Frequência cardíaca média durante o treino (batimentos por minuto).',
            'resting_bpm': 'Frequência cardíaca em repouso (batimentos por minuto).',
            'session_duration_(hours)': 'Duração da sessão de treino em horas.',
            'calories_burned': 'Calorias queimadas durante a sessão de treino.',
            'workout_type': 'Tipo de treino realizado (por exemplo, cardio, força, yoga).',
            'fat_percentage': 'Porcentagem de gordura corporal.',
            'water_intake_(liters)': 'Quantidade de água consumida em litros.',
            'workout_frequency_(days/week)': 'Frequência de treinos por semana (dias por semana).',
            'experience_level': 'Nível de experiência no treino (por exemplo, iniciante, intermediário, avançado).',
            'bmi': 'Índice de Massa Corporal (IMC) calculado como peso/altura².',
        }
        
        return descriptions.get(column_name, 'Descrição não disponível.')

def generate_visualizations(dataset, target_column):          

        X = dataset.drop('Experience_Level', axis=1)
        y = dataset['Experience_Level']

        # Aplicar a função SelectKBest para extrair as melhores features
        best_features = SelectKBest(k='all')
        fit = best_features.fit(X, y)
        df_scores = pd.DataFrame(fit.scores_)
        df_columns = pd.DataFrame(X.columns)

        # Concatenar dataframes
        feature_scores = pd.concat([df_columns, df_scores], axis=1)
        feature_scores.columns = ['Feature', 'Score']
        feature_scores = feature_scores.sort_values(by='Score', ascending=False).reset_index(drop=True)

        images = []

        # Distribuição das classes antes do balanceamento
        plt.figure(figsize=(10, 5))
        sns.countplot(x=y)
        plt.title('Distribuição das Classes Antes do Balanceamento')
        plt.xlabel('Classes')
        plt.ylabel('Contagem')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        images.append(base64.b64encode(buffer.read()).decode('utf-8'))
        plt.close()

        # Importância das Features
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Score', y='Feature', data=feature_scores)
        plt.title('Importância das Features')
        plt.xlabel('Score')
        plt.ylabel('Feature')
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        images.append(base64.b64encode(buffer.read()).decode('utf-8'))
        plt.close()        

        return images
    
def get_correlations(df, columns):
    """
    Calcula a correlação entre as colunas especificadas e outras variáveis do dataset.

    Args:
        df (pd.DataFrame): DataFrame original.
        columns (list): Lista de colunas para calcular a correlação.

    Returns:
        dict: Dicionário com as correlações das colunas especificadas com outras variáveis e imagem do heatmap em base64.
    """
    # Convertendo variáveis categóricas em variáveis dummy
    df = pd.get_dummies(df, drop_first=True)
    
    correlations = {}
    for column in columns:
        if column in df.columns:
            corr_series = df.corr()[column].drop(column).sort_values(ascending=False)
            correlations[column] = corr_series.to_dict()
        else:
            correlations[column] = "Column not found in dataset"
    
    # Gerar heatmap das correlações
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap de Correlações')

    # Salvar a figura em um buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return {
        "correlations": correlations,
        "heatmap": img_str
    }