from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import io
import json
import base64
import pandas as pd
import matplotlib.pyplot as plt
from api.models import Dataset
import seaborn as sns


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

    # Balancear o dataset
    X_resampled, y_resampled = balance_dataset(df, target_column)

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

    # Criar e treinar o pipeline
    pipeline = create_and_train_pipeline(X_train, y_train)

    # Avaliar o modelo
    evaluation_results = evaluate_model(pipeline, X_test, y_test)

    response_data = {
        "info": info,
        "describe": json.loads(describe),
        "missing_values": json.loads(missing_values),
        "histogram": img_str,
        "classification_report": evaluation_results["classification_report"],
        "confusion_matrix_image": evaluation_results["confusion_matrix_image"]
    }
    
    if dataset:
        dataset.status = 'PROCESSED'
        dataset.save()
        
    if 'error' in response_data:
        dataset.status = 'ERROR'
        dataset.save()

    return response_data

def balance_dataset(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    return X_resampled, y_resampled

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def create_and_train_pipeline(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    classification_report_str = classification_report(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return {
        "classification_report": classification_report_str,
        "confusion_matrix_image": img_str
    }
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

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