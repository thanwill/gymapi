import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import plotly.express as px
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go


def realizar_analise(df, features, target):
    # Verificar se a coluna alvo está no DataFrame
    if target not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(f"Column '{target}' not found in dataset. Available columns are: {available_columns}")

    # Verificar se todas as features estão no DataFrame
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        available_columns = df.columns.tolist()
        raise ValueError(f"Features {missing_features} not found in dataset. Available columns are: {available_columns}")

    X = df[features]
    y = df[target]

    # Identificar features numéricas e categóricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pré-processamento para features numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pré-processamento para features categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinar pré-processamentos
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    if is_categorical_dtype(df[target]) or df[target].dtype == 'object':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model_type = 'classifier'
    elif is_numeric_dtype(df[target]):
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_type = 'regressor'
    else:
        raise ValueError("Tipo de rótulo desconhecido. Talvez você esteja tentando ajustar um classificador em um alvo de regressão com valores contínuos.")

    # Criar pipeline completo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinar pipeline
    pipeline.fit(X_train, y_train)

    return pipeline, X_test, y_test, model_type    

def plot_confusion_matrix_old(pipeline, X_test, y_test, labels):
    # Fazer previsões
    y_pred = pipeline.predict(X_test)

    # Calcular matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # Criar um DataFrame para a matriz de confusão
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Gerar o heatmap da matriz de confusão
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues', labels=dict(x="Predicted", y="Actual", color="Count"))

    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted Label', yaxis_title='True Label')

    # Salvar a figura em um buffer de bytes
    buf = BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)

    # Codificar a imagem em base64
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')

    return img_base64

def plot_confusion_matrix(pipeline, X_test, y_test, labels, image_name):
    # Obter previsões
    y_pred = pipeline.predict(X_test)
    
    # Calcular a matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    # Criar a figura usando Plotly
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        hoverongaps=False,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title='Matriz de Confusão',
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label')
    )
    
    # Definir o caminho completo para salvar a imagem
    image_path = f"media/images/{image_name}"
    
    # Salvar a figura como imagem
    fig.write_image(image_path)
    
    return image_path