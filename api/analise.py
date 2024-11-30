import pandas as pd
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def realizar_analise(df, target):
    # Verificar se a coluna alvo está no DataFrame
    if target not in df.columns:
        available_columns = df.columns.tolist()
        raise ValueError(f"Column '{target}' not found in dataset. Available columns are: {available_columns}")

    # Convertendo as variáveis categóricas em numéricas
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(target, axis=1)
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