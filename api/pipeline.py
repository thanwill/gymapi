from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# ...existing imports...

def create_and_train_pipeline(X_train, y_train, model_type='classifier'):
    if model_type == 'classifier':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'regressor':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'categorical':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'continuous':
        model = RandomForestRegressor(random_state=42)
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=42)
    elif model_type == 'svm':
        model = SVC(random_state=42)
    else:
        raise ValueError("Tipo de modelo desconhecido. Use 'classifier', 'regressor', 'categorical', 'continuous', 'logistic' ou 'svm'.")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline
