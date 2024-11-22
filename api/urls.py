from django.urls import path
from .views import UploadDatasetView, ListDatasetsView, CreateAnalysisView, AnalysisResultsView

urlpatterns = [        
    path('datasets/upload', UploadDatasetView.as_view(), name='upload-dataset'),
    path('datasets', ListDatasetsView.as_view(), name='list-datasets'),
    path('analyses', CreateAnalysisView.as_view(), name='create-analysis'),
    path('results/<str:analysis_id>', AnalysisResultsView.as_view(), name='analysis-results'),
]