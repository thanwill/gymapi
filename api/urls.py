from django.urls import path
from .views import UploadDatasetView, ListDatasetsView, CreateAnalysisView, AnalysisResultsView, ListDatasetsViewByID, RemoveDatasetView

urlpatterns = [        
    path('datasets/upload', UploadDatasetView.as_view(), name='upload-dataset'),
    path('datasets', ListDatasetsView.as_view(), name='list-datasets'),
    path('datasets/<str:dataset_id>', ListDatasetsViewByID.as_view(), name='list-datasets-by-id'),
    path('datasets/<str:dataset_id>/remove', RemoveDatasetView.as_view(), name='remove-dataset'),
    path('analyses/<str:dataset_id>', CreateAnalysisView.as_view(), name='create-analysis'),
    path('results/<str:analysis_id>', AnalysisResultsView.as_view(), name='analysis-results'),
]