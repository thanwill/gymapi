from django.urls import path
from .views import UploadDatasetView, ListDatasetsView, CreateAnalysisView, AnalysisResultsView, ListDatasetsViewByID, RemoveAnalysesView, RemoveDatasetView, GetCorrelationsView , FilteredAnalysisResultsView, CreateMultipleAnalysesView, PredictView, index

urlpatterns = [        
    
    path('', index, name='index'),
    
    path('datasets/upload', UploadDatasetView.as_view(), name='upload-dataset'),
    path('datasets', ListDatasetsView.as_view(), name='list-datasets'),
    path('datasets/<str:dataset_id>', ListDatasetsViewByID.as_view(), name='list-datasets-by-id'),
    path('datasets/<str:dataset_id>/remove', RemoveDatasetView.as_view(), name='remove-dataset'),
    
    path('analyses', AnalysisResultsView.as_view(), name='analysis-results'),
    path('analyses/filtered', FilteredAnalysisResultsView.as_view(), name='filtered-analysis-results'),
    path('analyses/multiple', CreateMultipleAnalysesView.as_view(), name='create-multiple-analyses'),
    path('analyses/predict', PredictView.as_view(), name='predict'),
    path('analyses/<str:dataset_id>', CreateAnalysisView.as_view(), name='create-analysis'),
    path('analyses/<str:analysis_id>/remove', RemoveAnalysesView.as_view(), name='remove-analysis'),
    path('analyses/<str:dataset_id>/correlations', GetCorrelationsView.as_view(), name='get-correlations'),
]