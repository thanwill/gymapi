from django.urls import path
from .views import (
    UploadDatasetView, 
    ListDatasetsView, 
    CreateAnalysisView, 
    AnalysisResultsView, 
    ListDatasetsViewByID, 
    RemoveAnalysesView, 
    RemoveDatasetView, 
    GetCorrelationsView, 
    FilteredAnalysisResultsView, 
    CreateMultipleAnalysesView,     
    InsightsView, 
    GetInsightsTypesView, 
    GetImageView, 
    DownloadDatasetView,
    PredictByAttributeView, 
    PredictResultView,
    index, 
)

urlpatterns = [        
    
    path('', index, name='index'),
    
    path('datasets/upload', UploadDatasetView.as_view(), name='upload-dataset'),
    path('datasets', ListDatasetsView.as_view(), name='list-datasets'),
    path('datasets/<str:dataset_id>', ListDatasetsViewByID.as_view(), name='list-datasets-by-id'),
    path('datasets/<str:dataset_id>/remove', RemoveDatasetView.as_view(), name='remove-dataset'),
    path('datasets/<str:dataset_id>/download', DownloadDatasetView.as_view(), name='download-dataset'),
    
    path('analyses', AnalysisResultsView.as_view(), name='analysis-results'),
    path('analyses/filtered', FilteredAnalysisResultsView.as_view(), name='filtered-analysis-results'),
    path('analyses/multiple', CreateMultipleAnalysesView.as_view(), name='create-multiple-analyses'),    
    path('analyses/<str:analysis_id>/remove', RemoveAnalysesView.as_view(), name='remove-analysis'),
    path('analyses/<str:dataset_id>/correlations', GetCorrelationsView.as_view(), name='get-correlations'),
    
    path('analyses/predict/create', CreateAnalysisView.as_view(), name='create-analysis'),
    path('analyses/predict/result', PredictResultView.as_view(), name='predict-result'),
    
    
    path('insights', InsightsView.as_view(), name='insights'),
    path('insights/types', GetInsightsTypesView.as_view(), name='get-insights-types'),
    
    path('images/<str:image_name>', GetImageView.as_view(), name='get-image'),
    # endpoint resulta PredictByAttributeView 
    path('results/predict/attribute', PredictByAttributeView.as_view(), name='predict-by-attribute'),
]