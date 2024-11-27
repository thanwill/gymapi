from django.db import models

# criar um modelo para o dataset
class Dataset(models.Model):
    # id, filename, uploaded_at, columns (json), status (PENDING, PROCESSED, FAILED) (enum), error_message (string, opcional)
    id = models.AutoField(primary_key=True)
    filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    columns = models.JSONField()
    status = models.CharField(max_length=255)
    error_message = models.CharField(max_length=255)    
    
    def __str__(self):
        return self.filename
    
# criar um modelo para as análises
class Analyses(models.Model):
    # id, dataset_id (fk), analysis_type (enum: VISUALIZATION, PREDICTION), parameters (json), created_at, status (enum: PENDING, COMPLETED, FAILED), error_message (string, opcional)
    
    id = models.AutoField(primary_key=True)
    dataset_id = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    analysis_type = models.CharField(max_length=255)
    parameters = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=255, choices=[
        ('PENDING', 'Pending'),
        ('PROCESSING', 'Processing'),
        ('PROCESSED', 'Processed'),
        ('ERROR', 'Error'),
    ])
    error_message = models.CharField(max_length=255)    
    
    def __str__(self):
        return self.id
    
class ColumnMetadata(models.Model):
    dataset = models.ForeignKey(Dataset, related_name='column_metadata', on_delete=models.CASCADE)
    column_name = models.CharField(max_length=255)
    data_type = models.CharField(max_length=50)
    description = models.TextField(blank=True, null=True)

    def __str__(self):
        return f"{self.column_name} - {self.data_type}"
    
# criar um modelo para os resultados
# id, analysus_id (fk), result_data (json), created_at datetime
class Results(models.Model):
    id = models.AutoField(primary_key=True)
    analysis_id = models.ForeignKey(Analyses, on_delete=models.CASCADE)
    result_data = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.id