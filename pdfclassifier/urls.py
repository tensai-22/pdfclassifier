# pdfclassifier/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_model, name='train_model'),
    path('predict/', views.predict_pdf, name='predict_pdf'),
    path('upload/', views.upload_pdf, name='upload_pdf'),
]
