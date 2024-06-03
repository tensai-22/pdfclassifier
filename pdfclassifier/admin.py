# pdfclassifier/admin.py

from django.contrib import admin
from .models import PDFClassification

admin.site.register(PDFClassification)
