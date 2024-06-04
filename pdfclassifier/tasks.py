from celery import shared_task
import os
import fitz  # PyMuPDF para manejar PDFs
from django.conf import settings
import joblib

@shared_task
def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    processed_text = procesar_texto(text)
    classifier = joblib.load(os.path.join(settings.BASE_DIR, 'classifier.joblib'))
    vectorizer = joblib.load(os.path.join(settings.BASE_DIR, 'vectorizer.joblib'))
    X_new = vectorizer.transform([processed_text])
    prediction = classifier.predict(X_new)
    predicted_label = prediction[0]
    new_filename = f"{predicted_label} {os.path.basename(pdf_path)}"
    new_path = os.path.join(settings.BASE_DIR, new_filename)
    os.rename(pdf_path, new_path)
    return new_filename

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def procesar_texto(texto, idioma='spanish'):
    texto = texto.lower()
    tokens = word_tokenize(texto, language=idioma)
    tokens_limpios = [t for t in tokens if t.isalpha() and t not in stopwords.words(idioma)]
    texto_procesado = ' '.join(tokens_limpios)
    return texto_procesado
