import os
import fitz  # PyMuPDF para manejar PDFs
import nltk
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
from django.views.decorators.csrf import csrf_exempt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import zipfile
import io

# Paquetes necesarios de NLTK estén disponibles
nltk.download('punkt')
nltk.download('stopwords')

# Función para procesar texto
def procesar_texto(texto, idioma='spanish'):
    texto = texto.lower()  # Convertir a minúsculas
    tokens = word_tokenize(texto, language=idioma)  # Tokenizar
    tokens_limpios = [t for t in tokens if t.isalpha() and t not in stopwords.words(idioma)]  # Eliminar stopwords y no alfabéticos
    texto_procesado = ' '.join(tokens_limpios)  # Reconstruir el texto
    return texto_procesado

# Función para extraer texto de un PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
    except Exception as e:
        print(f"Error al procesar {pdf_file.name}: {e}")
    return text

# Vista principal
def index(request):
    return HttpResponse("Bienvenido al Clasificador de PDFs")

# Vista para mostrar el formulario de subida
def upload_pdf(request):
    print("Vista upload_pdf llamada")  # Mensaje de depuración
    try:
        response = render(request, 'pdfclassifier/upload.html')
        print(f"Renderizó correctamente la plantilla. Contenido: {response.content[:500]}")
        return response
    except Exception as e:
        print(f"Error al renderizar la plantilla: {e}")
        return HttpResponse(f"Error al renderizar la plantilla: {e}")

# Vista para predecir la clasificación de nuevos PDFs
@csrf_exempt
def predict_pdf(request):
    if request.method == 'POST':
        if 'pdfs' not in request.FILES:
            return JsonResponse({'error': 'No files provided'}, status=400)
        
        pdf_files = request.FILES.getlist('pdfs')
        zip_buffer = io.BytesIO()
        
        # Cargar el modelo y el vectorizador guardados
        classifier = joblib.load(os.path.join(settings.BASE_DIR, 'classifier.joblib'))
        vectorizer = joblib.load(os.path.join(settings.BASE_DIR, 'vectorizer.joblib'))

        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED) as zf:
            for pdf_file in pdf_files:
                text = extract_text_from_pdf(pdf_file)
                processed_text = procesar_texto(text)
                
                X_new = vectorizer.transform([processed_text])
                prediction = classifier.predict(X_new)
                
                predicted_label = prediction[0]
                new_filename = f"{predicted_label} {pdf_file.name}"
                
                pdf_file.seek(0)  # Reset the file pointer to the beginning
                zf.writestr(new_filename, pdf_file.read())
        
        zip_buffer.seek(0)
        response = HttpResponse(zip_buffer, content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="classified_pdfs.zip"'
        return response
    
    return HttpResponse("Método no permitido", status=405)

# Vista para entrenar el modelo
def train_model(request):
    directory = r'C:\Users\user\OneDrive\Desktop\DJANGO\clasificador\TextosExtraidos-20240603T215751Z-001\TextosExtraidos'
    
    texts = []
    labels = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_limpio.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                texts.append(text)
                label = os.path.basename(root)
                labels.append(label)
    
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Guardar el modelo entrenado y el vectorizador en un archivo
    joblib.dump(classifier, os.path.join(settings.BASE_DIR, 'classifier.joblib'))
    joblib.dump(vectorizer, os.path.join(settings.BASE_DIR, 'vectorizer.joblib'))
    
    return JsonResponse({'accuracy': accuracy})
