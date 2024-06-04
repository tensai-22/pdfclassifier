import os
import fitz  # PyMuPDF para manejar PDFs
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Especificar la ruta al directorio nltk_data
nltk.data.path.append(r"C:\Users\jdavila\Desktop\Tokenización-20240603T215751Z-001\Tokenización")

# Verificar si los recursos de NLTK están disponibles localmente
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("No se pudo encontrar los recursos de NLTK en las rutas especificadas. Verifique la configuración.")

# Cargar el vectorizador y el clasificador entrenados
vectorizer = joblib.load('vectorizer.joblib')
classifier = joblib.load('classifier.joblib')

# Función para extraer texto de un PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        print(f"Error al procesar {pdf_path}: {e}")
        text = ""
    return text

# Función para procesar el texto de un PDF
def preprocess_text(text, idioma='spanish'):
    texto = text.lower()  # Convertir a minúsculas
    tokens = word_tokenize(texto, language=idioma)  # Tokenizar
    stopwords_idioma = stopwords.words(idioma)
    tokens_limpios = [t for t in tokens if t.isalpha() and t not in stopwords_idioma]  # Eliminar stopwords y no alfabéticos
    texto_procesado = ' '.join(tokens_limpios)  # Reconstruir el texto
    return texto_procesado

# Directorio donde se encuentran los PDFs a clasificar
directory = r'C:\Users\jdavila\Downloads\casillas.pj.gob.pe'

# Iterar sobre los PDFs en el directorio
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.pdf'):  # Solo procesar archivos PDF
            pdf_path = os.path.join(root, file)
            text = extract_text_from_pdf(pdf_path)
            if text.strip():  # Verifica que hay texto y no solo espacios
                texto_preprocesado = preprocess_text(text)
                if texto_preprocesado:  # Verifica que el preprocesamiento resulte en texto útil
                    texto_vectorizado = vectorizer.transform([texto_preprocesado])
                    prediccion = classifier.predict(texto_vectorizado)
                    nuevo_nombre_pdf = f"{prediccion[0]} {file}"
                    nuevo_path_pdf = os.path.join(root, nuevo_nombre_pdf)
                    os.rename(pdf_path, nuevo_path_pdf)
                    print(f'El PDF {file} fue clasificado como: {prediccion} y renombrado a {nuevo_nombre_pdf}')
                else:
                    print(f'El PDF {file} parece ser un documento escaneado sin texto reconocible.')
            else:
                print(f'El PDF {file} parece ser un documento escaneado sin texto reconocible.')
