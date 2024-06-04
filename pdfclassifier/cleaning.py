import os
import fitz  # PyMuPDF, necesario para manejar archivos PDF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Asegúrate de descargar los recursos necesarios para NLTK la primera vez
nltk.download('punkt')
nltk.download('stopwords')

# Función para extraer texto de un archivo PDF
def extraer_texto_pdf(ruta_pdf):
    doc = fitz.open(ruta_pdf)  # Abre el documento PDF
    texto = ""
    for pagina in doc:  # Recorre cada página del PDF
        texto += pagina.get_text("text")  # Asegúrate de que se use el método correcto para extraer texto
    doc.close()  # Cierra el documento PDF
    return texto

# Función para procesar texto eliminando stopwords y filtrando tokens no alfabéticos
def procesar_texto(texto, idioma='spanish'):
    texto = texto.lower()  # Convierte el texto a minúsculas para estandarizar
    tokens = word_tokenize(texto, language=idioma)  # Tokeniza el texto
    tokens_limpios = [t for t in tokens if t.isalpha() and t not in stopwords.words(idioma)]  # Filtra tokens
    return ' '.join(tokens_limpios)  # Une los tokens limpios en un solo string

# Función para guardar el texto procesado en un nuevo archivo
def guardar_texto_en_archivo(texto, ruta_destino, nombre_archivo):
    ruta_archivo_texto = os.path.join(ruta_destino, f"{nombre_archivo}_limpio.txt")
    with open(ruta_archivo_texto, "w", encoding="utf-8") as archivo_texto:
        archivo_texto.write(texto)  # Escribe el texto en el archivo

# Directorio que contiene los PDFs
directorio_pdfs = r'C:\Users\user\OneDrive\Desktop\DJANGO\clasificador\TextosExtraidos-20240603T215751Z-001\TextosExtraidos'

print("Iniciando el procesamiento de archivos PDF...")
# Procesar todos los archivos PDF en el directorio y subdirectorios
for raiz, dirs, archivos in os.walk(directorio_pdfs):
    for archivo in archivos:
        if archivo.lower().endswith('.pdf'):
            print(f"Procesando el archivo: {archivo}")
            ruta_pdf = os.path.join(raiz, archivo)
            texto_pdf = extraer_texto_pdf(ruta_pdf)
            texto_procesado = procesar_texto(texto_pdf)
            guardar_texto_en_archivo(texto_procesado, raiz, os.path.splitext(archivo)[0])
            print(f"Archivo procesado y guardado como: {os.path.splitext(archivo)[0]}_limpio.txt")

print("Procesamiento completado.")
