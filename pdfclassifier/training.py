import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Directorio donde se encuentran los PDFs limpios y clasificados
directory = r'C:\Users\user\OneDrive\Desktop\DJANGO\clasificador\TextosExtraidos-20240603T215751Z-001\TextosExtraidos'
texts = []  # Listas para almacenar los textos
labels = []  # Listas para almacenar las etiquetas

# Iterar sobre las carpetas en el directorio
for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('_limpio.txt'):  # Solo procesar archivos limpios
            # Extraer el texto del archivo
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            texts.append(text)
            # Obtener la etiqueta del nombre de la carpeta
            label = os.path.basename(root)
            labels.append(label)

# Crear el vectorizador y vectorizar los textos
vectorizer = TfidfVectorizer(max_features=5000)  # Ajusta el número de características según sea necesario
X = vectorizer.fit_transform(texts)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Entrenamiento del clasificador (Random Forest como ejemplo)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Predicción en el conjunto de prueba y evaluación del modelo
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Guardar el vectorizador y el clasificador para uso futuro
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(classifier, 'classifier.joblib')
