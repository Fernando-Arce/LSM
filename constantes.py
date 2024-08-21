import os, cv2

# Declaración de variables básicas para el funcionamiento del programa

# Rutas para el proyecto
RUTA_RAIZ = os.path.dirname(os.path.abspath(__file__))  # Ruta del directorio raíz del proyecto
RUTA_FPS_ACCIONES = os.path.join(RUTA_RAIZ, "frames_acciones")  # Ruta para las acciones en frames
RUTA_ESTATICAS = os.path.join(RUTA_RAIZ, "frames_estaticos")  # Ruta para las imágenes estáticas
RUTA_DATOS = os.path.join(RUTA_RAIZ, "data")  # Ruta para el procesamiento y almacenamiento de archivos .h5
RUTA_MODEL = os.path.join(RUTA_RAIZ, "models")  # Ruta para guardar los modelos de IA

# Número máximo y mínimo de frames a evaluar
MAX_LENGTH_FRAMES = 15  # Número máximo de frames utilizados en la evaluación de secuencias
MIN_LENGTH_FRAMES = 5  # Número mínimo de frames necesarios para realizar una evaluación válida

# Datos para la creación del modelo de IA
LENGTH_KEYPOINTS = 1662  # Longitud de los keypoints (pendiente de determinar completamente)
NOMBRE_MODELO = f"acciones_{MAX_LENGTH_FRAMES}.keras"  # Nombre del modelo que se generará

# Parámetros básicos para mostrar texto en las imágenes de OpenCV
FUENTE = cv2.FONT_HERSHEY_PLAIN  # Fuente utilizada para mostrar texto en imágenes
FUENTE_SIZE = 1.5  # Tamaño de la fuente
FUENTE_POS = (10, 465)  # Posición del texto en la parte inferior izquierda de la imagen