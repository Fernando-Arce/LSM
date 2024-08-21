import os, cv2
import numpy as np
import pandas as pd
import ast
from mediapipe.python.solutions.holistic import HAND_CONNECTIONS, FACEMESH_CONTOURS, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_utils import draw_landmarks, DrawingSpec
from typing import NamedTuple

def resolucion_webcam(camara):
    """
    Configura la resolución y los fotogramas por segundo (FPS) de la cámara web.

    Este método ajusta la calidad de la captura de video para asegurar que la cámara web esté configurada en una resolución
    de 1280x720 píxeles y 30 FPS, proporcionando una calidad de video adecuada para la detección de puntos clave y otras tareas.

    Args:
        camara (cv2.VideoCapture): Objeto de captura de video que representa la cámara web.
    """
    # Configurar la resolución de la cámara web a 1280x720 píxeles
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Configurar la tasa de fotogramas (FPS) a 30
    camara.set(cv2.CAP_PROP_FPS, 30)

def crear_carpeta(path):
    """
    Crea una carpeta en la ruta especificada si no existe.

    Si la carpeta ya existe, no se realiza ninguna acción. Esta función asegura que la estructura de directorios necesaria
    esté disponible para guardar archivos.

    Args:
        path (str): Ruta de la carpeta que se desea crear.
    """
    if not os.path.exists(path):
        os.makedirs(path)  # Crear la carpeta y cualquier directorio padre necesario
        print(f"Carpeta creada: {path}")

def save_frames(frames, carpeta_salida):
    """
    Guarda una lista de fotogramas en la carpeta de salida especificada.

    Los fotogramas se guardan en formato JPG y se nombran secuencialmente en función de su orden en la lista. Cada 
    fotograma se almacena con un nombre que sigue el patrón 'frame_X.jpg', donde 'X' es el número del fotograma.

    Args:
        frames (list): Lista de fotogramas (imágenes en formato numpy array) que se desean guardar.
        carpeta_salida (str): Ruta de la carpeta donde se guardarán los fotogramas.
    """
    for num_frame, frame in enumerate(frames):
        # Construir el nombre del archivo basado en el número del fotograma
        filename = os.path.join(carpeta_salida, f'frame_{num_frame + 1}.jpg')
        
        # Guardar el fotograma en la carpeta de salida
        cv2.imwrite(filename, frame)

def mediapipe_detection(frame, model):
    """
    Realiza la detección de puntos clave en una imagen utilizando el modelo Holistic de MediaPipe.

    Convierte el frame de BGR a RGB para la detección, realiza el procesamiento con el modelo Holistic y luego
    convierte la imagen de nuevo a BGR para su visualización. Esto es necesario porque OpenCV utiliza el formato BGR
    para manejar imágenes, mientras que MediaPipe opera en el espacio de color RGB.

    Args:
        frame (numpy.ndarray): Imagen de entrada capturada desde la cámara.
        model (mediapipe.python.solutions.holistic.Holistic): Modelo Holistic de MediaPipe para la detección de puntos clave.

    Returns:
        tuple: Retorna la imagen procesada en formato BGR y los resultados de la detección (landmarks y conexiones).
    """
    # Convertir la imagen de BGR (formato de OpenCV) a RGB (formato esperado por MediaPipe)
    imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen.flags.writeable = False  # Marcar la imagen como no escribible para mejorar el rendimiento durante el procesamiento
    resultado = model.process(imagen)  # Realizar la detección con el modelo Holistic
    imagen.flags.writeable = True  # Restaurar la capacidad de escritura de la imagen
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)  # Convertir la imagen de vuelta a BGR para su visualización con OpenCV
    return imagen, resultado

def dibujar_keypoints(imagen, resultado):
    """
    Dibuja los puntos clave (landmarks) y sus conexiones en la imagen utilizando los resultados del modelo Holistic.

    Se dibujan los puntos clave para la cara, el cuerpo y ambas manos con colores y estilos específicos. Cada conjunto 
    de puntos clave (cara, pose, manos) se dibuja utilizando un esquema de color y grosor de línea que facilita la visualización.

    Args:
        imagen (numpy.ndarray): Imagen en la que se dibujarán los puntos clave.
        resultado (mediapipe.python.solutions.holistic.HolisticResults): Resultados del modelo Holistic, incluyendo landmarks y conexiones.
    """
    # Dibujar los puntos clave de la cara con colores y grosores específicos
    draw_landmarks(
        imagen,
        resultado.face_landmarks,
        FACEMESH_CONTOURS,
        DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    
    # Dibujar los puntos clave del cuerpo (pose) con colores y grosores específicos
    draw_landmarks(
        imagen,
        resultado.pose_landmarks,
        POSE_CONNECTIONS,
        DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    
    # Dibujar los puntos clave de la mano izquierda con colores y grosores específicos
    draw_landmarks(
        imagen,
        resultado.left_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    
    # Dibujar los puntos clave de la mano derecha con colores y grosores específicos
    draw_landmarks(
        imagen,
        resultado.right_hand_landmarks,
        HAND_CONNECTIONS,
        DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

def there_hand(resultado: NamedTuple) -> bool:
    """
    Verifica si hay manos detectadas en los resultados del modelo Holistic.

    Evalúa los resultados del modelo Holistic para determinar si se detectaron landmarks de la mano izquierda o derecha.
    La función retorna `True` si al menos una de las manos es detectada y `False` si no se detecta ninguna.

    Args:
        resultado (mediapipe.python.solutions.holistic.HolisticResults): Resultados del modelo Holistic.

    Returns:
        bool: Retorna True si se detectó al menos una mano, False en caso contrario.
    """
    # Retorna True si se detecta una mano (izquierda o derecha), de lo contrario, False
    return resultado.left_hand_landmarks or resultado.right_hand_landmarks

def get_actions(path):
    """
    Obtiene una lista de acciones disponibles en un directorio, basándose en los archivos HDF5 presentes.

    Recorre el directorio especificado y extrae los nombres de las acciones a partir de los archivos con extensión ".h5".
    Los nombres de las acciones se derivan de los nombres de los archivos sin la extensión.

    Args:
        path (str): Ruta del directorio que contiene los archivos HDF5 con las acciones.

    Returns:
        list: Lista de nombres de las acciones (sin la extensión .h5).
    """
    out = []
    for action in os.listdir(path):
        name, ext = os.path.splitext(action)  # Separar el nombre del archivo de su extensión
        if ext == ".h5":
            out.append(name)  # Añadir el nombre del archivo (sin la extensión) a la lista de acciones
    return out

def extract_keypoints(results):
    """
    Extrae los puntos clave (keypoints) de los resultados del modelo Holistic y los organiza en un solo vector.

    Se extraen los keypoints para el cuerpo (pose), la cara (face) y ambas manos (left hand y right hand). Si algún conjunto de 
    keypoints no está presente en los resultados, se completa con ceros, asegurando que el vector final tenga siempre la misma longitud.

    Args:
        results (mediapipe.python.solutions.holistic.HolisticResults): Resultados del modelo Holistic, incluyendo landmarks y conexiones.

    Returns:
        numpy.ndarray: Vector que contiene todos los keypoints extraídos, organizados de forma plana.
    """
    # Extraer los keypoints del cuerpo, o llenar con ceros si no hay datos de pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    
    # Extraer los keypoints de la cara, o llenar con ceros si no hay datos de la cara
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    
    # Extraer los keypoints de la mano izquierda, o llenar con ceros si no hay datos de la mano izquierda
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    
    # Extraer los keypoints de la mano derecha, o llenar con ceros si no hay datos de la mano derecha
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    
    # Concatenar todos los keypoints en un solo vector
    return np.concatenate([pose, face, lh, rh])

def get_keypoints(model, path):
    """
    Obtiene la secuencia de puntos clave para una muestra de imágenes en una carpeta específica.

    Carga cada imagen en la carpeta, realiza la detección de puntos clave con el modelo Holistic de MediaPipe y almacena 
    la secuencia resultante en una lista. Esta lista puede ser utilizada para análisis o como entrada para modelos de 
    aprendizaje profundo.

    Args:
        model (mediapipe.python.solutions.holistic.Holistic): Modelo Holistic de MediaPipe para la detección de puntos clave.
        path (str): Ruta de la carpeta que contiene las imágenes de la muestra.

    Returns:
        list: Lista de secuencias de puntos clave extraídos de las imágenes de la carpeta.
    """
    kp_seq = []
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)  # Construir la ruta completa de la imagen
        frame = cv2.imread(img_path)  # Cargar la imagen desde la ruta
        _, results = mediapipe_detection(frame, model)  # Realizar la detección de puntos clave usando el modelo Holistic
        kp_frame = extract_keypoints(results)  # Extraer los puntos clave del resultado
        # Convertir los puntos clave a lista y añadir a la secuencia
        kp_seq.append(kp_frame.tolist())
    return kp_seq

def insert_keypoints_sequence(df, n_sample: int, kp_seq):
    """
    Inserta una secuencia de puntos clave en un DataFrame, asociando cada secuencia con un número de muestra.

    Cada punto clave de la secuencia se guarda como una entrada en el DataFrame, con el número de muestra y el número de frame
    correspondiente. Esto permite organizar los puntos clave por muestra y frame para su posterior análisis o modelado.

    Args:
        df (pandas.DataFrame): DataFrame donde se insertarán los puntos clave.
        n_sample (int): Número de muestra asociada a la secuencia.
        kp_seq (list): Lista de secuencias de puntos clave.

    Returns:
        pandas.DataFrame: DataFrame actualizado con las secuencias de puntos clave insertadas.
    """
    for frame, keypoints in enumerate(kp_seq):
        # Crear un diccionario con los datos de la muestra, el número de frame y los puntos clave
        data = {'sample': n_sample, 'frame': frame + 1, 'keypoints': str(keypoints)}  # Convertimos los puntos clave a cadena
        
        # Convertir el diccionario en un DataFrame
        df_keypoints = pd.DataFrame([data])
        
        # Concatenar el DataFrame con la nueva entrada al DataFrame existente
        df = pd.concat([df, df_keypoints], ignore_index=True)
    
    return df

def get_sequences_and_labels(acciones, ruta_datos):
    """
    Obtiene las secuencias de puntos clave y las etiquetas asociadas para un conjunto de acciones.

    Carga los archivos HDF5 correspondientes a cada acción, extrae las secuencias de puntos clave y asigna una etiqueta
    numérica a cada secuencia. Estas secuencias se utilizan posteriormente para entrenar modelos de aprendizaje profundo.

    Args:
        acciones (list): Lista de nombres de las acciones.
        ruta_datos (str): Ruta de la carpeta que contiene los archivos HDF5 con los puntos clave y las etiquetas.

    Returns:
        tuple: Dos listas. La primera contiene las secuencias de puntos clave, y la segunda, las etiquetas correspondientes.
    """
    sequences, labels = [], []
    for label, action in enumerate(acciones):
        # Ruta del archivo HDF5 para la acción actual
        hdf_path = os.path.join(ruta_datos, f"{action}.h5")
        
        # Cargar los datos desde el archivo HDF5
        data = pd.read_hdf(hdf_path, key='data')
        
        # Agrupar los datos por 'sample' y extraer las secuencias de puntos clave
        for _, data_filtered in data.groupby('sample'):
            seq = [ast.literal_eval(fila['keypoints']) for _, fila in data_filtered.iterrows()]
            
            # Verificar que la secuencia no esté vacía antes de añadirla
            if len(seq) > 0:
                sequences.append(seq)
                labels.append(label)
            else:
                print(f"Advertencia: Secuencia vacía para la acción {action}")
    
    return sequences, labels