import os, cv2
import numpy as np
import pandas as pd
import tkinter as tk
import mediapipe as mp
import pickle
import math
import time
import threading
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
from natsort import natsorted
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
from keras.src.utils.sequence_utils import pad_sequences
from keras.src.utils.numerical_utils import to_categorical
from keras.src.saving.saving_api import load_model
from entrenamiento_lsm import start_training
from model import NUM_EPOCH, get_model
from constantes import RUTA_RAIZ, RUTA_FPS_ACCIONES, RUTA_ESTATICAS, RUTA_MODEL, RUTA_DATOS, NOMBRE_MODELO
from constantes import FUENTE, FUENTE_POS, FUENTE_SIZE, MAX_LENGTH_FRAMES, MIN_LENGTH_FRAMES
from helpers import crear_carpeta, save_frames, get_actions, insert_keypoints_sequence, get_sequences_and_labels
from helpers import mediapipe_detection, there_hand, dibujar_keypoints, extract_keypoints, get_keypoints, resolucion_webcam
from text_to_speach import text_to_speech

def capturar_frames(path, frames_ignorados=3, cantidad_minima_frames=5):
    """
    Captura secuencias de frames desde la cámara en tiempo real y las guarda en una carpeta específica.

    Durante la captura, se utilizan técnicas de detección de manos para identificar si hay una mano presente en el cuadro.
    Solo se guardan los frames donde se detecta una mano, y se ignoran un número específico de frames al inicio y al final
    de la secuencia. Si se detecta una secuencia válida (superando un umbral mínimo de frames), se guarda en una nueva carpeta.

    Args:
        path (str): Ruta de la carpeta donde se guardarán las secuencias de frames capturadas.
        frames_ignorados (int): Cantidad de frames que se ignoran al comienzo y al final de la captura, para evitar ruido.
        cantidad_minima_frames (int): Número mínimo de frames requeridos para que una secuencia sea considerada válida y guardada.
    """
    # Crear la carpeta de destino si no existe
    crear_carpeta(path)
    numero_muestras = len(os.listdir(path))  # Número de muestras ya existentes en la carpeta
    contador_muestras = 0  # Contador de secuencias de frames válidas capturadas
    contador_frame = 0  # Contador de frames capturados en la secuencia actual
    frames = []  # Lista para almacenar los frames de una secuencia

    # Inicializar el modelo Holistic de MediaPipe
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)  # Abrir la cámara
        while video.isOpened():
            ret, frame = video.read()  # Capturar un frame de la cámara
            if not ret:
                break
            
            # Realizar la detección de MediaPipe en el frame capturado
            imagen, resultado = mediapipe_detection(frame, holistic_model)

            # Verificar si se ha detectado una mano en el frame
            if there_hand(resultado):
                contador_frame += 1
                # Ignorar los primeros 'frames_ignorados' frames para evitar ruido
                if contador_frame > frames_ignorados:
                    cv2.putText(imagen, 'Capturando.....', FUENTE_POS, FUENTE, FUENTE_SIZE, (255, 50, 0))
                    frames.append(np.asanyarray(frame))  # Añadir el frame a la lista de secuencias
            else:
                # Si hay una secuencia válida, guardarla
                if len(frames) > cantidad_minima_frames + frames_ignorados:
                    frames = frames[:-frames_ignorados]  # Remover los últimos 'frames_ignorados' frames
                    carpeta_salida = os.path.join(path, f"sample_{numero_muestras + contador_muestras + 1}")
                    crear_carpeta(carpeta_salida)  # Crear la carpeta para guardar la secuencia
                    save_frames(frames, carpeta_salida)  # Guardar los frames en la carpeta
                    contador_muestras += 1
                
                # Reiniciar los contadores para la siguiente secuencia
                frames = []
                contador_frame = 0
                cv2.putText(imagen, 'Listo para capturar...', FUENTE_POS, FUENTE, FUENTE_SIZE, (0, 220, 100))
                cv2.rectangle(imagen, (8, 18), (270, 0), (245, 117, 16), -1)
                cv2.putText(imagen, 'Presiona q para salir', (10, 16), FUENTE, FUENTE_SIZE, (80, 256, 121))

            # Dibujar los puntos clave (keypoints) en la imagen
            dibujar_keypoints(imagen, resultado)
            cv2.imshow(f'Toma de muestras para "{os.path.basename(path)}"', imagen)  # Mostrar el frame en la ventana

            # Salir si se presiona 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Liberar recursos
        video.release()
        cv2.destroyAllWindows()

class CaptureWindow(tk.Toplevel):
    """
    Clase para la ventana de captura de imágenes, utilizada para capturar gestos de la Lengua de Señas Mexicana.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Captura de Imágenes - Programa para Sordo Mudos")
        self.resizable(False, False)
        self.iconbitmap("logo.ico")

        self.parent.withdraw()

        ancho_ventana = 1000
        alto_ventana = 700

        ancho_pantalla = self.winfo_screenwidth()
        alto_pantalla = self.winfo_screenheight()

        posicion_x = (ancho_pantalla - ancho_ventana) // 2
        posicion_y = (alto_pantalla - alto_ventana) // 3

        self.geometry(f"{ancho_ventana}x{alto_ventana}+{posicion_x}+{posicion_y}")

        bg_image_path = "img/background_estatica.png"
        bg_image = ImageTk.PhotoImage(Image.open(bg_image_path))
        background_label = tk.Label(self, image=bg_image)
        background_label.image = bg_image
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        self.captura_label = tk.Label(self)
        self.captura_label.place(x=26, y=156, width=640, height=480)

        self.imagen_capturar_label = tk.Label(self)
        self.imagen_capturar_label.place(x=683, y=156, width=300, height=300)

        self.contador_label = tk.Label(self, text="0", font=("Arial", 20), bg="white", fg="black")
        self.contador_label.place(x=840, y=549, width=80, height=50)

        self.capturar()

    def capturar(self):
        try:
            nombre_palabra = simpledialog.askstring("LSM", "     Ingrese el nombre de la letra:     ")
            if nombre_palabra:
                folder = os.path.join(RUTA_ESTATICAS, nombre_palabra)

                if not os.path.exists(folder):
                    os.makedirs(folder)

                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                detector = HandDetector(maxHands=1)
                offset = 50
                imgSize = 300
                self.counter = 0
                self.dataset_size = 100
                self.capturing = False  # Bandera para controlar el inicio de la captura

                img_default = Image.open("img/default.png")
                img_default = img_default.resize((300, 300), Image.Resampling.LANCZOS)
                self.img_default_tk = ImageTk.PhotoImage(img_default)  # Hacerlo un atributo de la clase
                self.imagen_capturar_label.configure(image=self.img_default_tk)
                self.imagen_capturar_label.image = self.img_default_tk

                def update_frame():
                    success, img = cap.read()

                    if success:
                        hands, img_processed = detector.findHands(img, draw=False)
                        aspect_ratio = img.shape[1] / img.shape[0]
                        new_width = 640
                        new_height = int(new_width / aspect_ratio)

                        if new_height > 480:
                            new_height = 480
                            new_width = int(new_height * aspect_ratio)

                        img_resized = cv2.resize(img_processed, (new_width, new_height))

                        img_display = np.zeros((480, 640, 3), dtype=np.uint8)
                        y_offset = (480 - new_height) // 2
                        x_offset = (640 - new_width) // 2
                        img_display[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

                        img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
                        self.captura_label.img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_display_rgb))
                        self.captura_label.configure(image=self.captura_label.img_tk)

                        self.current_frame = img

                        # Si hay manos detectadas, adaptar y mostrar en imagen_capturar_label
                        if hands:
                            hand = hands[0]
                            x, y, w, h = hand['bbox']

                            y1 = max(0, y - offset)
                            y2 = min(img.shape[0], y + h + offset)
                            x1 = max(0, x - offset)
                            x2 = min(img.shape[1], x + w + offset)

                            imgCrop = img[y1:y2, x1:x2]
                            imgWhite_np = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Fondo blanco

                            aspectRatio = h / w

                            if aspectRatio > 1:
                                k = imgSize / h
                                wCal = math.ceil(k * w)
                                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                                wGap = math.ceil((imgSize - wCal) / 2)
                                imgWhite_np[:, wGap:wCal + wGap] = imgResize
                            else:
                                k = imgSize / w
                                hCal = math.ceil(k * h)
                                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                                hGap = math.ceil((imgSize - hCal) / 2)
                                imgWhite_np[hGap:hCal + hGap, :] = imgResize

                            imgWhite_rgb = cv2.cvtColor(imgWhite_np, cv2.COLOR_BGR2RGB)
                            imgWhite_tk = ImageTk.PhotoImage(image=Image.fromarray(imgWhite_rgb))
                            self.imagen_capturar_label.configure(image=imgWhite_tk)
                            self.imagen_capturar_label.image = imgWhite_tk

                        else:
                            # Mostrar imagen por defecto si no se detecta una mano
                            self.imagen_capturar_label.configure(image=self.img_default_tk)
                            self.imagen_capturar_label.image = self.img_default_tk

                        # Si se ha iniciado la captura, guardar la imagen
                        if self.capturing and self.counter < self.dataset_size:
                            cv2.imwrite(os.path.join(folder, f'image_{time.time()}.jpg'), self.current_frame)
                            self.counter += 1
                            self.contador_label.config(text=str(self.counter))
                            if self.counter >= self.dataset_size:
                                messagebox.showinfo("Información", f"Se han capturado {self.dataset_size} imágenes.")
                                quit_program(None)

                    self.after(5, update_frame)

                def start_capture(event):
                    self.capturing = True  # Iniciar la captura al presionar 's'

                def quit_program(event):
                    cap.release()
                    self.destroy()
                    self.parent.deiconify()
                    messagebox.showinfo("Información", f"Captura de muestras para '{nombre_palabra}' completada.")

                self.bind('<s>', start_capture)
                self.bind('<S>', start_capture)
                self.bind('<q>', quit_program)
                self.bind('<Q>', quit_program)

                self.after(10, update_frame)
            else:
                messagebox.showinfo("Información", "No se ingresó ningún nombre de palabra o gesto.")
                self.parent.deiconify()
                self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")
            self.parent.deiconify()
            self.destroy()

def start_capture():
    """
    Inicia el proceso de captura de gestos, preguntando al usuario si desea utilizar la interfaz de captura gráfica o el flujo de captura original.

    Dependiendo de la respuesta del usuario, esta función abre una ventana gráfica para capturar imágenes con detección de manos o
    inicia el flujo tradicional de captura de frames sin interfaz gráfica. En ambos casos, los gestos capturados se guardan en carpetas
    designadas por el usuario.
    """
    try:
        # Preguntar al usuario si quiere seguir el flujo original o usar la interfaz de captura
        respuesta = messagebox.askquestion("Seleccionar Interfaz", "¿Desea usar la interfaz de captura para letras?")
        
        if respuesta == 'yes':
            # Ejecutar la interfaz gráfica de captura con detección de manos
            CaptureWindow(root)
        else:
            # Ejecutar el flujo original sin interfaz gráfica
            nombre_palabra = simpledialog.askstring("LSM - Captura de gestos", "         Ingrese el nombre de la palabra o gesto:               ")
            if nombre_palabra:
                # Definir la ruta donde se guardarán los frames capturados
                ruta_palabra = os.path.join(RUTA_RAIZ, RUTA_FPS_ACCIONES, nombre_palabra)
                # Iniciar la captura de frames utilizando la función capturar_frames
                capturar_frames(ruta_palabra)
                # Mostrar un mensaje informativo al usuario indicando que la captura ha finalizado
                messagebox.showinfo("Información", f"Captura de muestras para '{nombre_palabra}' completada.")
    except Exception as e:
        # Manejar cualquier excepción que ocurra durante el proceso de captura
        messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

def crear_keypoints(ruta_frames, hdf_ruta):
    """
    Genera keypoints a partir de secuencias de frames capturados y los guarda en un archivo HDF5.

    Esta función recorre todas las secuencias de gestos almacenadas en una carpeta y, para cada secuencia, extrae los keypoints
    (puntos de interés) utilizando un modelo de Mediapipe. Los keypoints generados se organizan en un DataFrame de pandas y
    se guardan en un archivo HDF5.

    Args:
        ruta_frames (str): Ruta de la carpeta donde se encuentran las secuencias de frames de los gestos.
        hdf_ruta (str): Ruta del archivo HDF5 donde se guardarán los keypoints generados.
    """
    # Verificar si la ruta de frames existe
    if not os.path.exists(ruta_frames):
        messagebox.showinfo("¡Aviso!", f"La ruta {ruta_frames} no existe.")
        return
    
    # Verificar si el archivo HDF5 ya existe y preguntar si se debe sobrescribir
    if os.path.exists(hdf_ruta):
        respuesta = messagebox.askyesno("Archivo existente", f"El archivo {hdf_ruta} ya existe. ¿Desea sobrescribirlo?")
        if not respuesta:
            return
    
    # Crear un DataFrame vacío para almacenar los keypoints
    datos = pd.DataFrame([])
    
    # Inicializar el modelo Holistic de MediaPipe
    with Holistic() as holistic_model:
        # Recorrer cada carpeta (gesto) dentro de la ruta de frames
        for nombre_carpeta in os.listdir(ruta_frames):
            ruta_carpeta = os.path.join(ruta_frames, nombre_carpeta)
            if os.path.isdir(ruta_carpeta):
                # Ordenar los archivos de la carpeta
                archivos_ordenados = natsorted(os.listdir(ruta_carpeta))
                for nombre_muestra in archivos_ordenados:
                    ruta_muestra = os.path.join(ruta_carpeta, nombre_muestra)
                    if os.path.isfile(ruta_muestra):
                        # Extraer los keypoints de la secuencia de frames
                        secuencia_keypoints = get_keypoints(holistic_model, ruta_carpeta)
                        # Insertar los keypoints en el DataFrame
                        datos = insert_keypoints_sequence(datos, len(datos) + 1, secuencia_keypoints)
    
    # Guardar el DataFrame de keypoints en un archivo HDF5
    datos.to_hdf(hdf_ruta, key="data", mode="w")

def start_generate_keypoints():
    """
    Inicia el proceso de generación de keypoints en un hilo separado y muestra una ventana de progreso.

    Esta función abre una ventana de progreso mientras los keypoints se generan en segundo plano. Esto permite que la interfaz
    gráfica de la aplicación se mantenga interactiva durante el proceso de generación, evitando que la aplicación se congele.
    """
    # Ocultar la ventana principal mientras se muestra la ventana de progreso
    root.withdraw()

    # Crear una nueva ventana para mostrar el progreso
    progress_window = tk.Toplevel()
    progress_window.title("Progreso de Generación de Keypoints")

    # Calcular las dimensiones y posición de la ventana de progreso para centrarla en la pantalla
    ancho_pantalla = root.winfo_screenwidth()
    alto_pantalla = root.winfo_screenheight()
    ancho_ventana = 400
    alto_ventana = 100
    posicion_x = (ancho_pantalla - ancho_ventana) // 2
    posicion_y = (alto_pantalla - alto_ventana) // 2
    progress_window.geometry(f"{ancho_ventana}x{alto_ventana}+{posicion_x}+{posicion_y}")

    # Añadir un Label para mostrar un mensaje de procesamiento en la ventana de progreso
    processing_label = tk.Label(progress_window, text="   Sea paciente, se están procesando los \ndatos para los keypoints...")
    processing_label.pack(pady=10)

    # Iniciar el proceso de generación de keypoints en segundo plano
    generate_keypoints(progress_window)

def generate_keypoints(progress_window):
    """
    Genera keypoints para todas las secuencias de gestos almacenadas en una carpeta y los guarda en archivos HDF5.

    Esta función recorre todas las subcarpetas de una carpeta principal que contiene secuencias de gestos, y para cada
    subcarpeta, genera keypoints y los guarda en un archivo HDF5 separado. Al finalizar, se muestra una notificación al
    usuario y se cierra la ventana de progreso.

    Args:
        progress_window (tk.Toplevel): Ventana que muestra el progreso del proceso de generación de keypoints.
    """
    # Definir la ruta donde se encuentran las secuencias de gestos
    ruta_palabras = os.path.join(RUTA_RAIZ, RUTA_FPS_ACCIONES)

    # Recorrer cada subcarpeta (gesto) dentro de la ruta de palabras
    for nombre_palabra in os.listdir(ruta_palabras):
        ruta_palabra = os.path.join(ruta_palabras, nombre_palabra)
        hdf_ruta = os.path.join(RUTA_DATOS, f"{nombre_palabra}.h5")
        print(f'Creando los keypoints de "{nombre_palabra}"....')
        # Generar y guardar los keypoints para la secuencia actual
        crear_keypoints(ruta_palabra, hdf_ruta)

    # Mostrar un mensaje informando al usuario que el proceso ha terminado
    messagebox.showinfo("¡Aviso!", "Keypoints creados!")
    # Cerrar la ventana de progreso
    progress_window.destroy()
    # Restaurar la ventana principal
    root.deiconify()

def modelo_entrenamiento(ruta_datos, ruta_model):
    """
    Entrena un modelo de aprendizaje profundo utilizando secuencias de keypoints y guarda el modelo entrenado.

    Esta función carga secuencias de keypoints y sus etiquetas correspondientes, las prepara para el entrenamiento y
    entrena un modelo de red neuronal para clasificar los gestos de la Lengua de Señas Mexicana. Una vez finalizado el
    entrenamiento, el modelo se guarda en una ubicación especificada.

    Args:
        ruta_datos (str): Ruta donde se encuentran los datos de entrenamiento (secuencias de keypoints y etiquetas).
        ruta_model (str): Ruta donde se guardará el modelo entrenado.
    """
    # Obtener las acciones (gestos) disponibles para el entrenamiento
    acciones = get_actions(ruta_datos)
    
    # Cargar las secuencias de keypoints y sus etiquetas
    secuencias, etiquetas = get_sequences_and_labels(acciones, ruta_datos)
    
    # Ajustar las secuencias para que tengan la misma longitud
    secuencias = pad_sequences(secuencias, maxlen=MAX_LENGTH_FRAMES, padding='post', truncating='post', dtype='float32')
    
    # Convertir las secuencias y etiquetas en arreglos NumPy para el entrenamiento
    x = np.array(secuencias)
    y = to_categorical(etiquetas).astype(int)
    
    # Obtener el modelo de red neuronal con la cantidad de salidas igual al número de acciones
    modelo = get_model(len(acciones))
    
    # Entrenar el modelo con los datos cargados
    modelo.fit(x, y, epochs=NUM_EPOCH)
    
    # Mostrar un resumen de la arquitectura del modelo
    modelo.summary()
    
    # Guardar el modelo entrenado en la ruta especificada
    modelo.save(ruta_model)

def train_model():
    """
    Inicia el proceso de entrenamiento del modelo en un hilo separado y muestra una ventana de progreso.

    Esta función inicia el proceso de entrenamiento del modelo de forma asíncrona, permitiendo que la interfaz gráfica siga
    siendo interactiva durante el entrenamiento. Al finalizar, se notifica al usuario y se guarda el modelo entrenado.
    """
    # Definir la ruta donde se encuentran los datos de entrenamiento
    ruta_datos = os.path.join(RUTA_RAIZ, RUTA_DATOS)
    
    # Definir la ruta donde se guardará el modelo entrenado
    ruta_guardada = os.path.join(RUTA_RAIZ, RUTA_MODEL)
    ruta_model = os.path.join(ruta_guardada, NOMBRE_MODELO)
    
    # Llamar a la función de entrenamiento del modelo
    modelo_entrenamiento(ruta_datos, ruta_model)
    
    # Mostrar un mensaje informando al usuario que el modelo ha sido entrenado y guardado
    messagebox.showinfo("Información", "Modelo entrenado y guardado exitosamente.")

def evaluar_modelo(modelo, threshold=0.8):
    """
    Evalúa un modelo de aprendizaje profundo en tiempo real utilizando la cámara, para reconocer gestos de la Lengua de Señas Mexicana.

    Esta función captura frames en tiempo real desde la cámara, extrae los keypoints utilizando Mediapipe, y utiliza un
    modelo de red neuronal previamente entrenado para predecir la clase de gesto realizado. Si la confianza de la predicción
    supera un umbral específico, se reproduce un audio correspondiente al gesto reconocido.

    Args:
        modelo (tf.keras.Model): Modelo de red neuronal entrenado para reconocer gestos.
        threshold (float): Umbral de confianza para la predicción. Las predicciones con una confianza inferior a este umbral se ignoran.
    """
    contador_frame = 0  # Contador de frames consecutivos con detección de manos
    keypoints_secuencias = []  # Lista para almacenar las secuencias de keypoints
    sentence = []  # Lista para almacenar las acciones detectadas
    acciones = get_actions(RUTA_DATOS)  # Obtener las acciones (gestos) disponibles para la predicción
    
    # Inicializar el modelo Holistic de Mediapipe para la detección de keypoints
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)  # Captura de video desde la cámara
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            # Realizar la detección de keypoints en el frame capturado
            imagen, resultado = mediapipe_detection(frame, holistic_model)
            keypoints_secuencias.append(extract_keypoints(resultado))
            
            # Evaluar si se ha alcanzado el número máximo de frames para la secuencia
            if len(keypoints_secuencias) > MAX_LENGTH_FRAMES and there_hand(resultado):
                contador_frame += 1
            else:
                # Si la secuencia es lo suficientemente larga, realizar la predicción
                if contador_frame >= MIN_LENGTH_FRAMES:
                    prediccion = modelo.predict(np.expand_dims(keypoints_secuencias[-MAX_LENGTH_FRAMES:], axis=0))[0]
                    prediccion_maxima = np.argmax(prediccion)
                    if prediccion[prediccion_maxima] > threshold:
                        accion_detectada = acciones[prediccion_maxima]
                        sentence.append(accion_detectada)  # Añadir la acción detectada a la oración
                        text_to_speech(accion_detectada)  # Reproducir el audio correspondiente al gesto detectado
                        
                    contador_frame = 0
                    keypoints_secuencias = []  # Reiniciar la secuencia de keypoints
            
            # Dibujar en la imagen los resultados y la oración formada
            cv2.rectangle(imagen, (0, 435), (640, 500), (245, 117, 16), -1)
            cv2.putText(imagen, ' | '.join(sentence), FUENTE_POS, FUENTE, FUENTE_SIZE, (255, 255, 255))
            dibujar_keypoints(imagen, resultado)
            cv2.imshow('Traductor LSM', imagen)

            # Manejar las teclas presionadas por el usuario
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):  # Salir si se presiona la tecla 'q'
                break
            elif key == ord('b'):  # Borrar la oración si se presiona la tecla 'b'
                sentence = []

        # Liberar recursos
        video.release()
        cv2.destroyAllWindows()

def start_evaluation():
    """
    Inicia el proceso de evaluación del modelo en tiempo real en un hilo separado.

    Esta función carga el modelo entrenado y luego inicia el proceso de evaluación de gestos en tiempo real en un hilo
    separado. Esto permite que la interfaz gráfica permanezca receptiva durante la evaluación del modelo.
    """
    # Cargar el modelo entrenado desde la ruta especificada
    ruta_model = os.path.join(RUTA_MODEL, NOMBRE_MODELO)
    lstm_model = load_model(ruta_model)
    
    # Iniciar la evaluación del modelo en tiempo real
    evaluar_modelo(lstm_model)

def single_capture_and_predict_window(parent):
    """
    Abre una ventana para capturar una sola imagen de un gesto, predecir su clase y mostrar el resultado en la interfaz gráfica.

    Args:
        parent (tk.Tk): Ventana principal de la aplicación, desde la cual se abre esta ventana de captura y predicción.
    """
    # Crear una nueva ventana para la captura y predicción
    window = tk.Toplevel(parent)
    window.title("Captura y Predicción de Imágenes - Programa para Sordo Mudos")
    window.resizable(False, False)
    window.iconbitmap("logo.ico")

    # Ocultar la ventana principal mientras la ventana de captura está abierta
    parent.withdraw()

    # Dimensiones de la ventana
    ancho_ventana = 1000
    alto_ventana = 700

    # Obtener las dimensiones de la pantalla para centrar la ventana
    ancho_pantalla = window.winfo_screenwidth()
    alto_pantalla = window.winfo_screenheight()
    posicion_x = (ancho_pantalla - ancho_ventana) // 2
    posicion_y = (alto_pantalla - alto_ventana) // 3

    # Establecer el tamaño y la posición de la ventana
    window.geometry(f"{ancho_ventana}x{alto_ventana}+{posicion_x}+{posicion_y}")

    # Añadir un fondo de pantalla a la ventana
    bg_image_path = "img/background_prediccion.png"
    bg_image = ImageTk.PhotoImage(Image.open(bg_image_path))
    background_label = tk.Label(window, image=bg_image)
    background_label.image = bg_image  # Guardar referencia para evitar que el recolector de basura lo elimine
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Crear etiquetas de Tkinter para mostrar las imágenes capturadas y la predicción
    captura_label = tk.Label(window)
    captura_label.place(x=15, y=130, width=640, height=480)

    # Crear un Label para mostrar la predicción
    prediccion_label = tk.Label(window, text="", font=("Arial", 20), bg="white", fg="black")
    prediccion_label.place(x=783, y=510, width=100, height=100)

    # Crear un Label para mostrar la frase construida con ajuste automático de líneas y margen simulado
    frase_label = tk.Label(window, text="", font=("Arial", 20), bg="white", fg="black", anchor="nw", justify="left", wraplength=350)
    frase_label.place(x=683, y=113, width=300, height=100)

    # Variable para almacenar la frase
    frase = ""

    # Cargar el modelo de predicción
    model_dict = pickle.load(open(os.path.join(RUTA_MODEL, 'model.p'), 'rb'))
    model = model_dict['model']

    # Iniciar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)

    # Configurar la resolución de la webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Inicializar MediaPipe para la detección de manos, asegurando que solo se detecte una mano
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    def update_frame():
        """
        Captura frames en tiempo real desde la cámara, predice la clase y dibuja el resultado en la interfaz gráfica.
        """
        success, img = cap.read()

        if success:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                hand_landmarks = results.multi_hand_landmarks[0]  # Solo considerar la primera mano detectada

                mp_drawing.draw_landmarks(
                    img,  # imagen donde se dibujan los landmarks
                    hand_landmarks,  # resultados del modelo
                    mp_hands.HAND_CONNECTIONS,  # conexiones entre landmarks
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(21):  # Usar solo los 21 landmarks de la mano
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(21):  # Usar solo los 21 landmarks de la mano
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Verificar que la longitud de data_aux es 42 (21 landmarks con x e y)
                if len(data_aux) == 42:
                    # Hacer la predicción
                    prediction = model.predict([np.asarray(data_aux)])

                    # Mostrar el resultado directamente, asumiendo que la predicción es una letra
                    predicted_character = prediction[0]

                    # Mostrar la predicción en la interfaz gráfica
                    prediccion_label.config(text=predicted_character, font=("Arial", 20))

            # Mantener la relación de aspecto original para el redimensionamiento
            aspect_ratio = img.shape[1] / img.shape[0]

            # Redimensionar la imagen para que se ajuste al Label (640x480)
            new_width = 640
            new_height = int(new_width / aspect_ratio)

            if new_height > 480:  # Si la altura excede el tamaño del Label, ajusta según la altura
                new_height = 480
                new_width = int(new_height * aspect_ratio)

            # Redimensionar la imagen procesada con las nuevas dimensiones
            img_resized = cv2.resize(img, (new_width, new_height))

            # Crear una imagen negra para llenar el Label y pegar la imagen redimensionada en el centro
            img_display = np.zeros((480, 640, 3), dtype=np.uint8)
            y_offset = (480 - new_height) // 2
            x_offset = (640 - new_width) // 2
            img_display[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

            # Convertir la imagen a RGB para Tkinter
            img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            captura_label.img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_display_rgb))
            captura_label.configure(image=captura_label.img_tk)

        window.after(2, update_frame)

    def add_to_frase(event):
        """
        Añadir la letra predicha a la frase cuando se presiona 'Enter'.
        """
        nonlocal frase
        predicted_character = prediccion_label.cget("text")
        if predicted_character:  # Solo añadir si hay un carácter predicho
            frase += predicted_character
            frase_label.config(text=frase)

    def update_frase_label():
        """
        Actualiza el texto en el frase_label con un margen simulado.
        """
        margin = " " * 2  # Margen simulado con 2 espacios
        # Añadir margen simulado al principio de cada línea
        frase_con_margen = "\n".join([margin + line for line in frase.splitlines()])
        frase_label.config(text=frase_con_margen)
        
    def add_to_frase(event):
        """
        Añadir la letra predicha a la frase cuando se presiona 'Enter'.
        """
        nonlocal frase
        predicted_character = prediccion_label.cget("text")
        if predicted_character:  # Solo añadir si hay un carácter predicho
            frase += predicted_character
            update_frase_label()

    def add_space(event):
        """
        Añadir un espacio a la frase cuando se presiona 'Espacio'.
        """
        nonlocal frase
        frase += " "
        update_frase_label()

    def add_newline(event):
        """
        Añadir un salto de línea a la frase cuando se presiona 'Enter'.
        """
        nonlocal frase
        frase += "\n"
        update_frase_label()

    def clear_frase(event):
        """
        Borrar la frase actual cuando se presiona 'b'.
        """
        nonlocal frase
        frase = ""
        update_frase_label()

    def delete_last_character(event):
        """
        Borrar el último carácter de la frase cuando se presiona 'Backspace'.
        """
        nonlocal frase
        frase = frase[:-1]  # Elimina el último carácter
        update_frase_label()

    def speak_frase(event):
        """
        Reproducir la frase utilizando la función text_to_speech cuando se presiona 'c'.
        Si la frase está vacía, reproducir un mensaje indicando que no hay texto.
        """
        def speak():
            if frase.strip():  # Verifica si hay algo en la frase (ignora espacios en blanco)
                text_to_speech(frase)
            else:
                text_to_speech("No hay texto para reproducir.")

        # Ejecutar la función speak en un hilo separado
        threading.Thread(target=speak).start()

    def quit_program():
        """
        Detiene la captura de video y cierra la ventana, restaurando la ventana principal.
        """
        cap.release()
        window.destroy()
        parent.deiconify()

    # Asignar teclas para salir del programa, para añadir letras, para añadir espacios, para borrar y para reproducir la frase
    window.bind('<Escape>', lambda event: quit_program())  # Asignar la tecla 'Esc' para salir del programa
    window.bind('<c>', add_to_frase)  # Asignar la tecla 'c' para añadir letras a la frase
    window.bind('<Return>', add_newline)  # Asignar la tecla 'Enter' para añadir un salto de línea a la frase
    window.bind('<space>', add_space)  # Asignar la tecla 'Espacio' para añadir un espacio a la frase
    window.bind('<b>', clear_frase)  # Asignar la tecla 'b' para borrar la frase
    window.bind('<BackSpace>', delete_last_character)  # Asignar la tecla 'Backspace' para borrar el último carácter
    window.bind('<f>', speak_frase)  # Asignar la tecla 'f' para reproducir la frase

    # Iniciar la actualización del frame
    window.after(2, update_frame)

def single_capture_and_predict_window_tflite(parent):
    window = tk.Toplevel(parent)
    window.title("Captura y Predicción de Imágenes - Programa para Sordo Mudos")
    window.resizable(False, False)
    window.iconbitmap("logo.ico")
    parent.withdraw()

    ancho_ventana = 1000
    alto_ventana = 700
    ancho_pantalla = window.winfo_screenwidth()
    alto_pantalla = window.winfo_screenheight()
    posicion_x = (ancho_pantalla - ancho_ventana) // 2
    posicion_y = (alto_pantalla - alto_ventana) // 3
    window.geometry(f"{ancho_ventana}x{alto_ventana}+{posicion_x}+{posicion_y}")

    bg_image_path = "img/background_prediccion.png"
    bg_image = ImageTk.PhotoImage(Image.open(bg_image_path))
    background_label = tk.Label(window, image=bg_image)
    background_label.image = bg_image
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    captura_label = tk.Label(window)
    captura_label.place(x=15, y=130, width=640, height=480)

    prediccion_label = tk.Label(window, text="", font=("Arial", 20), bg="white", fg="black")
    prediccion_label.place(x=783, y=510, width=100, height=100)

    frase_label = tk.Label(window, text="", font=("Arial", 20), bg="white", fg="black", anchor="nw", justify="left", wraplength=350)
    frase_label.place(x=683, y=113, width=300, height=100)

    frase = ""

    # Obtener las etiquetas directamente de los nombres de las carpetas
    folder_labels = sorted(os.listdir('frames_estaticos'))  # Las mismas etiquetas usadas durante el entrenamiento

    # Cargar el modelo TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=os.path.join(RUTA_MODEL, 'model.tflite'))
    interpreter.allocate_tensors()
    
    # Obtener detalles de entrada y salida del modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    def update_frame():
        success, img = cap.read()

        if success:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                hand_landmarks = results.multi_hand_landmarks[0]  

                mp_drawing.draw_landmarks(
                    img,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS,  
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                for i in range(21):  
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(21):  
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                if len(data_aux) == 42:
                    # Preparar la entrada para el modelo
                    input_data = np.array(data_aux, dtype=np.float32).reshape(1, -1)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()

                    # Obtener la predicción del modelo
                    output_data = interpreter.get_tensor(output_details[0]['index'])
                    predicted_probabilities = output_data[0]
                    predicted_class = np.argmax(predicted_probabilities)
                    confidence = predicted_probabilities[predicted_class]

                    # Aplicar un umbral de confianza
                    confidence_threshold = 0.6
                    if confidence > confidence_threshold:
                        predicted_character = folder_labels[predicted_class]  # Convertir el índice a la etiqueta correcta
                    else:
                        predicted_character = "?"  # Indica una predicción incierta

                    # Mostrar la predicción en la interfaz gráfica
                    prediccion_label.config(text=predicted_character, font=("Arial", 20))

            aspect_ratio = img.shape[1] / img.shape[0]
            new_width = 640
            new_height = int(new_width / aspect_ratio)

            if new_height > 480:  
                new_height = 480
                new_width = int(new_height * aspect_ratio)

            img_resized = cv2.resize(img, (new_width, new_height))
            img_display = np.zeros((480, 640, 3), dtype=np.uint8)
            y_offset = (480 - new_height) // 2
            x_offset = (640 - new_width) // 2
            img_display[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_resized

            img_display_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
            captura_label.img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_display_rgb))
            captura_label.configure(image=captura_label.img_tk)

        window.after(2, update_frame)

    def update_frase_label():
        margin = " " * 2  
        frase_con_margen = "\n".join([margin + line for line in frase.splitlines()])
        frase_label.config(text=frase_con_margen)

    def add_to_frase(event):
        nonlocal frase
        predicted_character = prediccion_label.cget("text")
        if predicted_character:  
            frase += predicted_character
            update_frase_label()

    def add_space(event):
        nonlocal frase
        frase += " "
        update_frase_label()

    def add_newline(event):
        nonlocal frase
        frase += "\n"
        update_frase_label()

    def clear_frase(event):
        nonlocal frase
        frase = ""
        update_frase_label()

    def delete_last_character(event):
        nonlocal frase
        frase = frase[:-1]  
        update_frase_label()

    def speak_frase(event):
        def speak():
            if frase.strip():  
                text_to_speech(frase)
            else:
                text_to_speech("No hay texto para reproducir.")

        threading.Thread(target=speak).start()

    def quit_program():
        cap.release()
        window.destroy()
        parent.deiconify()

    window.bind('<Escape>', lambda event: quit_program())  
    window.bind('<c>', add_to_frase)  
    window.bind('<Return>', add_newline)  
    window.bind('<space>', add_space)  
    window.bind('<b>', clear_frase)  
    window.bind('<BackSpace>', delete_last_character)  
    window.bind('<f>', speak_frase)  

    window.after(2, update_frame)


# Declaración de colores para los botones
color_aqua = '#1ABC9C'
color_azul = '#3498DB'
color_rojo = '#E74C3C'
color_amarillo = '#F39C12'

# Creación de carpetas antes de iniciar la ventana principal
crear_carpeta(RUTA_FPS_ACCIONES)
crear_carpeta(RUTA_ESTATICAS)
crear_carpeta(RUTA_DATOS)
crear_carpeta(RUTA_MODEL)

# Crear una ventana principal y configuración visual de esta misma
root = tk.Tk()
ancho_pantalla = root.winfo_screenwidth()
alto_pantalla = root.winfo_screenheight()
ancho_ventana = 800
alto_ventana = 650
posicion_x = (ancho_pantalla - ancho_ventana) // 2
posicion_y = (alto_pantalla - alto_ventana) // 3
root.title("Programa para Sordo Mudos")
root.geometry(f"{ancho_ventana}x{alto_ventana}+{posicion_x}+{posicion_y}")
root.title("Lengua de Señas Mexicana - Captura, Entrenamiento y Evaluación")
root.resizable(False, False)
root.iconbitmap("logo.ico")

# Añadir el fondo de pantalla de la ventana principal
bg_image_path = "img/background.png"
bg_image = Image.open(bg_image_path)
bg_photo = ImageTk.PhotoImage(bg_image)
background_label = tk.Label(root, image=bg_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Agregar botones a la ventana principal
btn_capturar = tk.Button(root, text="Iniciar captura", command=start_capture, font=("Helvetica", 15, "bold"), bg=color_aqua, fg='black')
btn_capturar.place(x=315, y=196)

btn_keypoints = tk.Button(root, text="Generar keypoints", command=start_generate_keypoints, font=("Helvetica", 15, "bold"), bg=color_azul, fg='black')
btn_keypoints.place(x=110, y=297)

btn_entrenar = tk.Button(root, text="Entrenar modelo", command=train_model, font=("Helvetica", 15, "bold"), bg=color_rojo, fg='black')
btn_entrenar.place(x=120, y=397)

btn_start = tk.Button(root, text="Evaluar modelo", command=start_evaluation, font=("Helvetica", 15, "bold"), bg=color_amarillo, fg='black')
btn_start.place(x=128, y=497)

btn_train_cnn = tk.Button(root, text="Entrenar Modelo", command=lambda: start_training(root), font=("Helvetica", 15, "bold"), bg=color_azul, fg='black')
btn_train_cnn.place(x=480, y=347)

btn_single_capture_predict = tk.Button(root, text="Evaluar modelo", command=lambda: single_capture_and_predict_window_tflite(root), font=("Helvetica", 15, "bold"), bg=color_amarillo, fg='black')
btn_single_capture_predict.place(x=490, y=449)

root.mainloop()