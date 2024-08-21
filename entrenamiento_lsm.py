import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from constantes import RUTA_DATOS, RUTA_MODEL
import tkinter as tk
from tkinter import Toplevel, Label, messagebox
import threading
from PIL import Image, ImageTk

def create_and_train_model():
    data_dir = 'frames_estaticos'

    # Inicializar MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    # Preparar listas para los datos y etiquetas
    data = []
    labels = []

    # Recorrer las carpetas y procesar las imágenes
    for dir_ in os.listdir(data_dir):
        for img_path in os.listdir(os.path.join(data_dir, dir_)):
            data_aux = []
            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(data_dir, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

    # Guardar los datos en un archivo pickle en RUTA_DATOS
    pickle_file = os.path.join(RUTA_DATOS, 'data.pickle')
    with open(pickle_file, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)

    # Cargar los datos desde el archivo pickle
    with open(pickle_file, 'rb') as f:
        data_dict = pickle.load(f)

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    # Entrenar el clasificador
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # Predecir y calcular la precisión
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)

    # Generar un reporte de clasificación
    report = classification_report(y_test, y_predict, target_names=sorted(set(labels)))

    # Generar y mostrar la matriz de confusión
    cm = confusion_matrix(y_test, y_predict, labels=sorted(set(labels)))
    cm_labels = sorted(set(labels))
    
    # Crear una imagen combinada del reporte
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.text(0.5, 0.5, f'{score * 100:.2f}% de las muestras fueron clasificadas correctamente!\n\n' + report, 
            fontsize=12, ha='center', va='center', wrap=True)
    ax.axis('off')

    # Guardar la imagen del reporte en RUTA_DATOS
    report_fig_path = os.path.join(RUTA_DATOS, "training_report.png")
    plt.savefig(report_fig_path)
    plt.close()

    # Crear una imagen de la matriz de confusión
    fig, ax = plt.subplots(figsize=(10, 7))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(cm_labels)))
    ax.set_yticks(np.arange(len(cm_labels)))
    ax.set_xticklabels(cm_labels, rotation=45)
    ax.set_yticklabels(cm_labels)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Verdadero')

    # Añadir los números dentro de los cuadros
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    # Guardar la imagen de la matriz de confusión en RUTA_DATOS
    cm_fig_path = os.path.join(RUTA_DATOS, "confusion_matrix.png")
    plt.savefig(cm_fig_path)
    plt.close()

    # Guardar el modelo entrenado en un archivo pickle en RUTA_MODEL
    model_file = os.path.join(RUTA_MODEL, 'model.p')
    with open(model_file, 'wb') as f:
        pickle.dump({'model': model}, f)

    return report_fig_path  # Devolver la ruta de la imagen del reporte

def start_training(root):
    # Verificar si los archivos ya existen antes de abrir la ventana de "Espere"
    data_exists = os.path.exists(os.path.join(RUTA_DATOS, 'data.pickle'))
    model_exists = os.path.exists(os.path.join(RUTA_MODEL, 'model.p'))

    if data_exists:
        respuesta = messagebox.askyesno("Archivo existente", "El archivo de datos ya existe. ¿Desea sobrescribirlo?")
        if not respuesta:
            return

    if model_exists:
        respuesta = messagebox.askyesno("Archivo existente", "El archivo del modelo ya existe. ¿Desea sobrescribirlo?")
        if not respuesta:
            return

    # Solo si el usuario decide sobrescribir los archivos, continuar con el entrenamiento
    # Ocultar la ventana principal
    root.withdraw()

    # Crear una ventana de "Espere" centrada
    wait_window = Toplevel(root)
    wait_window.title("Espere")
    wait_window.geometry("300x100")
    ancho_pantalla = wait_window.winfo_screenwidth()
    alto_pantalla = wait_window.winfo_screenheight()
    posicion_x = (ancho_pantalla - 300) // 2
    posicion_y = (alto_pantalla - 100) // 2
    wait_window.geometry(f"300x100+{posicion_x}+{posicion_y}")
    Label(wait_window, text="Entrenando el modelo, por favor espere...", font=("Arial", 12)).pack(pady=20)

    # Ejecutar el entrenamiento del modelo en un hilo separado
    threading.Thread(target=lambda: run_training(wait_window, root)).start()

def run_training(wait_window, root):
    report_image_path = create_and_train_model()
    
    # Ejecutar la parte de la interfaz gráfica en el hilo principal
    root.after(0, lambda: show_report(root, wait_window, report_image_path))

def show_report(root, wait_window, report_image_path):
    wait_window.destroy()
    root.deiconify()

    # Mostrar la imagen del reporte en un nuevo Toplevel
    report_window = Toplevel(root)
    report_window.title("Reporte de Entrenamiento")
    report_window.geometry("800x700")
    
    # Cargar la imagen usando Pillow
    img = Image.open(report_image_path)
    img = ImageTk.PhotoImage(img)
    
    # Crear un Label para mostrar la imagen
    img_label = Label(report_window, image=img)
    img_label.pack()
    
    # Mantener la referencia a la imagen para evitar que sea recolectada por el garbage collector
    img_label.image = img