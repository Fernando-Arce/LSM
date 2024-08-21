from keras.src.models.sequential import Sequential
from keras.src.layers.rnn.lstm import LSTM
from keras.src.layers.core.dense import Dense
from keras.src.layers import Dropout
from keras.src.optimizers import Nadam, RMSprop
from constantes import LENGTH_KEYPOINTS, MAX_LENGTH_FRAMES

NUM_EPOCH = 110  # Número de épocas para el entrenamiento del modelo

def get_model(output_length: int):
    """
    Crea y compila un modelo de red neuronal LSTM secuencial para la clasificación de secuencias de puntos clave.

    Este modelo está diseñado para trabajar con secuencias de puntos clave extraídas de videos, como gestos de
    la Lengua de Señas Mexicana. Utiliza varias capas LSTM para capturar la información temporal en las secuencias,
    seguidas de capas densas para realizar la clasificación final.

    Args:
        output_length (int): Número de clases en la salida del modelo (dimensión de la capa de salida).

    Returns:
        keras.Sequential: Un modelo secuencial de Keras ya compilado, listo para ser entrenado.
    
    El modelo incluye:
    - Tres capas LSTM con 64, 128 y 128 unidades, respectivamente, para procesar secuencias temporales.
    - Capas Dropout después de cada capa LSTM para reducir el sobreajuste.
    - Varias capas densas para refinar la representación aprendida.
    - Una capa de salida con activación softmax para la clasificación multiclase.
    """
    # Se crea un modelo secuencial de Keras
    model = Sequential()
    
    # Se añaden capas LSTM al modelo para capturar la información temporal en las secuencias
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(MAX_LENGTH_FRAMES, LENGTH_KEYPOINTS)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dropout(0.2))
    
    # Se añaden capas densas (totalmente conectadas) al modelo para refinar la representación aprendida
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    
    # Se añade la capa de salida con una activación softmax para la clasificación multiclase
    model.add(Dense(output_length, activation='softmax'))
    
    # Se compila el modelo especificando el optimizador Nadam, la función de pérdida categórica y las métricas
    model.compile(optimizer=Nadam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model