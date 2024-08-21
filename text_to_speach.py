from gtts import gTTS
import os
import pygame
from time import sleep

def text_to_speech(text):
    """
    Convierte texto en audio y lo reproduce usando pygame.

    Esta función toma un texto como entrada, lo convierte en un archivo de audio en formato MP3 utilizando la
    biblioteca gTTS (Google Text-to-Speech), y luego reproduce el archivo de audio usando pygame. 
    El archivo de audio se elimina automáticamente después de la reproducción.

    Args:
        text (str): El texto que se desea convertir en audio y reproducir.

    Este proceso incluye:
    - Generación de un archivo MP3 a partir del texto proporcionado.
    - Inicialización de pygame y su mezclador de audio para reproducir el archivo MP3.
    - Espera hasta que la reproducción del archivo de audio haya finalizado.
    - Eliminación del archivo de audio una vez completada la reproducción.
    """
    # Convertir el texto en un archivo de audio MP3 utilizando gTTS
    tts = gTTS(text=text, lang='es')
    filename = "speech.mp3"
    tts.save(filename)
    
    # Inicializar pygame y su mezclador de audio
    pygame.init()
    pygame.mixer.init()
    
    # Cargar y reproducir el archivo de audio MP3
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    # Esperar hasta que la reproducción haya finalizado
    while pygame.mixer.music.get_busy():
        sleep(0.1)

    # Liberar el archivo antes de eliminarlo y cerrar pygame
    pygame.mixer.music.unload()  
    pygame.mixer.quit()
    pygame.quit()

    # Eliminar el archivo de audio MP3
    os.remove(filename)

if __name__ == "__main__":
    text = "texto"
    text_to_speech(text)