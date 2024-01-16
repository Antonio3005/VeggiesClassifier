# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 22:30:02 2024

@author: anton
"""

import os
import numpy as np
from PIL import Image
from picamera import PiCamera
from picamera.array import PiRGBArray
from tensorflow.keras.applications.inception_v3 import preprocess_input
from edgetpu.basic.basic_engine import BasicEngine

# Imposta il percorso del modello Tensorflow Lite convertito per Edge TPU
model_path = 'saved_model.tflite'

# Carica il modello Edge TPU
engine = BasicEngine(model_path)

# Imposta le dimensioni dell'immagine in ingresso del modello
input_size = (224, 224)

# Inizializza la fotocamera Pi
camera = PiCamera()

# Configura le dimensioni della fotocamera
camera.resolution = (640, 480)

# Inizializza il buffer di array RGB per la fotocamera Pi
raw_capture = PiRGBArray(camera, size=(640, 480))

# Attendi che la fotocamera si inizializzi
time.sleep(0.1)

# Acquisisci l'immagine dalla fotocamera
camera.capture(raw_capture, format="rgb")

# Converte l'immagine in un array NumPy
image_array = np.array(raw_capture.array)

# Riduci l'immagine alle dimensioni di input del modello
img = Image.fromarray(image_array)
img = img.resize(input_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Esegui l'inferenza con il modello Edge TPU
inference_results = engine.RunInference(preprocess_input(img_array))

# Ottieni l'etichetta della classe predetta
predicted_class = np.argmax(inference_results)

# Mappa delle classi
category_vegetable = {
    0: 'Bean', 1: 'Broccoli', 2: 'Cabbage', 3: 'Capsicum', 4: 'Carrot',
    5: 'Cauliflower', 6: 'Cucumber', 7: 'Papaya', 8: 'Potato', 9: 'Tomato'
}

# Ottieni l'etichetta della classe predetta
predicted_class_label = category_vegetable[predicted_class]

# Stampa l'etichetta della classe predetta
print("Predicted class:", predicted_class_label)

# Ripulisci il buffer della fotocamera
raw_capture.truncate(0)