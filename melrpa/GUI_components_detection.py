#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
"""
Detección de componentes GUI

El objetivo de esta parte es poder detectar, delimitar y posteriormente recortar,
cada uno de los componentes gráficos (símbolos, imágenes o cuadros de texto)
que componen una captura de pantalla.

Importación de Librerías: Nos vamos a servir principalmente de dos librerías,
keras_ocr y Opencv (Cv2).
======================================
Detección de bordes y recortes: OpenCV

Hacemos uso de la libería OpenCV para llevar a cabo las siguientes tareas:
- Lectura de la imagen.
- Cálculo de intervalos ocupados por los cuadros de texto obtenidos a través de keras_ocr
- Tratamiento de la imagen:
  > Conversión a escala de grises.
  > Suavizado gaussiano a la imagen.
  > **Algoritmo de Canny** para la detección de bordes.
  > Obtención de contornos.
- Comparación entre contornos y cuadros de texto, otorgando más peso al cuadro de texto en
  el caso de que se solape espacialmente con algún contorno detectado.
- Recorte final de cada uno de los componentes.

"""

# Comando para hacer uso del script:
# ./GUI_components_detection.py media/log.csv media/screenshots/
import sys
import pickle
import keras_ocr
import pandas as pd
import os
from featureextraction.views import get_ocr_image, detect_images_components

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.


# Parametros que se pasan por consola

param_log_path = sys.argv[1] if len(sys.argv) > 1 else "media/log.csv"
param_img_root = sys.argv[2] if len(sys.argv) > 2 else "media/screenshots/"

# param_log_path="media/LogExample.csv"
# param_img_root="media/screenshots/"

# Leemos el log
log = pd.read_csv(param_log_path, sep=",")
# Extraemos los nombres de las capturas asociadas a cada fila del log
image_names = log.loc[:,"Screenshot"].values.tolist()
pipeline = keras_ocr.pipeline.Pipeline()
file_exists = os.path.exists(param_img_root + "images_ocr_info.txt")

if file_exists:
  print("\n\nReading images OCR info from file...")
  with open(param_img_root + "images_ocr_info.txt", "rb") as fp:   # Unpickling
    esquinas_texto = pickle.load(fp)
else:
  esquinas_texto = []
  for img in image_names:
    ocr_result = get_ocr_image(pipeline,param_img_root, img)
    esquinas_texto.append(ocr_result[0])
  with open(param_img_root + "images_ocr_info.txt", "wb") as fp:   #Pickling
    pickle.dump(esquinas_texto, fp)

# print(len(esquinas_texto))

path1 = param_img_root+"contornos/"
path2 = param_img_root+"components_npy/"
for p in [path1,path2]:
  if not os.path.exists(p):
    os.mkdir(p)
    
detect_images_components(param_img_root, image_names, esquinas_texto, path1, path2)

"""
RECURSOS UTILIZADOS

**Tutoriales OpenCV:**
Canny edge detection: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html#canny
Contours in OpenCV: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_table_of_contents_contours/py_table_of_contents_contours.html#table-of-content-contours

**Tutoriales Keras-OCR:**
https://keras-ocr.readthedocs.io/en/latest/
https://keras-ocr.readthedocs.io/en/latest/api.html

**Ejemplo de proyecto OpenCV:**
https://programarfacil.com/blog/vision-artificial/detector-de-bordes-canny-opencv/
"""