#!C:/Users/Antonio/Documents/TFM/melrpa/env/Scripts/python.exe
"""
Detección de componentes GUI

El objetivo de esta parte es poder detectar, delimitar y posteriormente recortar,
cada uno de los componentes gráficos (símbolos, imágenes o cuadros de texto)
que componen una captura de pantalla.

Importación de Librerías: Nos vamos a servir principalmente de dos librerías,
keras_ocr y Opencv (Cv2).
"""
# Comando para hacer uso del script:
# ./GUI_components_detection.py ../media/LogExample.csv ../media/screenshots/
import numpy as np
import matplotlib.pyplot as plt
import keras_ocr
import cv2
import sys
import pandas as pd
# from google.colab.patches import cv2_imshow # ESTO ES PARA GOOGLE COLABORATORY
# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
pipeline = keras_ocr.pipeline.Pipeline()

# Parametros que se pasan por consola
param_log_path=sys.argv[1]
param_img_root=sys.argv[2]

log = pd.read_csv(param_log_path, sep=";")

image_names = log.loc[:,"Screenshot"].values.tolist()

"""
Detección cuadros de texto: KERAS_OCR

Con el objetivo de detectar los cuadros de texto dentro las capturas de pantalla,
definimos la función get_keras_ocr_image.
Esta función tendrá como input una lista de imagénes, y como output,
las coordenadas de las esquinas que conforman cada uno de los cuadros de texto detectados.
"""

image_path = "../media/screenshots/Screenshot2.png"

def get_keras_ocr_image(images_input):
  """
  Hay que pasarle el path o url de la imagen a tratar, 
  o una lista con las urls en caso de ser varias.
  """
  if not isinstance(images_input, list):
    images_input = [images_input]
  # Get a set of three example images
  images = [
      keras_ocr.tools.read(url) for url in images_input
  ]
  # Each list of predictions in prediction_groups is a list of
  # (word, box) tuples.
  prediction_groups = pipeline.recognize(images)
  # Plot the predictions
  # fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))
  # for ax, image, predictions in zip(axs, images, prediction_groups):
  #     keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)
  return prediction_groups

"""
Hacemos uso de la función. Debido a que hay que pasarle un array de imágenes y en este
caso solamente tenemos una, vamos a construir un array de una sola imagen.
"""
imgs = [
  image_path
]

esquinas_texto = get_keras_ocr_image(imgs)

for img_name in image_names:
  image_path = param_img_root + img_name
  """
  Como ejemplo, mostramos las coordenadas de las esquinas de los cinco primeros cuadros 
  de texto devueltos por Keras-OCR

  esquinas_texto[0][0:5]
  """
  """
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
  #Leemos la imagen
  img = cv2.imread(image_path)
  img_copy = img.copy()
  # cv2_imshow(img_copy)

  #almaceno en global_y todas las coordenadas y de los cuadros de texto
  #cada fila es un cuadro de texto distinto, mucho más amigable que el formato que devuelve keras_ocr
  global_y = []
  global_x = []
  for j in range(0, len(esquinas_texto[0])):
    coordenada_y = []
    coordenada_x = []
    for i in range(0,len(esquinas_texto[0][j][1])):
      coordenada_y.append(esquinas_texto[0][j][1][i][1])
      coordenada_x.append(esquinas_texto[0][j][1][i][0])
    global_y.append(coordenada_y)
    global_x.append(coordenada_x)
    #print('Coord y, cuadro texto ' +str(j+1)+ str(global_y[j]))
    #print('Coord x, cuadro texto ' +str(j+1)+ str(global_x[j]))

  print("\n Numero cuadros de texto detectados " + str(len(esquinas_texto[0])))

  #Calculo los intervalos de los cuadros de texto
  intervalo_y=[]
  intervalo_x=[]
  for j in range(0, len(global_y)):
    intervalo_y.append([int(max(global_y[j])), int(min(global_y[j]))])
    intervalo_x.append([int(max(global_x[j])), int(min(global_x[j]))])
  print("intervalo y", intervalo_y)
  print("intervalo x", intervalo_x)
    
  # Convertimos a escala de grises
  gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # cv2_imshow(gris)

  # Aplicar suavizado Gaussiano
  gauss = cv2.GaussianBlur(gris, (5,5), 0)
  # cv2_imshow(gauss)

  # Detectamos los bordes con Canny
  canny = cv2.Canny(gauss, 50, 150)
  # cv2_imshow(canny)

  # Buscamos los contornos
  (contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  print("\n Número de componentes GUI detectados:", len(contornos), "\n")

  #y los dibujamos
  cv2.drawContours(img_copy,contornos,-1,(0,0,255), 2)
  # cv2_imshow(img_copy)
  cv2.imwrite('../media/screenshots/samplecontornos.png', img_copy)

  #Llevamos a cabo los recortes para cada contorno detectado
  recortes = []
  lista_prueba=[]

  for j in range(0,len(contornos)):
    cont_horizontal = []
    cont_vertical = []
    #obtenemos componentes máximas y mínimas (eje x,y) del contorno
    for i in range(0,len(contornos[j])-1):
      cont_horizontal.append(contornos[j][i][0][0])
      cont_vertical.append(contornos[j][i][0][1])
      x = min(cont_horizontal)
      w = max(cont_horizontal)
      y = min(cont_vertical)
      h = max(cont_vertical)
    #print('Coord x, componente' + str(j+1) + '  ' + str(x) + ' : ' + str(w))
    #print('Coord y, componente' + str(j+1) + '  ' + str(y) + ' : ' + str(h))
    
    #comprobamos que los contornos no solapen con cuadros de texto y optamos por recortar los cuadros de texto si solapan.
    condicion_recorte=True
    for k in range(0,len(intervalo_y)):
      solapa_y = 0
      solapa_x = 0
      if (min(intervalo_y[k]) <= y <= max(intervalo_y[k])) or (min(intervalo_y[k]) <= h <= max(intervalo_y[k])):
        solapa_y = 1
      if (min(intervalo_x[k]) <= x <= max(intervalo_x[k])) or (min(intervalo_x[k]) <= w <= max(intervalo_x[k])):
        solapa_x = 1
      if ((solapa_y == 1) and (solapa_x == 1)):
        if (lista_prueba.count(k) == 0):
          lista_prueba.append(k)
        else:
          condicion_recorte = False
        x = min(intervalo_x[k])
        w = max(intervalo_x[k])
        y = min(intervalo_y[k])
        h = max(intervalo_y[k])
        #crop_img = img[min(intervalo_y[k]) : max(intervalo_y[k]), min(intervalo_x[k]) : max(intervalo_x[k])]
        print("Componente " + str(j+1) + " solapa con cuadro de texto")

    #if (solapa_y == 1 and solapa_x == 1):
      #crop_img = img[min(intervalo_y[k]) : max(intervalo_y[k]), min(intervalo_x[k]) : max(intervalo_x[k])]
      #print("Componente " + str(j+1) + " solapa con cuadro de texto")
      #recortes.append(crop_img)
    #else:
    if (condicion_recorte):
      crop_img = img[y:h, x:w]
      recortes.append(crop_img)

  aux = np.array(recortes)
  np.save("../media/GUIcomponents/" + img_name + ".npy", aux)

# print("\n")

#Mostramos las imagenes recortada
# for i in range(0,len(recortes)):
#   if recortes[i].any():
#     print("Componente nº",i+1,  cv2_imshow(recortes[i]), "\n")
#   else:
#     print("componente vacío")

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