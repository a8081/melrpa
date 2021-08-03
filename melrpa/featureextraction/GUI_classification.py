#!C:/Users/Antonio/Documents/TFM/melrpa/env/Scripts/python.exe

# Comando para hacer uso del script:
# ./GUI_classification.py ../media/models/model.json ../media/models/model.h5 ../media/GUIcomponents/ ../media/LogExample.csv

# Importaciones
import sys
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import matplotlib.image as mpimg
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG19 #For Transfer Learning
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from keras.models import model_from_json

# Parametros que se pasan por consola
param_json_file_name=sys.argv[1]
param_model_weights=sys.argv[2]
param_images_root=sys.argv[3]
param_log_path=sys.argv[4]

# arr_balance = np.load('../media/models/preprocessed_50_50_balance.npy')
# arr_labels_balance = np.load('../media/models/test_labels_balance.npy')

# """
# Vemos que aunque no lleguemos a un balanceo perfecto, ya que cuando tratamos con
# datasets reales, es complicado llegar a situaciones ideales que vemos con datasets preparados. Vemos que la cantidad de imágenes de las clases ImageButton, ImageView y CheckBox han aumentado considerablemente.

# Aquí, dividiremos el conjunto de datos descargados en conjuntos de entrenamiento,
# prueba y validación
# """

# X_train, X_test, y_train, y_test = train_test_split(arr_balance, arr_labels_balance, test_size=0.2, random_state=1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# x_train = np.array(X_train)
# x_val = np.array(X_val)
# x_test = np.array(X_test)
# y_train = np.array(y_train)
# y_val = np.array(y_val)
# y_test = np.array(y_test)

# """
# Una vez divididos, veremos la forma de nuestros datos. Tenemos que hacer una codificación one hot porque
# tenemos 14 clases y debemos esperar que la dimensión de y_train,y_val y y_test cambie de 1 a 14 (una
# columna por cada clase, en las que se seteará a 1 la clase a la que corresponda cada instancia).
# """
# #One Hot Encoding
# onehot_encoder = OneHotEncoder(sparse=False)
# y_train = y_train.reshape(len(y_train), 1)
# y_train = onehot_encoder.fit_transform(y_train)
# y_val = y_val.reshape(len(y_val), 1)
# y_val = onehot_encoder.fit_transform(y_val)
# y_test = y_test.reshape(len(y_test), 1)
# y_test = onehot_encoder.fit_transform(y_test)

# column_names = onehot_encoder.get_feature_names()
column_names = ['x0_Button', 'x0_CheckBox', 'x0_CheckedTextView', 'x0_EditText',
 'x0_ImageButton', 'x0_ImageView', 'x0_NumberPicker', 'x0_RadioButton',
 'x0_RatingBar', 'x0_SeekBar', 'x0_Spinner', 'x0_Switch', 'x0_TextView',
 'x0_ToggleButton']

print("\n\n====== Column names =======================")
print(column_names)
print("===========================================\n\n")
# """
# Aquí realizaremos el data augmentation. Esta es la técnica que se utiliza para ampliar el tamaño de un conjunto
# de datos de entrenamiento creando versiones modificadas de las imágenes del conjunto de datos. En primer lugar,
# definiremos instancias individuales de ImageDataGenerator para el aumento y luego las ajustaremos con cada uno
# de los conjuntos de datos de entrenamiento, prueba y validación.
# """

# #Image Data Augmentation
# train_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

# val_generator = ImageDataGenerator(rotation_range=2, horizontal_flip=True, zoom_range=.1)

# test_generator = ImageDataGenerator(rotation_range=2, horizontal_flip= True, zoom_range=.1)

# #Fitting the augmentation defined above to the data
# train_generator.fit(x_train)
# val_generator.fit(x_val)
# test_generator.fit(x_test)

# """
# En este experimento utilizaremos el learning rate annealer. El learning rate annealer disminuye
# la tasa de aprendizaje después de un cierto número de epochs si la tasa de error no cambia.
# Aquí, a través de esta técnica, vamos a controlar la precisión de validación y si parece ser una
# "meseta" en 3 épocas, se reducirá la tasa de aprendizaje en 0,01.
# """
# #Learning Rate Annealer
# lrr= ReduceLROnPlateau(monitor='val_acc', factor=.01, patience=3, min_lr=1e-5)

#Initializing the hyperparameters
batch_size= 32
epochs=20
learn_rate=.001

# load json and create model
json_file = open(param_json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(param_model_weights)
print("=========== Loaded ML model from disk")

# sgd=SGD(lr=learn_rate,momentum=.9,nesterov=False)

# evaluate loaded model on test data
# loaded_model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
# score = loaded_model.evaluate(x_test, y_test, verbose=0)
# print("=========== %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

"""
Tras comprobar el accuracy obtenido tras la evaluación, vamos a aplicar el modelo a un experimento basado en un log,
con imágenes mockeadas a modo de screenchots del proceso, para extraerle las características a estas.

Cogemos los vectores de numpy correspondientes a cada imagen, obtenidos a través de la aplicación del notebook
de Detección y recorte de componentes GUI, que tendrán que ser almacenadas en la ruta "mockups_vector/".
"""

# images_root = "/content/drive/MyDrive/Proyecto MLE/Deteccion componentes GUI/mockups_vector/"
images_root = param_images_root #"mockups_vector/"
crop_imgs = {}
images_names = os.listdir(images_root)
for img_filename in os.listdir(images_root):
    crop_img_aux = np.load(images_root+img_filename, allow_pickle=True)
    crop_imgs[img_filename] = {'content': crop_img_aux}

# crop_imgs[images_names[0]]["content"][0].shape

"""
Una vez cargadas, reducimos su tamaño para adecuarlo a la entrada de la red neuronal convolucional producto de este
notebook.
"""
# Zero-padding
def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))

for crop_img in crop_imgs:
    print(crop_img)
    aux = []
    for index, img in enumerate(crop_imgs[crop_img]["content"]):
        print("=========== Original "+str(index)+": "+str(img.shape))
        if img.shape[1] > 150:
            img = img[0:img.shape[0], 0:150]
        if img.shape[0] > 150:
            img = img[0:150, 0:img.shape[1]]
        img_padded = pad(img, 150, 150)
        print("=========== Padded: "+str(img_padded.shape))
        img_resized = tf.image.resize(img_padded, [50,50], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=True,antialias=True)
        aux.append(img_resized)
    crop_imgs[crop_img]["content_preprocessed"] = aux

"""
Esta red devuelve como salida un número, indicando una de las clases. Este número tendrá que ser mapeado a su texto
(nombre de la clase) correspondiente.
"""

for i in range(0,len(images_names)):
    aux_2 = crop_imgs[images_names[i]]["content_preprocessed"]
    result = loaded_model.predict_classes(np.array(aux_2))
    result_mapped = [column_names[x] for x in result]
    crop_imgs[images_names[i]]["result"] = result_mapped
    crop_imgs[images_names[i]]["result_freq"] = pd.Series(result_mapped).value_counts()
    crop_imgs[images_names[i]]["result_freq_df"] = crop_imgs[images_names[i]]["result_freq"].to_frame().T

# crop_imgs[images_names[0]]["result"]

a = crop_imgs[images_names[0]]["result_freq_df"].columns.tolist()

# crop_imgs[images_names[0]]["result_freq_df"]

"""
Como todas las imágenes no tendrán todas las clases, generarán como salida un dataset que no tendrán siempre las mismas
columnas. Dependerá si en la imagen aparecen componentes GUI de todo tipo o solamente un subconjunto de ellos. Por ello, 
inicializamos un dataframe con todas las columnas posibles, y vamos incluyendo una fila por cada resultado obtenido de
cada imagen pasada por la red.
"""

nombre_clases=['x0_RatingBar', 'x0_ToggleButton', 'x0_Spinner', 'x0_Switch', 'x0_CheckBox', 'x0_TextView', 'x0_EditText', 'x0_ImageButton', 'x0_NumberPicker', 'x0_CheckedTextView', 'x0_SeekBar', 'x0_ImageView', 'x0_RadioButton', 'x0_Button']
df = pd.DataFrame([], columns=nombre_clases)

for i in range(0, len(images_names)):
    row1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # accedemos a las frecuencias almacenadas anteriormente
    df1 = crop_imgs[images_names[i]]["result_freq_df"]
    if len(df1.columns.tolist())>0:
        for x in df1.columns.tolist():
            uiui = nombre_clases.index(x)
            row1[uiui] = df1[x][0]
            df.loc[i] = row1

log = pd.read_csv(param_log_path, sep=";")

# log.head()

"""
Una vez obtenido el dataset correspondiente a la cantidad de elementos de cada clase contenidos en cada una de las
imágenes. Unimos este con el log completo, añadiendo las características extraídas de las imágenes.
"""

log_enriched = log.join(df).fillna(method='ffill')

# log_enriched

"""
Finalmente obtenemos un log enriquecido, que se torna como prueba de concepto de nuestra hipótesis basada en que, si
no solamente capturamos eventos de teclado o de ratón en la monitorización a través de un keylogger, sino que obtenemos
también capturas de pantalla. Podemos extraer mucha más información útil, pudiendo mejorar la minería de procesos sobre
dicho log. Nos ha quedado pendiente validar esta hipótesis mediante la comparación de resultados entre aplicar técnicas
de pricess mining aplicadas sobre el log inicial vs. el log enriquecido. Esperamos poder continuar con este proyecto
en fases posteriores del máster.
"""

log_enriched.to_csv('../media/enriched-log-feature-extracted.csv')
print("=========== ENRICHED LOG GENERATED: path=media/enriched-log-feature-extracted.csv")