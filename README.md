# Mining Event Logs RPA - MELRPA
Mining Event Logs for Robotic Process Automation: Looking for the why?


## Requirements:
Microsoft Visual C++ 14.0

### Download dataset ReDraw from Zenodo
´´´
wget https://zenodo.org/record/2530277/files/CNN-Data-Final.tar.gz -O CNNdata.tgz
tar -xzf CNNdata.tgz
´´´

Para que las capturas asociadas al log se procesen correctamente deben seguir el esquema de "image0001" para estar ordenadas alfabéticamente, ya que el clasificador de componentes generará una fila por cada una de las imágenes que se extraigan, procesándolas por orden alfabético y posteriormente la información asociada a esa imagen se añadirá como columnas adicionales a la fila correspondiente a su orden en el log. Si falta una imagen, por ejemplo la "image0005" habrá un descuadre en la información que se almacene a partir de la fila 5.

En el caso en el que se utilice la opción de entrenar el modelo:
Originalmente las etiquetas de las imágenes de los componentes GUI destinadas al entrenamiento de la CNN para su clasificación se encuentran contenida en el nombre de las propias imágenes de la siguiente manera:

Ejemplo: ._42-android.widget.TextView.png

Siendo la etiqueta la última palabra entre el penúltimo punto y el último punto. De esta manera lo interpretará el sistema para el entrenamiento de CNN.
