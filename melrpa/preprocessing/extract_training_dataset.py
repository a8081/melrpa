#!C:/Users/Antonio/Documents/TFM/melrpa/env/Scripts/python.exe

# Comando para hacer uso del script:
# ./ TODO

# Importaciones
import sys
import pandas as pd

# Parametros que se pasan por consola
param_log_path=sys.argv[1]
# TODO: parametro opcional, si viene busco 

case_column_name = "Case"
activity_column_name = "Activity"

"""
Averiguar cual es el punto de decisión ¿?: Preguntar reunión

Recorro cada fila del log:
    Por cada caso:
        Almaceno en un map todos los atributos de las actividades hasta llegar al punto de decisión
        Suponiendo que el punto de decisión estuviese en la actividad D,
        el map tendría la estructura:
    {
        "headers": ["timestamp", "MOUSE", "clipboard"...],
        "caso1":
            {"A": ["value1","value2","value3",...]}
            {"B": ["value1","value2","value3",...]}
            {"C": ["value1","value2","value3",...]},
        "caso2":
            {"A": ["value1","value2","value3",...]}
            {"B": ["value1","value2","value3",...]}
            {"C": ["value1","value2","value3",...]},...
    }

Una vez generado el map, para cada caso, concatenamos el header con la actividad para nombrar las columnas y asignamos los valores
Para cada caso se genera una fila nueva en el dataframe
"""

log = pd.read_csv(param_log_path, sep=";")

cases = log.loc[:, case_column_name].values.tolist()
actual_case = 0

dataset_map = {"headers": list(log.columns), "cases": {}}
for c, index in enumerate(cases, start=0):
    if c==actual_case:
        activity = log.at[index, activity_column_name]
        dataset_map["cases"][c][activity] = log.loc[index,:].values.tolist()
    else:
        activity = log.at[index, activity_column_name]
        dataset_map["cases"][c] = {activity: log.loc[index,:].values.tolist()}
        actual_case = c