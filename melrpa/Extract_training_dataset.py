#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe

# Comando para hacer uso del script:
# ./ Extract_training_dataset.py E

# Importaciones
import sys
import pandas as pd
from decisiondiscovery.views import flat_dataset_row

# Parametros que se pasan por consola

param_decision_point_activity = sys.argv[1]
param_log_path = sys.argv[2] if len(sys.argv) > 2 else "media/enriched-log-feature-extracted.csv"
param_activity_column_name = sys.argv[3] if len(sys.argv) > 5 else "Activity"
param_variant_column_name = sys.argv[4] if len(sys.argv) > 5 else "Variant"
param_case_column_name = sys.argv[5] if len(sys.argv) > 5 else "Case"
param_screenshot_column_name = sys.argv[6] if len(sys.argv) > 7 else "Screenshot"
param_timestamp_column_name = sys.argv[7] if len(sys.argv) > 7 else "Timestamp"

"""
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

log = pd.read_csv(param_log_path, sep=",", index_col=0)

cases = log.loc[:,param_case_column_name].values.tolist()
actual_case = 0

dataset_map = {"headers": list(log.columns), "cases": {}}
for index, c  in enumerate(cases, start=0):
    if c==actual_case:
        activity = log.at[index, param_activity_column_name]
        if c in dataset_map["cases"]:
            dataset_map["cases"][c][activity] = log.loc[index,:]
        else:
            dataset_map["cases"][c] = {activity: log.loc[index,:]}
    else:
        activity = log.at[index, param_activity_column_name]
        dataset_map["cases"][c] = {activity: log.loc[index,:]}
        actual_case = c

# import pprint
# pprint.pprint(dataset_map)

columns_to_drop = [param_case_column_name, param_activity_column_name, param_timestamp_column_name, param_screenshot_column_name]
columns = list(log.columns)
for c in columns_to_drop:
    columns.remove(c)

# Establecemos columnas comunes y al resto de columnas se le concatena el "_" actividad
data_flattened = flat_dataset_row(dataset_map, columns, param_timestamp_column_name, columns_to_drop, param_decision_point_activity)
print(data_flattened)
data_flattened.to_csv('media/preprocessed-dataset.csv')