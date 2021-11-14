from django.shortcuts import render
import pandas as pd
import os
from typing import List

import graphviz
import matplotlib.image as plt_img
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# Create your views here.
def flat_dataset_row(data, columns, param_timestamp_column_name, param_variant_column_name, columns_to_drop, param_decision_point_activity, actions_columns):
    """
    Con esta función convertimos el log en un dataset, de manera que aplanamos todos los registros que existan sobre un mismo caso,
    resultando una sola fila por cada caso. Para este aplanamiento solamente se tienen en cuenta los registros relativos a las actividades
    anteriores a la indicada en param_decision_point_activity, incluyendo a esta última. Se concatena el nombre de las columnas de las actividades
    con su identificación, por ejemplo, timestamp_A, timestamp_B, etc.

    :data: map cuyas claves corresponden con el identificador de cada cado y cuyos valores son cada actividad asociada a dicho caso, con su información asociada
    :type data: map
    :columns: nombre de las columnas del dataset que se quieren almacenar para cada actividad
    :type columns: list
    :param_timestamp_column_name: nombre de la columna donde se almacena el timestamp
    :type param_timestamp_column_name: str
    :param_variant_column_name: nombre de la columna donde se almacena la variante, que será la etiqueta de nuestro problema
    :type param_variant_column_name: str
    :columns_to_drop: nombre de las columnas a eliminar del dataset
    :type columns_to_drop: list
    :param_decision_point_activity: identificador de la actividad inmediatamente anterior al punto de decisión cuyo por qué se quiere descubrir
    :type param_decision_point_activity: str
    """
    df_content = []
    for case in data["cases"]:
        # print(case)
        timestamp_start = data["cases"][case]["A"].get(key=param_timestamp_column_name)
        timestamp_end = data["cases"][case][param_decision_point_activity].get(param_timestamp_column_name)
        variant = data["cases"][case]["A"].get(key=param_variant_column_name)
        row = [variant, case, timestamp_start, timestamp_end]
        headers = ["Variant", "Case", "Timestamp_start", "Timestamp_end"]
        for act in data["cases"][case]:
            # case
            # variant_id
            if act != param_decision_point_activity:
                # A todas las columnas les añado el sufijo con la letra de la actividad correspondiente
                row.extend(data["cases"][case][act].drop(columns_to_drop).values.tolist())
                for c in columns:
                    headers.append(c+"_"+act)
            else:
                new_list = [col_name for col_name in columns if col_name not in actions_columns]
                row.extend(data["cases"][case][act].drop(columns_to_drop).drop(actions_columns).values.tolist())
                for c in new_list:
                    headers.append(c+"_"+act)
                break
        # Introducimos la fila con la información de todas las actividades del caso en el dataset
        df_content.append(row)
        # print(row)
        # print(headers)
    df = pd.DataFrame(df_content, columns = headers)
    return df


# https://gist.github.com/j-adamczyk/dc82f7b54d49f81cb48ac87329dba95e#file-graphviz_disk_op-py
def plot_decision_tree(path: str,
                         clf: DecisionTreeClassifier,
                         feature_names: List[str],
                         class_names: List[str]) -> np.ndarray:
    # 1st disk operation: write DOT
    export_graphviz(clf, out_file=path+".dot",
                    feature_names=feature_names,
                    class_names=class_names,
                    label="all", filled=True, impurity=False,
                    proportion=True, rounded=True, precision=2)

    # 2nd disk operation: read DOT
    graph = graphviz.Source.from_file(path + ".dot")

    # 3rd disk operation: write image
    graph.render(path, format="png")

    # 4th disk operation: read image
    image = plt_img.imread(path + ".png")

    # 5th and 6th disk operations: delete files
    os.remove(path + ".dot")
    # os.remove("decision_tree.png")

    return image