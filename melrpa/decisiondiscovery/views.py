from django.shortcuts import render
import pandas as pd
import os
from typing import List

import graphviz
import matplotlib.image as plt_img
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# Create your views here.
def flat_dataset_row(data, columns, param_timestamp_column_name, param_variant_column_name, columns_to_drop, param_decision_point_activity):
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
                row.extend(data["cases"][case][act].drop(columns_to_drop).values.tolist())
                for c in columns:
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