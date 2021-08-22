from django.shortcuts import render
import pandas as pd

# Create your views here.
def flat_dataset_row(data, columns, param_timestamp_column_name, columns_to_drop, param_decision_point_activity):
    df_content = []
    for case in data["cases"]:
        timestamp_start = data["cases"][case]["A"].get(key=param_timestamp_column_name)
        timestamp_end = data["cases"][case][param_decision_point_activity].get(param_timestamp_column_name)
        row = [timestamp_start, timestamp_end, case]
        headers = ["Timestamp_start", "Timestamp_end", "Caso"]
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

def accumulate_values(data, row, case, act, headers, columns, columns_to_drop):
    row.append(data["cases"][case][act].drop(columns_to_drop).values.tolist())
    for c in columns:
        headers.append(c+"_"+act)