#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime
import sys

# Parametros que se pasan por consola
param_path = sys.argv[1] if len(sys.argv) > 1 else "..\\..\\agosuirpa\\CSV_exit\\version1636656263403\\"#"..\\..\\case-study\\"


folder_names = ["Basic_10_Balanced", "Basic_10_Imbalanced","Basic_100_Balanced","Basic_100_Imbalanced"]#,"Basic_1000_Balanced","Basic_1000_Imbalanced"]

times = {}

for n in folder_names:
    start = datetime.now().strftime("%H:%M:%S.%MS")
    times[n] = {"start": start}
    s1 = 'python GUI_components_detection.py '+param_path+n+'\\log.csv '+param_path+n+'\\'
    s2 = 'python GUI_classification.py media\\models\\model.json media\\models\\model.h5 '+param_path+n+'\\components_npy\\ '+param_path+n+'\\log.csv '+param_path+n+'\\enriched_log.csv'
    s3 = 'python Extract_training_dataset.py B '+param_path+n+'\\enriched_log.csv '+param_path+n+'\\'
    s4 = 'python Decision_tree.py '+param_path+n+'\\preprocessed_dataset.csv '+param_path+n+'\\ ' + 'autogeneration'
    os.system(s1)
    os.system(s2)
    os.system(s3)
    os.system(s4)
    #print(s)
    finish = datetime.now().strftime("%H:%M:%S.%MS")
    times[n]["finish"] = finish

    
# Serializing json 
json_object = json.dumps(times, indent = 4)
    
# Writing to sample.json
with open("media/GUI_components_detection_times.json", "w") as outfile:
    outfile.write(json_object)
    


# cada experimento una linea: csv
# almaceno los tiempos por cada fase y por cada experimento (por cada familia hay 30)
# ejecutar solamente los experimentos