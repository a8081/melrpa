#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime
import time
import sys

sep = "/"
version = "version1637144717955"
scenario_size = 30

# Parametros que se pasan por consola
orig_param_path = sys.argv[1] if len(sys.argv) > 1 else ".."+sep+".."+sep+"agosuirpa"+sep+"CSV_exit"+sep+"resources"+sep+version+sep+""#"..\\..\\case-study\\"

prefix_scenario = "scenario_"
scenarios = []
for i in range(0,scenario_size+1):
    scenarios.append("scenario_"+str(i))
    
family_names = ["Basic_10_Balanced", "Basic_10_Imbalanced","Basic_50_Balanced","Basic_50_Imbalanced","Basic_100_Balanced","Basic_100_Imbalanced"]#,"Basic_1000_Balanced","Basic_1000_Imbalanced"]

times = {}

for scenario in scenarios:
    param_path = orig_param_path + scenario + sep
    for n in family_names:
        times[n] = {}
        s1 = 'python GUI_components_detection.py '+param_path+n+sep+'log.csv '+param_path+n+sep
        s2 = 'python GUI_classification.py media'+sep+'models'+sep+'model.json media'+sep+'models'+sep+'model.h5 '+param_path+n+sep+'components_npy'+sep+' '+param_path+n+sep+'log.csv '+param_path+n+sep+'enriched_log.csv'
        s3 = 'python Extract_training_dataset.py B '+param_path+n+sep+'enriched_log.csv '+param_path+n+sep
        s4 = 'python Decision_tree.py '+param_path+n+sep+'preprocessed_dataset.csv '+param_path+n+sep+' ' + 'autogeneration'
        for index, s in enumerate([s1,s2,s3,s4]):
            start = datetime.now().strftime("%H:%M:%S.%MS")
            times[n][index] = {"start": start}
            os.system(s)
            times[n][index]["finish"] = datetime.now().strftime("%H:%M:%S.%MS")
    metadata_path = "media"+sep+scenario+"_"+str(round(time.time() * 1000))+"_metadata"+sep
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)
    # f = open(generate_path+"log.csv", 'w',newline='')
    # writer = csv.writer(f)
    
    # Serializing json 
    json_object = json.dumps(times, indent = 4)
    # Writing to .json
    with open(metadata_path+"GUI_components_detection_times.json", "w") as outfile:
        outfile.write(json_object)
    


# cada experimento una linea: csv
# almaceno los tiempos por cada fase y por cada experimento (por cada familia hay 30)
# ejecutar solamente los experimentos