
#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime
import time
import sys


def get_only_list_folders(path, sep):
    folders_and_files = os.listdir(path)
    family_names = []
    for f in folders_and_files:
        if os.path.isdir(path+sep+f):
            family_names.append(f)
    return family_names

# "version1637144717955", "/", True, None
def generate_case_study(version, sep, scenario_mode, p):
    if scenario_mode:
        # Parametros que se pasan por consola
        orig_param_path = p if p else ".."+sep+".."+sep+"agosuirpa"+sep+"CSV_exit"+sep+"resources"+sep+version#"..\\..\\case-study\\"

        prefix_scenario = "scenario_"
        scenarios = []
    
        family_names = get_only_list_folders(orig_param_path+sep+prefix_scenario+"0", sep)
    
        # for i in range(0,scenario_size+1):
        #     scenarios.append("scenario_"+str(i))
        scenarios = get_only_list_folders(orig_param_path, sep)
        print("Scenarios: " + str(scenarios))
    
    else:
        orig_param_path = p if p else ".."+sep+".."+sep+"agosuirpa"+sep+"CSV_exit"+sep+version
        family_names = ["Basic_10_Balanced", "Basic_10_Imbalanced"]#,"Basic_50_Balanced","Basic_50_Imbalanced","Basic_100_Balanced","Basic_100_Imbalanced"]#,"Basic_1000_Balanced","Basic_1000_Imbalanced"]

    times = {}

    metadata_path = "media"+sep+str(round(time.time() * 1000))+"_metadata"+sep
    if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
            
    for scenario in scenarios:
        param_path = orig_param_path + sep + scenario + sep
        for n in family_names:
            times[n] = {}
            s1 = 'python GUI_components_detection.py '+param_path+n+sep+'log.csv '+param_path+n+sep
            s2 = 'python GUI_classification.py media'+sep+'models'+sep+'model.json media'+sep+'models'+sep+'model.h5 '+param_path+n+sep+'components_npy'+sep+' '+param_path+n+sep+'log.csv '+param_path+n+sep+'enriched_log.csv'
            s3 = 'python Extract_training_dataset.py B '+param_path+n+sep+'enriched_log.csv '+param_path+n+sep
            s4 = 'python Decision_tree.py '+param_path+n+sep+'preprocessed_dataset.csv '+param_path+n+sep+' ' + 'autogeneration'# autogeneration mode is selected to not printing decision trees images
            for index, s in enumerate([s1,s2,s3,s4]):
                start = datetime.now().strftime("%H:%M:%S.%MS")
                times[n][index] = {"start": start}
                os.system(s)
                times[n][index]["finish"] = datetime.now().strftime("%H:%M:%S.%MS")
        
        # if not os.path.exists(scenario+sep):
        #     os.makedirs(scenario+sep)
    
        # Serializing json
        json_object = json.dumps(times, indent = 4)
        # Writing to .json
        with open(metadata_path+scenario+"-metainfo.json", "w") as outfile:
            outfile.write(json_object)
    # cada experimento una linea: csv
    # almaceno los tiempos por cada fase y por cada experimento (por cada familia hay 30)
    # ejecutar solamente los experimentos