
#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime
import time
import sys
import csv
import pandas as pd 
from basic_case_study_util import get_only_list_folders
import re

# Configuration data
sep = "/"
version = "version1637144717955"
scenario_size = True
orig_param_path =  ".."+sep+".."+sep+"agosuirpa"+sep+"CSV_exit"+sep+"resources"+sep+version+sep
prefix_scenario = "scenario_"
decision_tree_filename = "decision_tree.log"
experiment_path = "media" + sep
scenarios = []
times_info_path = "media"+sep+"1637149575778_metadata"+sep
preprocessed_log_filename = "preprocessed_dataset.csv"

# Expected results
gui_component_class = "ImageView"
activity = "B"
quantity_difference = 1

family_names = get_only_list_folders(orig_param_path+prefix_scenario+"0", sep)

scenarios = get_only_list_folders(orig_param_path, sep)
print("Scenarios: " + str(scenarios))


family = []
balanced = []
log_size = []
scenario_number = []
detection_time = []
classification_time = []
flat_time = []
tree_training_time = []
log_column = []
accuracy = []


for scenario in scenarios:
    scenario_path = orig_param_path + scenario
    family_size_balance_variations = get_only_list_folders(scenario_path, sep)
    json_f = open(times_info_path+scenario+sep+"GUI_components_detection_times.json")
    times = json.load(json_f)
    for n in family_size_balance_variations:
        metainfo = n.split("_")
        # path example of decision tree specification: agosuirpa\CSV_exit\resources\version1637144717955\scenario_1\Basic_10_Imbalanced\decision_tree.log
        decision_tree_path = scenario_path + sep + n + sep + decision_tree_filename
        
        family.append(metainfo[0])
        log_size.append(metainfo[1])
        scenario_number.append(scenario.split("_")[1])
        balanced.append(1 if metainfo[2]=="Balanced" else 0) # 1 == Balanced, 0 == Imbalanced
        detection_time.append(times[n]["0"]["start"])
        classification_time.append(times[n]["1"]["start"])
        flat_time.append(times[n]["2"]["start"])
        tree_training_time.append(times[n]["3"]["start"])

        with open(scenario_path + sep + n + sep + preprocessed_log_filename, newline='') as f:
            csv_reader = csv.reader(f)
            csv_headings = next(csv_reader)
        log_column.append(len(csv_headings))
        
        # Calculate level of accuracy
        f = open(decision_tree_path,"r").read()
        position = f.find(gui_component_class+"_"+activity)
        if position != -1:
            positions = [m.start() for m in re.finditer(gui_component_class+"_"+activity, f)]
            if len(positions) == 2:
                res_aux = {}
                for index, position_i in enumerate(positions):
                    position_aux = position_i + len(gui_component_class+"_"+activity)
                    s = f[position_aux:]
                    end_position = s.find("\n")
                    quantity = f[position_aux:position_aux+end_position]
                    for c in '<>= ':
                        quantity = quantity.replace(c, '')
                        res_aux[index] = quantity
                if float(res_aux[0])-float(res_aux[1]) < quantity_difference:
                    res = 1
            else:
                res = 0
        else:
            res = 0
        accuracy.append(res)

dict_results = {
    'family': family,
    'balanced': balanced,
    'log_size': log_size,
    'scenario_number': scenario_number,
    'detection_time': detection_time,
    'classification_time': classification_time,
    'flat_time': flat_time,
    'tree_training_time': tree_training_time,
    'log_column': log_column,
    'accuracy': accuracy
}
    
df = pd.DataFrame(dict_results)
df.to_csv(experiment_path+"experiment_results" + str(round(time.time() * 1000)) + ".csv")