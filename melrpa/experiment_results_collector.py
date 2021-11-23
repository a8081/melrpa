
#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime
import time
import sys
import csv
import pandas as pd 
from Case_study_util import get_only_list_folders
import re
from datetime import datetime

# Configuration data

#"version1637634767647_30_70": "version1637634822250_40_60"
# 1637524907108_metadata
# 1637562073752_metadata
sep = "/"
version = "version1637634822250_40_60"
times_path = "1637635231370_metadata"
scenario_size = True
orig_param_path =  ".."+sep+".."+sep+"agosuirpa"+sep+"CSV_exit"+sep+"resources"+sep+version+sep
prefix_scenario = "scenario_"
decision_tree_filename = "decision_tree.log"
experiment_path = "media" + sep
scenarios = []
times_info_path = "media"+sep+times_path+sep
preprocessed_log_filename = "preprocessed_dataset.csv"

# Expected results
gui_component_class = "ImageView"
activity = "B"
quantity_difference = 1

family_names = get_only_list_folders(orig_param_path+prefix_scenario+"0", sep)

scenarios = get_only_list_folders(orig_param_path, sep)
# print("Scenarios: " + str(scenarios))


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


def times_duration(times_dict):
    format = '%H:%M:%S.%fS'
    difference = datetime.strptime(times_dict["finish"], format) - datetime.strptime(times_dict["start"], format)
    # seconds_in_day = 24 * 60 * 60
    # res = difference.days * seconds_in_day + difference.seconds
    return difference.total_seconds()

for scenario in scenarios:
    scenario_path = orig_param_path + scenario
    family_size_balance_variations = get_only_list_folders(scenario_path, sep)
    json_f = open(times_info_path+scenario+"-metainfo.json")
    times = json.load(json_f)
    for n in family_size_balance_variations:
        metainfo = n.split("_")
        # path example of decision tree specification: agosuirpa\CSV_exit\resources\version1637144717955\scenario_1\Basic_10_Imbalanced\decision_tree.log
        decision_tree_path = scenario_path + sep + n + sep + decision_tree_filename
        
        family.append(metainfo[0])
        log_size.append(metainfo[1])
        scenario_number.append(scenario.split("_")[1])
        balanced.append(1 if metainfo[2]=="Balanced" else 0) # 1 == Balanced, 0 == Imbalanced
        detection_time.append(times_duration(times[n]["0"]))
        classification_time.append(times_duration(times[n]["1"]))
        flat_time.append(times_duration(times[n]["2"]))
        tree_training_time.append(times_duration(times[n]["3"]))

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