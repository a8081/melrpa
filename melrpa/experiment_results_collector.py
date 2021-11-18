
#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime
import time
import sys
from basic_case_study_util import get_only_list_folders

# Configuration data
sep = "/"
version = "version1637231842890"
scenario_size = True
orig_param_path =  ".."+sep+".."+sep+"agosuirpa"+sep+"CSV_exit"+sep+"resources"+sep+version+sep
prefix_scenario = "scenario_"
scenarios = []

# Expected results
gui_component_class = "ImageView"
activity = "B"
quantity_difference = 1

family_names = get_only_list_folders(orig_param_path+sep+prefix_scenario+"0")

scenarios = get_only_list_folders(orig_param_path)
print("Scenarios: " + str(scenarios))
   
for scenario in scenarios:
    param_path = orig_param_path + scenario + sep
    family_size_balance_variations = get_only_list_folders(orig_param_path)
    for n in family_size_balance_variations:
    # path example of decision tree specification: agosuirpa\CSV_exit\resources\version1637144717955\scenario_1\Basic_10_Imbalanced\decision_tree.log
    #     times[n] = {}
    #     for index, s in enumerate([s1,s2,s3,s4]):
    #         start = datetime.now().strftime("%H:%M:%S.%MS")
    #         times[n][index] = {"start": start}
    #         os.system(s)
    #         times[n][index]["finish"] = datetime.now().strftime("%H:%M:%S.%MS")
    # metadata_path = "media"+sep+scenario+"_"+str(round(time.time() * 1000))+"_metadata"+sep
    # if not os.path.exists(metadata_path):
    #     os.makedirs(metadata_path)
    # f = open(generate_path+"log.csv", 'w',newline='')
    # writer = csv.writer(f)
   
    # Serializing json
    # json_object = json.dumps(times, indent = 4)
    # Writing to .json
    # with open(metadata_path+"GUI_components_detection_times.json", "w") as outfile:
    #     outfile.write(json_object)