#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime

folder_names = ["Basic_10_Balanced", "Basic_10_Imbalanced(noV1ocurrences)","Basic_100_Balanced","Basic_100_Imbalanced","Basic_1000_Balanced","Basic_1000_Imbalanced"]

times = {}

for n in folder_names:
    start = datetime.now().strftime("%H:%M:%S.%MS")
    times[n] = {"start": start}
    s1 = 'python GUI_components_detection.py ../../case-study/'+n+'/log.csv ../../case-study/'+n+'/'
    s2 = 'python GUI_classification.py media/models/model.json media/models/model.h5 ../../case-study/'+n+'/components_npy/ ../../case-study/'+n+'/log.csv ../../case-study/'+n+'/enriched_log.csv'
    s3 = 'python Extract_training_dataset.py B ../../case-study/'+n+'/enriched_log.csv ../../case-study/'+n+'/'
    s4 = 'python Decision_tree.py ../../case-study/'+n+'/preprocessed_dataset.csv ../../case-study/'+n+'/'
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