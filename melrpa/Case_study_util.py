
#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import os
import json
from datetime import datetime
import sys
import csv
import pandas as pd 
import re
from tqdm import tqdm
from time import sleep
from melrpa.settings import sep, agosuirpa_path


def get_only_list_folders(path, sep):
    folders_and_files = os.listdir(path)
    family_names = []
    for f in folders_and_files:
        if os.path.isdir(path+sep+f):
            family_names.append(f)
    return family_names

# generate_case_study("version1637144717955", "/", True, None)
def generate_case_study(version, sep, p, experiment_name):
    # Parametros que se pasan por consola
    orig_param_path = p if p else agosuirpa_path+sep+"CSV_exit"+sep+"resources"+sep+version#"..\\..\\case-study\\"
    prefix_scenario = "scenario_"

    family_names = get_only_list_folders(orig_param_path+sep+prefix_scenario+"0", sep)

    # for i in range(0,scenario_size+1):
    #     scenarios.append("scenario_"+str(i))
    # print("Scenarios: " + str(scenarios))
    
    scenarios = get_only_list_folders(orig_param_path, sep)
    times = {}

    metadata_path = "media"+sep+experiment_name+"_metadata"+sep
    if not os.path.exists(metadata_path):
            os.makedirs(metadata_path)
            
    for scenario in tqdm(scenarios,
                desc ="Scenarios that have been processed"):
        sleep(.1)
        
        param_path = orig_param_path + sep + scenario + sep
        for n in family_names:
            times[n] = {}
            s1 = 'python GUI_components_detection.py '+param_path+n+sep+'log.csv '+param_path+n+sep
            s2 = 'python GUI_classification.py media'+sep+'models'+sep+'model.json media'+sep+'models'+sep+'model.h5 '+param_path+n+sep+'components_npy'+sep+' '+param_path+n+sep+'log.csv '+param_path+n+sep+'enriched_log.csv'
            s3 = 'python Extract_training_dataset.py B '+param_path+n+sep+'enriched_log.csv '+param_path+n+sep
            s4 = 'python Decision_tree.py '+param_path+n+sep+'preprocessed_dataset.csv '+param_path+n+sep+' ' + 'autogeneration'# autogeneration mode is selected to not # printing decision trees images
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


def times_duration(times_dict):
    format = '%H:%M:%S.%fS'
    difference = datetime.strptime(times_dict["finish"], format) - datetime.strptime(times_dict["start"], format)
    # seconds_in_day = 24 * 60 * 60
    # res = difference.days * seconds_in_day + difference.seconds
    return difference.total_seconds()


def calculate_accuracy_per_tree(decision_tree_path,levels,quantity_difference):
    f = open(decision_tree_path,"r").read()
    res = 1
    
    if not isinstance(levels, list):
        levels = [levels]
    
    for gui_component_class in levels:
        if len(gui_component_class)==1:
            gui_component_name_to_find = gui_component_class[0]
        else:
            gui_component_name_to_find = gui_component_class[0]+"_"+gui_component_class[1]
        position = f.find(gui_component_name_to_find)
        if position != -1:
            positions = [m.start() for m in re.finditer(gui_component_name_to_find, f)]
            if len(positions) == 2:
                res_aux = {}
                for index, position_i in enumerate(positions):
                    position_aux = position_i + len(gui_component_name_to_find)
                    s = f[position_aux:]
                    end_position = s.find("\n")
                    quantity = f[position_aux:position_aux+end_position]
                    for c in '<>= ':
                        quantity = quantity.replace(c, '')
                        res_aux[index] = quantity
                if float(res_aux[0])-float(res_aux[1]) > quantity_difference:
                    print("GUI component quantity difference greater than the expected")
                    res *= 0
            else:
                print("GUI component appears more than twice")
                res *= 0
        else:
            print("GUI component " + gui_component_name_to_find +" not found")
            res *= 0
    return res

def experiments_results_collectors(sep,version,times_path,gui_component_class,quantity_difference,decision_tree_filename,experiment_path,drop,orig_param_path):
    # Configuration data
    if not orig_param_path:
        orig_param_path =  agosuirpa_path+sep+"CSV_exit"+sep+"resources"+sep+version+sep
    decision_tree_filename = "decision_tree.log"
    
    scenarios = []
    times_info_path = "media"+sep+times_path+sep
    preprocessed_log_filename = "preprocessed_dataset.csv"


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


    for scenario in tqdm(scenarios,
                    desc ="Experiment results that have been processed"):
        sleep(.1)
        scenario_path = orig_param_path + scenario
        family_size_balance_variations = get_only_list_folders(scenario_path, sep)
        if drop and drop in family_size_balance_variations:
            family_size_balance_variations.remove(drop)
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
            accuracy.append(calculate_accuracy_per_tree(decision_tree_path,gui_component_class,quantity_difference))

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
    df.to_csv(experiment_path+experiment_name+"_results" + ".csv")
    return experiment_path+experiment_name+"_results" + ".csv"
    
    
#python Case_study_util.py version1637410905864_80_20 && python Case_study_util.py version1637410907926_70_30 && python Case_study_util.py version1637410968920_60_40
if __name__ == '__main__':
    # generate_case_study("version1637144717955", "/", True, None)
    version_name = sys.argv[1] if len(sys.argv) > 1 else "version1637695142490"
    mode = sys.argv[2] if len(sys.argv) > 2 else "generate"
    path_to_save_experiment = sys.argv[3] if len(sys.argv) > 3 else None
    
    experiment_name = "experiment_" + version_name
    
    if mode=="generate" or mode=="both":
        generate_case_study(version_name, sep, path_to_save_experiment, experiment_name)
    
    times_path = experiment_name + "_metadata"
    prefix_scenario = "scenario_"
    decision_tree_filename = "decision_tree.log"
    experiment_path = "media" + sep
    drop = None # ["Advanced_10_Balanced", "Advanced_10_Imbalanced"]
    
    # Expected results
    
    # It is necessary to specify first the name of the GUI component and next the activity where iit takes place
    # In case of other column, you must specify only its name: for example ["Case"]
    gui_component_class = [["Case"], ["Coor_Y","A"]]
    quantity_difference = 1
    
    if mode=="experiment" or mode=="both":
        if path_to_save_experiment and path_to_save_experiment.find(sep)==-1:
            path_to_save_experiment = path_to_save_experiment + sep
        experiments_results_collectors(sep,version_name,times_path,gui_component_class,quantity_difference,decision_tree_filename,experiment_path,drop,path_to_save_experiment)