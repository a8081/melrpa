import os
import json
import time
import csv
import pandas as pd
import re
from tqdm import tqdm
from time import sleep
from melrpa.settings import agosuirpa_path, times_calculation_mode, metadata_location
from datetime import datetime
from decisiondiscovery.views import decision_tree_training, extract_training_dataset
from featureextraction.views import gui_components_detection, classify_image_components

def get_only_list_folders(path, sep):
    folders_and_files = os.listdir(path)
    family_names = []
    for f in folders_and_files:
        if os.path.isdir(path+sep+f):
            family_names.append(f)
    return family_names

def generate_case_study(version, sep, p, experiment_name, decision_activity, scenarios, to_exec):
    # Parametros que se pasan por consola
    orig_param_path = p if p else agosuirpa_path+sep+"CSV_exit" + \
        sep+"resources"+sep+version  # "..\\..\\case-study\\"

    # prefix_scenario = "scenario_"
    # for i in range(0,scenario_size+1):
    #     scenarios.append("scenario_"+str(i))
    # print("Scenarios: " + str(scenarios))

    if not scenarios:
        scenarios = get_only_list_folders(orig_param_path, sep)
    times = {}

    family_names = get_only_list_folders(orig_param_path+sep+scenarios[0], sep)

    metadata_path = metadata_location+sep+experiment_name+"_metadata"+sep
    if not os.path.exists(metadata_path):
        os.makedirs(metadata_path)

    for scenario in tqdm(scenarios,
                         desc="Scenarios that have been processed"):
        sleep(.1)

        param_path = orig_param_path + sep + scenario + sep
        if to_exec and len(to_exec) > 0:
            for n in family_names:
                times[n] = {}

                to_exec_args = [(param_path+n+sep+'log.csv', param_path+n+sep),
                                ('media'+sep+'models'+sep+'model.json', 'media'+sep+'models'+sep+'model.h5', param_path+n +
                                sep+'components_npy'+sep, param_path+n+sep + 'log.csv', param_path+n+sep+'enriched_log.csv'),
                                (decision_activity, param_path + n+sep +
                                'enriched_log.csv', param_path+n+sep),
                                (param_path+n+sep + 'preprocessed_dataset.csv', param_path+n+sep, 'autogeneration')]
                for index, function_to_exec in enumerate(to_exec):
                    times[n][index] = {"start": time.time()}
                    output = eval(function_to_exec)(*to_exec_args[index])
                    times[n][index]["finish"] = time.time()
                    if index == len(to_exec)-1:
                        times[n][index]["decision_model_accuracy"] = output

            # if not os.path.exists(scenario+sep):
            #     os.makedirs(scenario+sep)

            # Serializing json
            json_object = json.dumps(times, indent=4)
            # Writing to .json
            with open(metadata_path+scenario+"-metainfo.json", "w") as outfile:
                outfile.write(json_object)
    # cada experimento una linea: csv
    # almaceno los tiempos por cada fase y por cada experimento (por cada familia hay 30)
    # ejecutar solamente los experimentos


def times_duration(times_dict):
    if times_calculation_mode == "legacy":
        format = "%H:%M:%S.%fS"
        difference = datetime.strptime(times_dict["finish"], format) - datetime.strptime(times_dict["start"], format)
        res = difference.total_seconds()
    else:
        res = float(times_dict["finish"]) - float(times_dict["start"])
    return res


def calculate_accuracy_per_tree(decision_tree_path, expression, quantity_difference):
    f = open(decision_tree_path, "r").read()
    res = {}
    
    # This code is useful if we want to get the expresion like: [["TextView", "B"],["ImageView", "B"]]
    # if not isinstance(levels, list):
    #     levels = [levels]
    levels = expression.replace("(", "")
    levels = levels.replace(")", "")
    levels = levels.split(" ")
    for op in ["and","or"]:
      while op in levels:
        levels.remove(op)

    for gui_component_name_to_find in levels:
    # This code is useful if we want to get the expresion like: [["TextView", "B"],["ImageView", "B"]]
    # for gui_component_class in levels:
        # if len(gui_component_class) == 1:
        #     gui_component_name_to_find = gui_component_class[0]
        # else:
        #     gui_component_name_to_find = gui_component_class[0] + \
        #         "_"+gui_component_class[1]
        position = f.find(gui_component_name_to_find)
        res[gui_component_name_to_find] = "False"
        if position != -1:
            positions = [m.start()
                         for m in re.finditer(gui_component_name_to_find, f)]
            number_of_nodes = int(len(positions)/2)
            if len(positions) != 2:
                print("GUI component appears more than twice")
            for n_nod in range(0, number_of_nodes):
                res_partial = {}
                for index, position_i in enumerate(positions):
                    position_i += 2*n_nod
                    position_aux = position_i + len(gui_component_name_to_find)
                    s = f[position_aux:]
                    end_position = s.find("\n")
                    quantity = f[position_aux:position_aux+end_position]
                    for c in '<>= ':
                        quantity = quantity.replace(c, '')
                        res_partial[index] = quantity
                if float(res_partial[0])-float(res_partial[1]) > quantity_difference:
                    print("GUI component quantity difference greater than the expected")
                else:
                    res[gui_component_name_to_find] = "True"

    s = expression
    print(res)
    for gui_component_name_to_find in levels:
        s = s.replace(gui_component_name_to_find, res[gui_component_name_to_find])
    
    res = eval(s)
    
    if not res:
      print("Condition " + str(expression) + " is not fulfilled")
    return int(res)


def experiments_results_collectors(scenarios, sep, version, times_path, gui_component_class, quantity_difference, decision_tree_filename, experiment_path, drop, orig_param_path, experiment_name):
    # Configuration data
    if not orig_param_path:
        orig_param_path = agosuirpa_path+sep+"CSV_exit"+sep+"resources"+sep+version+sep
    decision_tree_filename = "decision_tree.log"

    times_info_path = "media"+sep+times_path+sep
    preprocessed_log_filename = "preprocessed_dataset.csv"

    if not scenarios:
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
    tree_training_accuracy = []
    log_column = []
    accuracy = []

    for scenario in tqdm(scenarios,
                         desc="Experiment results that have been processed"):
        sleep(.1)
        scenario_path = orig_param_path + scenario
        family_size_balance_variations = get_only_list_folders(
            scenario_path, sep)
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
            # 1 == Balanced, 0 == Imbalanced
            balanced.append(1 if metainfo[2] == "Balanced" else 0)
            detection_time.append(times_duration(times[n]["0"]))
            classification_time.append(times_duration(times[n]["1"]))
            flat_time.append(times_duration(times[n]["2"]))
            tree_training_time.append(times_duration(times[n]["3"]))
            tree_training_accuracy.append(times[n]["3"]["decision_model_accuracy"])

            with open(scenario_path + sep + n + sep + preprocessed_log_filename, newline='') as f:
                csv_reader = csv.reader(f)
                csv_headings = next(csv_reader)
            log_column.append(len(csv_headings))

            # Calculate level of accuracy
            accuracy.append(calculate_accuracy_per_tree(
                decision_tree_path, gui_component_class, quantity_difference))

    dict_results = {
        'family': family,
        'balanced': balanced,
        'log_size': log_size,
        'scenario_number': scenario_number,
        'detection_time': detection_time,
        'classification_time': classification_time,
        'flat_time': flat_time,
        'tree_training_time': tree_training_time,
        'tree_training_accuracy': tree_training_accuracy,
        'log_column': log_column,
        'accuracy': accuracy
    }

    df = pd.DataFrame(dict_results)
    df.to_csv(experiment_path+experiment_name+"_results" + ".csv")
    return experiment_path+experiment_name+"_results" + ".csv"
