
#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe
import sys
from melrpa.settings import sep, scenarios_subset
from melrpa.views import generate_case_study, experiments_results_collectors
from featureextraction.views import check_npy_components_of_capture

# python Case_study_util.py version1637410905864_80_20 && python Case_study_util.py version1637410907926_70_30 && python Case_study_util.py version1637410968920_60_40


def case_study_generator(version_name, sep, path_to_save_experiment,
                         decision_activity, gui_component_class,
                         quantity_difference, drop):
    experiment_name = "experiment_" + version_name
    if mode == "generation" or mode == "both":
        to_execute = ['gui_components_detection'
                    # ,'classify_image_components',
                    # 'extract_training_dataset',
                    # 'decision_tree_training'
                    ]
        generate_case_study(version_name, sep, path_to_save_experiment,
                            experiment_name, decision_activity, scenarios_subset, to_execute)

    if mode == "results" or mode == "both":
        if path_to_save_experiment and path_to_save_experiment.find(sep) == -1:
            path_to_save_experiment = path_to_save_experiment + sep
        experiments_results_collectors(sep, version_name, experiment_name + "_metadata", gui_component_class,
                                       quantity_difference, "decision_tree.log", "media" + sep, drop, path_to_save_experiment, experiment_name)

def interactive_terminal(gui_component_class, quantity_difference, drop):
    version_name = input(
            'Enter the name of the folder generated by AGOSUIRPA with your experiment data (enter "UTILS" to check utilities): ')
    if version_name != "UTILS":
        decision_activity = input(
            'Enter the activity immediately preceding the decision point you wish to study: ')
        mode = input(
            'Enter if you want to obtain experiment "generation", "results" or "both": ')

        if(mode in 'generation results both'):
            if mode == "results" or mode == "both":
                input_exp_path = input(
                    'Enter path where you want to store experiment results (if nothing typed, it will be stored in "media/"): ')
                path_to_save_experiment = input_exp_path if input_exp_path != "" else None

            case_study_generator(version_name, sep, path_to_save_experiment,
                                    decision_activity, gui_component_class,
                                    quantity_difference, drop)
        else:
            print('Please enter valid input')
    else:
        check_npy_components_of_capture(None, None, True)

if __name__ == '__main__':
    # ## Expected results ## EXAMPLE: [["Case"],["ImageView", "D"]]
    # It is necessary to specify first the name of the GUI component and next the activity where iit takes place
    # In case of other column, you must specify only its name: for example ["Case"]
    gui_component_class = [["TextView", "D"]]
    quantity_difference = 1

    path_to_save_experiment = None
    drop = None  # ["Advanced_10_Balanced", "Advanced_10_Imbalanced"]
    interactive = False

    if interactive:
        interactive_terminal(gui_component_class, quantity_difference, drop)
    else:
        version_name = sys.argv[1] if len(
            sys.argv) > 1 else "Intermediate"
        decision_activity = sys.argv[2] if len(sys.argv) > 2 else "D"
        mode = sys.argv[3] if len(sys.argv) > 3 else "both"
        path_to_save_experiment = sys.argv[4] if len(sys.argv) > 4 else None
        
        case_study_generator(version_name, sep, path_to_save_experiment,
                                     decision_activity, gui_component_class,
                                     quantity_difference, drop)