#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe

# Comando para hacer uso del script:
# ./GUI_classification.py media/models/model.json media/models/model.h5 media/GUIcomponents/ media/LogExample.csv

# Importaciones
import sys
from featureextraction.views import pad, classify_image_components

# Parametros que se pasan por consola
param_json_file_name=sys.argv[1]
param_model_weights=sys.argv[2]
param_images_root=sys.argv[3]
param_log_path=sys.argv[4]

# arr_balance = np.load('../media/models/preprocessed_50_50_balance.npy')
# arr_labels_balance = np.load('../media/models/test_labels_balance.npy')

log_enriched = classify_image_components(param_json_file_name, param_model_weights, param_images_root, param_log_path, pad)
    
log_enriched.to_csv('media/enriched-log-feature-extracted.csv')
print("=========== ENRICHED LOG GENERATED: path=media/enriched-log-feature-extracted.csv")