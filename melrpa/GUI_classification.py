#!C:/Users/Antonio/Documents/TFM/melrpa/melrpa/env/Scripts/python.exe

# Comando para hacer uso del script:
# ./GUI_classification.py media/models/model.json media/models/model.h5 media/screenshots/components_npy/ media/LogExample.csv

# Importaciones
import sys
from featureextraction.views import pad, classify_image_components

# Parametros que se pasan por consola

param_json_file_name = sys.argv[1] if len(sys.argv) > 4 else "media/models/model.json"
param_model_weights = sys.argv[2] if len(sys.argv) > 4 else "media/models/model.h5"
param_images_root = sys.argv[3] if len(sys.argv) > 4 else "media/screenshots/components_npy/"
param_log_path = sys.argv[4] if len(sys.argv) > 4 else "media/log.csv"
param_output_path = sys.argv[5] if len(sys.argv) > 5 else "media/enriched_log_feature_extracted.csv"


log_enriched = classify_image_components(param_json_file_name, param_model_weights, param_images_root, param_log_path, pad)
    
log_enriched.to_csv(param_output_path)
# print("\n\n=========== ENRICHED LOG GENERATED: path=" + param_output_path)