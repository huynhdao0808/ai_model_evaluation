import os
import shutil
import datetime
import json
from draw_bndbox import process_images_in_folder
from filter_defect import process_images_in_directory
from filter_defect import load_config
from correct_classname import update_class_names_in_directory
from evaluation import evaluate_directory_structure_by_class
from evaluation import plot_precision_recall

# Define a dictionary of class names and their corresponding bounding box colors (BGR format)
class_colors = {
    'impression': (0, 255, 0),  # Green for 'impression'
    'einriss': (255, 0, 0),     # Blue for 'einriss'
    'asperity': (0, 0, 255),    # Red for 'asperity'
    'abriss': (255, 255, 0),    # Yellow for 'abriss'
    'ausseinriss': (0, 255, 255),    # Cyan for 'ausseinriss'
    'oil': (255, 0, 255)        # Magenta for 'oil'
}

# Load configurations from JSON files
defect_thresholds = load_config("config_defect_thresholds.json")
size_offsets = load_config("config_size_offsets.json")
confidence_thresholds = load_config("config_confidence_thresholds.json")

# Function to save a JSON config to a file
def save_config(config, file_path):
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)

# TODO - Update the model name and dataset version here
model_name = "rtdert_2.0"
dataset_version = "test1_v1"  # Full folder name of the dataset version

# Create a timestamped result folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = f"run_{timestamp}"

# Directories
image_directory = 'images\\' + dataset_version
label_gt_raw_directory = 'labels\\' + dataset_version
base_directory = 'prediction/' + model_name + '/' + result_folder
model_directory = 'prediction/' + model_name
label_raw_directory = base_directory + '\\labels'
result_imagecrop_rawlabel_directory = base_directory + '\\image_unfilter_crop'
labelcrop_raw_directory = base_directory + '\\label_xml_unfilter_crop'
label_filter_directory = base_directory + '\\label_xml_filter'
label_gt_filter_directory = base_directory + '\\gt_labels_filtered'

# Create the result folder and copy the configuration files
os.makedirs(base_directory, exist_ok=True)
config_folder = os.path.join(base_directory, 'configs')
os.makedirs(config_folder, exist_ok=True)

# Save current configurations to the result folder
save_config(defect_thresholds, os.path.join(config_folder, 'config_defect_thresholds.json'))
save_config(size_offsets, os.path.join(config_folder, 'config_size_offsets.json'))
save_config(confidence_thresholds, os.path.join(config_folder, 'config_confidence_thresholds.json'))

# Create directories for processing
os.makedirs(label_raw_directory, exist_ok=True)
os.makedirs(result_imagecrop_rawlabel_directory, exist_ok=True)
os.makedirs(labelcrop_raw_directory, exist_ok=True)
os.makedirs(label_filter_directory, exist_ok=True)
os.makedirs(label_gt_filter_directory, exist_ok=True)

# Copy prediction XML files to the run folder
print("Copying prediction files to the run folder...")
if os.path.exists(os.path.join(model_directory, 'labels')):
    for file in os.listdir(os.path.join(model_directory, 'labels')):
        if file.endswith('.xml'):
            shutil.copy(
                os.path.join(model_directory, 'labels', file),
                os.path.join(label_raw_directory, file)
            )

# Replace classname in case of mistake
old_class_name = 'ausenriss'  # Class name to be replaced
new_class_name = 'ausseinriss'  # New class name
print("Checking and correcting class names...")
update_class_names_in_directory(label_raw_directory, old_class_name, new_class_name)

# Draw all bounding boxes to the images (include all pre-filter)
print("Drawing raw label bounding boxes on the images...")
process_images_in_folder(image_directory, label_raw_directory, result_imagecrop_rawlabel_directory, labelcrop_raw_directory, class_colors)

# Filter small reject base on defect size for prediction labels
print("Filtering small reject base on defect size for prediction labels...")
process_images_in_directory(image_directory, label_raw_directory, label_filter_directory, defect_thresholds, size_offsets, confidence_thresholds)

# Filter small reject base on defect size for ground truth labels
print("Filtering small reject base on defect size for ground truth labels...")
process_images_in_directory(image_directory, label_gt_raw_directory, label_gt_filter_directory, defect_thresholds, {
    "impression": 1.0,
    "einriss": 1.0,
    "asperity": 1.0,
    "abriss": 1.0,
    "ausseinriss": 1.0
})

# Create result file using filtered ground truth labels
print("Creating result file...")
precision_recall_data = evaluate_directory_structure_by_class(base_directory, image_directory, label_gt_filter_directory, confidence_thresholds)

# Save precision-recall data to the result folder
save_config(precision_recall_data, os.path.join(config_folder, 'precision_recall_results.json'))

# Generate error analysis visualizations and confusion matrices
print("Generating error analysis and confusion matrices...")
from error_analysis import generate_error_analysis
analysis_dir = generate_error_analysis(model_name, dataset_version, result_folder)

print(f"\nEvaluation complete! Results saved to: {base_directory}")
print(f"Configuration files saved to: {config_folder}")
print(f"Error analysis and confusion matrices saved to: {analysis_dir}")