import os
import shutil
import datetime
import json
import argparse
import glob
from draw_bndbox import process_images_in_folder
from filter_defect import process_images_in_directory
from filter_defect import load_config
from correct_classname import update_class_names_in_directory
from evaluation import evaluate_directory_structure_by_class
from evaluation import plot_precision_recall
from error_analysis import generate_error_analysis

# Define a dictionary of class names and their corresponding bounding box colors (BGR format)
class_colors = {
    'impression': (0, 255, 0),  # Green for 'impression'
    'einriss': (255, 0, 0),     # Blue for 'einriss'
    'asperity': (0, 0, 255),    # Red for 'asperity'
    'abriss': (255, 255, 0),    # Yellow for 'abriss'
    'ausseinriss': (0, 255, 255),    # Cyan for 'ausseinriss'
    'oil': (255, 0, 255)        # Magenta for 'oil'
}

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run full evaluation pipeline for defect detection')
parser.add_argument('--test-config', action='store_true', help='Use ground truth from previous run')
args = parser.parse_args()

# Load configurations from JSON files
defect_thresholds = load_config("config_defect_thresholds.json")
size_offsets = load_config("config_size_offsets.json")
confidence_thresholds = load_config("config_confidence_thresholds.json")

print("Defect threshold: ",defect_thresholds)
print("Size offsets: ",size_offsets)
print("Confidence threshold: ", confidence_thresholds)

# Function to find latest run folder
def find_latest_full_run(model_dir):
    run_folders = glob.glob(os.path.join(model_dir, 'run_*'))
    if not run_folders:
        return None
    # Filter folders that contain the specified subfolder
    valid_folders = [folder for folder in run_folders if os.path.isdir(os.path.join(folder, "image_unfilter_crop"))]
    
    if not valid_folders:
        return None
    
    # Sort by creation time (newest first)
    valid_folders.sort(key=os.path.getctime, reverse=True)
    return os.path.basename(valid_folders[0])

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

# Find the latest run if using --test-config
last_run_folder = None
if args.test_config:
    last_run_folder = find_latest_full_run(f"prediction/{model_name}")
    if not last_run_folder:
        print("Warning: No previous run found. Will create new ground truth labels.")
    else:
        print(f"Using ground truth from previous run: {last_run_folder}")

# Directories
image_directory = 'images\\' + dataset_version
label_gt_raw_directory = 'labels\\' + dataset_version
base_directory = 'prediction/' + model_name + '/' + result_folder
model_directory = 'prediction/' + model_name
label_raw_directory = model_directory + '\\labels'
imagecrop_rawlabel_directory = base_directory + '\\image_unfilter_crop'
label_filter_directory = base_directory + '\\label_xml_filter'
label_gt_filter_directory = base_directory + '\\gt_labels_filtered'

# Set previous run directory if using --test-config
prev_run_directory = None
if last_run_folder:
    prev_run_directory = 'prediction/' + model_name + '/' + last_run_folder

# Create the result folder and copy the configuration files
os.makedirs(base_directory, exist_ok=True)
config_folder = os.path.join(base_directory, 'configs')
os.makedirs(config_folder, exist_ok=True)

# Save current configurations to the result folder
save_config(defect_thresholds, os.path.join(config_folder, 'config_defect_thresholds.json'))
save_config(size_offsets, os.path.join(config_folder, 'config_size_offsets.json'))
save_config(confidence_thresholds, os.path.join(config_folder, 'config_confidence_thresholds.json'))

# Replace classname in case of mistake
old_class_name = 'ausenriss'  # Class name to be replaced
new_class_name = 'ausseinriss'  # New class name
print("Checking and correcting class names...")
update_class_names_in_directory(label_raw_directory, old_class_name, new_class_name)

os.makedirs(label_filter_directory, exist_ok=True)
# Filter small reject base on defect size for prediction labels
print("Filtering small reject base on defect size for prediction labels...")
process_images_in_directory(image_directory, label_raw_directory, label_filter_directory, defect_thresholds, size_offsets, confidence_thresholds)

# Filter small reject base on defect size for ground truth labels
if args.test_config and prev_run_directory and os.path.exists(os.path.join(prev_run_directory, 'gt_labels_filtered')):
    prev_gt_dir = os.path.join(prev_run_directory, 'gt_labels_filtered')
    label_gt_filter_directory = prev_gt_dir
else:
    os.makedirs(label_gt_filter_directory, exist_ok=True)
    # Generate new ground truth labels
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

# Draw all bounding boxes to the images (include all pre-filter)
if args.test_config and prev_run_directory and os.path.exists(os.path.join(prev_run_directory, 'image_unfilter_crop')):
    prev_gt_dir = os.path.join(prev_run_directory, 'image_unfilter_crop')
    imagecrop_rawlabel_directory = prev_gt_dir
else:
    os.makedirs(imagecrop_rawlabel_directory, exist_ok=True)
    print("Drawing raw label bounding boxes on the images...")
    process_images_in_folder(image_directory, label_raw_directory, imagecrop_rawlabel_directory, class_colors)

# Generate error analysis visualizations and confusion matrices
print("Generating error analysis and confusion matrices...")
analysis_dir = generate_error_analysis(model_name, dataset_version, result_folder, imagecrop_rawlabel_directory)

print(f"\nEvaluation complete! Results saved to: {base_directory}")
print(f"Configuration files saved to: {config_folder}")
print(f"Error analysis and confusion matrices saved to: {analysis_dir}")