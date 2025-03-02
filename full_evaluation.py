import os
import shutil
from draw_bndbox import process_images_in_folder
from filter_defect import process_images_in_directory
from filter_defect import load_config
from correct_classname import update_class_names_in_directory
from evaluation import evaluate_directory_structure_by_class
from evaluation import suggest_best_threshold
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

# Setting for 
# size_offsets = {
#     'impression': 0.95,  # Green for 'impression'
#     'einriss': 1,     # Blue for 'einriss'
#     'asperity': 1,    # Red for 'asperity'
#     'abriss': 1.25,    # Cyan for 'abriss'
#     'ausseinriss': 1        # Magenta for 'oil'
# }

# # Define confidence thresholds per class
# confidence_thresholds = {
#     "impression": 0.3,
#     "asperity": 0,
#     "abriss": 0.25,
#     "einriss": 0.5,
#     "ausseinriss": 0.2
# }

size_offsets = {
    'impression': 1,  # Green for 'impression'
    'einriss': 1,     # Blue for 'einriss'
    'asperity': 1,    # Red for 'asperity'
    'abriss': 1.25,    # Cyan for 'abriss'
    'ausseinriss': 1        # Magenta for 'oil'
}

# Define confidence thresholds per class
confidence_thresholds = {
    "impression": 0,
    "asperity": 0,
    "abriss": 0,
    "einriss": 0,
    "ausseinriss": 0
}

# Load the JSON configuration from a file
config_data = load_config("config.json")

# TODO - Update the model name here
model_name = "result_rtmdet_0226"
confidence = "01"
batch = 1

# Directories
image_directory = 'split\\batch_'+str(batch)+'\\images'
label_gt_directory = 'split\\batch_'+str(batch)+'\\labels'
base_directory = 'split/batch_'+str(batch)+'/'+model_name
label_raw_directory = 'split\\batch_'+str(batch)+'\\'+model_name+'\\'+confidence+'\\label_xml_unfilter'
result_imagecrop_rawlabel_directory = 'split\\batch_'+str(batch)+'\\'+model_name+'\\'+confidence+'\\image_unfilter_crop'
labelcrop_raw_directory = 'split\\batch_'+str(batch)+'\\'+model_name+'\\'+confidence+'\\label_xml_unfilter_crop'
labelcrop_filter_directory = 'split\\batch_'+str(batch)+'\\'+model_name+'\\'+confidence+'\\labels'

# Replace classname incase of mistake
old_class_name = 'ausenriss'  # Class name to be replaced
new_class_name = 'ausseinriss'  # New class name
update_class_names_in_directory(label_raw_directory, old_class_name, new_class_name)

# if not os.path.exists(result_imagecrop_rawlabel_directory):
#     os.makedirs(result_imagecrop_rawlabel_directory)
# else:
#     shutil.rmtree(result_imagecrop_rawlabel_directory)
#     os.makedirs(result_imagecrop_rawlabel_directory)
    
# if not os.path.exists(labelcrop_raw_directory):
#     os.makedirs(labelcrop_raw_directory)
# else:
#     shutil.rmtree(labelcrop_raw_directory)
#     os.makedirs(labelcrop_raw_directory)
    
# # Draw all bounding boxes to the images (include all pre-filter)
# print("Drawing raw label bounding boxes on the images...")
# process_images_in_folder(image_directory, label_raw_directory, result_imagecrop_rawlabel_directory, labelcrop_raw_directory, class_colors)

if not os.path.exists(labelcrop_filter_directory):
    os.makedirs(labelcrop_filter_directory)
else:
    shutil.rmtree(labelcrop_filter_directory)
    os.makedirs(labelcrop_filter_directory)

# Filter small reject base on defect size
print("Filtering small reject base on defect size...")
process_images_in_directory(image_directory, label_raw_directory, labelcrop_filter_directory, config_data, size_offsets, confidence_thresholds)

# Create result file
print("Creating result file...")
precision_recall_data = evaluate_directory_structure_by_class(base_directory, image_directory, label_gt_directory, confidence_thresholds)

# Suggest Best Threshold
# best_threshold = suggest_best_threshold(base_directory, image_directory, label_gt_directory)

# Plot Precision-Recall Curve
# plot_precision_recall(precision_recall_data)