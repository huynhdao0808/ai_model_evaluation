import json
import os
import re
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to load JSON from a file
def load_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Function to extract camera name and defect threshold from the filename
def extract_info_from_filename(filename):
    # Regex to extract camera name and defect threshold (e.g., 06 -> AL06 and 20-12)
    match = re.match(r"(\d{2})_(\d{2}-\d{2}(?:AVO|PB)?)_IMG_LOG__CAM_(\d{1})", filename)
    if match:
        line_name = f"AL{match.group(1)}"
        element_type = match.group(2)
        cam_position = int(match.group(3))
        return line_name, element_type, cam_position
    print(filename)
    return None, None, None

# Function to load the XML label from a file
def load_xml_label(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

# Function to calculate object size in mm
def calculate_object_size_in_mm(bbox, resolution):
    """
    Calculate the size of the object in millimeters given its bounding box and the resolution of the camera.
    bbox format: (xmin, ymin, xmax, ymax) in pixels
    resolution format: {'x': mm per pixel, 'z': mm per pixel}
    """
    xmin, ymin, xmax, ymax = bbox
    width_px = xmax - xmin
    height_px = ymax - ymin
    width_mm = width_px * resolution['x']
    height_mm = height_px * resolution['z']
    return width_mm, height_mm

def crop_image(image_path):
    offset = 20
    
    image = cv2.imread(image_path)    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value = np.mean(gray) + np.std(gray)
    _, thresholded_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found!")
        return

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(0, x - offset)
    y = max(0, y - offset)
    w = min(image.shape[1] - x, w + 2 * offset)
    h = min(image.shape[0] - y, h + 2 * offset)
    
    saddle_y = y+h/2
    
    return x, x+w, largest_contour, saddle_y

# Function to remove objects smaller than the threshold size
def remove_small_objects(file_path, xml_file, thresholds, resolution, side, offsets, label_raw_directory, label_filter_directory, confidence_threshold = None):
    """
    Remove objects smaller than the threshold size from the XML file based on their bounding box.
    thresholds: dictionary containing class-specific size thresholds
    resolution: resolution dictionary for converting pixel to mm
    """
    xml_file_path = os.path.join(label_raw_directory, xml_file)
    if not os.path.exists(xml_file_path):
        print(f'{xml_file_path} not exist')
        return
    root = load_xml_label(xml_file_path)
    
    # Iterate through all object elements in the XML file
    objects_to_remove = []
    for obj in root.findall("object"):
        # Extract object class name
        class_name = obj.find("name").text
        
        if class_name == "oil":
            objects_to_remove.append(obj)
            continue
        else:
            if confidence_threshold != None:
                confidence = float(obj.find('confidence').text)
                if confidence < confidence_threshold[class_name]:
                    objects_to_remove.append(obj)
                    continue
        
        # Check if the class is in the thresholds dictionary
        if class_name in thresholds:
            # Extract the bounding box coordinates (xmin, ymin, xmax, ymax)
            bbox = obj.find("bndbox")
            if bbox is not None:
                xmin = int(float(bbox.find("xmin").text))
                ymin = int(float(bbox.find("ymin").text))
                xmax = int(float(bbox.find("xmax").text))
                ymax = int(float(bbox.find("ymax").text))
                
                # Calculate object size in mm
                width_mm, height_mm = calculate_object_size_in_mm((xmin, ymin, xmax, ymax), resolution)
                
                x, x_w, saddle_area, saddle_surface_y = crop_image(file_path)
                # image = cv2.imread(file_path)
                # x, x_w = 0, image.shape[1]

                # Check the box inside the saddle area or not
                corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
                outside_saddle = True
                 # Check each corner of the bounding box
                for point in corners:
                    px, py = point
                    result = cv2.pointPolygonTest(saddle_area, (px, py), False)
                    if result >= 0:  # Inside or on the contour
                        outside_saddle = False
                if outside_saddle:
                    # print(class_name, xml_file)
                    objects_to_remove.append(obj)
                    continue
                
                if outside_saddle:
                    objects_to_remove.append(obj)
                    continue
                
                # Check if the einriss defect is above or below the saddle surface, then reassign the class name to abriss or impression
                below_saddle = False
                above_saddle = False
                outside_saddle = False
                
                if class_name == "einriss":
                    # Check each corner of the bounding box
                    for point in corners:
                        px, py = point
                        result = cv2.pointPolygonTest(saddle_area, (px, py), False)
                        if result <= 0:  # Outsize the contour
                            outside_saddle = True
                            if py > saddle_surface_y:  # Assuming saddle_surface_y is the y-coordinate of the saddle surface
                                below_saddle = True
                                obj.find("name").text = "impression"
                            if py < saddle_surface_y:
                                above_saddle = True
                                obj.find("name").text = "abriss"
                            break
                    
                    # if  ("002950_005" in xml_file_path):
                    #     # Plotting using Matplotlib
                    #     plt.figure(figsize=(6, 6))
                    #     image = cv2.imread(file_path)
                    #     plt.imshow(image, cmap='gray')  # Show the image in grayscale
                    #     contour = saddle_area  # Assume there's only one contour

                    #     # Draw the contour using plt.plot
                    #     for c in contour:
                    #         plt.plot(c[0][0], c[0][1], color='red')  # Plot each point of the contour
                        
                    #     # Draw the bounding box (rectangle) around the contour
                    #     plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='blue', facecolor='none'))  # Blue bounding box

                    #     # Alternatively, if you want to draw the filled contour, you can use plt.fill:
                    #     contour = np.array(contour)  # Ensure it's in the right shape (Nx2)
                    #     reshaped_contour = contour.reshape(-1, 2)  # Reshape to (N, 2)
                    #     plt.fill(reshaped_contour[:, 0], reshaped_contour[:, 1], color='red', alpha=0.3)  # Fill contour with red color

                    #     # Add text labels for "impression" or "abriss"
                    #     if below_saddle:
                    #         plt.text(xmin, ymin - 10, "impression", color='green', fontsize=12, weight='bold')
                    #     if above_saddle:
                    #         plt.text(xmin, ymin - 30, "abriss", color='yellow', fontsize=12, weight='bold')
                        
                    #     plt.title("Contour Drawing")
                    #     plt.axis('off')  # Hide axes for better visualization
                    #     plt.show()
                
                # If defect in the outside of saddle, It would be ausseinriss
                if side == "right":
                    relative_x_outside = (x_w - xmax)*resolution['x'] 
                else:
                    relative_x_outside = (xmin - x)*resolution['x']
                if relative_x_outside < 1 and class_name != "ausseinriss":
                    obj.find("name").text = "ausseinriss"
                    # Get the size threshold for this object class
                    min_width_mm, min_height_mm = thresholds["ausseinriss"]['x'], thresholds["ausseinriss"]['z']
                    # If the object is smaller than the threshold, mark it for removal
                    if width_mm*offsets[obj.find("name").text] < min_width_mm and height_mm*offsets[obj.find("name").text] < min_height_mm:
                        objects_to_remove.append(obj)
                else:
                    # Get the size threshold for this object class
                    min_width_mm, min_height_mm = thresholds[obj.find("name").text]['x'], thresholds[obj.find("name").text]['z']
                    if width_mm*offsets[obj.find("name").text] < min_width_mm and height_mm*offsets[obj.find("name").text] < min_height_mm:
                        objects_to_remove.append(obj)
    
    # Remove the small objects from the XML tree
    for obj in objects_to_remove:
        root.remove(obj)
        
    dest_xml_file_path = os.path.join(label_filter_directory, xml_file)
    # Save the modified XML file
    tree = ET.ElementTree(root)
    tree.write(dest_xml_file_path)
    # print(f"Updated XML file saved: {dest_xml_file_path}")

# Function to process images in the directory
def process_images_in_directory(image_directory, label_raw_directory, label_filter_directory, config_data, offsets, confidence_threshold = None):
    
    # Loop through all the files in the directory
    for filename in tqdm(os.listdir(image_directory)):
        # Only process .bmp files
        if filename.endswith(".bmp") or filename.endswith(".jpg") or filename.endswith(".png"):
            # print(f"Processing file: {filename}")
            # Get the camera code and defect code from the filename
            line_name, element_type, cam_position = extract_info_from_filename(filename)
            
            if int(cam_position) == 1:
                side = "left"
            elif int(cam_position) == 2:
                side = "right"

            if line_name and element_type:
                # Get the camera resolution and defect thresholds from the config
                camera_resolution = config_data["camera_resolutions"].get(line_name)
                defect_threshold = config_data["defect_thresholds"].get(element_type)

                if camera_resolution and defect_threshold:
                    # print(f"Camera Resolution for {line_name}: {camera_resolution}")
                    # print(f"Defect Thresholds for {element_type}: {defect_threshold}")
                    pass
                else:
                    print(f"Error: Camera resolution or defect thresholds not found for {filename}.")
            else:
                print(f"Error: Invalid filename format for {filename}.")
        else:
            continue  # Skip non-bmp files
        
        label_filename = str(filename).replace("bmp", "xml")
        remove_small_objects(os.path.join(image_directory, filename), label_filename, defect_threshold, camera_resolution, side, offsets, label_raw_directory, label_filter_directory, confidence_threshold = confidence_threshold)

if __name__ == "__main__":
    # Load configurations from JSON files
    defect_thresholds = load_config("config_defect_thresholds.json")

    model_name = "rtdert_2.0"
    dataset_version = "test1_v1.1"  # Full folder name of the dataset version

    # Create a timestamped result folder
    result_folder = "run_20250306_224555"

    # Directories
    image_directory = 'images\\' + dataset_version
    label_gt_raw_directory = 'labels\\' + dataset_version
    base_directory = 'prediction/' + model_name + '/' + result_folder
    model_directory = 'prediction/' + model_name
    label_raw_directory = model_directory + '\\labels'
    result_imagecrop_rawlabel_directory = base_directory + '\\image_unfilter_crop'
    labelcrop_raw_directory = base_directory + '\\label_xml_unfilter_crop'
    label_filter_directory = base_directory + '\\label_xml_filter'
    label_gt_filter_directory = base_directory + '\\gt_labels_filtered'
    
    size_offsets = {
    'impression': 1, 
    'einriss': 1,    
    'asperity': 1,  
    'abriss': 1,   
    'ausseinriss': 1
    }
    confidence_thresholds = {
    "impression": 0.3,
    "asperity": 0.0,
    "abriss": 0.25,
    "einriss": 0.5,
    "ausseinriss": 0.2
    }

    # Filter small reject base on defect size for prediction labels
    print("Filtering small reject base on defect size for prediction labels...")
    process_images_in_directory(image_directory, label_raw_directory, label_filter_directory, defect_thresholds, size_offsets, confidence_thresholds)
