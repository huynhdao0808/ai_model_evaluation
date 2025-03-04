import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
import tqdm

def crop_image(image):
    """Crop the image based on the provided coordinates (x_min, y_min, x_max, y_max)."""
    offset = 20
    
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
    
    return image[y:y+h, x:x+w], (x,y)

def crop_xml_label(xml_path, crop_coords):
    """Crop the XML label file according to the cropping of the image."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get bounding box coordinates and adjust them
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float((bndbox.find('xmin').text)))
        ymin = int(float((bndbox.find('ymin').text)))
        xmax = int(float((bndbox.find('xmax').text)))
        ymax = int(float((bndbox.find('ymax').text)))
        
        # Adjust coordinates to the cropped image
        new_xmin = max(xmin - crop_coords[0], 0)
        new_ymin = max(ymin - crop_coords[1], 0)
        new_xmax = max(xmax - crop_coords[0], 0)
        new_ymax = max(ymax - crop_coords[1], 0)
        
        bndbox.find('xmin').text = str(new_xmin)
        bndbox.find('ymin').text = str(new_ymin)
        bndbox.find('xmax').text = str(new_xmax)
        bndbox.find('ymax').text = str(new_ymax)

    return tree

def draw_bounding_box(image, xml_tree, class_colors):
    """Draw bounding boxes from XML labels onto the image with different colors for each class."""
    root = xml_tree.getroot()
    
    offset = 0
    for obj in root.findall('object'):
        # Extract class name
        class_name = obj.find('name').text
        confidence = int(float(obj.find('confidence').text)*100)
        
        # Get bounding box coordinates
        bndbox = obj.find('bndbox')
        xmin = int(float((bndbox.find('xmin').text)))
        ymin = int(float((bndbox.find('ymin').text)))
        xmax = int(float((bndbox.find('xmax').text)))
        ymax = int(float((bndbox.find('ymax').text)))
        
        # Get the color for the class
        color = class_colors.get(class_name, (255, 0, 0))  # Default to blue if class not found
        
        # Draw rectangle on the image with the specific class color
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Add the class name text above the bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, class_name, (xmin+offset, ymin-10), font, 0.9, color, 1, cv2.LINE_AA)
        cv2.putText(image, str(confidence), (xmin+offset, ymin+20), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(image, str(xmax-xmin)+ "x"+str(ymax-ymin), (xmin+offset+20, ymin+20), font, 0.5, color, 1, cv2.LINE_AA)
        offset += 10

    return image

def process_images_in_folder(image_folder, xml_folder, output_image_folder, class_colors):
    """Process all images and their XML labels in a folder."""
    for filename in tqdm.tqdm(os.listdir(image_folder)):
        if filename.endswith(".jpg") or filename.endswith(".bmp"):
            # Image path
            image_path = os.path.join(image_folder, filename)
            xml_path = os.path.join(xml_folder, filename.replace('.jpg', '.xml').replace('.bmp', '.xml'))
            
            # Load and crop the image
            image = cv2.imread(image_path)
            cropped_image, crop_coords = crop_image(image)
            
            # Crop the XML label
            cropped_xml_tree = crop_xml_label(xml_path, crop_coords)
            
            # Draw bounding boxes on the cropped image
            final_image = draw_bounding_box(cropped_image, cropped_xml_tree, class_colors)
            
            filename = filename.replace('.bmp', '.jpg')
            # Save the final cropped image with bounding boxes
            output_image_path = os.path.join(output_image_folder, filename)
            cv2.imwrite(output_image_path, final_image)
            # print(f"Processed {filename} and saved to {output_image_path}")


if __name__ == "__main__":
    # Example usage
    image_folder = 'split\\batch_1\\images'
    xml_folder = 'split\\batch_1\\result_rtdert_2.0\\01\\label_xml_unfilter'
    output_image_folder = 'split\\batch_1\\result_rtdert_2.0\\01\\images_unfilter_crop'
    output_xml_folder = 'split\\batch_1\\result_rtdert_2.0\\01\\label_xml_unfilter_crop'

    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

    if not os.path.exists(output_xml_folder):
        os.makedirs(output_xml_folder)


    # Define a dictionary of class names and their corresponding bounding box colors (BGR format)
    class_colors = {
        'impression': (0, 255, 0),  # Green for 'impression'
        'einriss': (255, 0, 0),     # Blue for 'einriss'
        'asperity': (0, 0, 255),    # Red for 'asperity'
        'abriss': (255, 255, 0),    # Cyan for 'abriss'
        'oil': (255, 0, 255)        # Magenta for 'oil'
    }

    process_images_in_folder(image_folder, xml_folder, output_image_folder, output_xml_folder, class_colors)
