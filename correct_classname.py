import os
import xml.etree.ElementTree as ET

def update_class_name_in_xml(xml_path, old_class, new_class):
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate through each object in the XML
    for obj in root.findall('object'):
        # Find the class name
        class_name = obj.find('name').text
        
        # If the class name matches the old class, update it
        if class_name == old_class:
            obj.find('name').text = new_class
            print(f"Updated class name in {xml_path}")
    
    # Save the modified XML
    tree.write(xml_path)

def update_class_names_in_directory(directory, old_class, new_class):
    # List all XML files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.xml'):
            xml_path = os.path.join(directory, filename)
            update_class_name_in_xml(xml_path, old_class, new_class)

if __name__ == "__main__":
    # Example usage
    directory_path = "ok_images\\AL07\\A\\label"  # Path to the directory containing XML files
    old_class_name = 'ausenriss'  # Class name to be replaced
    new_class_name = 'ausseinriss'  # New class name

    update_class_names_in_directory(directory_path, old_class_name, new_class_name)
