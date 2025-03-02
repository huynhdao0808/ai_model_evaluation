import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import csv
from pathlib import Path

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / float(union_area)
    
    return iou

def calculate_iobox1(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    
    union_area = box1_area + box2_area - inter_area
    
    iobox1 = inter_area / box1_area
    
    return iobox1

def calculate_iou_matrix(gt_boxes, pred_boxes):
    """Calculate IoU matrix between ground truth boxes and predicted boxes."""
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = calculate_iou(gt[1], pred[1])
    
    return iou_matrix

def find_best_matches(gt_boxes, pred_boxes, iou_threshold=0.3, overlap_threshold=0.001):
    """Find the best match between ground truth and predicted boxes using the Hungarian algorithm.
    
    Args:
        gt_boxes (list): Ground truth boxes.
        pred_boxes (list): Predicted boxes with confidence scores.
        iou_threshold (float): Minimum IoU to consider a match as TP.
        overlap_threshold (float): Minimum IoU with an already matched GT to not consider as FP.
    
    Returns:
        true_positives (list): Matched boxes with IoU and confidence.
        false_negatives (list): Unmatched ground truth boxes.
        false_positives (list): Unmatched predicted boxes with confidence.
    
    Example of a box: ('impression', [863.0, 349.0, 890.0, 390.0], confidence)
    """
    if not gt_boxes or not pred_boxes:
        return [], gt_boxes, pred_boxes  # No matches possible if either list is empty

    # Step 1: Calculate IoU matrix
    iou_matrix = calculate_iou_matrix(gt_boxes, pred_boxes)
    
    # Step 2: Apply Hungarian algorithm to find the best matches
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)  # Hungarian for max IoU
    
    # Step 3: Initialize result lists
    true_positives = []
    false_negatives = gt_boxes.copy()  # Start with all ground truth boxes
    false_positives = pred_boxes.copy()  # Start with all predicted boxes
    matched_gt_boxes = []  # To track already matched ground truth boxes

    # Step 4: Filter matches based on IoU threshold
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[gt_idx, pred_idx]
        if iou >= iou_threshold:  # This is a true positive
            gt_box = gt_boxes[gt_idx]
            pred_box = pred_boxes[pred_idx]
            confidence = pred_box[2] if len(pred_box) > 2 else None  # Extract confidence if available
            true_positives.append((gt_box, pred_box, iou, confidence))
            matched_gt_boxes.append(gt_box)  # Track matched ground truth
            # Remove matched boxes from false negatives and false positives
            false_negatives.remove(gt_box)
            false_positives.remove(pred_box)
    
    # Step 5: Check unmatched detections for IoU > overlap_threshold with matched GTs
    for pred in false_positives[:]:  # Iterate over a copy of false_positives
        for matched_gt in matched_gt_boxes:
            iou = calculate_iou(matched_gt[1], pred[1])
            if iou > overlap_threshold:  # If IoU > 0.1, remove from false positive
                false_positives.remove(pred)
                break  # No need to check further if it's matched to any GT

    # Return only three lists
    return true_positives, false_negatives, false_positives

import xml.etree.ElementTree as ET

def parse_voc_labels(xml_file, confidence_thresholds={}):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if confidence_thresholds:  
            confidence = float(obj.find('confidence').text)  # Extract confidence score

            # Check if class is valid and meets confidence threshold
            if class_name in confidence_thresholds and confidence >= confidence_thresholds[class_name]:
                bbox = obj.find('bndbox')
                x1 = float(bbox.find('xmin').text)
                y1 = float(bbox.find('ymin').text)
                x2 = float(bbox.find('xmax').text)
                y2 = float(bbox.find('ymax').text)
                labels.append((class_name, [x1, y1, x2, y2], confidence))  
        else:
            bbox = obj.find('bndbox')
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            labels.append((class_name, [x1, y1, x2, y2]))  

    return labels

def parse_kitti_labels(label_file):
    labels = []
    # Classes to process
    valid_classes = {"impression", "asperity", "abriss", "einriss", "ausseinriss"}
    
    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 15:
                class_name = parts[0]  # Assuming class name is the first entry
                if class_name in valid_classes:  # Filter for valid classes
                    x1, y1, x2, y2 = map(float, parts[4:8])
                    labels.append((class_name, [x1, y1, x2, y2]))  # Include class name with bbox
    return labels

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=1):
    """
    Draw bounding boxes and their class names with class-specific text positions.

    Args:
        image (np.array): The image to draw on.
        boxes (list): A list of tuples, where each tuple contains
                      the class name and bounding box coordinates.
                      Example: [("class_name", [x1, y1, x2, y2]), ...]
        color (tuple): Default color for the box.
        thickness (int): The thickness of the box lines.

    Returns:
        np.array: The image with the bounding boxes and class names drawn.
    """
    # Define relative text positions for each class
    text_positions = {
        "impression": "above",  # Text above the box
        "asperity": "below",    # Text below the box
        "abriss": "left",       # Text on the left of the box
        "einriss": "right",     # Text on the right of the box
        "ausseinriss": "inside"         # Text inside the box
    }
    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i][1])

        # Draw the bounding box
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Determine text position based on class
        if boxes[i][0] in text_positions:
            position = text_positions[boxes[i][0]]
        else:
            position = "above"  # Default to "above" if class not found
        
        text = boxes[i][0] + " (" +str(x2-x1) + "x" +str(y2-y1) +")"

        # Calculate text position
        if position == "above":
            text_position = (x1, max(y1 - 10, 0))
        elif position == "below":
            text_position = (x1, y2 + 15)
        elif position == "left":
            text_position = (max(x1 - 60, 0), y1)
        elif position == "right":
            text_position = (x2 + 5, y1)
        elif position == "inside":
            text_position = (x1 + 5, y1 + 15)
        else:
            text_position = (x1, max(y1 - 10, 0))  # Default to "above"

        # Put the class name
        image = cv2.putText(image, text, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image

def evaluate_directory_structure_by_class(base_dir, images_dir, gt_dir, confidence_thresholds, iou_threshold=0.001):
    precision_recall_by_class = {class_name: {} for class_name in ["impression", "asperity", "abriss", "einriss", "ausseinriss"]}
    
    for thresh_dir in sorted(os.listdir(base_dir)):
        thresh_path = os.path.join(base_dir, thresh_dir)
        model = Path(base_dir).name
        if os.path.isdir(thresh_path):
            labels_dir = os.path.join(thresh_path, 'labels')
            output_dir = os.path.join(thresh_path, 'misdetections')
            os.makedirs(output_dir, exist_ok=True)
            
            file_path = model + "_" + thresh_dir+".csv"
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' has been deleted.")
            else:
                print(f"File '{file_path}' does not exist.")
            with open(file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerows([["filename", "gt", "gt_location", "pred", "pred_location", "confidence", "eval"]])
            
            for img_file in tqdm(sorted(os.listdir(images_dir)), desc=f"Processing Images ({thresh_dir})", leave=False):
                img_name = os.path.splitext(img_file)[0]
                gt_label_file = os.path.join(gt_dir, f'{img_name}.xml')
                pred_label_file = os.path.join(labels_dir, f'{img_name}.xml')
                img_path = os.path.join(images_dir, img_file)
                
                gt_labels = []
                if os.path.exists(gt_label_file):
                    gt_labels = parse_voc_labels(gt_label_file)
                
                pred_labels = []
                if os.path.exists(pred_label_file):
                    pred_labels = parse_voc_labels(pred_label_file, confidence_thresholds)
                    
                original_img = cv2.imread(img_path)
                img_fp_total = 0
                img_fn_total = 0
                f_list = []
                
                for class_name in precision_recall_by_class:
                    gt_class_labels = [label for label in gt_labels if label[0] == class_name]
                    pred_class_labels = [label for label in pred_labels if label[0] == class_name]
                    
                    true_positives, false_negatives, false_positives = find_best_matches(
                        gt_class_labels, pred_class_labels, iou_threshold)
                    
                    wrong_predictions = []
                    wrong_pred_locations = []
                    for box in false_negatives:
                        for pred_label in pred_labels:
                            if calculate_iobox1(box[1], pred_label[1]) >= 0.6:
                                wrong_predictions.append(pred_label[0])
                                wrong_pred_locations.append(pred_label[1])
                                break
                        else:
                            wrong_predictions.append("")
                            wrong_pred_locations.append("")
                    
                    wrong_gts = []
                    wrong_gt_locs = []
                    for box in false_positives:
                        for gt_label in gt_labels:
                            if gt_label[0] != class_name:
                                if calculate_iobox1(box[1], gt_label[1]) >= 0.6:
                                    wrong_gts.append(gt_label[0])
                                    wrong_gt_locs.append(gt_label[1])
                                    break
                        else:
                            wrong_gts.append("")
                            wrong_gt_locs.append("")
                    
                    log = []
                    for box in true_positives:
                        log.append([img_name, box[0][0], box[0][1], box[1][0], box[1][1], box[1][2], "TP"])
                    for box, wrong_prediction, wrong_pred_location in zip(false_negatives, wrong_predictions, wrong_pred_locations):
                        log.append([img_name, box[0], box[1], wrong_prediction, wrong_pred_location, "", "FN"])
                    for box, wrong_gt, wrong_gt_loc in zip(false_positives, wrong_gts, wrong_gt_locs):
                        log.append([img_name, wrong_gt, wrong_gt_loc, box[0], box[1], box[2], "FP"])
                    
                    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerows(log)
                    
                    img_tp = len(true_positives)
                    img_fp = len(false_positives)
                    img_fn = len(false_negatives)
                    
                    tp = precision_recall_by_class[class_name].get('tp', 0) + img_tp
                    fp = precision_recall_by_class[class_name].get('fp', 0) + img_fp
                    fn = precision_recall_by_class[class_name].get('fn', 0) + img_fn
                    
                    precision_recall_by_class[class_name]['tp'] = tp
                    precision_recall_by_class[class_name]['fp'] = fp
                    precision_recall_by_class[class_name]['fn'] = fn
                    
                    original_img = draw_boxes(original_img, gt_class_labels, color=(0, 255, 0))  
                    original_img = draw_boxes(original_img, pred_class_labels, color=(255, 255, 0), thickness=1)
                    
                    if img_fp > 0 or img_fn > 0:
                        f_list.append(class_name)
                        img_fp_total += img_fp
                        img_fn_total += img_fn
                        original_img = draw_boxes(original_img, false_positives, color=(255, 0, 0), thickness=1)
                        original_img = draw_boxes(original_img, false_negatives, color=(0, 0, 255), thickness=1)
                
                prefix = "OK"
                if img_fp_total > 0 and img_fn_total > 0:
                    prefix = "FN_FP_"
                elif img_fp_total > 0:
                    prefix = "FP_"
                elif img_fn_total > 0:
                    prefix = "FN_"
                
                if img_fp_total > 0 or img_fn_total > 0:
                    cv2.imwrite(os.path.join(output_dir, f'{prefix}_{"_".join(f_list)}_{img_name}.jpg'), original_img)
    
    for class_name, counts in precision_recall_by_class.items():
        tp = counts['tp']
        fp = counts['fp']
        fn = counts['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_recall_by_class[class_name]['precision'] = precision
        precision_recall_by_class[class_name]['recall'] = recall
        
        print(f"Class: {class_name} - TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision:0.2f}, Recall: {recall:0.2f}")
    
    return precision_recall_by_class


def plot_precision_recall(precision_recall):
    plt.figure(figsize=(10, 8))  # Doubled the size of the plot
    
    thresholds = []
    precisions = []
    recalls = []
    
    for thresh, (precision, recall) in precision_recall.items():
        thresholds.append(int(thresh))
        precisions.append(precision)
        recalls.append(recall)
    
    plt.plot(recalls, precisions, marker='o')
    
    for i, txt in enumerate(thresholds):
        plt.annotate(f'{txt}%', (recalls[i], precisions[i]), fontsize=16)  # Doubled annotation font size
    
    plt.xlabel('Recall', fontsize=24)  # Doubled x-axis label font size
    plt.ylabel('Precision', fontsize=24)  # Doubled y-axis label font size
    plt.title('Precision-Recall Curve', fontsize=28)  # Doubled title font size
    plt.grid(True)
    plt.show()

def suggest_best_threshold(base_dir, images_dir, gt_dir, iou_threshold=0.01):
    best_threshold_by_class = {class_name: {"threshold": None, "recall": 0} for class_name in ["impression", "asperity", "abriss", "einriss", "ausseinriss"]}
    
    # Iterate over different threshold directories
    for thresh_dir in sorted(os.listdir(base_dir)):
        thresh_path = os.path.join(base_dir, thresh_dir)
        if os.path.isdir(thresh_path):
            labels_dir = os.path.join(thresh_path, 'label')
            os.makedirs(labels_dir, exist_ok=True)

            precision_recall_by_class = {class_name: {'tp': 0, 'fp': 0, 'fn': 0} for class_name in ["impression", "asperity", "abriss", "einriss", "ausseinriss"]}

            for img_file in tqdm(sorted(os.listdir(images_dir)), desc=f"Processing Images ({thresh_dir})", leave=False):
                img_name = os.path.splitext(img_file)[0]
                gt_label_file = os.path.join(gt_dir, f'{img_name}.xml')
                pred_label_file = os.path.join(labels_dir, f'{img_name}.txt')
                img_path = os.path.join(images_dir, img_file)
                
                gt_labels = []
                if os.path.exists(gt_label_file):
                    gt_labels = parse_voc_labels(gt_label_file)
                
                pred_labels = []
                if os.path.exists(pred_label_file):
                    pred_labels = parse_voc_labels(pred_label_file)
                
                # Process each class separately
                for class_name in precision_recall_by_class:
                    # Filter the labels based on the class
                    gt_class_labels = [label for label in gt_labels if label[0] == class_name]
                    pred_class_labels = [label for label in pred_labels if label[0] == class_name]
                    
                    # Step 1: Find the best matches using the Hungarian algorithm
                    true_positives, false_negatives, false_positives = find_best_matches(
                        gt_class_labels, pred_class_labels, iou_threshold)
                    
                    # Update counts
                    img_tp = len(true_positives)
                    img_fp = len(false_positives)
                    img_fn = len(false_negatives)
                    
                    precision_recall_by_class[class_name]['tp'] += img_tp
                    precision_recall_by_class[class_name]['fp'] += img_fp
                    precision_recall_by_class[class_name]['fn'] += img_fn

            # Now calculate recall and find the best threshold for each class
            for class_name, counts in precision_recall_by_class.items():
                tp = counts['tp']
                fp = counts['fp']
                fn = counts['fn']
                
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Check if the current recall is better than the previous best recall
                if recall > best_threshold_by_class[class_name]["recall"]:
                    best_threshold_by_class[class_name]["recall"] = recall
                    best_threshold_by_class[class_name]["threshold"] = float(thresh_dir)

    # Print the best threshold for each class
    for class_name, values in best_threshold_by_class.items():
        print(f"Class: {class_name} - Best Threshold: {values['threshold']}, Recall: {values['recall']:.2f}")

    return best_threshold_by_class



if __name__ == "__main__":
    base_directory = "split/batch_1/result_rtdert"
    images_directory = "split/batch_1/images"
    gt_directory = "split/batch_1/labels"
    
    
    # Get the current working directory
    cwd = os.getcwd()

    # Print the current working directory
    print(f"Current working directory: {cwd}")
    
    precision_recall_data = evaluate_directory_structure_by_class(base_directory, images_directory, gt_directory)
    
    # Suggest Best Threshold
    best_threshold = suggest_best_threshold(base_directory, images_directory, gt_directory)
    
    # Plot Precision-Recall Curve
    # plot_precision_recall(precision_recall_data)
