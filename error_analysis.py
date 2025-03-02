import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import warnings
import json
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Suppress all warnings
warnings.filterwarnings("ignore")

def crop_image(original_image):
    offset = 20
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    threshold_value = np.mean(gray) + np.std(gray)
    _, thresholded_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found!")
        return original_image

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x_crop = max(0, x - offset)
    y_crop = max(0, y - offset)
    w = min(original_image.shape[1] - x_crop, w + 2 * offset)
    h = min(original_image.shape[0] - y_crop, h + 2 * offset)
    cropped_image = original_image[y_crop:y_crop+h, x_crop:x_crop+w]
    
    return cropped_image 

def load_config(config_file):
    """Load a configuration file from disk."""
    try:
        with open(config_file, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {config_file}: {e}")
        return {}

def show_image_pairs(image_pairs, mode, def_name, model_path, run_folder, output_dir=None, save_images=True, show_plot=False):
    """
    Generate and optionally save/display image pairs showing defect comparisons.
    
    Args:
        image_pairs: List of tuples with (filename, img2_path, gt, pred)
        mode: Analysis mode ('TP', 'FN', 'FP', etc.)
        def_name: Defect type name
        model_path: Path to the model directory
        run_folder: Current run folder name
        output_dir: Directory to save output images (required if save_images=True)
        save_images: Whether to save the generated images (default: True)
        show_plot: Whether to display the plots (default: False, set to True in Jupyter)
    
    Returns:
        List of generated figure objects if show_plot=True, otherwise None
    """
    # Create output directory if saving images
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    figures = []
    for idx, (filename, img2_path, gt, pred) in enumerate(image_pairs):
        # Create figure for this pair
        fig_each = plt.figure(figsize=(20, 5))
        figures.append(fig_each)
        plt.suptitle(f"{def_name} - {mode}", fontsize=16)
        
        # Try to find the misdetection image in the current run folder
        misdetections_path = f"{model_path}/{run_folder}/misdetections"
        img1_path = None
        
        if os.path.exists(misdetections_path):
            for file_name in os.listdir(misdetections_path):
                if filename in file_name:  # Check if part_of_name is in file_name
                    img1_path = os.path.join(misdetections_path, file_name)
                    break
        
        # If not found in current run, try the model's general misdetection folder
        if not img1_path:
            model_misdetections = f"{model_path}/misdetections"
            if os.path.exists(model_misdetections):
                for file_name in os.listdir(model_misdetections):
                    if filename in file_name:
                        img1_path = os.path.join(model_misdetections, file_name)
                        break
        
        # If still not found, use the original image
        if not img1_path or not os.path.exists(img1_path):
            print(f"Warning: Misdetection image not found for {filename}, using original image")
            img1_path = img2_path
            
        # Check if the second image exists
        if not os.path.exists(img2_path):
            print(f"Error: Image not found -> {img2_path}")
            plt.close(fig_each)
            continue
            
        # Load and process images
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img1 = crop_image(img1)
        img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        
        if mode == "TP":
            img1 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
        
        # Create 2 subplots
        plt.subplot(1, 2, 1)
        plt.title(f"{filename} {mode} {pred}")
        plt.imshow(img1)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"Output")
        plt.imshow(img2)
        plt.axis('off')
        
        # Save the error pair image if requested
        if save_images and output_dir:
            output_file = os.path.join(output_dir, f"{mode}_{def_name}_{Path(img2_path).name.split('.')[0]}.jpg")
            fig_each.savefig(output_file)
        
        # Close the figure if not displaying it
        if not show_plot:
            plt.close(fig_each)
    
    # Return the figures if showing them (for Jupyter notebook)
    return figures if show_plot else None

def generate_confusion_matrices(df_eval, full_list, classes, results, output_dir, model_name):
    """
    Generate and save confusion matrices for the evaluation results.
    
    Args:
        df_eval: DataFrame with evaluation results
        full_list: List of all image filenames
        classes: List of defect classes
        results: Dictionary with precision/recall results
        output_dir: Directory to save output images
        model_name: Name of the model being evaluated
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # OVERALL IMAGE LEVEL CONFUSION MATRIX
    # Get lists of rejected images (based on ground truth and predictions)
    ground_truth_rejected = df_eval[df_eval['gt'].notnull()]['filename'].drop_duplicates().to_list()
    predicted_rejected = df_eval[df_eval['pred'].notnull()]['filename'].drop_duplicates().to_list()

    # Calculate true positives, false positives, true negatives, and false negatives
    tp = len(set(ground_truth_rejected) & set(predicted_rejected))
    fn = len(set(ground_truth_rejected) - set(predicted_rejected))
    fp = len(set(predicted_rejected) - set(ground_truth_rejected))
    tn = len(set(full_list) - set(ground_truth_rejected) - set(predicted_rejected))

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Save metrics to a text file
    with open(os.path.join(output_dir, "overall_metrics.txt"), "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Total images: {len(full_list)}\n")
        f.write(f"Images with defects (from GT): {len(ground_truth_rejected)}\n")
        f.write(f"Images with defects (predicted): {len(predicted_rejected)}\n")
        f.write(f"True Positives: {tp}, False Positives: {fp}, False Negatives: {fn}, True Negatives: {tn}\n")
        f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")
    
    # Overall confusion matrix
    fig1 = plt.figure(figsize=(10, 8))
    cm = np.array([[tp, fn], [fp, tn]])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rejected', 'Accepted'])
    disp.plot(cmap='viridis')
    plt.title(f'Overall Confusion Matrix\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}')
    fig1.savefig(os.path.join(output_dir, "overall_confusion_matrix.png"))
    plt.close(fig1)
    
    # CLASS LEVEL CONFUSION MATRICES
    fig2 = plt.figure(figsize=(15, 12))
    
    # Extract data for all defect types
    defect_data = {}
    for defect in classes:
        if defect in results:
            metrics = results[defect]
            tp = metrics.get('tp', 0)
            fp = metrics.get('fp', 0)
            fn = metrics.get('fn', 0)
            tn = len(full_list) - (tp + fp + fn)  # Approximate TN
            defect_data[defect] = {
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0)
            }
    
    # Create confusion matrix per defect type
    for i, defect in enumerate(classes):
        if defect in defect_data:
            data = defect_data[defect]
            cm = np.array([[data['tp'], data['fn']], [data['fp'], data['tn']]])
            
            plt.subplot(2, 3, i+1)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Defect', 'No Defect'])
            disp.plot(cmap='viridis', ax=plt.gca())
            
            plt.title(f"{defect.capitalize()}\nP: {data['precision']:.2f}, R: {data['recall']:.2f}")
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "defect_confusion_matrices.png"))
    plt.close(fig2)
    
    # Save per-class metrics to a text file
    with open(os.path.join(output_dir, "per_class_metrics.txt"), "w") as f:
        for defect, data in defect_data.items():
            f.write(f"=== {defect.upper()} ===\n")
            f.write(f"TP: {data['tp']}, FP: {data['fp']}, FN: {data['fn']}\n")
            f.write(f"Precision: {data['precision']:.4f}, Recall: {data['recall']:.4f}\n\n")

def generate_error_analysis(model_name, dataset_version, run_folder, output_dir=None):
    """
    Generate error analysis visualizations and confusion matrices.
    
    Args:
        model_name: Name of the model
        dataset_version: Dataset version folder name
        run_folder: Run folder name
        output_dir: Optional output directory (defaults to run_folder/analysis)
    """
    # Setup paths
    model_path = f"prediction/{model_name}"
    run_path = f"{model_path}/{run_folder}"
    result_file = f"{run_path}/{model_name}_evaluation.csv"
    
    if output_dir is None:
        output_dir = os.path.join(run_path, "analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation data
    df_eval = pd.read_csv(result_file)
    
    # Load configurations
    config_folder = os.path.join(run_path, "configs")
    confidence_thresholds = load_config(os.path.join(config_folder, 'config_confidence_thresholds.json'))
    results = load_config(os.path.join(config_folder, 'precision_recall_results.json'))
    
    # Define defect classes
    classes = ["impression", "einriss", "abriss", "asperity", "ausseinriss"]
    
    # Get list of all images
    image_dir = f'images/{dataset_version}'
    file_names = os.listdir(image_dir)
    full_list = [f.split(".")[0] for f in file_names if os.path.isfile(os.path.join(image_dir, f))]
    
    # Generate confusion matrices
    generate_confusion_matrices(df_eval, full_list, classes, results, output_dir, model_name)
    
    # Generate error pair visualizations for FN cases
    fn_output_dir = os.path.join(output_dir, "false_negatives")
    os.makedirs(fn_output_dir, exist_ok=True)
    
    for def_name in classes:
        # Look for "not detected" false negatives
        fil_df_eval = df_eval[(df_eval['gt']==def_name) & (df_eval['eval']=="FN") & (df_eval['pred'].isnull())]
            
        if fil_df_eval.shape[0] != 0:
            fil_df_eval['gt_path'] = f"{run_path}/image_unfilter_crop/" + fil_df_eval['filename'] + ".bmp"
            fil_df_eval = fil_df_eval.sort_values("filename")
            image_pairs = list(zip(fil_df_eval['filename'], fil_df_eval['gt_path'], fil_df_eval['gt'], fil_df_eval['pred']))
            
            # Limit to 10 examples maximum per defect type to avoid too many visualizations
            show_image_pairs(image_pairs[:10], "notdetect", def_name, model_path, run_folder, fn_output_dir, save_images=True, show_plot=False)
    
    # Generate error pair visualizations for FP cases
    fp_output_dir = os.path.join(output_dir, "false_positives")
    os.makedirs(fp_output_dir, exist_ok=True)
    
    for def_name in classes:
        # Look for "redundant" false positives where no ground truth exists
        fil_df_eval = df_eval[(df_eval['pred']==def_name) & (df_eval['eval']=="FP") & (df_eval['gt'].isnull())]
            
        if fil_df_eval.shape[0] != 0:
            fil_df_eval['gt_path'] = f"{run_path}/image_unfilter_crop/" + fil_df_eval['filename'] + ".bmp"
            fil_df_eval = fil_df_eval.sort_values("filename")
            image_pairs = list(zip(fil_df_eval['filename'], fil_df_eval['gt_path'], fil_df_eval['pred'], fil_df_eval['gt']))
            
            # Limit to 10 examples maximum per defect type
            show_image_pairs(image_pairs[:10], "redundant", def_name, model_path, run_folder, fp_output_dir, save_images=True, show_plot=False)
    
    print(f"Error analysis complete. Results saved to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Example usage when run directly
    model_name = "result_rtmdet_0226"
    dataset_version = "dataset_v1"
    
    # Find the latest run folder
    model_path = f"prediction/{model_name}"
    run_folders = sorted([f for f in os.listdir(model_path) if f.startswith("run_")], reverse=True)
    
    if run_folders:
        run_folder = run_folders[0]
        print(f"Analyzing latest run: {run_folder}")
        generate_error_analysis(model_name, dataset_version, run_folder)
    else:
        print("No run folders found.")