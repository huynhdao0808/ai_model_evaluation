# AI Model Evaluation - Technical Documentation

## System Overview

This evaluation framework provides a comprehensive pipeline for assessing object detection models, with a focus on defect detection applications. The system ensures consistent evaluation by:

1. Filtering both predictions and ground truth using the same size criteria
2. Applying configurable confidence thresholds to predictions
3. Generating detailed metrics and visualizations
4. Preserving evaluation settings in timestamped runs

## Components Architecture

### Core Modules

| Module | Description |
|--------|-------------|
| `full_evaluation.py` | Main pipeline orchestrator - handles the entire evaluation flow |
| `evaluation.py` | Core metrics calculation and comparison logic |
| `filter_defect.py` | Applies size/confidence filtering to prediction and ground truth labels |
| `draw_bndbox.py` | Visualizes bounding boxes on images |
| `error_analysis.py` | Generates confusion matrices and visualizes error cases |
| `correct_classname.py` | Utility for fixing class name inconsistencies |

### Run Commands
```bash
# Main evaluation pipeline
python full_evaluation.py

# Individual components:
python evaluation.py                   # Run evaluation only
python draw_bndbox.py                  # Draw bounding boxes 
python filter_defect.py                # Filter defects based on thresholds
python error_analysis.py               # Generate error analysis
jupyter notebook error_analysis.ipynb  # Interactive error analysis
```

## Configuration System

Three JSON configuration files control evaluation behavior:

### 1. Confidence Thresholds (`config_confidence_thresholds.json`)
Controls minimum confidence scores required for each defect type.
```json
{
  "impression": 0.3,
  "asperity": 0.0,
  "abriss": 0.25,
  "einriss": 0.5, 
  "ausseinriss": 0.2
}
```

### 2. Size Thresholds (`config_defect_thresholds.json`)
Defines physical size thresholds for each defect by product type and camera.
```json
{
  "camera_resolutions": {
    "AL06": {"x": 0.0116, "z": 0.0165},
    "AL07": {"x": 0.0094, "z": 0.0143}
  },
  "defect_thresholds": {
    "24-06": {
      "impression": {"x": 0.5, "z": 0.5},
      "einriss": {"x": 3, "z": 0.87}
    }
  }
}
```

### 3. Size Offsets (`config_size_offsets.json`)
Size multipliers applied during filtering to adjust thresholds for specific defect types.
```json
{
  "impression": 1.0,
  "einriss": 1.0,
  "asperity": 1.0,
  "abriss": 1.25,
  "ausseinriss": 1.0
}
```

## Directory Structure & Data Flow

```
/images
  /dataset_v1        # Original input images grouped by dataset
/labels
  /dataset_v1        # Original ground truth labels
/prediction
  /model_name
    /labels          # Original prediction XML files
    /run_timestamp   # Created for each evaluation run
      /configs       # Copy of JSON configs used for the run
      /gt_labels_filtered    # Filtered ground truth labels
      /image_unfilter_crop   # Cropped images with raw labels
      /label_xml_unfilter    # Raw prediction labels copied from /labels
      /labels                # Filtered prediction labels 
      /misdetections         # Images with detection errors
      /analysis              # Results of error analysis
        /false_positives     # Example FP visualizations
        /false_negatives     # Example FN visualizations
```

## Evaluation Process Details

1. **Setup Phase**
   - Creates timestamped run folder (`run_YYYYMMDD_HHMMSS`)
   - Copies configurations to preserve evaluation settings
   - Copies prediction labels from model folder to run folder

2. **Preprocessing**
   - Updates class names if needed (eg. ausenriss → ausseinriss)
   - Draws bounding boxes on images with original labels
   - Crops images to focus on relevant areas

3. **Filtering Phase**
   - Filters predictions based on:
     - Size thresholds (adjusted by size_offsets)
     - Confidence thresholds
   - Filters ground truth using the same size criteria (but no confidence thresholds)

4. **Evaluation Phase**
   - Compares filtered predictions against filtered ground truth
   - Calculates precision, recall, and F1 metrics
   - Identifies true positives, false positives, and false negatives

5. **Analysis & Visualization**
   - Generates confusion matrices (overall and per-class)
   - Creates visualizations of error cases
   - Saves all metrics in text and JSON formats

## Output Files & Formats

| File/Directory | Description | Format |
|----------------|-------------|--------|
| `configs/` | Configuration copies | JSON |
| `precision_recall_results.json` | Detailed metrics | JSON |
| `overall_metrics.txt` | Summary statistics | Text |
| `per_class_metrics.txt` | Per-class statistics | Text |
| `overall_confusion_matrix.png` | Image-level confusion matrix | PNG |
| `defect_confusion_matrices.png` | Per-defect confusion matrices | PNG |
| `false_positives/` | FP example images | JPG |
| `false_negatives/` | FN example images | JPG |

## XML Label Format

The system uses VOC-style XML format for both predictions and ground truth:

```
<annotation>
  <filename>image_name.jpg</filename>
  <object>
    <name>defect_class</name>
    <confidence>0.85</confidence>
    <bndbox>
      <xmin>123</xmin>
      <ymin>45</ymin>
      <xmax>167</xmax>
      <ymax>89</ymax>
    </bndbox>
  </object>
</annotation>
```

## Code Style Guidelines

- Use 4 spaces for indentation (no tabs)
- Function/variable names: snake_case
- Constants: UPPER_CASE
- Include docstrings for all functions
- Follow PEP 8 conventions
- Use descriptive variable names
- Organize imports: standard, third-party, then local
- Handle errors with try/except blocks where appropriate

## Dependencies

- Python 3.6+
- numpy
- matplotlib
- opencv-python (cv2)
- pandas
- scikit-learn
- xml.etree.ElementTree
- tqdm
