# AI Model Evaluation

A comprehensive framework for evaluating object detection models, particularly focused on defect detection. This system handles the full evaluation pipeline from prediction filtering to detailed error analysis.

## Setup and Configuration

### 1. Prepare Directory Structure

Create the following directory structure:
```
/images
  /dataset_v1        # Your images organized by dataset version
/labels
  /dataset_v1        # Ground truth labels in XML format
/prediction
  /model_name        # Your model name
    /labels          # Model prediction XML files
```

### 2. Configure Evaluation Parameters

Customize the configuration files to match your evaluation requirements:

- **config_confidence_thresholds.json**: Set minimum confidence scores per defect type
  ```json
  {
    "impression": 0.3,
    "asperity": 0.0,
    "abriss": 0.25,
    "einriss": 0.5, 
    "ausseinriss": 0.2
  }
  ```

- **config_defect_thresholds.json**: Define size thresholds for each defect by product and camera
  ```json
  {
    "camera_resolutions": { ... },
    "defect_thresholds": { ... }
  }
  ```

- **config_size_offsets.json**: Set size multipliers for different defect classes
  ```json
  {
    "impression": 1.0,
    "einriss": 1.0,
    "asperity": 1.0,
    "abriss": 1.25,
    "ausseinriss": 1.0
  }
  ```

### 3. Set Model and Dataset Version

In `full_evaluation.py`, update:

```python
# Update the model name and dataset version here
model_name = "your_model_name"
dataset_version = "dataset_v1"  # Full folder name of the dataset version
```

## Running Evaluation

The main evaluation pipeline is executed with:

```bash
python full_evaluation.py
```

This will:
1. Create a timestamped run folder (`run_YYYYMMDD_HHMMSS`)
2. Copy configurations to preserve evaluation settings
3. Copy and correct prediction labels
4. Draw bounding boxes on images
5. Filter predictions based on size and confidence thresholds
6. Filter ground truth using the same size criteria
7. Compare filtered predictions against filtered ground truth
8. Generate confusion matrices and error analysis visualizations
9. Save all results in the run folder

## Evaluation Results

After running the evaluation, you'll find results in:
```
/prediction/model_name/run_YYYYMMDD_HHMMSS/
  /configs                  # Saved configuration files
  /labels                   # Filtered prediction labels
  /gt_labels_filtered       # Filtered ground truth labels
  /image_unfilter_crop      # Images with bounding boxes
  /misdetections            # Images with detection errors
  /analysis                 # Analysis results
    overall_confusion_matrix.png
    defect_confusion_matrices.png
    overall_metrics.txt
    per_class_metrics.txt
    /false_positives        # False positive examples
    /false_negatives        # False negative examples
```

## Manual Error Analysis

For interactive error analysis, you can use:

```bash
python error_analysis.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook error_analysis.ipynb
```

## Additional Documentation

For more detailed information, see [CLAUDE.md](CLAUDE.md) which provides comprehensive technical documentation about the codebase organization, configuration options, and evaluation process.
