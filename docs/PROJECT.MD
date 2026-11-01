# RoadWatcher: YOLOv9 Vehicle Detection Project Plan

## 1. Project Overview

RoadVision AI develops artificial intelligence solutions for road-traffic analysis. The goal of this project is to build and evaluate a prototype that identifies several vehicle categories in images and localizes each vehicle with a bounding box.

The solution will use YOLOv9-C, a pretrained object-detection model, and an annotated road-vehicle dataset. The project should demonstrate the complete machine-learning workflow: data preparation, model training, evaluation, prediction analysis, and critical discussion of the results.

## 2. Objectives

- Prepare and validate the Road Vehicle Images Dataset.
- Understand and verify the YOLO annotation format.
- Create a reproducible train, validation, and test split.
- Fine-tune YOLOv9-C on a GPU using pretrained weights.
- Evaluate the model globally and for each vehicle category.
- Visualize predictions and investigate detection errors.
- Explain the model's limitations and propose justified improvements.

## 3. Project Constraints and Fixed Parameters

| Item | Requirement |
| --- | --- |
| Model | YOLOv9-C |
| Training script | `train_dual.py` |
| Model configuration | `models/detect/yolov9-c.yaml` |
| Pretrained weights | `yolov9-c.pt` |
| Hyperparameters | `data/hyps/hyp.scratch-high.yaml` |
| Image size | 640 pixels |
| Batch size | 4 |
| Number of epochs | 10 |
| Visualization confidence threshold | 0.25 |
| Random split seed | `random_state=42` |
| Training images | 2,297 |
| Validation images | 298 |
| Test images | 405 |

## 4. Technical Environment

- Python and PyTorch
- YOLOv9
- OpenCV
- Google Colab with GPU acceleration
- Kaggle dataset download tools
- Git and GitHub

## 5. Execution Plan

### Phase 1: Set Up the Environment and Download the Dataset

1. Create a Google Colab notebook and enable GPU acceleration.
2. Verify the GPU, Python version, PyTorch version, and CUDA availability.
3. Download the Road Vehicle Images Dataset from Kaggle.
4. Inspect the dataset structure and locate:
   - Images
   - YOLO label files
   - `data.yaml`
   - Training and validation directories
5. Install and configure the dependencies required to run YOLOv9.

**Deliverable:** A working notebook with a verified GPU environment and an accessible dataset.

### Phase 2: Explore and Clean the Data

1. Count images and annotation files in every dataset split.
2. Inspect sample images and their bounding boxes.
3. Verify the YOLO label format:
   ```text
   class_id center_x center_y width height
   ```
4. Confirm that coordinates are normalized between 0 and 1.
5. Validate `data.yaml`, class names, and class identifiers.
6. Check image-label correspondence.
7. Detect and document:
   - Corrupted images
   - Missing labels
   - Empty annotation files
   - Invalid annotation rows
   - Out-of-range coordinates
   - Unknown class identifiers
8. Clean invalid data and repeat the validation checks.
9. Calculate and visualize the distribution of annotations by class.
10. Identify class imbalance and classes with few examples.

**Deliverable:** A data-quality report and a validated, cleaned dataset.

### Phase 3: Build the Train, Validation, and Test Sets

1. Use the cleaned training data as the source for the test split.
2. Select exactly 405 test images with a reproducible random selection using `random_state=42`.
3. Keep the remaining 2,297 images for training.
4. Preserve the 298 validation images in a separate validation set.
5. Copy each image together with its corresponding label file.
6. Organize the data into:
   ```text
   train/images       train/labels
   valid/images       valid/labels
   test/images        test/labels
   ```
7. Update `data.yaml` with the correct paths and class names.
8. Verify the final image counts and annotation consistency.
9. Check for exact duplicates and, where practical, highly similar images across splits.

**Deliverable:** A reproducible dataset split containing 2,297 training images, 298 validation images, and 405 test images.

### Phase 4: Install and Configure YOLOv9

1. Clone the official YOLOv9 repository.
2. Locate `train_dual.py` and `models/detect/yolov9-c.yaml`.
3. Download the compatible `yolov9-c.pt` pretrained weights.
4. Locate `data/hyps/hyp.scratch-high.yaml`.
5. Confirm all paths for the dataset, model configuration, weights, and hyperparameters.
6. Explain the difference between:
   - The model architecture
   - The pretrained weights
   - The training script
7. Explain why starting from pretrained weights is transfer learning and why the training stage is fine-tuning.

**Deliverable:** A configured YOLOv9 environment with all required files and paths documented.

### Phase 5: Train YOLOv9-C

1. Configure training with the fixed parameters:
   - Image size: 640
   - Batch size: 4
   - Epochs: 10
   - GPU execution
2. Launch training with `train_dual.py`.
3. Monitor training losses and validation metrics.
4. Explain the roles of epochs, batches, hyperparameters, and checkpoints.
5. Confirm that training produces:
   - `best.pt`
   - `last.pt`
6. Record the training command, runtime environment, and final metrics.

**Deliverable:** A completed training run with available checkpoints and recorded training results.

### Phase 6: Evaluate the Model on the Test Set

1. Load the `best.pt` checkpoint.
2. Run evaluation on the 405 test images.
3. Record global and per-class results for:
   - Precision
   - Recall
   - mAP@50
   - mAP@50-95
4. Explain what each metric measures.
5. Compare vehicle categories and identify the strongest and weakest classes.
6. Relate performance to class imbalance, annotation quality, and the number of examples per class.
7. Clearly distinguish evaluation parameters from visualization parameters.

**Deliverable:** An evaluation table with global metrics, per-class metrics, and an initial interpretation.

### Phase 7: Generate and Analyze Predictions

1. Run inference on images in the test directory using the trained model.
2. Use a confidence threshold of 0.25 for visualization.
3. Save annotated images with predicted classes and bounding boxes.
4. Display at least five prediction examples.
5. Inspect:
   - Correct detections
   - Missed objects
   - False positives
   - Poorly localized bounding boxes
6. Compare visual observations with the quantitative evaluation results.

**Deliverable:** Saved prediction images and a short error-analysis section with at least five examples.

### Phase 8: Critical Analysis and Improvements

1. Interpret the test-set results by vehicle category.
2. Explain why some classes perform better than others.
3. Discuss the effects of:
   - Class distribution
   - Annotation quality
   - Dataset size
   - Ten training epochs
   - Image scale and visual complexity
4. Propose at least two improvements and explain why they could help. Examples include:
   - More balanced or larger training data
   - Improved annotation quality
   - Longer training with controlled validation
   - Data augmentation
   - Hyperparameter tuning
   - Comparison with another YOLO model
5. Separate measured results from improvements that have only been proposed and still need testing.

**Deliverable:** A critical analysis with evidence-based conclusions and at least two justified improvement proposals.

### Optional Phase 9: Ultralytics Comparison

1. Use the Ultralytics Python API to test vehicle detection.
2. Generate and visualize predictions.
3. Compare the Ultralytics workflow and results with the YOLOv9 implementation.

**Deliverable:** An optional comparison section documenting differences in setup, inference, and results.

## 6. Notebook Structure

The final notebook should contain the following sections:

1. Project introduction and objectives
2. Environment and GPU verification
3. Dataset download and structure
4. Data exploration and cleaning
5. Dataset split creation
6. YOLOv9 installation and configuration
7. Model training
8. Test-set evaluation
9. Prediction visualization
10. Error analysis
11. Critical analysis and improvements
12. Optional Ultralytics comparison
13. Conclusion

Each section should include short explanations, executable code, outputs, and a clear distinction between measured results and assumptions.

## 7. Five-Day Schedule

| Day | Main work | Expected result |
| --- | --- | --- |
| Day 1 - 26/10/2026 | Environment, dataset download, structure inspection | Working GPU notebook and dataset inventory |
| Day 2 - 27/10/2026 | Data validation, cleaning, class analysis, split creation | Clean and correctly partitioned dataset |
| Day 3 - 28/10/2026 | YOLOv9 setup and training | `best.pt`, `last.pt`, and training logs |
| Day 4 - 29/10/2026 | Test evaluation, predictions, and error analysis | Metrics, visualizations, and error examples |
| Day 5 - 30/10/2026 | Critical analysis, documentation, and presentation rehearsal | Final notebook and presentation-ready conclusions |

The final submission is due on 30/10/2026 before 17:00.

## 8. Final Acceptance Checklist

- [ ] GPU availability and library versions are documented.
- [ ] Dataset structure and YOLO annotation format are explained.
- [ ] Invalid, missing, empty, or corrupted data has been checked and documented.
- [ ] The final split contains 2,297 train, 298 validation, and 405 test images.
- [ ] `data.yaml` points to the correct directories.
- [ ] YOLOv9-C was trained with image size 640, batch size 4, and 10 epochs.
- [ ] `best.pt` and `last.pt` are available.
- [ ] Test metrics are reported globally and by class.
- [ ] At least five prediction examples are displayed.
- [ ] Correct detections, missed objects, and false positives are discussed.
- [ ] At least two improvements are proposed and justified.
- [ ] Measured results are separated from untested hypotheses.
- [ ] The notebook can be presented and explained in 30 minutes.

## 9. Assessment Format

The assessment lasts 30 minutes:

1. **Solution demonstration - 10 minutes**
   - Present the notebook.
   - Show data preparation, the trained model, and prediction examples.
2. **Technical review - 10 minutes**
   - Explain YOLO annotations, transfer learning, training parameters, and evaluation metrics.
3. **Practical discussion - 10 minutes**
   - Analyze a prediction error, explain a technical issue, or justify a possible improvement.

## 10. Expected Learning Outcomes

At the end of the project, the student should be able to:

- Prepare and validate an object-detection dataset.
- Read and explain YOLO annotations.
- Configure and fine-tune a pretrained YOLOv9 model.
- Evaluate object-detection performance with standard metrics.
- Interpret predictions and diagnose model errors.
- Critically discuss dataset and model limitations.
