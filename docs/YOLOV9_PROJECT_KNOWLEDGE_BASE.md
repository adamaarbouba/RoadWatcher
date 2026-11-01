# RoadVision AI: Vehicle Detection with YOLOv9-C
## Project Knowledge Base (source of truth)

Consolidated from the working session of 27 Oct 2026. It combines the project brief, the concepts behind it, the YOLO/YOLOv9 configuration, the annotation knowledge, step-by-step guides, evaluation theory and a framework for the critical analysis.

**Evidence labels used throughout**

| Label | Meaning |
|---|---|
| **[BRIEF]** | Stated in the project brief provided by the project owner. Authoritative. |
| **[KNOWN]** | Well-established general knowledge. Safe, but not tied to one source here. |
| **[SOURCED]** | Taken from a web source listed in section 14. Reported by that source. |
| **[VERIFY]** | From memory or likely to vary. Check against your own files, repo or `--help` output before relying on it. |

> Rule for the report: only claim as a **result** what you measured. Anything else is a **hypothesis to test**.

---

## Table of contents

1. [Project brief](#1-project-brief)
2. [Vision tasks: where detection sits](#2-vision-tasks-where-detection-sits)
3. [What YOLO is](#3-what-yolo-is)
4. [YOLO versions](#4-yolo-versions)
5. [YOLOv9-C: architecture, files, configuration](#5-yolov9-c-architecture-files-configuration)
6. [Annotations](#6-annotations)
7. [Train / validation / test split](#7-train--validation--test-split)
8. [Evaluation theory](#8-evaluation-theory)
9. [Step-by-step guide (brief steps 1 to 9)](#9-step-by-step-guide-brief-steps-1-to-9)
10. [Critical analysis framework](#10-critical-analysis-framework)
11. [Pitfalls checklist](#11-pitfalls-checklist)
12. [Schedule](#12-schedule)
13. [Glossary](#13-glossary)
14. [Sources and verification list](#14-sources-and-verification-list)

---

## 1. Project brief

**Context [BRIEF].** RoadVision AI specialises in AI for road-traffic analysis. The goal is a prototype that identifies several categories of vehicles in images and localises each with a bounding box. You develop and evaluate a first solution based on **YOLOv9-C**, using a **pre-trained model** and an **annotated dataset**. Performance must be **critically analysed**, taking into account annotation quality, class distribution and the limits of the available data.

**Logistics [BRIEF].** Individual work. 5 working days, **26/10/2026 to 30/10/2026, before 17:00**. Technologies: Python, PyTorch, YOLOv9, OpenCV, Google Colab, Kaggle, Git/GitHub.

**Fixed numbers [BRIEF]**

| Item | Value |
|---|---|
| Dataset | Road Vehicle Images Dataset (Kaggle) |
| Test set | 405 images, taken from the cleaned training data, `random_state=42` |
| Train set | 2,297 images (the remainder) |
| Validation set | 298 images, preserved as a separate set |
| Image size | 640 |
| Batch size | 4 |
| Epochs | 10 |
| Visualisation confidence | 0.25 |
| Training script | `train_dual.py` |
| Architecture config | `models/detect/yolov9-c.yaml` |
| Pre-trained weights | `yolov9-c.pt` |
| Hyperparameters | `data/hyps/hyp.scratch-high.yaml` |

### The 9 steps [BRIEF], translated

1. **Environment and dataset.** Create a Colab notebook with GPU. Check GPU and library versions (notably PyTorch). Download the Road Vehicle Images Dataset from Kaggle. Explore the folder tree and identify images, annotations and config files. Prepare the environment for YOLOv9.
2. **Exploration and cleaning.** Explore images and annotations. Examine the YOLO format (class id, normalised centre, width, height). Check `data.yaml`, class names and ids used in the annotations. Check image/label correspondence. Identify corrupted images, missing annotations, invalid lines, empty annotations. Clean, then verify after cleaning. Examine the class distribution and imbalances.
3. **Train/validation/test sets.** Identify data after cleaning. Create a 405-image test set from the cleaned training data (reproducible random selection, `random_state=42`). Keep the remaining 2,297 for training. Preserve the 298 validation images as a distinct set. Organise into `train`, `valid`, `test`. Verify counts and annotation consistency. Update `data.yaml`. Check, as far as possible, for exact duplicates or very similar images across sets. *Resulting split: 2,297 / 298 / 405.*
4. **YOLOv9 setup.** Clone the official repo. Identify `train_dual.py` and `models/detect/yolov9-c.yaml`. Download compatible `yolov9-c.pt`. Identify `data/hyps/hyp.scratch-high.yaml`. Verify paths to weights, config and dataset. Understand pre-trained weights and transfer learning. Identify the differences between architecture, pre-trained weights and training script.
5. **Training.** Configure with pre-trained YOLOv9-C weights; image size 640, batch 4, 10 epochs; train on GPU with `train_dual.py`. Follow losses and validation metrics. Understand epochs, batches, hyperparameters, checkpoints. Identify `best.pt` and `last.pt`; verify they exist at the end. Explain why this is transfer learning by fine-tuning.
6. **Evaluation on the test set.** Load `best.pt`, evaluate on the test set. Record global and per-class metrics. Understand Precision, Recall, mAP@50, mAP@50-95. Compare vehicle categories; identify best and weakest classes. Examine limits from class imbalance and few examples per category. Distinguish parameters used for evaluation from those used for visualisation.
7. **Predictions and visualisation.** Predict on test images with **confidence 0.25**. Save images with boxes and predicted classes. Show **at least 5** examples. Examine correct detections, missed objects, false detections. Compare visual observations with the computed metrics.
8. **Critical analysis and improvements.** Interpret test results. Compare main vehicle categories. Explain why some classes are detected better. Analyse limits (data distribution, number of epochs, annotation quality). Propose **at least two** improvements and justify them. Distinguish what was **measured** from what **remains to be tested**.
9. **Bonus: Ultralytics.** Test vehicle detection with the Ultralytics Python API. Visualise predictions and compare the two approaches.

---

## 2. Vision tasks: where detection sits

Four tasks, in increasing detail **[SOURCED: GeeksforGeeks]**:

| Task | Input to output | Handles several objects? | Gives location? |
|---|---|---|---|
| **Image classification** | Image to one class label (with probability) | No (one label) | No |
| **Object localisation** | Image to one box (position, height, width) | One object | Box |
| **Object detection** | Image to several boxes, each with a class label | Yes | Boxes |
| **Image segmentation** | Image to pixel-wise masks per object | Yes | Exact shape |

Segmentation has two forms: **semantic** (all objects of a class share one label/colour) and **instance** (each individual object is separate).

**Why detection is the right task here.** A road scene contains many vehicles of several classes, and RoadVision AI needs to count and locate each. A plain classifier outputs one label per image and cannot do this.

**Limits of detection (use in your analysis)** [SOURCED]:

- Boxes are rectangles, so they do not describe the shape of curved or irregular objects, and include background pixels.
- Boxes cannot give accurate area or perimeter of an object.
- Possible untested improvement: **instance segmentation**, if box shapes limit performance (for example crowded scenes with heavy overlap).

**Terminology caution.** The source uses "object recognition" as an umbrella term. Other sources use it only for "naming what is in the image". In the report, define the terms you use.

---

## 3. What YOLO is

**Name [KNOWN].** YOLO = **"You Only Look Once"**. A family of object detectors, first published in 2015 by Joseph Redmon and colleagues.

**Meaning [KNOWN].** The network looks at the image **once**: one forward pass returns all boxes, classes and confidences.

| | Two-stage detectors (R-CNN family) | One-stage detectors (YOLO, SSD) |
|---|---|---|
| Procedure | Propose thousands of regions, classify each, refine boxes | One network, one pass |
| Passes over the image | Many | **One** |
| Speed | Slower | Faster |
| Historical accuracy | Higher | Lower at first, now competitive |
| Complexity | Multi-stage | Single network, single loss |

**Benefits [KNOWN].**
- **Speed:** suits real-time video (traffic cameras).
- **Global context:** sees the whole image, so it uses context (a vehicle sits on a road) and is less likely to confuse background with an object.
- **Simplicity:** trained end to end.

**Classic mechanism (grid) [KNOWN].**
1. Divide the image into a grid of cells.
2. Each cell is responsible for objects whose **centre** falls inside it.
3. Each cell predicts box values (x, y, w, h), a confidence that an object is present, and class scores.
4. This yields thousands of candidate boxes, mostly low confidence.
5. **Confidence filtering** drops weak boxes.
6. **Non-Maximum Suppression (NMS)** keeps the best of overlapping boxes on the same object.

YOLO label files store the box **centre** plus width and height because of this "which cell holds the centre" logic.

**Modern versions [KNOWN].** Predict at three scales (large, medium, small objects). Coarse grids suit large objects; fine grids suit small, distant ones. Small objects are where detectors typically lose accuracy.

**Pipeline inside the model.**
```
Image 640x640 -> Backbone (features) -> Neck (multi-scale fusion) -> Head (3 scales)
  -> candidate boxes + class scores -> confidence filter -> NMS -> final detections
```

---

## 4. YOLO versions

**YOLO is not one project.** It has been forked and extended by different groups, so numbering is not a single linear line of development **[SOURCED]**. A higher number does not automatically mean "newer by date" or "from the same team".

### Eras

| Era | Versions | What changed |
|---|---|---|
| **1. Original idea** (about 2015-2018) | v1 to v3 | One-pass grid detection (v1). Anchors and batch normalisation (v2, general knowledge). **Multi-scale detection with a Feature Pyramid Network** in v3, so small objects became detectable **[SOURCED]**. Darknet framework in C. |
| **2. Engineering and speed** (about 2020-2022) | v4 to v7 | Different teams competed on training tricks and fast architectures. **YOLOv5 (Ultralytics) was the first fully PyTorch implementation** **[SOURCED]**, which made training and deployment far easier. v3 to v7 are **anchor-based** (predict offsets from predefined reference boxes) **[SOURCED]**. |
| **3. Simplification, anchor-free** (2023 onward) | v8 and later | **YOLOv8 dropped anchors** and predicts coordinates directly **[SOURCED]**. Later versions keep removing hand-designed pipeline parts. |

### Notable recent versions

| Version | Key points | Evidence |
|---|---|---|
| **YOLOv9** (2024) | **GELAN** backbone/neck design and **PGI** (Programmable Gradient Information): an auxiliary reversible branch used **only in training** to keep useful gradient information flowing to deep layers. Different group from the Ultralytics line. The auxiliary branch is why the script is `train_dual.py`. | [KNOWN] |
| **YOLOv10** | Introduced NMS-free training (no post-processing step). | [KNOWN] |
| **YOLO26** (Sept 2025, Ultralytics) | Removes NMS and Distribution Focal Loss (DFL); end-to-end NMS-free; MuSGD optimiser; ProgLoss and STAL aimed at stability and small-object fidelity; focus on low-power deployment. One review reports up to 43% faster CPU inference than YOLO11. | [SOURCED] |
| **YOLOv13** | Hypergraph-enhanced correlation modelling; reported +3.0 mAP@[0.5:0.95] over YOLO11-N (38.6% to 41.6%). Figures are source-reported, not hardware-normalised re-benchmarks. | [SOURCED] |

### The change log in one table

| Change | Meaning | First seen |
|---|---|---|
| Single regression pass | Detection in one network | v1 |
| Multi-scale prediction | Big and small objects | v3 |
| PyTorch implementation | Easy training and deployment | v5 |
| Anchor-free heads | No predefined box shapes | v8 onward |
| Auxiliary gradient branch (PGI) and GELAN | Better learning in deep layers | v9 |
| NMS-free, end-to-end | No post-processing step | v10, YOLO26 |

### What this means for your project
- The brief **fixes YOLOv9-C**. Train and evaluate that.
- Do **not** claim a newer version would automatically do better. That is a hypothesis to test (step 8), and should be justified (for example small-object handling if small vehicles are missed).
- The **official YOLOv9 repo** (`train_dual.py`) and the **Ultralytics API** (bonus) are different codebases, even if both can load a YOLOv9-C model. The Ultralytics pretrained model only knows the 80 COCO classes unless you fine-tune it.

---

## 5. YOLOv9-C: architecture, files, configuration

### 5.1 Three things not to confuse [BRIEF step 4]

| Thing | File | What it is | Analogy |
|---|---|---|---|
| **Architecture** | `models/detect/yolov9-c.yaml` | The blueprint: layers, sizes, number of classes | Building plan |
| **Pre-trained weights** | `yolov9-c.pt` | Learned numbers (from COCO) | The knowledge |
| **Training script** | `train_dual.py` | Code running the learning loop | The teacher |

The architecture can be built with random weights ("from scratch"). Supplying the weights is what makes it **transfer learning**.

### 5.2 Transfer learning by fine-tuning [KNOWN]

- COCO weights have an output layer for **80 classes**. Your dataset has a different number of classes (**[VERIFY]** from your `data.yaml`; about 21 from memory).
- **Backbone and neck** (edges, textures, wheels, vehicle shapes) are **reused**: this is the valuable knowledge.
- **Final classification layers do not match in shape**, so they are re-initialised. The training log reports how many items were transferred.
- **All layers keep training** on your data at a modest learning rate: that is **fine-tuning** (as opposed to freezing the backbone).
- Why it works with about 2,300 images: the model does not have to learn what an edge or wheel looks like from scratch.

### 5.3 Training command

```bash
python train_dual.py \
  --workers 2 --device 0 --batch 4 --epochs 10 --img 640 \
  --data /content/dataset/data.yaml \
  --cfg models/detect/yolov9-c.yaml \
  --weights yolov9-c.pt \
  --hyp data/hyps/hyp.scratch-high.yaml \
  --name yolov9c_vehicles \
  --min-items 0 --close-mosaic 2
```

| Argument | Meaning | Notes |
|---|---|---|
| `--img 640` | Images resized (letterboxed) to 640x640 | Small objects lose detail when downscaled |
| `--batch 4` | Images per weight update | About 575 updates per epoch (2,297 / 4) |
| `--epochs 10` | Full passes over the training set | About 5,750 updates total. Short. |
| `--weights` | Starting point | `''` means from scratch |
| `--cfg` | Architecture | Must match the weights |
| `--data` | Dataset config | Paths and class names |
| `--hyp` | Hyperparameter file | See 5.4 |
| `--close-mosaic N` | Turns mosaic augmentation off for the last N epochs | **[VERIFY]** The repo's own example uses 15, which exceeds 10 epochs. Check behaviour in code; a small value (for example 2) is sensible. |
| `--name` | Output folder `runs/train/<name>` | |

**[VERIFY]** Exact argument names and defaults with `python train_dual.py --help`. The example above follows the repo's published pattern from memory.

### 5.4 Hyperparameter file `hyp.scratch-high.yaml` [VERIFY values in the file]

A recommended starting set, despite "scratch" in the name. Groups:
- **Optimisation:** initial learning rate (about 0.01), final LR fraction, momentum (about 0.937), weight decay (about 0.0005), **warmup epochs (about 3)**. With only 10 epochs, a 3-epoch warmup is a large share of training. Worth mentioning in the analysis.
- **Loss weights:** `box`, `cls`, `dfl`.
- **Augmentation:** HSV colour shifts, translate, scale, left-right flip, **mosaic** (4 images stitched), **mixup**, **copy-paste**.

### 5.5 Reading the training log

Per epoch you see `box_loss`, `cls_loss`, `dfl_loss` (should fall), plus Precision, Recall, mAP@0.5, mAP@0.5:0.95 on the **validation** set.

| Pattern | Likely meaning |
|---|---|
| Train loss falls, validation mAP stalls or drops | Overfitting |
| Both still improving at epoch 10 | **Under-training**: justifies "train longer" |
| Per-epoch gains shrink | Normal: big early gains, smaller later |

### 5.6 The training loop (what an "update" is)

1. Take a **batch** (4 images). 2. **Predict** boxes. 3. **Compare** with annotations (loss). 4. **Adjust weights** to lower the loss. One epoch is every training image seen once. Loss is a "how wrong" score; YOLOv9 combines box, class and DFL losses.

### 5.7 Checkpoints

In `runs/train/<name>/weights/`:
- **`last.pt`**: state at the final epoch (used to resume).
- **`best.pt`**: epoch with best validation fitness (a weighted mix of mAP@0.5 and mAP@0.5:0.95). **Use this for evaluation.**
- Copy both to Google Drive: Colab storage is temporary.

### 5.8 Which weights file [VERIFY]
The official repo's releases offer a training ("dual") version and a "converted" inference-only version. Use the dual `yolov9-c.pt` with `train_dual.py`. Confirm the filename and release link in the repo README.

---

## 6. Annotations

### 6.1 The dataset [VERIFY all from your own files]
Road Vehicle Images Dataset (Kaggle), already annotated in YOLO format. From memory: about 3,000 images in total (the brief's numbers: 2,702 + 298), about 21 vehicle classes, collected on Bangladeshi roads (classes such as auto-rickshaw, CNG/three-wheeler, human hauler, wheelbarrow). **You do not draw boxes; you inspect, clean and critique existing labels.**

### 6.2 YOLO label format

One `.txt` per image, same name. One line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All four values **normalised to 0-1** (divided by image width/height). Example: `5 0.481 0.612 0.230 0.310` means class 5, centre at 48.1% of width and 61.2% of height, box covering 23% of width and 31% of height.

Pixel corners:
```
x1 = (xc - w/2) * W      x2 = (xc + w/2) * W
y1 = (yc - h/2) * H      y2 = (yc + h/2) * H
```

- **`data.yaml`** holds the class list. The **order defines the id** (first name is 0). If ids and names drift apart, the model learns wrong labels without any crash. Check it first.
- **Image-label pairing:** same filename, different extension.
- **One line = one object.** An empty `.txt` means "no vehicles": valid for a background image, but a possible error if vehicles are present.

### 6.3 Format errors (what step 2 hunts)

| Problem | Symptom | Effect |
|---|---|---|
| Missing label file | Image with no `.txt` | Treated as pure background, so real vehicles are taught as "nothing" |
| Empty label file | `.txt` with 0 lines | Same, if vehicles are present |
| Invalid line | Not 5 values, non-numeric, class id out of range | Crash or silently skipped |
| Out-of-range coordinates | Values < 0 or > 1, or width/height = 0 | Corrupt boxes |
| Duplicate boxes | Identical lines | Double-counted target |
| Corrupted image | File will not open | Training crash |
| Orphan label | `.txt` with no image | Useless but harmless |

### 6.4 Semantic quality issues (no script catches these; discuss in step 8)

- **Visually overlapping classes** (car/SUV/taxi, van/minivan, bus/minibus, motorbike/scooter, auto-rickshaw/CNG). If humans cannot separate them reliably, labels are inconsistent and the model will confuse them: expect low per-class scores and confusion-matrix off-diagonals.
- **Classes absent from COCO** (rickshaw, CNG, human hauler, wheelbarrow, ambulance): pre-training gives less of a head start.
- **Unknown box rules** (tightness, occlusion, truncation): infer from overlays.
- **Possible missing boxes** (unlabelled background vehicles): the most damaging noise, because correct detections are scored as false positives.

Habit: overlay boxes on 20-30 random images (OpenCV) and write down what you see. These observations become evidence.

### 6.5 If you ever annotate yourself: guideline checklist

Define classes (mutually exclusive, visually separable), minimum object size, occlusion rule (for example label if at least 30% visible), truncation at borders, parked vs moving, box rule (tight around visible extent, include mirrors, exclude shadows), edge cases (trailers, rider on motorbike, bus vs minibus, crowds). Annotate every instance. Review 10-20% with a second annotator and measure IoU agreement. Document every decision.

### 6.6 Annotating with a pre-trained model: bias warning
Pre-labelling with the pretrained model and correcting is faster, but it biases labels toward what that model already finds. Document it if used.

---

## 7. Train / validation / test split

| Set | Purpose | Analogy |
|---|---|---|
| **Train** (2,297) | The model learns from it | Practice exercises |
| **Validation** (298) | Monitor progress, choose the best checkpoint | Mock exam |
| **Test** (405) | One final, honest score | The real exam |

- The test set is carved out **before** training and never used for tuning.
- `random_state=42` makes the selection reproducible.
- **Data leakage:** if near-identical images (for example consecutive video frames or the same scene) land in both train and test, metrics are inflated. Check exact duplicates (file hashes) and near-duplicates (perceptual hashes) across sets.
- A random split does **not** preserve class proportions: compute the test class distribution too, because a rare class may have very few test examples.
- Recommended folder layout:
```
dataset/
  train/images  train/labels
  valid/images  valid/labels
  test/images   test/labels
  data.yaml
```

---

## 8. Evaluation theory

### 8.1 Core definitions

- **IoU (Intersection over Union)** = overlap area / combined area of predicted and true box, from 0 to 1.
- **True positive (TP):** correct class and IoU at or above the threshold. **False positive (FP):** prediction that is wrong class or too little overlap, or has no real object. **False negative (FN):** a real vehicle with no matching prediction.
- **Precision** = TP / (TP + FP). **Recall** = TP / (TP + FN).
- **AP:** area under the precision-recall curve for one class. **mAP:** mean of AP over classes.
- **mAP@50:** IoU threshold 0.5 (lenient). **mAP@50-95:** average over ten thresholds 0.50, 0.55, ..., 0.95 (strict, **rewards tight boxes**), so it is always lower.

### 8.2 Why overall mAP can mislead
A dataset dominated by a few classes can give a good overall number while rare classes perform badly. Always read **per-class** results, and relate them to instance counts (step 2). Rare classes also have few test examples, so their metrics are statistically fragile.

### 8.3 Confidence threshold and NMS

- The network proposes thousands of boxes. **Confidence threshold** drops uncertain boxes. **NMS** keeps the best of several overlapping boxes. You need both: the threshold alone leaves duplicates; NMS alone leaves low-score background guesses.
- **Evaluation** uses a very low threshold (about **0.001**) so mAP can sweep every score.
- **Visualisation** uses **0.25** [BRIEF] for readable images.
- Same model, very different pictures. **Distinguishing these is explicitly required [BRIEF step 6].**
- NMS-free models (YOLOv10, YOLO26) are trained so raw output is already one box per object.

### 8.4 Confusion matrix
Generated in the run folder. Shows which classes get confused with which. Use it to back claims about visually similar classes.

---

## 9. Step-by-step guide (brief steps 1 to 9)

### Step 1. Environment and dataset

1. Colab: *Runtime, Change runtime type, GPU (T4)*.
2. Verify:
```python
!nvidia-smi
import torch
print(torch.__version__, torch.cuda.is_available())
```
3. Kaggle: create an API token (*Account, Create New Token*), upload `kaggle.json` (or use `kagglehub`), search Kaggle for "Road Vehicle Images Dataset" and download and unzip it. **[VERIFY]** the exact dataset slug on Kaggle.
4. Explore the tree (`!find <dataset> -maxdepth 3 | head -50`), count files per folder, locate `data.yaml`.
5. **Mount Google Drive** and save checkpoints there: Colab sessions are temporary.

### Step 2. Exploration and cleaning

Read `data.yaml` (number of classes, names, paths). Then run a validator over every split.

```python
from pathlib import Path
import cv2, collections

def check_split(img_dir, lbl_dir, nc):
    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
    exts = {".jpg", ".jpeg", ".png"}
    imgs = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in exts}
    lbls = {p.stem: p for p in lbl_dir.glob("*.txt")}
    rep, counts = collections.defaultdict(list), collections.Counter()
    rep["missing_label"] = sorted(imgs.keys() - lbls.keys())
    rep["orphan_label"] = sorted(lbls.keys() - imgs.keys())
    for stem, p in imgs.items():
        if cv2.imread(str(p)) is None:
            rep["corrupt_image"].append(stem)
        if stem not in lbls:
            continue
        lines = [l.strip() for l in lbls[stem].read_text().splitlines() if l.strip()]
        if not lines:
            rep["empty_label"].append(stem)
            continue
        seen = set()
        for l in lines:
            parts = l.split()
            try:
                if len(parts) != 5:
                    raise ValueError
                c = int(parts[0]); x, y, w, h = map(float, parts[1:])
                ok = (0 <= c < nc and w > 0 and h > 0
                      and all(0 <= v <= 1 for v in (x, y, w, h))
                      and x - w/2 >= -1e-3 and x + w/2 <= 1 + 1e-3
                      and y - h/2 >= -1e-3 and y + h/2 <= 1 + 1e-3)
                if not ok:
                    raise ValueError
            except ValueError:
                rep["invalid_line"].append((stem, l))
                continue
            if l in seen:
                rep["duplicate_line"].append((stem, l))
            seen.add(l)
            counts[c] += 1
    return rep, counts
```

- Decide per problem: remove the image, delete the bad line, or fix it. **Log every decision and count.**
- **Re-run the checks after cleaning** to prove it worked.
- **Class distribution:** count **instances** per class (not images per class), plot a bar chart; also look at box-size distribution and boxes per image. Record these numbers: they explain per-class results later.
- Overlay boxes on 20-30 random images and write observations.

### Step 3. Splits

```python
from sklearn.model_selection import train_test_split
stems = sorted(p.stem for p in Path("train/images").iterdir())   # adapt to your cleaned set
train_stems, test_stems = train_test_split(stems, test_size=405, random_state=42)
# move each image AND its label together into test/images and test/labels
```
- Expected counts: train 2,297, valid 298, test 405. **Verify** every image still has its label.
- Rewrite `data.yaml` with absolute Colab paths:
```yaml
path: /content/dataset
train: train/images
val: valid/images
test: test/images
nc: <number of classes>
names: [<class 0>, <class 1>, ...]   # keep the original order and ids
```
- **Leakage check:** exact duplicates with MD5 hashes; near-duplicates with perceptual hashes (for example the `imagehash` library). Report findings and remove or note them.
- Compute the **class distribution of each split**.

### Step 4. YOLOv9 setup

1. `git clone https://github.com/WongKinYiu/yolov9` and install `requirements.txt`.
2. Download `yolov9-c.pt` from the repo's releases page (see 5.8; **[VERIFY]** link and filename).
3. Open and skim `train_dual.py`, `models/detect/yolov9-c.yaml`, `data/hyps/hyp.scratch-high.yaml`.
4. Verify all paths with `!ls`.
5. Write in the notebook: architecture vs weights vs script (5.1) and transfer learning (5.2).
6. Pitfall: library version conflicts on Colab (especially NumPy). Read the error and pin the version it names.

### Step 5. Training

1. Run the command in 5.3.
2. Check the first-epoch log: "transferred items", number of classes, images and labels found, corrupt/missing label warnings.
3. Plot `results.csv` (losses and metrics per epoch).
4. Confirm `best.pt` and `last.pt` exist; **copy to Drive**.
5. In the notebook, define: epoch, batch, iteration, hyperparameter, checkpoint; explain why this is fine-tuning.
6. If GPU memory runs out, lower the batch size (4 normally fits).

### Step 6. Evaluation

```bash
python val_dual.py --data /content/dataset/data.yaml --img 640 --batch 4 \
  --weights runs/train/yolov9c_vehicles/weights/best.pt \
  --task test --conf 0.001 --iou 0.7 --device 0 --name eval_test
```
- **`--task test`** is essential: otherwise the script evaluates the validation split.
- `--conf 0.001` is deliberate (see 8.3).
- **[VERIFY]** argument names with `python val_dual.py --help`.
- Record overall and per-class Precision, Recall, mAP@50, mAP@50-95. Sort by class, inspect the confusion matrix. Tie each weak class to its instance count (step 2) and to visual similarity with other classes.

### Step 7. Predictions and visualisation

```bash
python detect_dual.py --source /content/dataset/test/images \
  --weights runs/train/yolov9c_vehicles/weights/best.pt \
  --img 640 --conf-thres 0.25 --device 0 --name pred_test
```
- **[VERIFY]** argument names with `--help`.
- Show **at least 5** images inline (OpenCV/matplotlib). Choose deliberately: good detections, a rare class, crowded scenes, small or distant vehicles.
- For each, note: correct boxes, **missed** vehicles (FN), **spurious** boxes (FP), **wrong-class** boxes. Link observations to metrics.

### Step 8. Critical analysis
See section 10.

### Step 9. Bonus: Ultralytics
```python
from ultralytics import YOLO
model = YOLO("yolov9c.pt")                        # COCO classes only
results = model.predict("image.jpg", conf=0.25)
results[0].show()
```
Optionally fine-tune: `model.train(data=..., epochs=10, imgsz=640, batch=4)` and `model.val(split="test")`. **[VERIFY]** model filename and arguments against current Ultralytics docs. Compare: ease of use, results, and that the pretrained model only knows COCO classes while yours knows your vehicle classes.

---

## 10. Critical analysis framework

**Structure for step 8**

1. **Interpretation:** overall numbers; best and worst classes.
2. **Why classes differ:** number of training examples, visual similarity, object size, presence in COCO, annotation consistency.
3. **Limits:** class imbalance; only 10 epochs and batch 4 (and warmup share); possible label noise; one region, limited lighting/cameras; few test examples for rare classes; boxes cannot describe shape.
4. **At least two improvements, each justified**, for example:

| Improvement | Justified if |
|---|---|
| Train longer | Validation curves still rising at epoch 10 |
| Handle imbalance (more images of rare classes, targeted augmentation, class weighting) | Rare classes have low recall/AP and few instances |
| Merge or relabel ambiguous classes | Confusion matrix shows systematic confusion |
| Higher resolution or a larger model | Small or distant vehicles are missed |
| Audit and correct annotations | Many apparent false positives are actually missing labels |
| Instance segmentation | Box shape or heavy overlap limits usefulness |
| A newer YOLO version | Documented limitation it claims to address (an untested hypothesis) |

5. **Measured vs proposed:** label every statement. "mAP@50 of class X is Y" is **measured**. "More epochs would help" is a **hypothesis to be tested**.

**Evidence to collect while you work:** class counts per split; overlay observations; training curves; per-class table; confusion matrix; 5+ prediction images with notes.

---

## 11. Pitfalls checklist

- [ ] `data.yaml` class order matches the ids in labels.
- [ ] Every image has a label file (and the reverse); empty labels reviewed.
- [ ] Cleaning verified by re-running the checks.
- [ ] Test split is made once, with seed 42, before training.
- [ ] No exact or near-duplicate images across train/valid/test.
- [ ] `data.yaml` paths rewritten for Colab after the split.
- [ ] Dual weights used with `train_dual.py`; paths verified.
- [ ] Checkpoints copied to Drive (Colab is temporary).
- [ ] `--task test` used for evaluation.
- [ ] Evaluation (conf 0.001) not confused with visualisation (conf 0.25).
- [ ] Per-class results read, not only overall mAP.
- [ ] Results and hypotheses clearly separated in the report.
- [ ] README and notebook pushed to GitHub before the deadline.

---

## 12. Schedule

The brief allows 26-30 Oct 2026, deadline **Friday 30/10/2026 before 17:00**. This session took place on Tuesday 27/10. If starting from scratch now, a compressed plan:

| Day | Work |
|---|---|
| Tue 27 | Steps 1-3 (environment, exploration, cleaning, splits) |
| Wed 28 | Step 4, start step 5 training (let it run; keep checkpoints on Drive) |
| Thu 29 | Steps 6-7 (evaluation, visualisation); begin step 8 |
| Fri 30 (morning) | Finish step 8, bonus step 9, README, GitHub push. Submit before 17:00. |

Original 5-day plan: Day 1 steps 1-2; Day 2 steps 3-4; Day 3 step 5; Day 4 steps 6-7; Day 5 steps 8-9 and submission.

---

## 13. Glossary

| Term | Definition |
|---|---|
| **Annotation / ground truth** | Human-provided correct boxes and classes the model learns from and is scored against |
| **Anchor** | Predefined reference box shape (anchor-based detectors v3 to v7) |
| **Anchor-free** | Predicts box coordinates directly (v8 onward, YOLOv9 head) |
| **AP / mAP** | Area under precision-recall curve per class / mean over classes |
| **Backbone / neck / head** | Feature extractor / multi-scale feature fusion / prediction layers |
| **Batch** | Group of images processed per weight update |
| **Bounding box** | Rectangle around an object |
| **Checkpoint** | Saved model state (`best.pt`, `last.pt`) |
| **COCO** | Large benchmark dataset (80 classes) used to pre-train the weights |
| **Confidence** | Model's score that a box contains an object of a class |
| **Confusion matrix** | Table of true vs predicted classes |
| **DFL** | Distribution Focal Loss; refines box edge precision |
| **Epoch** | One full pass over the training set |
| **Fine-tuning** | Continuing training of a pre-trained model on a new dataset |
| **GELAN** | YOLOv9 efficient layer-aggregation network design |
| **Hyperparameter** | Training setting not learned by the model (learning rate, augmentation strength) |
| **IoU** | Intersection over Union |
| **Data leakage** | Near-identical data in both train and test, inflating metrics |
| **Loss** | "How wrong" score the optimiser tries to lower |
| **Mosaic** | Augmentation stitching four images together |
| **NMS** | Non-Maximum Suppression: removes duplicate overlapping boxes |
| **Overfitting / under-training** | Memorising training data / not trained enough |
| **PGI** | Programmable Gradient Information: training-only auxiliary branch in YOLOv9 |
| **Precision / Recall** | TP/(TP+FP) / TP/(TP+FN) |
| **Segmentation** | Pixel-level masks per object (semantic or instance) |
| **Transfer learning** | Reusing knowledge from a model trained on another task or dataset |

---

## 14. Sources and verification list

**Project brief:** provided by the project owner (French original), translated in section 1.

**Web sources consulted in this session**
- GeeksforGeeks, *Object Detection vs Object Recognition vs Image Segmentation* (published 2020, updated July 2025): https://www.geeksforgeeks.org/machine-learning/object-detection-vs-object-recognition-vs-image-segmentation/
- YOLO versions survey (arXiv 2510.09653): https://arxiv.org/abs/2510.09653
- arXiv 2602.14582: https://arxiv.org/pdf/2602.14582
- Preprint review, YOLO evolution (Preprints.org 202602.1844): https://www.preprints.org/manuscript/202602.1844/v1
- Preprint (Preprints.org 202607.1076): https://www.preprints.org/manuscript/202607.1076
- Datature, YOLO glossary: https://datature.io/glossary/yolo

**Items to verify before relying on them**
1. Dataset details: number of classes, class names, origin, exact Kaggle slug.
2. Exact script argument names (`train_dual.py`, `val_dual.py`, `detect_dual.py`): run each with `--help`.
3. `--close-mosaic` value suitable for 10 epochs.
4. Hyperparameter values in `hyp.scratch-high.yaml`.
5. Correct `yolov9-c.pt` release file (dual vs converted) and download link.
6. Ultralytics model filename and API arguments (changes between releases).
7. Version-comparison numbers (YOLO26, YOLOv13): reported by sources, not independently benchmarked.
8. Era-1 and YOLOv10 details marked [KNOWN] are from general knowledge, not from a cited source here.

**Companion material:** an interactive animated guide (7 chapters: one look vs many, annotation format, grid and scales, confidence and NMS, IoU and mAP thresholds, training loop, class imbalance) was published at https://claude.ai/artifact/B7pPBTRPnjCouR7QGNMhG6 and is visible only to its owner unless shared.
