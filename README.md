# RoadWatcher

RoadWatcher is a beginner-friendly YOLOv9-C vehicle detector. It cleans and splits the road-image dataset, fine-tunes YOLOv9-C, evaluates the 405-image test set, and saves visual predictions.

## Folder map

```text
docs/                 project brief, knowledge base, and YOLO guide
notebooks/            01_prepare_data and 02_train_and_evaluate
trafic_data/          original dataset (read-only source)
data_clean/           generated cleaned dataset
roadwatcher.yaml      generated dataset configuration
yolov9/               trimmed YOLOv9 implementation and runs/
```

## One-time setup

1. Connect VS Code to the Vast.ai GPU machine with Remote-SSH.
2. Open the `RoadWatcher` project folder on the remote machine.
3. Install the Python and Jupyter extensions on the remote machine.
4. Open the `notebooks/` folder and select the remote GPU Python kernel.

The data notebook creates the cleaned dataset and `roadwatcher.yaml`. The training notebook installs YOLOv9 requirements in the selected kernel.

## Run order

1. Run `notebooks/01_prepare_data.ipynb` from top to bottom.
2. Run `notebooks/02_train_and_evaluate.ipynb` from top to bottom.

The fixed project settings are YOLOv9-C, `train_dual.py`, image size 640, batch size 4, 10 epochs, seed 42, and confidence 0.25 for visualisation.

## If your connection drops

Run the same training command from `yolov9/` inside tmux so training continues after SSH disconnects:

```bash
tmux new -s train
python train_dual.py --device 0 --workers 2 --img 640 --batch-size 4 --epochs 10 --data ../roadwatcher.yaml --cfg models/detect/yolov9-c.yaml --weights yolov9-c.pt --hyp data/hyps/hyp.scratch-high.yaml --name roadwatcher_yolov9c --min-items 0 --close-mosaic 2 --exist-ok
```

Detach with `Ctrl+B` then `D`. Reconnect and run `tmux attach -t train`.

## Troubleshooting

- `libGL.so.1` error: run `apt-get install -y libgl1 libglib2.0-0`, or install `opencv-python-headless` instead.
- `bus error` or shared-memory errors: use `--workers 0` or `--workers 2`.
- CUDA out of memory: keep batch size 4.
- Vast.ai instance storage is lost when the instance is destroyed. Download `yolov9/runs/` first, especially `best.pt`, `last.pt`, plots, and predictions.