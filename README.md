# RoadWatcher

RoadWatcher is a beginner-friendly YOLOv9-C vehicle detection system trained on Bangladeshi road traffic images. It cleans and splits the dataset, trains YOLOv9-C, evaluates the test set, and deploys a live Streamlit web application for interactive predictions.

## Folder Map

```text
docs/                 Cell-by-cell guides and documentation
notebooks/            01_prepare_data, 02_train_and_evaluate, 03_deploy_streamlit
streamlit_app/        Interactive Streamlit web app (app.py)
trafic_data/          Original raw dataset (21 vehicle classes)
data_clean/           Generated clean dataset (2,297 train, 298 valid, 405 test)
roadwatcher.yaml      Generated dataset configuration for YOLOv9
yolov9/               Trimmed YOLOv9 code and checkpoints (runs/)
```

## Running on Google Colab (Recommended)

1. Open **Google Colab** ([colab.research.google.com](https://colab.research.google.com)).
2. Make sure you enable a GPU runtime: **Runtime > Change runtime type > T4 GPU**.
3. Upload and run the notebooks in sequential order:
   - `notebooks/01_prepare_data.ipynb` — Cleans data, creates 405-image test split, and generates `roadwatcher.yaml`.
   - `notebooks/02_train_and_evaluate.ipynb` — Trains YOLOv9-C (10 epochs), evaluates test set, and saves `best.pt`.
   - `notebooks/03_deploy_streamlit.ipynb` — Launches the interactive Streamlit web app with a public localtunnel link.

## Running Locally

1. Install requirements:
   ```bash
   pip install -r yolov9/requirements.txt
   pip install streamlit
   ```
2. Run notebooks 01 and 02 in Jupyter or VS Code.
3. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app/app.py
   ```
4. Open `http://localhost:8501` in your browser, upload an image, and test detections!

## Fixed Training Parameters

- Architecture: `models/detect/yolov9-c.yaml`
- Pretrained weights: `yolov9-c.pt`
- Image size: 640
- Batch size: 4
- Epochs: 10
- Hyperparameters: `data/hyps/hyp.scratch-high.yaml`
- Evaluation confidence: 0.001 (for mAP calculation)
- Visualization confidence: 0.25 (for human-readable boxes)

## Documentation

Detailed cell-by-cell explanations are available in `docs/`:
- [01_prepare_data_cells.md](file:///d:/Coding/RoadWatcher/docs/01_prepare_data_cells.md)
- [02_train_and_evaluate_cells.md](file:///d:/Coding/RoadWatcher/docs/02_train_and_evaluate_cells.md)
- [03_deploy_streamlit_cells.md](file:///d:/Coding/RoadWatcher/docs/03_deploy_streamlit_cells.md)
