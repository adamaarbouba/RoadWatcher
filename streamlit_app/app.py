import os
import sys
import types
from pathlib import Path
import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image

# Add yolov9 directory to sys.path so we can import its modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
yolo_dir = project_root / "yolov9"

if str(yolo_dir) not in sys.path:
    sys.path.insert(0, str(yolo_dir))

# Ensure export_formats is available in sys.modules so models/common.py never fails
if "export" not in sys.modules:
    export_mod = types.ModuleType("export")
    def export_formats():
        x = [
            ['PyTorch', '-', '.pt', True, True],
            ['TorchScript', 'torchscript', '.torchscript', True, True],
            ['ONNX', 'onnx', '.onnx', True, True],
            ['ONNX END2END', 'onnx_end2end', '_end2end.onnx', True, True],
            ['OpenVINO', 'openvino', '_openvino_model', True, False],
            ['TensorRT', 'engine', '.engine', False, True],
            ['CoreML', 'coreml', '.mlmodel', True, False],
            ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
            ['TensorFlow GraphDef', 'pb', '.pb', True, True],
            ['TensorFlow Lite', 'tflite', '.tflite', True, False],
            ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
            ['TensorFlow.js', 'tfjs', '_web_model', False, False],
            ['PaddlePaddle', 'paddle', '_paddle_model', True, True],
        ]
        return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])
    export_mod.export_formats = export_formats
    sys.modules["export"] = export_mod

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

# Streamlit page layout
st.set_page_config(page_title="RoadWatcher - Vehicle Detection", page_icon="🚗", layout="wide")
st.title("🚗 RoadWatcher - Road Vehicle Detection")
st.write("Upload an image of road traffic to detect vehicles using the trained YOLOv9-C model.")

# Sidebar controls
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.05, max_value=1.0, value=0.25, step=0.05)
iou_threshold = st.sidebar.slider("IoU (NMS) Threshold", min_value=0.1, max_value=1.0, value=0.45, step=0.05)

# Function to search for available model weights
def get_weights_path():
    candidate_paths = [
        project_root / "yolov9" / "runs" / "train" / "roadwatcher_yolov9c" / "weights" / "best.pt",
        project_root / "best.pt",
        Path.cwd() / "best.pt",
        Path("/content/drive/MyDrive/RoadWatcher/best.pt"),
        yolo_dir / "runs" / "train" / "roadwatcher_yolov9c" / "weights" / "best.pt",
        yolo_dir / "yolov9-c.pt",
        project_root / "yolov9-c.pt"
    ]
    for p in candidate_paths:
        if p.exists():
            return p
    return None

weights_file = get_weights_path()

# If no weights found, offer file upload directly in Streamlit
if weights_file is None:
    st.warning("⚠️ No trained weights found automatically on disk.")
    st.write("Please upload your `best.pt` file below, or run Notebook 02 to train the model:")
    uploaded_weights = st.file_uploader("Upload model file (best.pt)", type=["pt"])
    if uploaded_weights is not None:
        save_path = project_root / "best.pt"
        with open(save_path, "wb") as f:
            f.write(uploaded_weights.read())
        st.success("Weights saved to " + save_path.name + "! Reloading application...")
        st.rerun()
    else:
        st.info("Tip: In Colab, if you saved best.pt to Google Drive, run Cell 3 in Notebook 03 to mount Drive and restore it.")
        st.stop()

st.sidebar.success("Using model: " + weights_file.name)

# Cache model loading so it loads only once
@st.cache_resource
def load_yolo_model(weights_path):
    device = select_device("0" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(str(weights_path), device=device, dnn=False, data=None, fp16=False)
    return model, device

model, device = load_yolo_model(weights_file)
stride = int(model.stride.max() if hasattr(model.stride, "max") else model.stride)
imgsz = check_img_size(640, s=stride)
class_names = model.names

st.sidebar.write("Device: " + ("GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"))
st.sidebar.write("Total Classes: " + str(len(class_names)))

# File uploader
uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Open the image
    image = Image.open(uploaded_file).convert("RGB")
    original_numpy = np.array(image)
    
    # 2. Preprocess image with letterbox (resize to 640 while maintaining aspect ratio)
    img_letterboxed = letterbox(original_numpy, imgsz, stride=stride, auto=True)[0]
    
    # Convert BGR/RGB to tensor (channels first: HWC -> CHW)
    img_tensor = img_letterboxed.transpose((2, 0, 1))
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor).to(device)
    img_tensor = img_tensor.float() / 255.0  # Normalize 0 - 255 to 0.0 - 1.0
    
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension: (1, 3, 640, 640)
        
    # 3. Model forward pass
    with torch.no_grad():
        prediction = model(img_tensor)
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0][1] if isinstance(prediction[0], (list, tuple)) else prediction[0]
        
        # 4. Non-Max Suppression (NMS)
        detections = non_max_suppression(
            prediction,
            conf_thres=confidence_threshold,
            iou_thres=iou_threshold,
            max_det=1000
        )
        
    # 5. Draw bounding boxes on image
    det = detections[0]
    annotator = Annotator(original_numpy.copy(), line_width=3, example=str(class_names))
    
    detection_records = []
    if len(det):
        # Rescale bounding boxes back to original image size
        det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], original_numpy.shape).round()
        
        for *xyxy, conf, cls in reversed(det):
            cls_id = int(cls)
            if isinstance(class_names, dict) and cls_id in class_names:
                class_name = class_names[cls_id]
            elif isinstance(class_names, (list, tuple)) and 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id)
                
            score = round(float(conf) * 100, 1)
            
            box_label = class_name + " " + str(score) + "%"
            annotator.box_label(xyxy, box_label, color=colors(cls_id, True))
            
            detection_records.append({
                "Vehicle Type": class_name,
                "Confidence": str(score) + "%",
                "Class ID": cls_id
            })
            
    result_image = annotator.result()
    
    # 6. Display Side-by-Side Images
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detections (" + str(len(detection_records)) + " vehicles)")
        st.image(result_image, use_container_width=True)
        
    # 7. Summary Table of Detections
    if len(detection_records) > 0:
        st.subheader("Detailed Detections")
        st.table(detection_records)
    else:
        st.info("No vehicles detected at confidence " + str(confidence_threshold) + ". Try lowering the confidence slider.")
else:
    st.info("Please upload an image above to run vehicle detection.")
