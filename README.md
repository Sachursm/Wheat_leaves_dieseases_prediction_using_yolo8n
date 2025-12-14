
### 🌾 Wheat Leaf Disease Detection using YOLOv8

This project detects wheat leaf diseases using a YOLOv8 object detection model trained on labeled images with bounding boxes.
The trained model is later used inside a web application for real-time or image-based disease detection.



## 📁 Dataset

Due to large file size, the dataset is not included in this repository.

### 🔗 Download Link
Google Drive:
[https://drive.google.com](https://drive.google.com/file/d/1NGGp7IVQm5E9Z4epuHWx4WD8MX17RuLx/view?usp=sharing)

### 📦 How to Use the Dataset

1. Download `image.zip`
2. Extract it into the project root folder

Final structure should look like this:

### 📌 Disease Classes
## The dataset contains 5 classes:
```
Class ID 	Disease Name
0	        BrownRust
1	        Healthy
2	        Mildew
3	        Septoria
4	        YellowRust
```
📂 Project Structure
```
.
├── split_dataset.py
├── test_gpu.py
│
├── model_training/
│   ├── wheat.yaml
│   ├── run_this_code_in_terminal.txt
│
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt
│               └── last.pt
│
├── model_application/
│   ├── app.py
│   ├── templates/
│   ├── static/
│
└── README.md
```
### 🧪 Dataset Preparation

The dataset is split into train and validation sets using:
```
python split_dataset.py
```
Expected folder structure after splitting:
```kotlin
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
```
### 🏷️ Data Labeling

Labeling tool used: https://www.makesense.ai

Annotation type: Bounding Boxes
Export format: YOLO
One image contains:
  Original image (no bounding box)
  Labeled image (with bounding boxes)

  
### ⚙️ GPU Setup (Recommended)


## 1️⃣ Create Virtual Environment
```
python -m venv yolovenv
```
## 2️⃣ Activate Virtual Environment
Windows
```
yolovenv\Scripts\activate
```
## 3️⃣ Install PyTorch with CUDA (GPU)

Make sure you have an NVIDIA GPU and CUDA-compatible drivers installed.
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
## 4️⃣ Install YOLOv8 (Ultralytics)
```
pip install ultralytics
```
## 5️⃣ Verify GPU Availability

Run:
```
python test_gpu.py
```
Expected output:
```
True
NVIDIA RTX XXXX
```
### 🧠 Model Selection

Model used: yolov8n (Nano)
Reason:
  Fast
  Lightweight
  Suitable for deployment and edge devices


### 🧾 Dataset Configuration (wheat.yaml)

Located inside model_training/

```yaml
path: D:/AI_projects/wheat_diseases_detection/model_training/wheat_dataset  

train: images/train
val: images/val

names:
  0: BrownRust
  1: Healthy
  2: Mildew
  3: Septoria
  4: YellowRust
```

### 🚀 Model Training

Navigate to the training folder:
```
cd model_training
```

Run the training command:

```
yolo detect train model=yolov8n.pt data=wheat.yaml epochs=50 imgsz=640
```

### 📈 Training Output

After training completes, YOLO automatically creates:

```bash
runs/detect/train/
└── weights/
    ├── best.pt
    └── last.pt
```

## 🔑 Important

best.pt → Best performing model (use this)

last.pt → Final epoch model

### 🌐 Web Application

The model_application/ folder contains the web app files.

Purpose

Load best.pt

Upload an image

Detect wheat leaf diseases

Display bounding boxes with disease labels

You can build this using Flask / FastAPI / Streamlit.


### 📦 Deployment Note

Always use best.pt for inference

yolov8n is suitable for:

    Web apps
    Low-latency inference
    Edge / embedded systems
