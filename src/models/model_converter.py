
import torch
from ultralytics import YOLO
import os

def convert_to_tflite(model_path, output_dir="./models/tflite", img_size=(640, 640)):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "best_model_int8.tflite")
    model.export(format="tflite", int8=True, imgsz=img_size) # Export to TFLite with int8 quantization
    print(f"TensorFlow Lite model saved to {output_path}")
    return output_path

def convert_to_onnx(model_path, output_dir="./models/onnx", img_size=(640, 640)):
    model = YOLO(model_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "best_model.onnx")
    model.export(format="onnx", imgsz=img_size) # Export to ONNX
    print(f"ONNX model saved to {output_path}")
    return output_path


