from ultralytics import YOLO
import os

class ModelConverter:
    def __init__(self, model_path, device, data_config_path, output_dir="./models", img_size=(640, 640)):
        """
        Initialize the ModelConverter with model path, output directory, and image size.

        Args:
            model_path (str): Path to the YOLO model file.
            output_dir (str): Base directory to save converted models.
            img_size (tuple): Image size for model export (width, height).
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.img_size = img_size
        self.model = None
        self.device = device
        self.data_config_path = data_config_path

    def load_model(self):
        """Load the YOLO model from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = YOLO(self.model_path)

    def convert_to_tflite(self):
        """Convert the YOLO model to TFLite format with int8 quantization."""
        try:
            if self.model is None:
                self.load_model()
            tflite_dir = os.path.join(self.output_dir, "tflite")
            os.makedirs(tflite_dir, exist_ok=True)
            output_path = os.path.join(tflite_dir, "best_model_int8.tflite")
            self.model.export(format="tflite", device = self.device, data=self.data_config_path, int8=True, imgsz=self.img_size)
            print(f"TensorFlow Lite model saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error during TFLite conversion: {e}")
            return None

    def convert_to_onnx(self):
        """Convert the YOLO model to ONNX format."""
        try:
            if self.model is None:
                self.load_model()
            onnx_dir = os.path.join(self.output_dir, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            output_path = os.path.join(onnx_dir, "best_model.onnx")
            self.model.export(format="onnx", device = self.device, data=self.data_config_path, imgsz=self.img_size)
            print(f"ONNX model saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error during ONNX conversion: {e}")
            return None

    def convert_all(self):
        """Convert the YOLO model to both TFLite and ONNX formats."""
        tflite_path = self.convert_to_tflite()
        onnx_path = self.convert_to_onnx()
        return {"tflite": tflite_path, "onnx": onnx_path}