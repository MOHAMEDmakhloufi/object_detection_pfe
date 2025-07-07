import os
import shutil
from ultralytics import YOLO, settings
import mlflow
import mlflow.pytorch

class YOLOTrainer:
    def __init__(self, data_config_path, epochs, imgsz, batch_size, device, model_name, output_dir, mlflow_tracking_uri,  mlflow_experiment_name, weights_path=None):
        """
        Initialize the YOLOv11Trainer with training parameters.

        Args:
            data_config_path (str): Path to the data configuration file.
            epochs (int): Number of training epochs.
            imgsz (int): Image size for training.
            batch_size (int): Batch size for training.
            device (str): Device to use for training (e.g., 'cpu', 'cuda').
            model_name (str): Name of the model for MLflow run.
            output_dir (str): Directory to save training outputs.
            weights_path (str, optional): Path to pre-trained weights. If None, uses default 'yolo11m.pt'.
        """
        self.data_config_path = data_config_path
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.device = device
        self.model_name = model_name
        self.output_dir = output_dir
        self.weights_path = weights_path
        self.model = None
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self. mlflow_experiment_name =  mlflow_experiment_name

    def log_parameters(self):
        """Log training hyperparameters to MLflow."""
        mlflow.log_param("epochs", self.epochs)
        mlflow.log_param("imgsz", self.imgsz)
        mlflow.log_param("batch_size", self.batch_size)
        mlflow.log_param("device", self.device)
        mlflow.log_param("data_config", self.data_config_path)

    def load_model(self):
        """Load the YOLO model, either from provided weights or default."""
        if self.weights_path:
            #if not os.path.exists(self.weights_path):
            #    raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
            self.model = YOLO(self.weights_path)
        else:
            self.model = YOLO("yolo11n.pt")

    def train_model(self):
        """Train the YOLO model and return the results."""
        results = self.model.train(
            data=self.data_config_path,
            epochs=self.epochs,
            imgsz=self.imgsz,
            batch=self.batch_size,
            device=self.device,
            name=self.model_name,
            project=self.output_dir
        )
        return results

    def log_metrics(self, results):
        """Log training metrics to MLflow."""
        if hasattr(results, 'metrics') and results.metrics:
            mlflow.log_metrics(results.metrics)

    def save_best_model(self):
        """Save the best model to MLflow if it exists."""

        best_model_path = os.path.join(self.model.trainer.save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            mlflow.pytorch.log_model(self.model, name="best_yolov11_model", registered_model_name="YOLOv11_WOTR_Model")
            print(f"Best model saved to MLflow: {best_model_path}")
            destination_path = os.path.join(self.output_dir, os.path.basename(best_model_path))
            shutil.copy2(best_model_path, destination_path)
        else:
            print("Best model not found at expected path.")

    def train(self):
        """Execute the training process with MLflow tracking."""
        settings.update({"mlflow": True})
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        mlflow.set_experiment(self.mlflow_experiment_name)
        with mlflow.start_run(run_name=self.model_name):
            self.log_parameters()
            self.load_model()
            results = self.train_model()
            self.log_metrics(results)
            self.save_best_model()
            return self.model