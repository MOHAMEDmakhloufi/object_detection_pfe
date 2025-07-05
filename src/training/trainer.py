
import os
from ultralytics import YOLO
import mlflow
import mlflow.pytorch

def train_model(data_config_path, epochs, imgsz, batch_size, device, model_name, output_dir, weights_path=None):
    mlflow.set_experiment("YOLOv11_Object_Detection")

    with mlflow.start_run(run_name=model_name):
        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("imgsz", imgsz)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("device", device)
        mlflow.log_param("data_config", data_config_path)

        # Load model
        if weights_path:
            model = YOLO(weights_path)
        else:
            model = YOLO("yolo11m.pt")

        # Train the model
        results = model.train(
            data=data_config_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            name=model_name,
            project=output_dir
        )

        # Log metrics (example, adjust based on YOLO output)
        # You might need to parse the results object for specific metrics
        # For now, let's assume `results` contains accessible metrics like map50
        if hasattr(results, 'metrics') and results.metrics:
            mlflow.log_metrics(results.metrics)
        
        # Save the best model artifact
        best_model_path = os.path.join(model.trainer.save_dir, 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            mlflow.pytorch.log_model(model, "best_yolov11_model", registered_model_name="YOLOv11_WOTR_Model")
            print(f"Best model saved to MLflow: {best_model_path}")
        else:
            print("Best model not found at expected path.")

    return model


