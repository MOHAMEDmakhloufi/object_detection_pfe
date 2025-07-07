import os
import yaml
from src.data_processing.data_processor import DataProcessor
from src.data_exploration.data_explorer import DataExplorer
from src.models.model_converter import ModelConverter
from src.training.yolo_trainer import YOLOTrainer
from src.data_processing.data_downloader import DataDownloader

if __name__ == "__main__":
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]
    training_config = config["training"]
    mlflow_config = config["mlflow"]
    deployment_config = config["deployment"]

    # Data Downloading
    data_downloader = DataDownloader(
        file_id=dataset_config["file_id"],
        output_dir=config["output_dir_path"],
        output_file=dataset_config["output_file"],
        extract_dir=dataset_config["extract_dir"],
        extract_folder=dataset_config["extract_folder"],
    )
    data_downloader.download()
    # Data Exploration
    print("\n--- Data Exploration ---")
    annotations_dir = os.path.join(dataset_config["raw_data_path"], 'Annotations')
    data_explorer = DataExplorer(
        annotations_dir=annotations_dir,
        classes_names=dataset_config["classes"],
        output_dir=config["output_dir_path"]
    )
    data_explorer.plot_class_distribution()
    data_explorer.plot_objects_per_image_distribution()
    data_explorer.plot_image_dimensions()

    # Data Processing
    print("\n--- Data Processing ---")
    data_processor = DataProcessor(
        dataset_root=dataset_config["raw_data_path"],
        output_root=dataset_config["processed_data_path"],
        classes_names=dataset_config["classes"]
    )
    data_processor.convert_voc_to_yolo()

    # Training
    print("\n--- Model Training ---")

    yolo_trainer = YOLOTrainer(
        data_config_path=dataset_config["wotr_config_path"],
        epochs=training_config["epochs"],
        imgsz=training_config["imgsz"],
        batch_size=training_config["batch_size"],
        device=training_config["device"],
        model_name=training_config["model_name"],
        output_dir=training_config["training_output_dir"],
        weights_path=training_config["weights_path"],
        mlflow_tracking_uri=mlflow_config["tracking_uri"],
        mlflow_experiment_name=mlflow_config["experiment_name"]
    )

    trained_model = yolo_trainer.train()

    # Convert Model to tflite and onnx
    model_converter = ModelConverter(
        model_path= os.path.join(deployment_config["models_dir"], 'best.pt'),
        device= training_config["device"],
        data_config_path= dataset_config["wotr_config_path"],
        output_dir= deployment_config["models_dir"],
        img_size=training_config["imgsz"],
    )

    model_converter.convert_to_tflite()