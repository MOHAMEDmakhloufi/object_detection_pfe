import os
import yaml
from src.data_processing.data_processor import voc_to_yolo_wotr
from src.data_exploration.data_explorer import class_counts_VOC_format, plot_class_distribution
from src.training.trainer import train_model

if __name__ == "__main__":
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]



    # Training
    print("\n--- Model Training ---")
    training_config = config["training"]
    data_yaml_path = dataset_config["wotr_config_path"]

    trained_model = train_model(
        data_yaml_path,
        training_config["epochs"],
        training_config["imgsz"],
        training_config["batch_size"],
        training_config["device"],
        training_config["model_name"],
        training_config["training_output_dir"],
        training_config["weights_path"]
    )