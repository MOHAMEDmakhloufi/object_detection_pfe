import os
import yaml
from src.data_processing.data_processor import voc_to_yolo_wotr
from src.data_exploration.data_explorer import class_counts_VOC_format, plot_class_distribution
from src.training.trainer import train_model
from src.data_processing.downloader import download_wotr_dataset, unzip_wotr_dataset

if __name__ == "__main__":
    # Load configuration
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]

    # Data Downloading
    zip_path = download_wotr_dataset(dataset_config["file_id"], "outputs", dataset_config["output_file"])
    unzip_wotr_dataset(zip_path, dataset_config["extract_dir"], dataset_config["extract_folder"])
    # Data Exploration
    print("\n--- Data Exploration ---")
    annotations_dir = os.path.join(dataset_config["raw_data_path"], 'Annotations')
    class_counts = class_counts_VOC_format(annotations_dir, dataset_config["classes"])
    plot_class_distribution(class_counts, dataset_config["classes"])

    # Data Processing
    print("\n--- Data Processing ---")

    voc_to_yolo_wotr(
        dataset_config["raw_data_path"],
        dataset_config["processed_data_path"],
        dataset_config["classes"]
    )

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