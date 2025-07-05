import os
from ultralytics import YOLO


def evaluate_yolo_model(model_path, model_name, data_config_path, output_dir):

    try:
        print(f"Loading model from: {model_path}")
        model = YOLO(model_path)

        print(f"Using dataset configuration: {data_config_path}")
        print(f"Saving evaluation results to: {os.path.join(output_dir, 'evaluation_metrics')}")

        # Ensure the base output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Run validation
        metrics = model.val(
            device="mps",
            data=data_config_path,
            name=model_name,
            project=output_dir,
            save_json=True,  # Save metrics results to a JSON file
            save_hybrid=False,  # Save hybrid format labels (labels + predictions)
            plots=True
        )


        print("\n--- Evaluation Metrics ---")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP75: {metrics.box.map75:.4f}")
        # You can access per-class mAP50 via metrics.box.maps

        results_dir = os.path.join(output_dir, 'evaluation_metrics')
        print(f"\nDetailed results, plots (including confusion matrix), and metrics saved in: {results_dir}")

        return metrics

    except FileNotFoundError as e:
        print(f"Error: File not found. Please check model path and data YAML path. {e}")
        return None
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        return None