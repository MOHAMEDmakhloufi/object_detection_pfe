
import torch
from ultralytics import YOLO

class YOLOv7Predictor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
        return results

    def visualize_predictions(self, results, output_dir="./predictions"):
        for i, r in enumerate(results):
            # Plot results image
            im_bgr = r.plot()  # BGR numpy array
            # Save results to disk
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f"prediction_{i}.jpg")
            cv2.imwrite(output_path, im_bgr)
            print(f"Prediction saved to {output_path}")


