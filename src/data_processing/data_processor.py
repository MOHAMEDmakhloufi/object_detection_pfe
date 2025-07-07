import os
import xml.etree.ElementTree as ET
from shutil import copy2

class DataProcessor:
    def __init__(self, dataset_root, output_root, classes_names):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.classes_names = classes_names
        self.annotations_dir = os.path.join(dataset_root, "Annotations")
        self.images_dir = os.path.join(dataset_root, "JPEGImages")

    def process_image(self, file_id, label_out_dir, image_out_dir):
        xml_path = os.path.join(self.annotations_dir, f"{file_id}.xml")
        if not os.path.exists(xml_path):
            print(f"Skipping missing XML file: {xml_path}")
            return
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename = root.find("filename").text
        img_path = os.path.join(self.images_dir, filename)
        if not os.path.exists(img_path):
            print(f"Skipping missing image file: {img_path}")
            return
        base_name = os.path.splitext(filename)[0]
        label_out_path = os.path.join(label_out_dir, f"{base_name}.txt")
        image_out_path = os.path.join(image_out_dir, filename)
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in self.classes_names:
                continue
            class_id = self.classes_names.index(class_name)
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        with open(label_out_path, "w") as f:
            f.write("\n".join(yolo_lines))
        copy2(img_path, image_out_path)

    def convert_voc_to_yolo(self):
        sets = ['train', 'val', 'test']
        for split in sets:
            split_txt = os.path.join(self.dataset_root, "ImageSets", "Main", f"{split}.txt")
            if not os.path.exists(split_txt):
                print(f"Split file not found: {split_txt}. Skipping {split} split.")
                continue
            label_out_dir = os.path.join(self.output_root, "labels", split)
            image_out_dir = os.path.join(self.output_root, "images", split)
            os.makedirs(label_out_dir, exist_ok=True)
            os.makedirs(image_out_dir, exist_ok=True)
            with open(split_txt) as f:
                file_ids = [line.strip() for line in f.readlines()]
            for file_id in file_ids:
                self.process_image(file_id, label_out_dir, image_out_dir)
        print(f"✅ VOC to YOLO conversion done. Data saved in '{self.output_root}'.")