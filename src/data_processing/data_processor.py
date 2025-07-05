
import os
import xml.etree.ElementTree as ET
from shutil import copy2

def voc_to_yolo_wotr(dataset_root, output_root, classes_names):
    sets = ['train', 'val', 'test']

    for split in sets:
        # Paths
        split_txt = os.path.join(dataset_root, "ImageSets", "Main", f"{split}.txt")
        annotations_dir = os.path.join(dataset_root, "Annotations")
        images_dir = os.path.join(dataset_root, "JPEGImages")

        # Output directories
        label_out_dir = os.path.join(output_root, "labels", split)
        image_out_dir = os.path.join(output_root, "images", split)
        os.makedirs(label_out_dir, exist_ok=True)
        os.makedirs(image_out_dir, exist_ok=True)

        with open(split_txt) as f:
            file_ids = [line.strip() for line in f.readlines()]

        for file_id in file_ids:
            xml_path = os.path.join(annotations_dir, f"{file_id}.xml")
            img_path = os.path.join(images_dir, f"{file_id}.jpg")
            label_out_path = os.path.join(label_out_dir, f"{file_id}.txt")
            image_out_path = os.path.join(image_out_dir, f"{file_id}.jpg")

            if not os.path.exists(xml_path) or not os.path.exists(img_path):
                print(f"Skipping missing file: {file_id}")
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image dimensions
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)

            yolo_lines = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in classes_names:
                    continue
                class_id = classes_names.index(class_name)
                bndbox = obj.find("bndbox")
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))

                # Convert to YOLO format
                x_center = ((xmin + xmax) / 2) / width
                y_center = ((ymin + ymax) / 2) / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

            # Save label file
            with open(label_out_path, "w") as f:
                f.write("\n".join(yolo_lines))

            # Copy image
            copy2(img_path, image_out_path)

    print(f"✅ VOC to YOLO conversion done. Data saved in '{output_root}'.")

def class_counts_VOC_format(annotations_dir, classes_names):
    # Count class occurrences
    from collections import Counter
    class_counts = Counter()
    
    for xml_file in os.listdir(annotations_dir):
        if not xml_file.endswith(".xml"):
            continue
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()

        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name in classes_names:
                class_counts[class_name] += 1

    return class_counts

def plot_class_distribution(class_counts, classes_names):
    # Ensure all classes are included (even with zero count)
    import matplotlib.pyplot as plt
    import seaborn as sns

    for cls in classes_names:
        class_counts.setdefault(cls, 0)

    # Sort counts by class order in classes_names
    sorted_counts = [class_counts[cls] for cls in classes_names]
    total = sum(sorted_counts)
    percentages = [f"{(count / total) * 100:.1f}%" if count > 0 else "0%" for count in sorted_counts]

    # Plot
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(x=classes_names, y=sorted_counts, palette="viridis")

    # Annotate each bar with the percentage
    for i, (count, pct) in enumerate(zip(sorted_counts, percentages)):
        ax.text(i, count + max(sorted_counts) * 0.01, pct, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel("Number of Instances")
    ax.set_xlabel("Classes")
    ax.set_title("Dataset Class Distribution with Percentages")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


