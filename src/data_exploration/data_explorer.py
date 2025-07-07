import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import xml.etree.ElementTree as ET

class DataExplorer:
    def __init__(self, annotations_dir, classes_names, output_dir):
        """Initialize the DataExplorer with directory paths and class names."""
        self.annotations_dir = annotations_dir
        self.classes_names = classes_names
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_class_counts(self):
        """Compute the count of each class in the annotations."""
        class_counts = Counter()
        for xml_file in os.listdir(self.annotations_dir):
            if not xml_file.endswith(".xml"):
                continue
            tree = ET.parse(os.path.join(self.annotations_dir, xml_file))
            root = tree.getroot()
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name in self.classes_names:
                    class_counts[class_name] += 1
        return class_counts

    def plot_class_distribution(self):
        """Plot the distribution of classes with percentages."""
        class_counts = self.get_class_counts()
        for cls in self.classes_names:
            class_counts.setdefault(cls, 0)
        sorted_counts = [class_counts[cls] for cls in self.classes_names]
        total = sum(sorted_counts)
        percentages = [f"{(count / total) * 100:.1f}%" if count > 0 else "0%" for count in sorted_counts]

        plt.figure(figsize=(16, 6))
        ax = sns.barplot(x=self.classes_names, y=sorted_counts, hue=self.classes_names, palette="viridis", legend=False)
        for i, (count, pct) in enumerate(zip(sorted_counts, percentages)):
            ax.text(i, count + max(sorted_counts) * 0.01, pct, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Number of Instances")
        ax.set_xlabel("Classes")
        ax.set_title("Dataset Class Distribution with Percentages")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "class_distribution.png"))
        plt.close()

    def get_image_dimensions(self):
        """Retrieve image dimensions from annotations."""
        img_dimensions = []
        for xml_file in os.listdir(self.annotations_dir):
            if not xml_file.endswith(".xml"):
                continue
            tree = ET.parse(os.path.join(self.annotations_dir, xml_file))
            root = tree.getroot()
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            img_dimensions.append((width, height))
        return img_dimensions

    def plot_image_dimensions(self):
        """Plot the distribution of image dimensions."""
        img_dims = self.get_image_dimensions()
        widths = [d[0] for d in img_dims]
        heights = [d[1] for d in img_dims]

        plt.figure(figsize=(12, 6))
        sns.histplot(widths, kde=True, color="blue", label="Width")
        sns.histplot(heights, kde=True, color="red", label="Height")
        plt.title("Distribution of Image Dimensions")
        plt.xlabel("Pixels")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "image_dimensions_distribution.png"))
        plt.close()

    def get_objects_per_image(self):
        """Compute the number of objects per image."""
        objects_per_image = Counter()
        for xml_file in os.listdir(self.annotations_dir):
            if not xml_file.endswith(".xml"):
                continue
            tree = ET.parse(os.path.join(self.annotations_dir, xml_file))
            root = tree.getroot()
            obj_count = len(root.findall("object"))
            objects_per_image[obj_count] += 1
        return objects_per_image

    def plot_objects_per_image_distribution(self):
        """Plot the distribution of objects per image."""
        objects_per_image = self.get_objects_per_image()
        counts = list(objects_per_image.keys())
        frequencies = list(objects_per_image.values())

        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts, y=frequencies, hue=counts, palette="viridis", legend=False)
        plt.title("Distribution of Objects per Image")
        plt.xlabel("Number of Objects")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "objects_per_image_distribution.png"))
        plt.close()