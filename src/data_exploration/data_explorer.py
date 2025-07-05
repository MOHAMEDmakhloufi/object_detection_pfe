import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import xml.etree.ElementTree as ET


def class_counts_VOC_format(annotations_dir, classes_names):
    # Count class occurrences
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
    for cls in classes_names:
        class_counts.setdefault(cls, 0)

    # Sort counts by class order in classes_names
    sorted_counts = [class_counts[cls] for cls in classes_names]
    total = sum(sorted_counts)
    percentages = [f"{(count / total) * 100:.1f}%" if count > 0 else "0%" for count in sorted_counts]

    # Plot
    plt.figure(figsize=(16, 6))
    ax = sns.barplot(x=classes_names, y=sorted_counts, hue=classes_names, palette="viridis")

    # Annotate each bar with the percentage
    for i, (count, pct) in enumerate(zip(sorted_counts, percentages)):
        ax.text(i, count + max(sorted_counts) * 0.01, pct, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel("Number of Instances")
    ax.set_xlabel("Classes")
    ax.set_title("Dataset Class Distribution with Percentages")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join("./outputs", "class_distribution.png"))
    plt.close()