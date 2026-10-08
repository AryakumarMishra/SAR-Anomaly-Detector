import json
from pathlib import Path
from collections import Counter

# File/Folder Paths
PROJECT_ROOT = Path(__file__).resolve().parents[0]
ANNOTATION_FILE = (PROJECT_ROOT/"data"/"SSDD"/"annotations"/"train.json")


# Loading the Annotations
with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

print(f"Number of Images: {len(images)}")
print(f"Number of Annotations: {len(annotations)}")
print(f"Number of Categories: {len(categories)}")


# Image dimensions
widths = [img["width"] for img in images]
heights = [img["height"] for img in images]

print("\nImage dimensions:")

print(f"Width  - min : {min(widths)}")
print(f"Width  - max : {max(widths)}")

print(f"Height - min : {min(heights)}")
print(f"Height - max : {max(heights)}")


# Counting the Ships per Image
ships_per_image = Counter(
    annotation["image_id"] for annotation in annotations
)

ship_counts = list(ships_per_image.values())

print(f"Minimum ships/image: {min(ship_counts)}")
print(f"Maximum ships/image: {max(ship_counts)}")
print(f"Average ships/image: {sum(ship_counts)/len(ship_counts):.2f}")


# Images without ships
img_ids = {img["id"] for img in images}
annotated_img_ids = set(ships_per_image.keys())

images_without_ship = annotated_img_ids - img_ids
print(f"Images without Annotations: {len(images_without_ship)}")


# Bounding Box Sizes
bbox_widths = []
bbox_heights = []
bbox_areas = []

for annotation in annotations:
    _, _, width, height = annotation["bbox"]

    bbox_widths.append(width)
    bbox_heights.append(height)
    bbox_areas.append(annotation["area"])

print("\nBounding boxes:")
print(f"Average bbox width  : {sum(bbox_widths) / len(bbox_widths):.2f}")
print(f"Average bbox height : {sum(bbox_heights) / len(bbox_heights):.2f}")
print(f"Average bbox area   : {sum(bbox_areas) / len(bbox_areas):.2f}")


print("\nCategories:")
for category in categories:
    print(
        f"ID: {category['id']} | "
        f"Name: {category['name']}"
    )