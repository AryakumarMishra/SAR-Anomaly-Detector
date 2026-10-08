import json
import cv2
import random
import collections

from pathlib import Path
import matplotlib.pyplot as plt



# File/Folder Paths
PROJECT_ROOT = Path(__file__).resolve().parents[0]
ANNOTATION_FILE = (PROJECT_ROOT/"data"/"SSDD"/"annotations"/"train.json")
IMAGE_DIR = (PROJECT_ROOT/"data"/"SSDD"/"images"/"train")


# Loading the annotations
with open(ANNOTATION_FILE, "r") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]


# Mapping the images with annotations

annotations_by_image = collections.defaultdict(list)

for annotation in annotations:
    image_id = annotation["image_id"]
    annotations_by_image[image_id].append(annotation)


# Selecting random images   
random.seed(42)
selected_images = random.sample(images, 6)


# Plotting the images
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, image_info in zip(axes, selected_images):
    image_id = image_info["id"]
    filename = image_info["file_name"]
    image_path = IMAGE_DIR / filename

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not load: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ax.imshow(image)

    image_annotations = annotations_by_image.get(image_id, [])

    for annotation in image_annotations:
        x, y, width, height = annotation["bbox"]
        rectangle = plt.Rectangle((x, y), width, height, fill=False, linewidth=1.5)
        ax.add_patch(rectangle)

    ax.set_title(f"{filename} | Ships: {len(image_annotations)}")

    ax.axis("off")


plt.tight_layout()

output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "annotation_examples.png"

plt.savefig(output_path, dpi=150, bbox_inches="tight")

print(f"Saved visualization to: {output_path}")

plt.show()