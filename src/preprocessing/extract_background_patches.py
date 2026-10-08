import cv2
import csv
import json
import random
import collections

from pathlib import Path



# Configurations
PATCH_SIZE = 64
EXCLUSION_MARGIN = 16 # Extra area around ships that will be discarded. This helps avoid ship wakes and immediate surroundings 
PATCHES_PER_IMAGE = 15
MAX_TRIES_PER_IMAGE = 300 # Maximum random attemps per image
RANDOM_SEED = 42


# File/Folder Paths
PROJECT_ROOT = Path(__file__).resolve().parents[0]
ANNOTATION_FILE = (PROJECT_ROOT/"data"/"SSDD"/"annotations"/"train.json")
IMAGE_DIR = (PROJECT_ROOT/"data"/"SSDD"/"images"/"train")
OUTPUT_DIR = (PROJECT_ROOT/"data"/"SSDD"/"background_patches")
METADATA_FILE = OUTPUT_DIR/"metadata.csv"

# Random generator
rng = random.Random(RANDOM_SEED)


# Helper Functions
def expand_bbox(bbox, margin, image_width, image_height):
    """
    Expand a bounding box by `margin` pixels on every side.
    Returns:
        (x1, y1, x2, y2)
    """
    x, y, w, h = bbox

    x1 = max(0, int(x - margin))
    y1 = max(0, int(y - margin))

    x2 = min(image_width, int(x + w + margin))
    y2 = min(image_height, int(x + h + margin))

    return x1, y1, x2, y2


def patch_intersects_bbox(patch_x1, patch_y1, patch_x2, patch_y2, bbox):
    """
    Checks whether a candidate patch intersects an exclusion bounding box
    """
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox

    if patch_x2 <= bbox_x1:
        return False
    
    if patch_x1 <= bbox_x2:
        return False
    
    if patch_y2 <= bbox_y1:
        return False
    
    if patch_y1 <= bbox_y2:
        return False
    
    return True


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



# Preparing output directory and metadata
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

metadata = []

total_patches = 0
total_images = 0 



# Processing the images
for image_info in images:
    image_id = image_info["id"]
    filename = image_info["file_name"]

    image_path = IMAGE_DIR / filename

    image = cv2.imread(str(image_path),cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"[WARNING] Could not load: {image_path}")
        continue

    height, width = image.shape

    image_annotations = annotations_by_image.get(image_id,[])

    # Creating exclusion regions
    exclusion_boxes = []

    for annotation in image_annotations:
        bbox = expand_bbox(annotation["bbox"], EXCLUSION_MARGIN, width, height)
        exclusion_boxes.append(bbox)


    # Generating patches
    patches_for_image = 0
    attempts = 0

    while (
        patches_for_image < PATCHES_PER_IMAGE
        and attempts < MAX_TRIES_PER_IMAGE
    ):

        attempts += 1

        # Candidate top-left corner
        x1 = rng.randint(0, width - PATCH_SIZE)
        y1 = rng.randint(0, height - PATCH_SIZE)

        x2 = x1 + PATCH_SIZE
        y2 = y1 + PATCH_SIZE

        # Checking overlap with every ship exclusion zone
        rejected = False

        for exclusion_box in exclusion_boxes:
            if patch_intersects_bbox(x1, y1, x2, y2, exclusion_box):
                rejected = True
                break

        if rejected:
            continue

        # Extracting patch
        patch = image[y1:y2, x1:x2]

        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            continue

        
        # Saving the patch
        patch_name = (
            f"{Path(filename).stem}"
            f"_x{x1}_y{y1}.png"
        )

        patch_path = OUTPUT_DIR / patch_name

        cv2.imwrite(str(patch_path), patch)

        metadata.append({
            "patch_filename": patch_name,
            "source_image": filename,
            "source_image_id": image_id,
            "x": x1,
            "y": y1,
            "width": PATCH_SIZE,
            "height": PATCH_SIZE,
        })

        patches_for_image += 1
        total_patches += 1

    total_images += 1

    print(
        f"{filename:15s} | "
        f"ships={len(image_annotations):2d} | "
        f"patches={patches_for_image:2d}"
    )