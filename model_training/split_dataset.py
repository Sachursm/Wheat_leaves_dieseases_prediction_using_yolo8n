import os
import random
import shutil

# adjust if needed
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

BASE_DIR = r"C:\\Users\\sachu\\Desktop\\wheat_dataset\\wheat_dataset"  # <-- change this
images_dir = os.path.join(BASE_DIR, "images")
labels_dir = os.path.join(BASE_DIR, "labels")

train_images_dir = os.path.join(images_dir, "train")
val_images_dir = os.path.join(images_dir, "val")
train_labels_dir = os.path.join(labels_dir, "train")
val_labels_dir = os.path.join(labels_dir, "val")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# collect image files in images/ root (not inside train/ or val/)
all_images = [
    f for f in os.listdir(images_dir)
    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
]

print("Total images found:", len(all_images))

random.shuffle(all_images)

train_ratio = 0.8
train_count = int(len(all_images) * train_ratio)
train_files = all_images[:train_count]
val_files = all_images[train_count:]

def move_pair(filename, target_images_dir, target_labels_dir):
    image_src = os.path.join(images_dir, filename)
    image_dst = os.path.join(target_images_dir, filename)

    base, _ = os.path.splitext(filename)
    label_name = base + ".txt"
    label_src = os.path.join(labels_dir, label_name)
    label_dst = os.path.join(target_labels_dir, label_name)

    if not os.path.exists(label_src):
        print("WARNING: label not found for", filename)
    else:
        shutil.move(label_src, label_dst)

    shutil.move(image_src, image_dst)

for f in train_files:
    move_pair(f, train_images_dir, train_labels_dir)

for f in val_files:
    move_pair(f, val_images_dir, val_labels_dir)

print("Done!")
print("Train images:", len(train_files))
print("Val images:", len(val_files))
