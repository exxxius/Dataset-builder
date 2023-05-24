import os
import shutil

# Define the directories
images_dir = 'L:\Programming\IMAGES\YOLOv8 Dataset/validation\images'
labels_dir = 'L:\Programming\IMAGES\YOLOv8 Dataset/validation\labels'
error_dir = 'L:\Programming\IMAGES\YOLOv8 Dataset/no-match'

# Get the list of files in the images and labels directories
image_files = os.listdir(images_dir)
label_files = os.listdir(labels_dir)

# Create the error directory if it doesn't exist
if not os.path.exists(error_dir):
    os.makedirs(error_dir)


# Define a function to move the file to the error directory
def move_to_error_dir(file_path):
    file_name = os.path.basename(file_path)
    error_path = os.path.join(error_dir, file_name)
    shutil.move(file_path, error_path)
    print(f"Moved {file_path} to {error_path}")


# Create sets of base file names for images and labels
image_base_names = set([os.path.splitext(file)[0] for file in image_files])
label_base_names = set([os.path.splitext(file)[0] for file in label_files])
print("image_base_names: ", image_base_names)
print("label_base_names: ", label_base_names)

# Find image files without a matching label file
for file in image_files:
    if os.path.splitext(file)[0] not in label_base_names:
        # Move the image file to the error directory
        image_path = os.path.join(images_dir, file)
        if os.path.exists(image_path):
            move_to_error_dir(image_path)

# Find label files without a matching image file
for file in label_files:
    if os.path.splitext(file)[0] not in image_base_names:
        # Move the label file to the error directory
        label_path = os.path.join(labels_dir, file)
        if os.path.exists(label_path):
            move_to_error_dir(label_path)


image_files = set([os.path.splitext(file)[0] for file in os.listdir(images_dir)])
label_files = set([os.path.splitext(file)[0] for file in os.listdir(labels_dir)])

print(f"Number of files in images directory: {len(image_files)}")
print(f"Number of files in labels directory: {len(label_files)}")

# Find files in labels directory that are not in images directory
extra_label_files = label_files - image_files
extra_label_files = [file + '.txt' for file in extra_label_files]
if len(extra_label_files) > 0:
    print(f"Extra label files: {extra_label_files}")

# Find files in images directory that are not in labels directory
extra_image_files = image_files - label_files
extra_image_files = [file + '.jpg' for file in extra_image_files] + [file + '.png' for file in extra_image_files]
if len(extra_image_files) > 0:
    print(f"Extra image files: {extra_image_files}")