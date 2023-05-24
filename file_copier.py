import os
import shutil

# Define the source and destination directories
src_dir = 'L:/Programming/IMAGES/salmonids-main'
dst_dir = 'L:/Programming\IMAGES/final_imgsrc'


# Define a function to recursively traverse the directory and rename files
def rename_files(src_dir, dst_dir):
    # Initialize a counter for the file names
    i = 1

    # Traverse the source directory and its subdirectories
    for root, dirs, files in os.walk(src_dir):
        # For each file in the current directory
        for file in files:
            # Get the full path of the current file
            src_path = os.path.join(root, file)

            # Generate a new unique file name
            dst_file = str(i) + os.path.splitext(file)[1]
            dst_path = os.path.join(dst_dir, dst_file)

            # If the file already exists in the destination directory, generate a new unique file name
            while os.path.exists(dst_path):
                i += 1
                dst_file = str(i) + os.path.splitext(file)[1]
                dst_path = os.path.join(dst_dir, dst_file)

            try:
                # Copy the file to the destination directory with the new name
                shutil.copy(src_path, dst_path)
                print(f"Copying {src_path} to {dst_path}")
            except Exception as e:
                print(f"Error copying {src_path} to {dst_path}: {e}")

            # Increment the counter for the file names
            i += 1

# Call the function to rename and copy the files
rename_files(src_dir, dst_dir)