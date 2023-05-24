import os

# Define the directory path and the character to replace the first non-empty character with
directory_path = "L:/Programming/IMAGES/re-sort/steelhead"
replacement_char = "5"

# Loop through all the files in the directory
for filename in os.listdir(directory_path):
    # Check if the file is a text file
    if filename.endswith(".txt"):
        # Open the file
        with open(os.path.join(directory_path, filename), "r") as file:
            # Read the contents of the file
            contents = file.readlines()

        # Replace the first non-empty character in each line with the replacement character
        for i in range(len(contents)):
            if contents[i].strip():
                contents[i] = replacement_char + contents[i][1:]

        # Write the modified contents back to the file
        with open(os.path.join(directory_path, filename), "w") as file:
            file.writelines(contents)
