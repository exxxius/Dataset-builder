# import os
# import tensorflow as tf
#
# # Load the pre-trained TFLite model
# interpreter = tf.lite.Interpreter(model_path='L:/Programming/Dataset-builder/assets/full_salmon_ssd_inception.tflite')
# interpreter.allocate_tensors()
#
# # Define the input and output tensors
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Define the input size of the model
# input_shape = input_details[0]['shape'][1:3]
#
# # Define the directories for input and output files
# input_dir = 'L:/Programming/IMAGES/salmonids'
# output_dir = 'L:/Programming/IMAGES/hasfish'
#
# # Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Loop through all files in the input directory and its subdirectories
# for root, dirs, files in os.walk(input_dir):
#     for file in files:
#         # Check if the file is an image
#         if file.endswith((".jpg", ".jpeg", ".png")):
#             print("Processing file:", os.path.join(root, file))
#
#             # Load the image and resize it to the input shape of the model
#             img = tf.io.read_file(os.path.join(root, file))
#             img = tf.image.decode_jpeg(img, channels=3)
#             img = tf.image.resize(img, input_shape)
#             img = tf.cast(img, tf.float32) / 255.0
#             img = tf.expand_dims(img, axis=0)
#
#             # Run the model on the input image
#             interpreter.set_tensor(input_details[0]['index'], img)
#             interpreter.invoke()
#             output_data = interpreter.get_tensor(output_details[0]['index'])
#             predicted_class = tf.argmax(output_data, axis=-1).numpy()[0]
#
#             # Check if the model predicts fish in the image
#             if predicted_class[0] == 0:
#                 print("Fish detected in image!")
#
#                 # Copy the file to the output directory
#                 output_path = os.path.join(output_dir, file)
#                 tf.io.write_file(output_path, tf.io.read_file(os.path.join(root, file)))
#                 print("File copied to:", output_path)
#             else:
#                 print("No fish detected in image.")
#         else:
#             print("Skipping file:", os.path.join(root, file), "- Not an image.")


# This program is used to detect fish in images using a pre-trained TFLite model. It is used to filter out images that
# do not contain fish from a dataset of images. The program loops through all files in the input directory and its
# subdirectories and checks if the model predicts fish in the image. If the model predicts fish, the image is copied to
# the output directory. It's beneficial to use this program to filter out images that do not contain fish from a large
# dataset of images, as it will speed up the training process of the model.

# import os
# import bbox as bbox
# import cv2
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
#
# interpreter = tf.lite.Interpreter(model_path='L:/Programming/Dataset-builder/assets/full_salmon_ssd_inception.tflite')
# interpreter.allocate_tensors()
#
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# input_shape = input_details[0]['shape'][1:3]
#
# input_dir = 'L:/Programming/IMAGES/test'
# output_dir = 'L:/Programming/IMAGES/hasfish'
#
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# for root, dirs, files in os.walk(input_dir):
#     for file in files:
#         if file.endswith((".jpg", ".jpeg", ".png", ".gif")):
#             print("Processing file:", os.path.join(root, file))
#
#             try:
#                 img = tf.io.read_file(os.path.join(root, file))
#                 img = tf.image.decode_image(img, channels=3)
#                 img = tf.image.resize(img, input_shape)
#                 img = tf.cast(img, tf.float32) / 255.0
#                 img = tf.expand_dims(img, axis=0)
#
#                 interpreter.set_tensor(input_details[0]['index'], img)
#                 interpreter.invoke()
#                 boxes = interpreter.get_tensor(output_details[0]['index'])
#                 classes = interpreter.get_tensor(output_details[1]['index'])
#                 scores = interpreter.get_tensor(output_details[2]['index'])
#                 num_detections = interpreter.get_tensor(output_details[3]['index'])
#
#                 if any(score > 0.5 for score in scores[0]):
#                     print("Fish detected in image!")
#
#                     img = np.asarray(Image.open(os.path.join(root, file)))
#                     img = np.array(img, dtype=np.uint8)
#                     for i in range(int(num_detections[0])):
#                         if scores[0][i] > 0.5:
#                             ymin, xmin, ymax, xmax = boxes[0][i]
#                             im_height, im_width, _ = img.shape
#                             left = int(xmin * im_width)
#                             top = int(ymin * im_height)
#                             right = int(xmax * im_width)
#                             bottom = int(ymax * im_height)
#                             #label = classes[0][i]
#                             class_names = {
#                                 0: 'Chinook Salmon',
#                                 1: 'Coho Salmon',
#                                 2: 'Sockeye Salmon',
#                                 3: 'Pink Salmon',
#                                 4: 'Chum Salmon',
#                                 5: 'Lingcod',
#                                 6: 'Pacific Halibut',
#                                 7: 'Steelhead'
#                             }
#                             label = class_names[classes[0][i]]
#                             cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
#                             cv2.putText(img, str(label), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                             # Save the processed image to the output directory
#                             output_filename = os.path.splitext(file)[0] + '_hasfish' + os.path.splitext(file)[1]
#                             output_path = os.path.join(output_dir, output_filename)
#                             cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#
#                     plt.imshow(img)
#                     plt.show()
#
#                     # Complete the output_path variable with the desired output file path
#                     output_path = os.path
#             except Exception as e:
#                 print(f"Error processing {os.path.join(root, file)}: {e}")
#                 continue

########################################################################################################################
# this variation copies each classified image into a folder named after the class it was classified as. GPU
# import os
# import bbox as bbox
# import cv2
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX
# from pathlib import Path
#
#
# # This function generates a unique filename for the output file
# def generate_unique_filename(directory, filename):
#     base, ext = os.path.splitext(filename)
#     counter = 1
#     new_filename = f"{base}_{counter}{ext}"
#     while Path(directory, new_filename).exists():
#         counter += 1
#         new_filename = f"{base}_{counter}{ext}"
#     return new_filename
#
#
# # Cpu interpreter
# interpreter = tf.lite.Interpreter(model_path='L:/Programming/Dataset-builder/assets/full_salmon_ssd_inception.tflite')
# interpreter.allocate_tensors()
#
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# input_shape = input_details[0]['shape'][1:3]
#
# input_dir = 'L:/Programming/IMAGES/salmonids-main'
# output_dir = 'L:/Programming/IMAGES/hasfish'
#
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# for root, dirs, files in os.walk(input_dir):
#     for file in files:
#         if file.endswith((".jpg", ".jpeg", ".png", ".gif")):
#             print("Processing file:", os.path.join(root, file))
#
#             try:
#                 img = tf.io.read_file(os.path.join(root, file))
#                 img = tf.image.decode_image(img, channels=3)
#                 img = tf.image.resize(img, input_shape)
#                 img = tf.cast(img, tf.float32) / 255.0
#                 img = tf.expand_dims(img, axis=0)
#
#                 interpreter.set_tensor(input_details[0]['index'], img)
#                 interpreter.invoke()
#                 boxes = interpreter.get_tensor(output_details[0]['index'])
#                 classes = interpreter.get_tensor(output_details[1]['index'])
#                 scores = interpreter.get_tensor(output_details[2]['index'])
#                 num_detections = interpreter.get_tensor(output_details[3]['index'])
#
#                 if any(score > 0.9 for score in scores[0]):
#                     print("Fish detected in image!")
#
#                     img = np.asarray(Image.open(os.path.join(root, file)))
#                     img = np.array(img, dtype=np.uint8)
#                     for i in range(int(num_detections[0])):
#                         if scores[0][i] > 0.9:
#                             ymin, xmin, ymax, xmax = boxes[0][i]
#                             im_height, im_width, _ = img.shape
#                             left = int(xmin * im_width)
#                             top = int(ymin * im_height)
#                             right = int(xmax * im_width)
#                             bottom = int(ymax * im_height)
#                             class_names = {
#                                 0: 'Chinook Salmon',
#                                 1: 'Coho Salmon',
#                                 2: 'Sockeye Salmon',
#                                 3: 'Pink Salmon',
#                                 4: 'Chum Salmon',
#                                 5: 'Lingcod',
#                                 6: 'Pacific Halibut',
#                                 7: 'Steelhead'
#                             }
#                             label = class_names[classes[0][i]]
#                             cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
#                             cv2.putText(img, str(label), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#                             # Create and save the processed image in the class-specific subdirectory
#                             class_dir = os.path.join(output_dir, label.replace(' ', '-'))
#                             if not os.path.exists(class_dir):
#                                 os.makedirs(class_dir)
#                             output_filename = os.path.splitext(file)[0] + '_hasfish' + os.path.splitext(file)[1]
#                             unique_output_filename = generate_unique_filename(class_dir, output_filename)
#                             output_path = os.path.join(class_dir, unique_output_filename)
#                             cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#
#                     plt.imshow(img)
#                     plt.show()
#
#             except Exception as e:
#                 print(f"Error processing {os.path.join(root, file)}: {e}")
#                 continue

########################################################################################################################
# this variation copies each classified image into a folder named after the class it was classified as and also
# creates an annotation file for each image in the class-specific folder that contains the bounding box coordinates for
# YOLOv8 model training.

import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path


# This function generates a unique filename for the output file
def generate_unique_filename(directory, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{base}_{counter}{ext}"
    while Path(directory, new_filename).exists():
        counter += 1
        new_filename = f"{base}_{counter}{ext}"
    return new_filename


# Cpu interpreter
interpreter = tf.lite.Interpreter(model_path='L:/Programming/Dataset-builder/assets/full_salmon_ssd_inception.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]

input_dir = 'L:/Programming/IMAGES/final_imgsrc'
output_dir = 'L:/Programming/IMAGES/yolo_annotations'
image_output_dir = 'L:/Programming/IMAGES/hasfish'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

# Add the first class_names dictionary for the SSD model classifications
class_names = {
    0: 'Chinook Salmon',
    1: 'Coho Salmon',
    2: 'Sockeye Salmon',
    3: 'Pink Salmon',
    4: 'Chum Salmon',
    5: 'Lingcod',
    6: 'Pacific Halibut',
    7: 'Steelhead'
}

# Add the second class_names dictionary for the YOLOv8 model annotations
new_class_names = {
    0: 'Chinook Salmon',
    1: 'Coho Salmon',
    2: 'Sockeye Salmon',
    3: 'Pink Salmon',
    4: 'Chum Salmon',
    5: 'Steelhead'
}

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png", ".gif")):
            print("Processing file:", os.path.join(root, file))

            try:
                img = tf.io.read_file(os.path.join(root, file))
                img = tf.image.decode_image(img, channels=3)
                img = tf.image.resize(img, input_shape)
                img = tf.cast(img, tf.float32) / 255.0
                img = tf.expand_dims(img, axis=0)

                interpreter.set_tensor(input_details[0]['index'], img)
                interpreter.invoke()
                boxes = interpreter.get_tensor(output_details[0]['index'])
                classes = interpreter.get_tensor(output_details[1]['index'])
                scores = interpreter.get_tensor(output_details[2]['index'])
                num_detections = interpreter.get_tensor(output_details[3]['index'])

                yolo_annotations = []

                if any(score > 0.9 for score in scores[0]):
                    img = np.asarray(Image.open(os.path.join(root, file)))
                    im_height, im_width, _ = img.shape

                    for i in range(int(num_detections[0])):
                        if scores[0][i] > 0.9:
                            ymin, xmin, ymax, xmax = boxes[0][i]
                            left = int(xmin * im_width)
                            top = int(ymin * im_height)
                            right = int(xmax * im_width)
                            bottom = int(ymax * im_height)

                            # Use the original class_names dictionary for classification
                            label = class_names[classes[0][i]]

                            # Check if the detected class is in the new_class_names dictionary
                            new_class_id = None
                            for key, value in new_class_names.items():
                                if value == label:
                                    new_class_id = key
                                    break

                            # If the detected class is not in the new_class_names dictionary, skip it
                            if new_class_id is None:
                                continue

                            #cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                            #cv2.putText(img, str(label), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            class_id = int(classes[0][i])
                            x_center = (left + right) / 2
                            y_center = (top + bottom) / 2
                            width = right - left
                            height = bottom - top

                            # Normalize coordinates
                            x_center /= im_width
                            y_center /= im_height
                            width /= im_width
                            height /= im_height

                            # Create YOLOv8-compatible annotation using original class_names dictionary
                            # yolo_annotation = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

                            # Use new_class_names dictionary when saving annotations
                            yolo_annotation = f"{new_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                            yolo_annotations.append(yolo_annotation)

                            # Create and save the processed image in the class-specific subdirectory
                            class_dir = os.path.join(image_output_dir, label.replace(' ', '-'))
                            if not os.path.exists(class_dir):
                                os.makedirs(class_dir)
                            # output_filename = os.path.splitext(file)[0] + '_hasfish' + os.path.splitext(file)[1]
                            output_filename = os.path.splitext(file)[0] + os.path.splitext(file)[1]
                            unique_output_filename = generate_unique_filename(class_dir, output_filename)

                            output_path = os.path.join(class_dir, output_filename)
                            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # Save YOLOv8-compatible annotations
                if yolo_annotations:
                    annotation_filename = os.path.splitext(file)[0] + '.txt'
                    annotation_output_path = os.path.join(class_dir, annotation_filename)
                    with open(annotation_output_path, 'w') as annotation_file:
                        annotation_file.write('\n'.join(yolo_annotations))

            except Exception as e:
                print(f"Error processing {os.path.join(root, file)}: {e}")
                continue
