# Dataset-builder

The Dataset-builder is a Python-based project designed to preprocess images for the creation of the Salmonidae dataset V1.0. This repository contains the first version of the Dataset-builder, including an SSD-Inception TFLite model.

## Repository Contents

- `assets`: Holds necessary resources for the project.
- `class_index_changer.py`: Python script for changing class indices in label files.
- `file_copier.py`: Python script for copying files.
- `file_nomatch_finder.py`: Python script for finding files without matches.
- `main.py`: The main Python script of the project. 

## About the Project

The Dataset-builder project is a Python-based solution for preprocessing images, specifically designed for the creation of the Salmonidae dataset V1.0. The project includes a MobileNet SSD-Inception TFLite model for fish object detection tasks used in this project to filter and sort out images that contain fish objects from the total images downloaded for building the dataset.

## Languages

- Python: 100%
