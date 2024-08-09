# Face Detection with InsightFace
This project provides a tool for detecting faces in images using the InsightFace library with the RetinaFace model. The tool processes images from a specified input folder, detects faces, and saves the results to a specified output folder with bounding boxes drawn around detected faces.

## Features
Face detection using InsightFace's RetinaFace model.
Batch processing of images in a specified folder.
Automatic creation of output folders.
Support for common image formats: PNG, JPG, JPEG, BMP, and TIFF.
## Prerequisites
1. Python 3.6 or higher

2. OpenCV

3. InsightFace
## Installation
Clone the repository:

```bash
git clone https://github.com/MahnazRazavi/FaceDetection.git
cd face-detection
```
Install required packages:

```bash
pip install opencv-python insightface
```
## Usage
1. Prepare the input folder: Place the images you want to process in a folder. For example, create a folder named input and add your images to this folder.

2. Run the script: Execute the script with the input and output folder paths.

```
python face_detection.py
```
The script will read images from the input folder, detect faces, and save the results to the output folder.

## Example
1. Place your images in the input folder.
2. Run the script:
```
python face_detection.py
```
3. The processed images with detected faces will be saved in the output folder.
## Code Overview
FaceDetection Class:

init: Initializes the face detection model.

detect_faces: Detects faces in a given image and saves the result with bounding boxes drawn around detected faces.

process_images: Processes all images in the specified input folder.
