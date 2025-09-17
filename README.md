
<p align="center">
   <img width="650" height="350" alt="icon_3" src="https://github.com/user-attachments/assets/8a3e7694-b632-4506-9343-f45166ebaf80" />
</p>

# Face Detection and Emotion Recognition

This project implements a computer vision system capable of detecting human faces in static images and recognizing their emotional expressions. The system combines a classical approach for face detection thath is the Viola Jones algorithm, with a convolutional neural network (EfficientNetV2B0) for emotion classification.

## Table of Contents

- [Project Goal](#project-goal)
- [System Architecture](#system-architecture)
- [Dataset Used](#dataset-used)
- [Extra Features: Webcam and Video processing](#extra-features-webcam-and-video-processing)
- [Project Structure](#project-structure)
- [How to install and run the project](#how-to-install-and-run-the-project)

## Project Goal

The main objective is to develop a comprehensive facial analysis system that integrates two fundamental tasks:
1.  **Face Detection**: To identify and isolate facial regions in an image.
2.  **Emotion Recognition**: To classify the emotional expression of each detected face.

The system is designed to analyze static images, providing the original image with detected faces circled and annotated with the predicted emotion as output.

## System Architecture

The project is divided into two main modules that work sequentially.

### 1. Face Detection (C++ with OpenCV)

The first component uses the **Viola-Jones algorithm**, a classic method known for its efficiency and accuracy. This part is implemented in **C++** using the **OpenCV** library and its cascade classifier framework to locate faces within an image.

### 2. Emotion Recognition (Python with Keras)

Once a face is detected, its region is passed to the second module. This component employs a **Convolutional Neural Network (CNN)** pre-trained on the FER-2013 dataset. The module, implemented in **Python**, classifies the facial expression into one of seven predefined categories:
* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## Dataset Used

The CNN for emotion recognition was trained on the **FER-2013 dataset**. We applied a basic transfer learning and fine tuning technique to a pre-trained EfficientNetV2B0 architecture.

## Extra Features: Webcam and Video processing

As a bonus the program can track faces in a video or using the webcam. In the following there are some examples.

https://github.com/user-attachments/assets/59a87002-e1ba-4f44-a51a-571a10870527

## Project Structure

``` bash
Face_detection_and_emotion_recognition/
├── data/
│   ├── dataset_classification/    # datasets for classification
│   ├── dataset_detection/         # datasets for detection
│   ├── haarcascades/              # Haar Cascade xml files
│   └── trained_models/            # Our fine tuned model
├── include/                       # C++ header files
├── lib/                           # C++ definitions/implementation of the headers
├── log/                           # log files
├── output/                        # generated outputs like detections and metrics
├── src/                           
│   ├── cpp/                       # C++ source code
│   └── python/                    # Python source code
├── tmp/                           # temporary files
├── .gitignore                     
├── CMakeLists.txt                  
└── README.md                      # main project documentation
```

## How to install and run the project

*For detailed instructions on installing dependencies and running the code, please refer to the specific documentation within the repository.*

Firstly you need to install the following dependecies if are not already installed in your machine.
**Dependencies:**

* C++ Compiler (GCC, Clang, etc.)
* OpenCV
* Python 3.x
* Keras
* NumPy
* Matplotlib
* Sklearn
* Seaborn

After that for running the code you have to make the following commands:

---
**Build**

```bash
# Create the build directory
mkdir build
```
```bash
# Compile the entire pipeline
cmake ..
make
```
---
**Run**

```bash
# Run the pipeline on images specifying the labels for the calculation of the metrics
./out --imgsdir <path> --labelsdir <path> 
```
Or if you don't need to output the metrics you can also execute
```bash
# Run the pipeline specifying on images
./out --imgsdir <path> 
```
If you want to process the pipeline on the video of one camera device connected with your machine you have to execute:
```bash
# Run the pipeline with one camera of your computer
./out --webcam 
```
Otherwise if you want to do the same but on a video, you must to pass the path of it with the folowing command:
```bash
# Run the pipeline on a video specifying the path
./out --video <path> 
```

Help message:
```bash
Usage:
 ./out --imgsdir <path> --labelsdir <path> --video <path> --webcam
  Where:
    --imgsdir is the input image directory path
    --labelsdir is the label directory path (OPTIONAL)
    --video is the path of a video
    --webcam to detect faces using webcam
```
    
