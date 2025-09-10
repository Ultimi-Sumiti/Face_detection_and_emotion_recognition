# Face Detection and Emotion Recognition

This project implements a computer vision system capable of detecting human faces in static images and recognizing their emotional expressions. The system combines a classical approach for face detection with a pre-trained convolutional neural network (CNN) for emotion classification.

## 🎯 Project Goal

The main objective is to develop a comprehensive facial analysis system that integrates two fundamental tasks:
1.  **Face Detection**: To identify and isolate facial regions in an image.
2.  **Emotion Recognition**: To classify the emotional expression of each detected face.

The system is designed to analyze static images, providing the original image with detected faces circled and annotated with the predicted emotion as output.

## ⚙️ System Architecture

The project is divided into two main modules that work sequentially.

### 1. Face Detection (C++ and OpenCV)

The first component uses the **Viola-Jones algorithm**, a classic method known for its efficiency and accuracy. This part is implemented in **C++** using the **OpenCV** library and its cascade classifier framework to locate faces within an image.

### 2. Emotion Recognition (Python and CNN)

Once a face is detected, its region is passed to the second module. This component employs a **Convolutional Neural Network (CNN)** pre-trained on the FER-2013 dataset. The module, implemented in **Python**, classifies the facial expression into one of seven predefined categories:
* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

## 📊 Datasets Used

### Training Dataset (FER-2013)

The CNN for emotion recognition was trained on the **FER-2013 dataset**. This dataset consists of 35,887 grayscale images of 48x48 pixels, each labeled with one of the seven emotions.

### Test Dataset

For the overall system evaluation, a separate test dataset containing 46 images was used. Each image is annotated with bounding boxes for all visible faces and the corresponding emotion labels.

## 📈 Performance Metrics

The system's evaluation was conducted at multiple levels to measure the effectiveness of each component and the complete workflow.

* **Face Detection Evaluation**:
    * **Intersection over Union (IoU)**: To measure the accuracy of the predicted bounding boxes against the ground truth.
    * **Precision and Recall**: To evaluate the trade-off between correct, missed, and false detections, based on an IoU threshold (commonly 0.5).

* **Emotion Recognition Evaluation**:
    * **Classification Accuracy**: To measure the proportion of correctly predicted emotions across all detected faces.
    * **Confusion Matrix**: To analyze the detailed performance of the CNN for each emotion category.

* **System-Level Evaluation**:
    * The primary metric is the **percentage of detected faces with correctly classified emotions**, to assess the end-to-end performance of the system.

## 💡 Potential Applications

This system can be applied in various domains:
* **Human-Computer Interaction (HCI)**: To create more empathetic and responsive systems capable of interpreting users' emotional states.
* **Behavioral Analysis**: To provide insights in contexts such as education, retail, and security by monitoring people's emotional responses.
* **Photo Management**: To automatically organize and tag images based on facial expressions, simplifying searches in large photo archives.

## ⚠️ Limitations

The system has some important limitations:
* It works **exclusively on static images** and does not support real-time processing.
* Emotion recognition is entirely dependent on the success of face detection. If the Viola-Jones algorithm does not detect a face, it **will not be analyzed** for emotion.

## 🚀 How to Run the Project

*For detailed instructions on installing dependencies and running the code, please refer to the specific documentation within the repository.*

**Prerequisites (example):**
* C++ Compiler (GCC, Clang, etc.)
* OpenCV
* Python 3.x
* TensorFlow/Keras
* NumPy

**Execution (example):**
```bash
# Compile and run the face detection module
g++ -o face_detection main.cpp `pkg-config --cflags --libs opencv4`
./face_detection input.jpg

# Run the emotion recognition script
python emotion_recognition.py --image output_faces/face_1.jpg
