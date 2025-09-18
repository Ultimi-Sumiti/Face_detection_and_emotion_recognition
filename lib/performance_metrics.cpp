// Mattia Scantamburlo

#include "../include/performance_metrics.h"

#include "../include/utils.h"

#include <iostream>

#include <fstream>

#include <filesystem>

#include <numeric>

#define BLUE "\033[34m"
#define RESET "\033[0m"

namespace fs = std::filesystem;

// -------------- MEMBER FUNCTIONS --------------
// This member function compute the IOUs of the detected faces.
std::vector < float > PerformanceMetrics::get_label_IOUs(const std::vector < cv::Rect > & detection,
    const std::vector < cv::Rect > & labels, std::vector < int > & ordering) {
    float current_IoU;
    std::vector < float > IOUs(labels.size(), 0.0f);
    ordering = std::vector < int > (detection.size());
    for (int i = 0; i < labels.size(); i++) {
        for (int j = 0; j < detection.size(); j++) {
            current_IoU = compute_IOU(labels[i], detection[j]);
            if (current_IoU > IOUs[i]) {
                IOUs[i] = current_IoU;
                ordering[j] = i;
            }
        }
    }
    return IOUs;
}

// Function to write in a file and computing in terminal the metrics for the scenepath.
void PerformanceMetrics::print_metrics(bool verbose) {
    // Print in a file and in the terminal
    std::ofstream outfile(metrics_file_path, std::ios::app);
    std::vector < float > all_IOUs;
    float precision = 0.0f;
    float recall = 0.0f;
    float avg_IOU = 0.0f;
    float class_accuracy = 0.0f;
    float system_accuracy = 0.0f;
    int count_detected = 0;
    std::vector < int > ordering;
    if (outfile.is_open()) {
        outfile << "IOUs of labels: \n";
        for (int j = 0; j < face_labels.size(); j++) {
            outfile << "\nIOUs of image " << j << ": \n";

            count_detected += detected_faces[j].size();
            std::vector < float > IOUs = get_label_IOUs(detected_faces[j], face_labels[j], ordering);
            orderings.push_back(ordering);
            all_IOUs.insert(all_IOUs.end(), IOUs.begin(), IOUs.end());

            for (int i = 0; i < face_labels[j].size(); i++) {
                outfile << "    IOU of box number " << i << " is: " << IOUs[i] << std::endl;
            }
        }
        outfile << std::endl;
        avg_IOU = compute_MIOU(all_IOUs);
        precision = compute_precision(all_IOUs, IOU_THRESHOLD, count_detected);
        recall = compute_recall(all_IOUs, IOU_THRESHOLD);
        class_accuracy = compute_emotions_accuracy(detected_emotions, emotion_labels, orderings);
        system_accuracy = compute_system_accuracy(detected_emotions, emotion_labels, orderings);
        outfile << "The precision over analized images is: " << precision << std::endl;
        outfile << "The recall over analized images is: " << recall << std::endl;
        outfile << "The avarage IOUs is: " << avg_IOU << std::endl;
        outfile << "The accuracy of the emotion recognition process is: " << class_accuracy << std::endl;
        outfile << "The system accuracy is: " << system_accuracy << std::endl;
        outfile.close();
    } else {
        std::cerr << "Impossibile to open the file\n";
    }
    if (verbose) {
        std::cout << std::endl << BLUE << "The avarage IOUs is: " << RESET << avg_IOU << std::endl;
        std::cout << BLUE << "The precision over analized images is: " << RESET << precision << std::endl;
        std::cout << BLUE << "The recall over analized images is: " << RESET << recall << std::endl;
        std::cout << BLUE << "The accuracy of the emotion recognition process is: " << RESET << class_accuracy << std::endl;
        std::cout << BLUE << "The system accuracy is: " << RESET << system_accuracy << std::endl;
    }
}

// This function clean the metrics file for a new analysis.
void PerformanceMetrics::clean_metrics() {
    try {
        // The remove function returns true if a file was deleted, false otherwise
        fs::remove(metrics_file_path);
    } catch (const fs::filesystem_error & e) {
        // This catch block handles errors like permission issues
        std::cerr << "Error deleting file: " << e.what() << std::endl;
        return;
    }
}

// This function adds a new image data to the performance metrics class.
void PerformanceMetrics::add_image_detections(const std::vector < cv::Rect > & detection,
    const std::vector < cv::Rect > & labels,
        const std::vector < int > & emotions,
            const std::vector < int > & emotion_labs) {
    face_labels.push_back(labels);
    detected_faces.push_back(detection);
    emotion_labels.push_back(emotion_labs);
    detected_emotions.push_back(emotions);
}

// -------------- HELPER FUNCTIONS --------------

// Function to compute the IOU (intersection over union) between 2 given boxes.
float compute_IOU(const cv::Rect & box1,
    const cv::Rect & box2) {
    // Define the variable to store the areas of intersection, union and the respective IoU.
    double areas_int;
    double areas_union;

    // Compute intersection union of boxes. 
    cv::Rect intersect = box1 & box2;
    areas_int = intersect.area();
    areas_union = box1.area() + box2.area() - areas_int;

    // Compute and return the IoU.
    float IoU = areas_int / areas_union;
    return IoU;
}

// Function to compute the mean over IOUs (MIOU).
float compute_MIOU(const std::vector < float > & IOUs) {

    // Handle the edge case of an empty vector to prevent division by zero.
    if (IOUs.empty()) {
        return 0.0f;
    }

    int detection_count = 0;
    for (int i = 0; i < IOUs.size(); i++) {
        if (IOUs[i] > 0) {
            detection_count++;
        }
    }

    // Calculate the sum of all elements in the vector.
    float total_iou = std::accumulate(IOUs.begin(), IOUs.end(), 0.0f);

    // Divide the sum by the number of elements to get the mean.
    return total_iou / detection_count;
}

// Function to compute the precision.
float compute_precision(const std::vector < float > & all_IOUs, float threshold, int all_detections) {
    float correct_detections = 0;
    for (int i = 0; i < all_IOUs.size(); i++) {
        if (all_IOUs[i] > threshold) {
            correct_detections++;
        }
    }
    return (correct_detections / all_detections);
}

// Function to compute the recall.
float compute_recall(const std::vector < float > & all_IOUs, float threshold) {
    float correct_detections = 0;
    float all_labels = all_IOUs.size();
    for (int i = 0; i < all_labels; i++) {
        if (all_IOUs[i] > threshold) {
            correct_detections++;
        }
    }
    return (correct_detections / all_labels);
}

// Function to compute emotions labels.
std::vector < int > get_label_emotion(const std::string & filename) {
    // Get the labels for the given filename.
    std::vector < std::vector < float >> labels = parse_labels(filename);
    std::vector < int > emotions;
    for (int i = 0; i < labels.size(); i++) {
        emotions.push_back(labels[i][0]);
    }
    return emotions;
}

// Function to compute accuracy of classification task.
float compute_emotions_accuracy(const std::vector < std::vector < int >> & detected_emotions,
    const std::vector < std::vector < int >> & emotion_labels,
        const std::vector < std::vector < int >> & orderings) {
    float count_equal = 0;
    int total_detected = 0;
    int corrected = 0;
    for (int i = 0; i < emotion_labels.size(); i++) {
        std::vector < int > ordering = orderings[i];
        if (emotion_labels[i].empty() || detected_emotions[i].empty() || ordering.empty()) {
            continue;
        }
        for (int j = 0; j < detected_emotions[i].size(); j++) {
            if (detected_emotions[i][j] == emotion_labels[i][ordering[j]]) {
                count_equal++;
                corrected++;
            }
        }
        total_detected += detected_emotions[i].size();
        corrected = 0;
    }

    if (total_detected == 0) {
        return 0.0f;
    }
    return count_equal / total_detected;
}

// This function is used to compute overall system accuracy, so how many faces are correctly 
// classified over all present faces.
float compute_system_accuracy(const std::vector < std::vector < int >> & detected_emotions,
    const std::vector < std::vector < int >> & emotion_labels,
        const std::vector < std::vector < int >> & orderings) {
    float count_equal = 0;
    int total_present = 0;
    int corrected = 0;
    for (int i = 0; i < emotion_labels.size(); i++) {
        std::vector < int > ordering = orderings[i];
        total_present += emotion_labels[i].size();

        if (emotion_labels[i].empty() || detected_emotions[i].empty() || ordering.empty()) {
            continue;
        }

        for (int j = 0; j < detected_emotions[i].size(); j++) {
            if (detected_emotions[i][j] == emotion_labels[i][ordering[j]]) {
                count_equal++;
                corrected++;
            }
        }
        corrected = 0;
    }

    if (total_present == 0) {
        return 0.0f;
    }
    return count_equal / total_present;
}