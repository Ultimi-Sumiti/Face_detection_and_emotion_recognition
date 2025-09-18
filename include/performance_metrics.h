#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <opencv2/core/types.hpp>

const float IOU_THRESHOLD = 0.4;

/*
    This class relize the purpose of summing up all the performance metrics 
    functions, data and functionalities. 
*/
class PerformanceMetrics {

    public:

        // CONSTRUCTORS:

        // Main constructor: initializes both the detected faces positions and label faces positions.
        PerformanceMetrics(
            const std::vector < std::vector < cv::Rect >> & detected_faces,
                const std::vector < std::vector < cv::Rect >> & face_labels,
                    const std::vector < std::vector < int >> & emotion_labels,
                        const std::vector < std::vector < int >> & detected_emotions,
                            const std::string & out_file
        ): detected_faces(detected_faces),
        face_labels(face_labels),
        emotion_labels(emotion_labels),
        detected_emotions(detected_emotions),
        metrics_file_path(out_file) {
            clean_metrics();
        }

        // Empty constructor.
        PerformanceMetrics(const std::string & out_file): metrics_file_path(out_file),
        face_labels({}),
        detected_faces({}),
        emotion_labels({}),
        detected_emotions({}) {
            clean_metrics();
        }

        //MEMBER FUNCTIONS:
        // Setter.
        void add_image_detections(const std::vector < cv::Rect > & detection,
            const std::vector < cv::Rect > & labels,
                const std::vector < int > & emotions,
                    const std::vector < int > & emotion_labels);

        // This member function compute the IOUs of the detected faces.
        std::vector < float > get_label_IOUs(const std::vector < cv::Rect > & detection,
            const std::vector < cv::Rect > & labels, std::vector < int > & ordering);

        // Function to write in a file and computing in terminal the metrics for the scenepath.
        void print_metrics(bool verbose);

        void clean_metrics();

    private:

        // DATA MEMEBERS: 

        // Vectors in which memorize the coordinate of the read values from the label txt file.
        std::vector < std::vector < cv::Rect >> detected_faces;
        std::vector < std::vector < cv::Rect >> face_labels;
        std::vector < std::vector < int >> detected_emotions;
        std::vector < std::vector < int >> emotion_labels;
        std::vector < std::vector < int >> orderings;

        std::string metrics_file_path;

};

// HELPER FUNCTIONS: 

// Function to compute the IOU (intersection over union) between 2 given boxes.
float compute_IOU(const cv::Rect & box1,
    const cv::Rect & box2);

// Function to compute the mean over IOUs of recatangles (MIOU).
float compute_MIOU(const std::vector < float > & IOUs);

// Function to compute the precision.
float compute_precision(const std::vector < float > & all_IOUs, float threshold, int all_detections);

// Function to compute the recall.
float compute_recall(const std::vector < float > & all_IOUs, float threshold);

// Function to compute emotions labels.
std::vector < int > get_label_emotion(const std::string & filename);

// Function to compute accuracy of classification task.
float compute_emotions_accuracy(const std::vector < std::vector < int >> & detected_emotions,
    const std::vector < std::vector < int >> & emotion_labels,
        const std::vector < std::vector < int >> & orderings);

// Function used to compute entire system accuaracy.
float compute_system_accuracy(const std::vector < std::vector < int >> & detected_emotions,
    const std::vector < std::vector < int >> & emotion_labels,
        const std::vector < std::vector < int >> & orderings);
#endif