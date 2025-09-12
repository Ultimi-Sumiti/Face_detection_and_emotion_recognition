#include <iostream>
#include <string>
#include <fstream>
#include <thread> 
#include <vector>
#include <opencv2/imgcodecs.hpp>

#include "../../include/utils.h"
#include "../../include/performance_metrics.h"
#include "../../include/face_detector.h"


// Haarcascades that can be used for face detection.
const std::vector<std::string> HAARCASCADES_PATHS = {
    "../data/haarcascades/haarcascade_frontalface_alt.xml",
    "../data/haarcascades/haarcascade_frontalface_alt_tree.xml",
    "../data/haarcascades/haarcascade_frontalface_default.xml",
    "../data/haarcascades/haarcascade_frontalface_alt2.xml",
    "../data/haarcascades/haarcascade_profileface.xml",
};

// Folder path in which cropped images with faces are (temporary) stored.
const std::string CROPPED_IMGS_PATH = "../tmp/cropped_imgs/";

// Folder in which the images with the bounding boxes drawn are stored.
const std::string OUTPUT_DETECTIONS_PATH = "../output/detections/";
// File where metrics are stored.
const std::string METRICS_OUT = "../output/metrics.txt";

// Path to the fifo file where messages are sent.
const std::string SEND_FIFO = "../tmp/cpp_to_py.fifo";
// Path to the fifo file where messages are received.
const std::string RECEIVE_FIFO = "../tmp/py_to_cpp.fifo";

// Command used to start the emotion recognition model.
const std::string EMOTION_REC_CMD = "python3 ../src/python/emotion_classifier.py"
                                    " 2> ../log/emotion_classifier_log.txt";

// Function used to run the emotion recognition model (in Python).
void run_emotion_rec() {
    int ret = system(EMOTION_REC_CMD.c_str());
}


int main(int argc, char* argv[]) {

    // Create fifo files used for Inter Process Communication (CPP, Python).
    if (fifo_creation(SEND_FIFO) || fifo_creation(RECEIVE_FIFO)) {
        std::cerr << "Error in creating the fifo file... aborting" << std::endl;
        return EXIT_FAILURE;
    }

    // Clean all images in the output dir from previous run.
    remove_images(get_all_filenames(OUTPUT_DETECTIONS_PATH));

    // Parse command line, get image directory and (optinally) labels directory.
    std::string imgs_dir_path{}, labels_dir_path{};
    if (parse_command_line(argc, argv, imgs_dir_path, labels_dir_path)) {
        std::cout << help_msg << std::endl;
        return EXIT_FAILURE;
    }

    if (imgs_dir_path.empty()) {
        std::cerr << "Error: input directory argument is missing." << std::endl;
        std::cout << help_msg << std::endl;
        return EXIT_FAILURE;
    }

    // Retreive all filenames inside the directories.
    std::vector<std::string> imgs_paths = get_all_filenames(imgs_dir_path);
    std::vector<std::string> labels_paths{};
    if (!labels_dir_path.empty())
        labels_paths = get_all_filenames(labels_dir_path);

    if (imgs_paths.empty()) {
        std::cerr << "ERROR: directory '" << imgs_dir_path 
                  << "' is empty or doesn't exists." << std::endl;
        return EXIT_FAILURE;
    }


    // ------------------------- FACE DETECTION -------------------------------

    // Define the FaceDetector passing it the path of the classifier to load.
    FaceDetector detector;
    try {
        detector = FaceDetector(HAARCASCADES_PATHS);
    } catch(const std::runtime_error& e) {
        std::cerr << "Exception caught, impossible to upload the cascades: " 
                  << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Call the python pipeline to classify the faces.
    std::thread emotion_rec_thread = std::thread(run_emotion_rec);

    // Start processing all images.
    for (int itr = 0; itr < imgs_paths.size(); itr++) {

        // Processing the current image.
        cv::Mat img = cv::imread(imgs_paths[itr]);
        std::cout<<std::endl<< "Analyzing: "<< imgs_paths[itr];

        if (img.empty()) {
            std::cerr<<"Error: cannot open image!"<<std::endl;
            continue;
        }

        // Detect and save the faces in a specific folder.
        //std::vector<cv::Rect> faces = detector.face_detect(img);
        std::vector<cv::Rect> labels_rect;
        std::vector<cv::Rect> faces = detector.face_detect(img);
        std::cout<<std::endl<<"Detected: "<< faces.size()<< " faces."<<std::endl;
        // Crop images and save it in a vector.
        std::vector<std::string> cropped_paths = crop_images(img, faces, CROPPED_IMGS_PATH);

        // -------------------- EMOTION RECOGNITION ---------------------------
        // Signal (to Python).
        std::cout<<"Prima di python\n";   
        std::ofstream to_server(SEND_FIFO);

        if(faces.empty()){
            std::cout <<"No faces are detected, going to next image." << std::endl;
            // Singal (to Python) for closing its pipeline. 
            to_server << "continue" << std::endl; 
            // Go to next iteration (next image).
/**/        if (!labels_paths.empty()) { // TODO: modificare sto codice doppio orribile.
                labels_rect = compute_rectangles(labels_paths[itr], img.cols, img.rows);
                PerformanceMetrics pm = PerformanceMetrics(faces, labels_rect, METRICS_OUT);
                if (itr == 0) pm.clean_metrics();
                pm.print_metrics(imgs_paths[itr]);
            }
            continue;
        }
        to_server << "Required Emotion recognition" << std::endl;
        to_server.close();

        // **** Python program is currently detecting *****

        // Waiting (from Python)
        std::cout << "Waiting Python Response...\n";
        std::ifstream from_server(RECEIVE_FIFO);
        std::string line;
        std::vector<std::string> labels;

        // Read all the output stream
        while (std::getline(from_server, line)) {
            std::cout << "Python output: " << line << std::endl;
            labels.push_back(line);
        } 
        from_server.close();

        // Draw the detection with the labels on current image.
        detector.draw_bbox(img, faces, labels);

        // Store the image with boxes drawn.
        std::string full_detection_path = OUTPUT_DETECTIONS_PATH + "image_" + std::to_string(itr) + ".png";
        if (cv::imwrite(full_detection_path, img))
            std::cout << "Image: " << full_detection_path << " saved." << std::endl;
        else
            std::cerr << "Error: couldn't save " << full_detection_path << " to disk" << std::endl;

        //  ------------------ PERFORMANCE METRICS ----------------------------

        // Compute and store metrics in a file, if necessary.
        if (!labels_paths.empty()) {
            labels_rect = compute_rectangles(labels_paths[itr], img.cols, img.rows);
            PerformanceMetrics pm = PerformanceMetrics(faces, labels_rect, METRICS_OUT);
            if (itr == 0) pm.clean_metrics();
            pm.print_metrics(imgs_paths[itr]);
        }

        // Remove cropped images.
        remove_images(cropped_paths); 
    }

    // Sending exit message to python.
    std::ofstream to_server(SEND_FIFO);
    to_server << "exit" << std::endl;

    // Wait the thread ends.
    emotion_rec_thread.join();

    return EXIT_SUCCESS;
}
