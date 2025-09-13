#include <iostream>
#include <string>
#include <fstream>
#include <thread> 
#include <vector>
#include <numeric>
#include <opencv2/imgcodecs.hpp>

#include "../../include/utils.h"
#include "../../include/performance_metrics.h"
#include "../../include/face_detector.h"


// Haarcascades that can be used for face detection.
const std::vector<std::string> HAARCASCADES_PATHS = {
    "../data/haarcascades/haarcascade_frontalface_alt.xml",
    //"../data/haarcascades/haarcascade_frontalface_alt_tree.xml",
    //"../data/haarcascades/haarcascade_frontalface_default.xml",
    //"../data/haarcascades/haarcascade_frontalface_alt2.xml",
    //"../data/haarcascades/haarcascade_profileface.xml",
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

    // Parse command line, get image directory and (optinally) labels directory.
    std::string imgs_dir_path{}, labels_dir_path{};
    if (parse_command_line(argc, argv, imgs_dir_path, labels_dir_path)) {
        std::cout << help_msg << std::endl;
        return EXIT_FAILURE;
    }

    // If input dir is missing => quit.
    if (imgs_dir_path.empty()) {
        std::cerr << "ERROR: Input directory argument is missing." << std::endl;
        std::cout << help_msg << std::endl;
        return EXIT_FAILURE;
    }

    // Retreive all filenames inside the directories.
    std::vector<std::string> imgs_paths = get_all_filenames(imgs_dir_path);
    std::vector<std::string> labels_paths{};

    if (!labels_dir_path.empty())
        labels_paths = get_all_filenames(labels_dir_path);

    // If the niput directory is empty => quit.
    if (imgs_paths.empty()) {
        std::cerr << "ERROR: Directory '" << imgs_dir_path 
                  << "' is empty or doesn't exists." << std::endl;
        return EXIT_FAILURE;
    }

    // Create fifo files used for Inter Process Communication (CPP <-> Python).
    if (fifo_creation(SEND_FIFO) || fifo_creation(RECEIVE_FIFO)) {
        std::cerr << "ERROR: Cannot create fifo files... aborting" << std::endl;
        return EXIT_FAILURE;
    }

    // Define the FaceDetector.
    FaceDetector detector;
    try {
        detector = FaceDetector(HAARCASCADES_PATHS);
    } catch(const std::runtime_error& e) {
        std::cerr << "Exception caught, impossible to upload the cascades: " 
                  << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Clean all images in the output dir from previous run.
    remove_images(get_all_filenames(OUTPUT_DETECTIONS_PATH));

    // Define the sending channel and the receiving channel.
    std::ofstream chan_send;
    std::ifstream chan_receive;

    // Start concurrent thread with the emotion recognizer.
    std::thread emotion_rec_thread = std::thread(run_emotion_rec);

    // Store IOU of each image (if necessary).
    std::vector<float> IOUs; 

    // Process all images.
    for (int itr = 0; itr < imgs_paths.size(); itr++) {
        std::cout << "\n### ITR: " << itr << " ###"<< std::endl;

        // ---------------------- FACE DETECTION ------------------------------
        // Open input image.
        std::cout << "INFO: Analyzing '" << imgs_paths[itr] << "'" << std::endl;
        cv::Mat img = cv::imread(imgs_paths[itr]);
        if (img.empty()) {
            std::cerr << "ERROR: Cannot open " << imgs_paths[itr] << std::endl;
            continue;
        }

        // Detect faces in the image.
        std::vector<cv::Rect> faces = detector.face_detect(img);
        std::cout << "INFO: Detected "<< faces.size() << " faces." << std::endl;

        // Crop detected faces, store them to disk.
        std::vector<std::string> cropped_paths = crop_images(img, faces, CROPPED_IMGS_PATH);

        std::vector<cv::Rect> labels_rect; // Used to compute the metrics.

        //  ------------------ PERFORMANCE METRICS ----------------------------

        // Compute and store metrics in a file, if necessary.

        if (!labels_paths.empty()) { // TODO: modificare sto codice doppio orribile.
            labels_rect = compute_rectangles(labels_paths[itr], img.cols, img.rows);
            PerformanceMetrics pm = PerformanceMetrics(faces, labels_rect, METRICS_OUT);
            if (itr == 0) pm.clean_metrics();
            pm.print_metrics(imgs_paths[itr]);
            std::vector<float> label_IOUS = pm.get_label_IOUs();
            IOUs.insert(IOUs.end(), label_IOUS.begin(), label_IOUS.end());
        }

        if(faces.empty()) continue;

        // -------------------- EMOTION RECOGNITION ---------------------------
        //std::cout<<"Prima di python\n"; // TODO: debug comment.
        // Open communication, send start message.
        chan_send.open(SEND_FIFO);
        chan_send << "start" << std::flush;
        chan_send.close();

        //std::cout << "Waiting Python Response...\n"; // TODO: debug comment.
        // Wait for response.
        chan_receive.open(RECEIVE_FIFO);

        // Read all the messages (i.e. emotions) and close channel.
        std::vector<std::string> emotions;
        std::string line;
        while (std::getline(chan_receive, line))
            emotions.push_back(line);
        chan_receive.close();

        // Draw boxes around detected faces and write emotions.
        detector.draw_bbox(img, faces, emotions);

        // Store the image with boxes drawn.
        std::string out_path = 
            OUTPUT_DETECTIONS_PATH + "image_" + std::to_string(itr) + ".png";
        if (cv::imwrite(out_path, img))
            std::cout << "INFO: '" << out_path << "' saved." << std::endl;
        else
            std::cerr << "ERROR: Couldn't save '" << out_path << "' to disk." << std::endl;

        // Clean cropped image folder for next interation.
        remove_images(cropped_paths); 
    }

    // Open communication, send exit message.
    chan_send.open(SEND_FIFO);
    chan_send << "exit" << std::flush;
    chan_send.close();

    // Wait the thread.
    emotion_rec_thread.join();

    if (!labels_paths.empty()) {
        float avg_IOU = (std::accumulate(IOUs.begin(), IOUs.end(), 0.0))/(IOUs.size());
        std::cout<<std::endl<<"The avarage IOU obtained with current detector configuration: "<< avg_IOU<<std::endl;
    }

    return EXIT_SUCCESS;
}
