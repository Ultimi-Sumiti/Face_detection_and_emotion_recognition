#include <iostream>
#include <string>
#include <fstream>
#include <thread> 
#include <vector>
#include <algorithm>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "../../include/utils.h"
#include "../../include/performance_metrics.h"
#include "../../include/face_detector.h"

#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define RED     "\033[31m"
#define INV     "\033[7m"

// Haarcascades that can be used for face detection.
const std::vector<std::string> HAARCASCADES_PATHS = {
    "../data/haarcascades/haarcascade_frontalface_alt.xml",
    //"../data/haarcascades/haarcascade_frontalface_alt_tree.xml",
    //"../data/haarcascades/haarcascade_frontalface_default.xml",
    "../data/haarcascades/haarcascade_frontalface_alt2.xml",
    //"../data/haarcascades/haarcascade_profileface.xml",
};

// Folder path in which cropped images with faces are (temporary) stored.
const std::string CROPPED_IMGS_PATH = "../tmp/cropped_imgs/";

// Folder in which the images with the bounding boxes drawn are stored.
const std::string OUTPUT_DETECTIONS_PATH = "../output/detections/";
// File where metrics are stored.
const std::string METRICS_OUT = "../output/metrics.txt";
// Output video.
const std::string VIDEO_OUT = "../output/video.avi";

// Path to the fifo file where messages are sent.
const std::string SEND_FIFO = "../tmp/cpp_to_py.fifo";
// Path to the fifo file where messages are received.
const std::string RECEIVE_FIFO = "../tmp/py_to_cpp.fifo";

// Command used to start the emotion recognition model.
const std::string EMOTION_REC_CMD = "python3 ../src/python/emotion_classifier.py"
                                    " 2> ../log/emotion_classifier_log.txt";


// Function used to run the emotion recognition model (in Python).
void run_emotion_rec(void) {
    system(EMOTION_REC_CMD.c_str());
}


int main(int argc, char* argv[]) {

    // Parse command line options.
    std::string imgs_dir_path{}, labels_dir_path{}, video{};
    int webcam = 0;

    int status = parse_command_line(
            argc,
            argv,
            imgs_dir_path,
            labels_dir_path,
            video,
            webcam
    );

    if (status) {
        std::cout << help_msg << std::endl;
        return EXIT_FAILURE;
    }
    
    if (!webcam && imgs_dir_path.empty() && video.empty()) {
        std::cerr << RED "ERROR: You must either set webcam or provide a "
                         "video file or provide an input images directory." RESET 
                  << std::endl;
        std::cout << help_msg << std::endl;
        return EXIT_FAILURE;
    }

    // Set the mode:
    //   mode = 0 => webcam
    //   mode = 1 => video
    //   mode = 2 => images directory
    int mode;
    
    if (webcam) mode = 0;
    else if (!video.empty()) mode = 1;
    else if (!imgs_dir_path.empty()) mode = 2;

    // Retreive all filenames inside the directories (for mode 2).
    std::vector<std::string> imgs_paths, labels_paths;

    if (mode == 2) {

        // Retreive all images to process.
        imgs_paths = get_all_filenames(imgs_dir_path);
        std::sort(imgs_paths.begin(), imgs_paths.end());

        // If associated labels are provided.
        if (!labels_dir_path.empty()) {

            // Retreive labels.
            labels_paths = get_all_filenames(labels_dir_path);
            std::sort(labels_paths.begin(), labels_paths.end());

            // Manage errors.
            if (labels_paths.empty()) {
                std::cerr << RED "ERROR: Labels directory is empty." RESET
                          << std::endl;
                return EXIT_FAILURE;
            } else if (labels_paths.size() != imgs_paths.size()) {
                std::cerr << RED "ERROR: Images directory and labels directory "
                                 "sizes must concide." RESET << std::endl;
                return EXIT_FAILURE;
            }
        }
    }

    // Define video capture and video writer (for mode 0 or 1).
    cv::VideoCapture capture;
    cv::VideoWriter writer;

    switch (mode) {
        case 0: // Open webcam.
            capture.open(0);
            if (!capture.isOpened()) {
                std::cerr << RED "ERROR: Couldn't open webcam." RESET << std::endl;
                return EXIT_FAILURE;
            }
            break;

        case 1: // Open video and define writer.
            capture.open(video);
            if (!capture.isOpened()) {
                std::cerr << RED "ERROR: Couldn't open video '" 
                          <<  video << "'." RESET << std::endl;
                return EXIT_FAILURE;
            }

            // Setup writer.
            int w = (int) capture.get(cv::CAP_PROP_FRAME_WIDTH);
            int h = (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = capture.get(cv::CAP_PROP_FPS);
            writer = cv::VideoWriter(
                    VIDEO_OUT,
                    cv::VideoWriter::fourcc('M','J','P','G'),
                    fps,
                    cv::Size(w, h),
                    true
            );

            if (!writer.isOpened()) {
                std::cerr << RED "Could not open the output video "
                                 "file for write" RESET <<std::endl;
                return EXIT_FAILURE;
            }
            break;
    }

    // Create fifo files used for Inter Process Communication (CPP <-> Python).
    if (fifo_creation(SEND_FIFO) || fifo_creation(RECEIVE_FIFO)) {
        std::cerr << RED "ERROR: Cannot create fifo files... aborting" RESET <<std::endl;
        std::cerr << "errno: " << errno << std::endl 
                  << std::strerror(errno) << std::endl;
        return EXIT_FAILURE;
    }

    // Define the FaceDetector.
    FaceDetector detector;
    try {
        detector = FaceDetector(HAARCASCADES_PATHS);
    } catch(const std::runtime_error& e) {
        std::cerr << RED "Exception caught, impossible to upload the cascades: " RESET 
                  << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Clean all images in the output dir from previous run.
    remove_images(get_all_filenames(OUTPUT_DETECTIONS_PATH));
    remove_images(get_all_filenames(CROPPED_IMGS_PATH));

    // Store IOU of each image (if necessary).
    std::vector<float> IOUs; 

    // This object hold all the functionalities for performance metrics.
    PerformanceMetrics pm = PerformanceMetrics(METRICS_OUT);
    int correct_detection = 0;
    int detection_count = 0;

    // Start concurrent thread with the emotion recognizer.
    std::thread emotion_rec_thread = std::thread(run_emotion_rec);


    std::vector<int> emotions_val;
    std::vector<int> emotions_labels;
    std::vector<std::string> emotions;
    // Process all images.
    for (int itr = 0; true; itr++) {

        std::string info;

        // Load image.
        cv::Mat img;

        // Choose next image to process.
        switch(mode) {
            case 0:
            case 1:
                capture.read(img);
                info = "INFO: Processing next frame...";
                break;

            case 2:
                if (itr >= imgs_paths.size())
                    break;

                img = cv::imread(imgs_paths[itr]);
                if (img.empty()) {
                    std::cerr << RED "ERROR: Cannot open '" << imgs_paths[itr]  
                              << "'." RESET << std::endl;
                    continue;
                }

                info = "INFO: Processing '" + imgs_paths[itr] + "'.";
                break;
        }


        // Quit if no image has been loaded.
        if (img.empty()) break;

        // ---------------------- FACE DETECTION ------------------------------
        std::cout << INV GREEN "\n### ITR: " << itr << " ###" RESET << std::endl;
        std::cout << info << std::endl;


        // Detect faces in the image.
        std::vector<cv::Rect> faces = detector.face_detect(img);
        std::cout << "INFO: Detected "<< faces.size() << " faces." << std::endl;
        detection_count += faces.size();

        // Crop detected faces, store them to disk.
        std::vector<std::string> cropped_paths = crop_images(img, faces, CROPPED_IMGS_PATH);

        // Get emotions for each face found.
        if (!faces.empty()) {
            // ------------------ EMOTION RECOGNITION -------------------------
            // Open communication, send start message.
            std::ofstream chan_send(SEND_FIFO);
            chan_send << "start" << std::flush;
            chan_send.close();

            // Wait for response.
            std::ifstream chan_receive(RECEIVE_FIFO);

            // Read all the messages (i.e. emotions) and close channel.
            std::string line;
            emotions.clear();
            std::cout << "INFO: Received emotions:" << std::endl;
            while (std::getline(chan_receive, line)) {
                emotions.push_back(line);
                std::cout << "\t" << line << std::endl;
            }
            chan_receive.close();

            if (emotions.size() != faces.size()) {
                std::cerr << RED "ERROR: Not all faces were classified. "
                                 "Skipping image." RESET << std::endl;
                continue;
            }

            // Draw boxes around detected faces and write emotions.
            detector.draw_bbox(img, faces, emotions);
        }

        //  ------------------ PERFORMANCE METRICS ----------------------------

        // Compute and store metrics in a file, if necessary.
        if (!labels_paths.empty()) { 
            std::vector<cv::Rect> labels_rect = compute_rectangles(
                    labels_paths[itr],
                    img.cols,
                    img.rows
            );
            emotions_val = parse_emotions(emotions);
            emotions_labels = get_label_emotion(labels_paths[itr]);
            pm.add_image_detections(faces, labels_rect, emotions_val, emotions_labels);
        }



        // Store the image with boxes drawn.
        switch (mode) {
            case 0:
                cv::imshow("Capture - Face detection", img);
                if( cv::waitKey(10) == 27 )  break; 
                break;

            case 1:
                writer.write(img);
                break;

            case 2:
                if (faces.empty())
                    break;
                std::string out_path = 
                    OUTPUT_DETECTIONS_PATH + "image_" + std::to_string(itr) + ".png";
                if (cv::imwrite(out_path, img))
                    std::cout << "INFO: '" << out_path << "' saved." << std::endl;
                else
                    std::cerr << RED "ERROR: Couldn't save '" << out_path 
                              << "' to disk." RESET << std::endl;
                break;

        }

        // Clean cropped image folder for next interation.
        remove_images(cropped_paths); 
    }

    // Open communication, send exit message.
    std::ofstream chan_send(SEND_FIFO);
    chan_send << "exit" << std::flush;
    chan_send.close();

    // Wait the thread.
    emotion_rec_thread.join();

    // Printing metrics in a file, if necessary.
    if (!labels_paths.empty()) {
        pm.print_metrics(true);
    }

    return EXIT_SUCCESS;
}
