#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include <string>
#include <fstream>
#include <errno.h>
#include <unistd.h> 
#include <thread> 
#include <filesystem>
//#include <Python.h>

#include "../include/utils.h"
#include "../include/performance_metrics.h"
#include "../include/face_detector.h"


void recognition_pipeline_call(){
        int ret = system("python3 ../python/emotion_classifier.py 2>/dev/null");

}


const std::vector<std::string> classifiers_paths = {
        "../classifiers/haarcascade_frontalface_alt_tree.xml",
        "../classifiers/haarcascade_frontalface_alt.xml",
        "../classifiers/haarcascade_frontalface_alt2.xml",
        "../classifiers/haarcascade_frontalface_default.xml",
        "../classifiers/haarcascade_profileface.xml",
};


const std::string detections_path = "../detections/";
const std::string image_extension = ".jpg";


int main(int argc, char* argv[]) {

    // Create fifo files for Inter Process Communication (CPP, Python).
    fifo_creation("cpp_to_py.fifo");
    fifo_creation("py_to_cpp.fifo");

    // Parse command line, get image directory and (optinally) labels directory.
    std::string imgs_dir_path{}, labels_dir_path{};
    parse_command_line(argc, argv, imgs_dir_path, labels_dir_path);
        
    if (imgs_dir_path.empty()) {
        std::cerr << "Error in parsing the command line...\n";
        return EXIT_FAILURE;
    }
    
    // Retreive all filenames inside the directories.
    std::vector<std::string> imgs_paths = get_all_filenames(imgs_dir_path);
    std::vector<std::string> labels_paths{};
    if (!labels_dir_path.empty())
        labels_paths = get_all_filenames(labels_dir_path);


    // -------------------------------------- FACE DETECTION --------------------------------------
    
    // Define the FaceDetector passing it the path of the classifier to load.
    FaceDetector detector;    
    try
    {
        detector = FaceDetector(classifiers_paths);
    }
    catch(const std::runtime_error& e)
    {
        std::cerr << "Exception caught, impossible to upload the cascades: " << e.what() << std::endl;
        return 1;
    }

    // Call the python pipeline to classify the faces
    std::thread emotion_rec_thread = std::thread(recognition_pipeline_call);

    // Start processing all images.
    int count = 0;
    for(const auto& path : imgs_paths){

        // Processing the current image
        cv::Mat img = cv::imread(path);
        std::cout<<std::endl<< "Analyzing: "<<path;

        if(img.empty()){
            std::cerr<<"Error: cannot open image!"<<std::endl;
            continue;
        }
        
        // Detect and save the faces in a specific folder.
        std::vector<cv::Rect> faces = detector.face_detect(img);
        std::cout<<std::endl<<"Detected: "<< faces.size()<< " faces."<<std::endl;
        // Crop images and save it in a vector.
        std::vector<std::string> cropped_paths = crop_images(img, faces);

        
        // ------------------------------------ EMOTION RECOGNITION ------------------------------------
        // Signal (to Python)
        std::cout<<"Prima di python\n";   
        std::ofstream to_server("cpp_to_py.fifo");
        
        if(faces.empty()){
            std::cout <<"No faces are detected, the program terminates\n";
            // Singal (to Python) for closing its pipeline 
            to_server << "continue" << std::endl; 
            // Go to next iteration (next image).
            continue;
        }
        to_server << "Required Emotion recognition" << std::endl;
        to_server.close();

        // **** Python program is currently detecting *****

        // Waiting (from Python)
        std::cout << "In attesa della risposta da Python...\n";
        std::ifstream from_server("py_to_cpp.fifo");
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

        std::string full_detection_path = detections_path + "image_" + std::to_string(count) + image_extension;
        std::cout<<full_detection_path<<std::endl;
        if(cv::imwrite( full_detection_path, img)){
            std::cout<<"Image: "<<"image_" + std::to_string(count)<<" saved."<<std::endl;
        }else{
            std::cout<<"Image: "<<"image_" + std::to_string(count)<<" not saved."<<std::endl;
        }
        count++;
        
        //  ------------------------------------ PERFORMANCE METRICS ------------------------------------ 
        // Performance metrics, if necessary.
        //if (!file_name.empty()) {
        //    std::vector<cv::Rect> label_rect = compute_rectangles(labels_paths.back(), img.cols, img.rows);
        //    PerformanceMetrics pm(faces, label_rect);
        //    pm.print_metrics();
        //}

        //draw_bbox(img, faces, labels);
        //namedWindow("Window", cv::WINDOW_NORMAL);
        //cv::imshow("Window", img);
        //cv::waitKey(0);

        // Remove cropped
        remove_images(cropped_paths); 

    }

    // Sending exit message to python.
    std::ofstream to_server("cpp_to_py.fifo");
    to_server << "exit" << std::endl;

    // Wait the thread ends
    emotion_rec_thread.join();

    return 0;
}







   
