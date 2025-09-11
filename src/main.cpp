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
#include "face_detector.h"


namespace fs = std::filesystem;

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

const std::string image_dir = "images";
const std::string label_dir = "labels";
const std::string image_extension = ".jpg";
const std::string label_extension = ".txt";
const std::string detection_dir = "detections";


int main(int argc, char* argv[]) {

    // Creation of the 2 fifo to communicate with the emotion_classifier pipeline
    fifo_creation("cpp_to_py.fifo");
    fifo_creation("py_to_cpp.fifo");

    // Parse command line.
    std::string input_path{}, file_name{};
    parse_command_line(argc, argv, input_path, file_name);
        
    
    std::vector<std::string> complete_paths;
    std::vector<std::string> label_paths;
    std::string detection_path = input_path + "/" + detection_dir + "/";


    if (input_path.empty()) {
        std::cerr << "Error in parsing the command line...\n";
        return -1;
    }

    if (!file_name.empty()){
        complete_paths.push_back(input_path + "/" + image_dir + "/"+ file_name + image_extension);
        label_paths.push_back(input_path + "/" + label_dir + "/"+ file_name + label_extension);

        // Print args found.
        std::cout << "INPUT FILE PATH " << input_path << "\n";
        std::cout << "FILE NAME " << file_name << "\n";

    }else{

        try {
            // Create a directory iterator
            for (const auto& entry : fs::directory_iterator(input_path + "/" + image_dir )) {
                // Check if the entry is a regular file
                if (entry.is_regular_file()) {
                    // Get the path and extract the filename
                    complete_paths.push_back(entry.path().string());
                    //std::cout<<entry.path().filename().string();
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error accessing directory: " << e.what() << std::endl;
            return 1;
        }

        try {
            // Create a directory iterator
            for (const auto& entry : fs::directory_iterator(input_path + "/" + label_dir)) {
                // Check if the entry is a regular file
                if (entry.is_regular_file()) {
                    // Get the path and extract the filename
                    label_paths.push_back(entry.path().string());
                    //std::cout<<entry.path().filename().string();
                }
            }
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Error accessing directory: " << e.what() << std::endl;
            return 1;
        }

    }


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
    for(const auto& path : complete_paths){

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

        std::string full_detection_path = detection_path + "image_" + std::to_string(count) + image_extension;
        std::cout<<full_detection_path<<std::endl;
        if(cv::imwrite( full_detection_path, img)){
            std::cout<<"Image: "<<"image_" + std::to_string(count)<<" saved."<<std::endl;
        }else{
            std::cout<<"Image: "<<"image_" + std::to_string(count)<<" not saved."<<std::endl;
        }
        count++;
        
        //  ------------------------------------ PERFORMANCE METRICS ------------------------------------ 
        // Performance metrics, if necessary.
        if (!file_name.empty()) {
            std::vector<cv::Rect> label_rect = compute_rectangles(label_paths.back(), img.cols, img.rows);
            PerformanceMetrics pm(faces, label_rect);
            pm.print_metrics();
        }

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







   
