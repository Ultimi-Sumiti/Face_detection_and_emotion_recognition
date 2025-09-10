#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include "../include/utils.h"
#include "../include/performance_metrics.h"


void parse_command_line(
    int argc,
    char **argv,
    std::string& input_path,
    std::string& label_path
) {
    int opt;
    while ((opt = getopt(argc, argv, "i:l:")) != -1) {
        switch (opt) {
            case 'i':
                input_path = optarg;
                break;
            case 'l':
                label_path = optarg;
                break;
            case '?':
                std::cerr << "Usage: " << argv[0] 
                    << " -i <path> -l <path>\n"
                    << "  Where:\n"
                    << "    -i is the input image path\n"
                    << "    -l is the label of the input image (OPTIONAL)\n";
                break;
        }
    }
}




// Function to parse the labels of the positions and emotion from the given textual file.
std::vector<std::vector<float>> parse_labels(const std::string& filename){
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<float>> faces;
    int count = 0;
    int line_count = 0;
    std::vector<float> current_face;

    while(getline(file, line)){
        std::stringstream ss(line);
        std::string word;
        
        while(ss >> word && count < 5){
            current_face.push_back(std::stof(word));
            count++;
        }

        if(!current_face.empty()){
            faces.push_back(current_face);
            current_face.clear();
        }

        count = 0;
        line_count++;
    }

    return faces;
}

std::vector<cv::Rect> compute_rectangles(std::string& filename, int img_width, int img_height){
    std::vector<std::vector<float>> labels = parse_labels(filename);
    int x, y, width, height;
    std::vector<cv::Rect> rects_label;

    for (const auto& face: labels){
        x = (face[1] - face[3]/2) * img_width;
        y = (face[2] - face[4]/2) * img_height;
        //std::cout<<x<<" "<<y<<std::endl;
        width = face[3] * img_width;
        height = face[4] * img_height;
        rects_label.push_back(cv::Rect(x, y, width, height));
    }
    return rects_label;
}

void printRectDetails(const std::vector<cv::Rect>& rects) {
    // Check if the vector is empty first
    if (rects.empty()) {
        std::cout << "The vector of rectangles is empty." << std::endl;
        return;
    }

    // Use an index-based loop to easily number the rectangles
    for (size_t i = 0; i < rects.size(); ++i) {
        const cv::Rect& r = rects[i];

        // Calculate the center coordinates
        int centerX = r.x + r.width / 2;
        int centerY = r.y + r.height / 2;

        // Print the details in a formatted way
        std::cout << "Rect " << i << ":"
                  << " Center=[" << centerX << ", " << centerY << "],"
                  << " Width=" << r.width << ","
                  << " Height=" << r.height
                  << std::endl;
    }
}


double calculateBlurScore(const cv::Mat& image, const cv::Rect& roi) {
    // Isolate the Region of Interest (ROI)
    cv::Mat roi_mat = image(roi);

    // Convert to Grayscale
    cv::Mat gray_roi;
    cv::cvtColor(roi_mat, gray_roi, cv::COLOR_BGR2GRAY);

    // Apply the Laplacian Operator
    cv::Mat laplacian_image;
    // We use CV_64F (double) to avoid losing negative values from the operator
    cv::Laplacian(gray_roi, laplacian_image, CV_64F);

    // Calculate the Mean and Standard Deviation
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian_image, mean, stddev);

    // The variance is the standard deviation squared
    return stddev.val[0] * stddev.val[0];
}


// Detection function using the ViolaJones algorithm.
std::vector<cv::Rect> face_detect(cv::Mat& frame){

    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 

    // Plotting the image equalized.
    cv::namedWindow("Window",cv::WINDOW_NORMAL);
    cv::imshow("Window", frame);
    cv::waitKey(0);

    // Load the cascades.
    std::vector<cv::CascadeClassifier> f_cascades (file_paths.size());
    for(int i = 0; i < file_paths.size(); i++ ){
        if (!f_cascades[i].load(file_paths[i])){
            std::cout << "Error loading face cascade\n";
            exit(1);
        }
    }

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    std::vector<int> rejectLevels;   
    std::vector<double> levelWeights;   // This will hold the confidence scores.
    std::vector<double> blurScore;


    cv::CascadeClassifier final_classifier;
    int score = 0;
    std::vector<cv::Rect> filtered_faces; 
    std::vector<cv::Rect> best_detections; 
    int min_score = 1000000;
    int min_area = frame.rows * frame.cols /100;
    int best_score = 0;
    int best_index = 0;
    int actual_score = 0;
    int best_count = 0;
    float blur_score = 0.0f;
    cv::Rect img_rect = cv::Rect(0,0, frame.cols,frame.rows);
    float avg_blur = calculateBlurScore(frame, img_rect);

    // Checking all possible faces cascades. 
    for(int i = 0; i < f_cascades.size(); i++){
        std::cout<<std::endl<< "Testing classifier number: "<<i<<std::endl;
        faces.clear();
        f_cascades[i].detectMultiScale(
            frame_gray,
            faces,
            rejectLevels,
            levelWeights,
            1.1, // scaleFactor
            5,   // minNeighbors
            0,   // flags
            cv::Size(70, 70), // minSize
            cv::Size(),       // maxSize
            true              // outputRejectLevels true
        );


        std::cout << "Found " << faces.size() << " faces." << std::endl;
        // Loop through each detected face.
        for(int j = 0; j < faces.size(); j++){
            // Print the confidence score for the corresponding face.
            blur_score = calculateBlurScore(frame, faces[j]);
            score =  blur_score * faces[j].area();
            std::cout << "Face " << j
                << " -> Score: " << score <<std::endl;
            // Here we filter the detection: if they're both not defined and small we filter out.
            if((score > min_score || blur_score > avg_blur) && faces[i].area() > min_area){
                filtered_faces.push_back(faces[j]);
                actual_score += score;
            }
        }
        
        if(faces.size() > 0){
            std::cout<< "(Selected "<<filtered_faces.size()<<")"<<std::endl;
        }


        // Storing new best performance if current classifier perfomance are the best.
        if (actual_score > best_score){
            best_detections = filtered_faces;
            best_score = actual_score;
            best_index = i;
        }

        // Removing the previouse detections for the next classifier.
        filtered_faces.clear();
        actual_score = 0;
    }

    std::cout<<std::endl<<"Best classifier is: "<<file_paths[best_index]<<std::endl; 
    std::cout<<"  with score: "<<best_score<<std::endl<<std::endl;

    final_classifier = f_cascades[best_index];

    std::cout<<"Detected rectangles position and size: "<<std::endl;
    printRectDetails(best_detections);
    std::cout<<std::endl;

    if(best_detections.size() == 0){
        std::cout<<"Error: no detection!";
        exit(1);
    }
    
    /*
    // Folder path in which will be saved the images.
    std::string folder_path_cropped_imgs = "../cropped_imgs/";
    // Vector of cropped images and vector of bounding boxes.
    std::vector<cv::Mat>  cropped_imgs; 

    for (size_t i = 0; i < faces.size(); i++){

        // Cropping the detected faces.
        cv::Mat faceROI = frame(faces[i]);
        cropped_imgs.push_back(faceROI.clone());
        // Saving the cropped images.
        cv::imwrite(folder_path_cropped_imgs + "cut_" + std::to_string(i)+".png", cropped_imgs[i]);        
    }*/

    // Draw the box over the detection.
    /*for (int i = 0; i < best_detections.size(); i++){
        cv::rectangle(frame, best_detections[i], cv::Scalar(255, 0, 255), 4);      
    }*/
    
    // Show the images detected.
    /*
    double scale = 0.5;
    cv::resize(frame, frame, cv::Size(), scale, scale);
    cv::imshow("Window", frame);
    cv::waitKey(0);*/

    /*
    for (size_t i = 0; i < cropped_imgs.size(); i++){

        std::string window_name = "Window " + std::to_string(i);
        cv::imshow(window_name, cropped_imgs[i]);
    }

    cv::waitKey(0);
    */
   return best_detections;
}



/*
int main(){
    std::vector<std::vector<float>> faces = parse_labels("../dataset_detection/labels/angry_1.txt");
    if(!faces.empty()){
        std::cout<<faces[0][0]<<std::endl;
        std::cout<<faces[0][1]<<std::endl;
        std::cout<<faces[0][2]<<std::endl;
        std::cout<<faces[0][3]<<std::endl;
        std::cout<<faces[0][4]<<std::endl;
    }

}*/