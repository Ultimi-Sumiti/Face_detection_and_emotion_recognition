#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <numeric>
#include "../include/utils.h"


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



// Function to compute the IOU (intersection over union) between 2 given boxes.
float compute_IOU(cv::Rect& box1, cv::Rect& box2){
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

std::vector<float> get_label_IOUs(std::string& filename, std::vector<cv::Rect>& boxes, int img_width, int img_height){
    std::vector<cv::Rect> rects_label = compute_rectangles(filename, img_width, img_height);
    float current_IoU;
    std::vector<float> IOUs(boxes.size(), 0.0f); 

    for(int i = 0; i < rects_label.size(); i++){
        for(int j = 0; j < boxes.size(); j++){
            current_IoU = compute_IOU(rects_label[i], boxes[j]);
            if(current_IoU > IOUs[i]){
                IOUs[i] = current_IoU;
            }
        }
    }
    return IOUs;
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

float compute_MIOU(std::vector<float>& IOUs){
    // Handle the edge case of an empty vector to prevent division by zero.
    if (IOUs.empty()) {
        return 0.0f;
    }

    // Calculate the sum of all elements in the vector.
    //    std::accumulate(begin, end, initial_value)
    float total_iou = std::accumulate(IOUs.begin(), IOUs.end(), 0.0f);

    // Divide the sum by the number of elements to get the mean.
    return total_iou / IOUs.size();
}


// Detection function using the ViolaJones algorithm.
std::vector<cv::Rect> vj_detect(std::string& filename){
    std::string image_path = image_dir + filename + image_extension;
    std::string label_path = label_dir + filename + label_extension;

    cv::Mat frame = cv::imread(image_path);

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
    std::vector<double> levelWeights;   // This will hold the confidence scores
    std::vector<double> blurScore;


    cv::CascadeClassifier final_classifier;
    int score = 0;
    std::vector<cv::Rect> filtered_faces; 
    std::vector<cv::Rect> best_detections; 
    int min_score = 1000000;
    int best_score = 0;
    int best_index = 0;
    std::vector<float> IOUs (f_cascades.size());
    float best_MIOU = 0.0f;
    float curr_MIOU = 0.0f;

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
            cv::Size(30, 30), // minSize
            cv::Size(),       // maxSize
            true              // outputRejectLevels true
        );


        std::cout << "Found " << faces.size() << " faces." << std::endl;
        // Loop through each detected face
        for(int j = 0; j < faces.size(); j++){
            // Print the confidence score (level weight) for the corresponding face
            score = calculateBlurScore(frame, faces[j]) * faces[j].area();
            std::cout << "Face " << j
                << " -> Score: " << score <<std::endl;
            // Here we filter the detection: if they're both not defined and small we filter out.
            if(score > min_score){
                filtered_faces.push_back(faces[j]);
            }
        }
        
        if(faces.size() > 0){
            std::cout<< "(Selected "<<filtered_faces.size()<<")"<<std::endl;
        }

        IOUs = get_label_IOUs(label_path, filtered_faces, frame.cols, frame.rows);
        for(int k = 0; k < IOUs.size(); k++){
            std::cout<<"Rect "<<k<<" IOUs: "<<IOUs[k]<<std::endl;
        }
        curr_MIOU = compute_MIOU(IOUs);
        if (curr_MIOU > best_MIOU){
            best_index = i;
            best_MIOU = curr_MIOU;
            best_detections = filtered_faces;
        }

        filtered_faces.clear();
    }

    std::cout<<std::endl<<"Best classifier is: "<<file_paths[best_index]<<std::endl; 
    std::cout<<"  with MIOU: "<<best_MIOU<<std::endl<<std::endl;

    final_classifier = f_cascades[best_index];

    std::cout<<"Detected rectangles position and size: "<<std::endl;
    printRectDetails(best_detections);
    std::cout<<std::endl;
    std::vector<cv::Rect> label_rects = compute_rectangles(label_path, frame.cols, frame.rows);
    std::cout<<"Label rectangles position and size: "<<std::endl;
    printRectDetails(label_rects);

    if(best_detections.size() == 0){
        exit(1);
    }
    get_label_IOUs(label_path, best_detections, frame.cols, frame.rows);
    
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
    for (int i = 0; i < best_detections.size(); i++){
        cv::rectangle(frame, best_detections[i], cv::Scalar(255, 0, 255), 4);      
    }
    // Draw the label box.
    for (int i = 0; i < label_rects.size(); i++){
        cv::rectangle(frame, label_rects[i], cv::Scalar(255, 255, 0), 4);      
    }
    
    // Show the images detected.

    double scale = 0.5;
    cv::resize(frame, frame, cv::Size(), scale, scale);
    cv::imshow("Window", frame);
    cv::waitKey(0);

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