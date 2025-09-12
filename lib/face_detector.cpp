#include "../include/face_detector.h"

#include <iostream>
#include <opencv2/imgproc.hpp>

#include "../include/utils.h"


//-------------- MEMBER FUNCTIONS --------------
void FaceDetector::draw_bbox(
        cv::Mat frame, std::vector<cv::Rect> faces,
        const std::vector<std::string> &labels
) {
    for (size_t i = 0; i < faces.size(); i++) {
        cv::Scalar color = cv::Scalar(255, 255, 255); // default = white
        if (this->label_color.find(labels[i]) != this->label_color.end())
            color = this->label_color[labels[i]];

        // Draw the Bounding box of the detected face.
        cv::rectangle(frame, faces[i], color, 3);

        // Font parameters
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 0.7;  
        int thickness = 1;

        // Calculating size of the text.
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(labels[i], font_face, font_scale, thickness, &baseline);

        // Position of the text.
        cv::Point text_org(faces[i].x, faces[i].y - 5);

        // Background rectangle for text label.
        cv::rectangle(frame,
                      text_org + cv::Point(0, baseline),
                      text_org + cv::Point(text_size.width, -text_size.height),
                      color, cv::FILLED);

        // If color is yellow or white change the text color into balck.
        cv::Scalar text_color = cv::Scalar(255, 255, 255);
        if (labels[i] == "neutral" || labels[i] == "disgust")
            text_color = cv::Scalar(0, 0, 0);

        // Draw the text over the colored backround.
        cv::putText(frame, labels[i], text_org,
                    font_face, font_scale, text_color, thickness, cv::LINE_AA);
    }
}


std::vector<cv::Rect> FaceDetector::vj_detect(cv::Mat frame) {
    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    this->f_cascades[0].detectMultiScale(frame_gray, faces);

    return faces;   

}

std::vector<cv::Rect> FaceDetector::face_detect(cv::Mat& frame) {
    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 
    // Computing the area of the frame.
    int area = frame.rows * frame.cols;

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    std::vector<int> rejectLevels;   
    std::vector<double> levelWeights;   // This will hold the confidence scores.
    std::vector<double> blurScore;


    cv::CascadeClassifier final_classifier;
    int score = 0;
    std::vector<cv::Rect> filtered_faces; 
    std::vector<cv::Rect> best_detections; 
    int min_area = area / 100;
    int min_side = static_cast<int>(std::sqrt(min_area));
    int best_score = 0;
    int best_index = 0;
    int actual_score = 0;
    int best_count = 0;
    float blur_score = 0.0f;
    cv::Rect img_rect = cv::Rect(0,0, frame.cols,frame.rows);
    float min_blur = calculateBlurScore(frame, img_rect) / 2;
    int min_score = min_blur * min_area;
    int min_weight = 80;

    // Checking all possible faces cascades. 
    for(int i = 0; i < this->f_cascades.size(); i++){
        // std::cout<<std::endl<< "Testing classifier number: "<<i<<std::endl;
        faces.clear();
        this->f_cascades[i].detectMultiScale(
            frame_gray,
            faces,
            rejectLevels,
            levelWeights,
            1.02, // scaleFactor
            10,   // minNeighbors
            0,   // flags
            cv::Size(min_side, min_side), // minSize
            cv::Size(),       // maxSize
            true              // outputRejectLevels true
        );

        //std::cout << "Found " << faces.size() << " faces." << std::endl;
        // Loop through each detected face.
        for(int j = 0; j < faces.size(); j++){
            // Print the confidence score for the corresponding face.
            blur_score = calculateBlurScore(frame, faces[j]);
            score =  blur_score * faces[j].area();
            // std::cout << std::endl << "Face " << j << " -> Score: " << levelWeights[j] << std::endl;
            // Here we filter the detection: if they're both not defined and small we filter out.
            if((score >= min_score) && levelWeights[j] > min_weight){
                filtered_faces.push_back(faces[j]);
                actual_score += score;
            }
        }
        
        if(faces.size() > 0){
            //std::cout<< "(Selected "<<filtered_faces.size()<<")"<<std::endl;
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

    //std::cout<<std::endl<<"Best classifier is: "<<file_paths[best_index]<<std::endl; 
    //std::cout<<"  with score: "<<best_score<<std::endl<<std::endl;

    final_classifier = this->f_cascades[best_index];

    //std::cout<<"Detected rectangles position and size: "<<std::endl;
    //printRectDetails(best_detections);
    //std::cout<<std::endl;

   
   return best_detections;
}

//-------------- HELPER FUNCTION --------------

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
