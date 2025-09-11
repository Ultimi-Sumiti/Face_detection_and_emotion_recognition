#include "../include/face_detector.h"

//-------------- MEMBER FUNCTIONS --------------
void FaceDetector:: draw_bbox(cv::Mat frame, std::vector<cv::Rect> faces, const std::vector<std::string> &labels)
{
    // Draw the box over the detection.
    for (size_t i = 0; i < faces.size(); i++)
    {
        cv::Scalar color = cv::Scalar(255, 255, 255); // default = white
        if (this->label_color.find(labels[i]) != this->label_color.end())
        {
            color = this->label_color[labels[i]];
        }

        // Draw the Bounding box
        cv::rectangle(frame, faces[i], color, 4);
        // Draw the corrispective label on the BBox
        cv::putText(frame, labels[i],
                    cv::Point(faces[i].x, faces[i].y - 5), // 5 pixels above the top-left
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5, // font scale
                    color,
                    1,
                    cv::LINE_AA);
    }
}

//######TODO DA CONTROLLLARE ANCHE SE NON LA USIAMO PIU ho cambiato una roba si può anche rimuovere potenzialmente
std::vector<cv::Rect> FaceDetector::vj_detect(cv::Mat frame){

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

std::vector<cv::Rect> FaceDetector::face_detect(cv::Mat& frame){
    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    // Histogram equalization.
    cv::equalizeHist(frame_gray, frame_gray); 

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
    for(int i = 0; i < this->f_cascades.size(); i++){
        // std::cout<<std::endl<< "Testing classifier number: "<<i<<std::endl;
        faces.clear();
        this->f_cascades[i].detectMultiScale(
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


        //std::cout << "Found " << faces.size() << " faces." << std::endl;
        // Loop through each detected face.
        for(int j = 0; j < faces.size(); j++){
            // Print the confidence score for the corresponding face.
            blur_score = calculateBlurScore(frame, faces[j]);
            score =  blur_score * faces[j].area();
            //std::cout << "Face " << j
            //   << " -> Score: " << score <<std::endl;
            // Here we filter the detection: if they're both not defined and small we filter out.
            if((score >= min_score || blur_score > avg_blur) && faces[j].area() > min_area){
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

