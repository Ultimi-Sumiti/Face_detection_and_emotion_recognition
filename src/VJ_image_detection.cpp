#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

void vj_detect( cv::Mat frame, cv::CascadeClassifier f_cascade);

int main(void){

    cv::CascadeClassifier face_cascade;

    // Load the cascades.
    if (!face_cascade.load("../classifiers/haarcascade_frontalface_alt.xml")){

        std::cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    std::string img_path = "../dataset_detection/images/angry (1).jpg";
    cv::Mat img = cv::imread(img_path);
    vj_detect(img, face_cascade);


    return 0;
}


void vj_detect( cv::Mat frame , cv::CascadeClassifier f_cascade){

    cv::Mat frame_gray;
    // Convert into GRAY the frame passed.
    cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
    // Histogram equalization.
    cv::equalizeHist( frame_gray, frame_gray ); 

    // Plotting the image equalized.
    cv::namedWindow("Window");
    cv::imshow( "Window", frame_gray );
    cv::waitKey(0);

    // Detect faces on the frame in gray scale.
    std::vector<cv::Rect> faces;
    f_cascade.detectMultiScale( frame_gray, faces );

    // Folder path in which will be saved the images
    std::string folder_path_cropped_imgs = "../cropped_imgs/";
    // Vector of cropped images and vector of bounding boxes
    std::vector<cv::Mat>  cropped_imgs; 

    for (size_t i = 0; i < faces.size(); i++){

        cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        
        // Cropping the detected faces.
        cv::Mat faceROI = frame( faces[i] );
        cropped_imgs.push_back(faceROI.clone());
        // Saving the cropped images.
        cv::imwrite(folder_path_cropped_imgs + "cut_" + std::to_string(i)+".png", cropped_imgs[i]);        
    }

    // Draw the box over the detection.
    for (size_t i = 0; i < faces.size(); i++){

        cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 255), 4);      
    }
    
    // Show what you got.
    cv::imshow( "Window", frame );
    cv::waitKey(0);

    for (size_t i = 0; i < cropped_imgs.size(); i++){

        std::string window_name = "Window " + std::to_string(i);
        cv::imshow(window_name, cropped_imgs[i]);
    }

    cv::waitKey(0);
}