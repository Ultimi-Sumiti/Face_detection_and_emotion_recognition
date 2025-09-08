#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>

void detectAndDisplay( cv::Mat frame, cv::CascadeClassifier f_cascade);

int main(void){

    cv::CascadeClassifier face_cascade;

    //-- 1. Load the cascades
    if (!face_cascade.load("../classifiers/haarcascade_frontalface_alt.xml"))
    {
        std::cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    std::string img_path = "../dataset_detection/images/disgust (2).jpg";
    cv::Mat img = cv::imread(img_path);
    detectAndDisplay(img, face_cascade);


    return 0;
}


void detectAndDisplay( cv::Mat frame , cv::CascadeClassifier f_cascade)
{
    cv::Mat frame_gray;
    cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
    cv::equalizeHist( frame_gray, frame_gray ); // Histogram equalization
    cv::namedWindow("Window");
    cv::imshow( "Window", frame_gray );
    cv::waitKey(0);

    //-- Detect faces
    std::vector<cv::Rect> faces;
    f_cascade.detectMultiScale( frame_gray, faces );


    for ( size_t i = 0; i < faces.size(); i++ )
    {
        cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        cv::Rect face_rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
        cv::rectangle(frame, face_rect, cv::Scalar(255, 0, 255), 4);  // colore e spessore

        cv::Mat faceROI = frame_gray( faces[i] );
    }
    //-- Show what you got
    cv::imshow( "Window", frame );
    cv::waitKey(0);
}