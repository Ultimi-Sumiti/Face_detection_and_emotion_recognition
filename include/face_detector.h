#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <map>
#include "utils.h"
#include <iostream>


class FaceDetector{

    public:

    //CONSTRUCTOR

        FaceDetector(const std::vector<std::string> &paths){

            this->f_cascades = std::vector<cv::CascadeClassifier>(paths.size());

            for (int i = 0; i < paths.size(); i++){
                if (!f_cascades[i].load(paths[i])){
                    throw std::runtime_error("Error loading face cascade: " + paths[i]);
                }
            }
        }

    //MEMBER FUNCTIONS

        std::vector<cv::CascadeClassifier> get_detectors(){
            return this->f_cascades;
        }

    void draw_bbox(cv::Mat frame, std::vector<cv::Rect> faces, const std::vector<std::string>& labels);
    std::vector<cv::Rect> face_detect(cv::Mat& frame);
    std::vector<cv::Rect> vj_detect(cv::Mat frame); // POTENZIALMENTE SI PUÒ togliere TODO
    


    private:

    // DATA MEMBERS

    std::vector<cv::CascadeClassifier> f_cascades;
    std::vector<std::string> cropped_paths;
    

    std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 0, 255),     // red
        cv::Scalar(0, 255, 255),   // yellow
        cv::Scalar(0, 0, 0),       // black
        cv::Scalar(0, 255, 0),     // green
        cv::Scalar(255, 255, 255), // white
        cv::Scalar(255, 0, 0),     // blue
        cv::Scalar(128, 0, 128),   // purple
    };

    std::map<std::string, cv::Scalar> label_color = {
        {"angry", colors[0]},
        {"disgust", colors[1]},
        {"fear", colors[2]},
        {"happy", colors[3]},
        {"neutral", colors[4]},
        {"sad", colors[5]},
        {"surprise", colors[6]}};
};

// HELPER FUNCTIONS
// Function used to compute a score that tells how a rectangle is defined in a frame.
double calculateBlurScore(const cv::Mat& image, const cv::Rect& roi);
// Function used to print the details regarding a given vector of rectangles.
void printRectDetails(const std::vector<cv::Rect>& rects);
// Function used to compute a rectangles from a file labels.
std::vector<cv::Rect> compute_rectangles(std::string& filename, int img_width, int img_height);





#endif
