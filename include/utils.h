#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"


const std::vector<std::string> file_paths= {
"../classifiers/haarcascade_eye_tree_eyeglasses.xml",
"../classifiers/haarcascade_eye.xml",
"../classifiers/haarcascade_frontalface_alt_tree.xml",
"../classifiers/haarcascade_frontalface_alt.xml",
"../classifiers/haarcascade_frontalface_alt2.xml",
"../classifiers/haarcascade_frontalface_default.xml",
"../classifiers/haarcascade_lefteye_2splits.xml",
"../classifiers/haarcascade_profileface.xml",
"../classifiers/haarcascade_righteye_2splits.xml",
};

const std::string image_dir = "../dataset_detection/images/";

const std::string label_dir = "../dataset_detection/labels/";

const std::string image_extension = ".jpg";

const std::string label_extension = ".txt";

// Function used to parse the command line arguments.
void parse_command_line(int argc, char* argv[], std::string& input_path, 
        std::string& label_path);

// Function to compute the IOU (intersection over union) between 2 given boxes.
float compute_IOU(cv::Rect& box1, cv::Rect& box2);

// Function to parse the labels of the positions and emotion from the given textual file.
std::vector<std::vector<float>> parse_labels(const std::string& filename);


void printRectDetails(const std::vector<cv::Rect>& rects);

std::vector<float> get_label_IOUs(std::string& filename, std::vector<cv::Rect>& boxes, int img_width, int img_height);
  

double calculateBlurScore(const cv::Mat& image, const cv::Rect& roi);

std::vector<cv::Rect> compute_rectangles(std::string& filename, int img_width, int img_height);

float compute_MIOU(std::vector<float>& IOUs);


// Detection function using the ViolaJones algorithm.
std::vector<cv::Rect> vj_detect(std::string& filename);
#endif
