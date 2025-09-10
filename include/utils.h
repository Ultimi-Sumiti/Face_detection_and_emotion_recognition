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

// Function used to print the details regarding a given vector of rectangles.
void printRectDetails(const std::vector<cv::Rect>& rects);

// Function used to compute a score that tells how a rectangle is defined in a frame.
double calculateBlurScore(const cv::Mat& image, const cv::Rect& roi);

// Function used to compute a rectangles from a file labels.
std::vector<cv::Rect> compute_rectangles(std::string& filename, int img_width, int img_height);

// Function to parse the labels of the positions and emotion from the given textual file.
std::vector<std::vector<float>> parse_labels(const std::string& filename);

// Detection function using the ViolaJones algorithm.
std::vector<cv::Rect> face_detect(std::string& filename);
#endif
