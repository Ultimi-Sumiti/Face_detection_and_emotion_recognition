#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

// Function used to parse the command line arguments.
void parse_command_line(int argc, char* argv[], std::string& input_path, 
        std::string& label_path);

// Function to compute the IOU (intersection over union) between 2 given boxes.
float compute_IOU(cv::Rect& box1, cv::Rect& box2);

// Function to parse the labels of the positions and emotion from the given textual file.
std::vector<std::vector<float>> parse_labels(const std::string& filename);


void printRectDetails(const std::vector<cv::Rect>& rects);

void print_IOU(std::string& filename, std::vector<cv::Rect>& boxes);
#endif
