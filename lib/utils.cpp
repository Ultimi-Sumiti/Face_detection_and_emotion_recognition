#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include "../include/utils.h"

const int IMG_WIDTH = 1280;
const int IMG_HEIGHT = 853;

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

void print_IOU(std::string& filename, std::vector<cv::Rect>& boxes){
    std::vector<std::vector<float>> labels = parse_labels(filename);
    int x, y, width, height;
    std::vector<cv::Rect> rects_label;

    for (const auto& face: labels){
        x = (face[1] - face[3]) * IMG_WIDTH;
        y = (face[2] - face[4]) * IMG_HEIGHT;
        width = face[3] * IMG_WIDTH;
        height = face[4] * IMG_HEIGHT;
        rects_label.push_back(cv::Rect(x, y, width, height));
    }
    float current_IoU;
    std::vector<float> IOUs(boxes.size(), 0.0f); 

    for(int i = 0; i < rects_label.size(); i++){
        for(int j = 0; j < boxes.size(); j++){
            current_IoU = compute_IOU(rects_label[i], boxes[j]);
            if(current_IoU > IOUs[i]){
                IOUs[i] = current_IoU;
            }
        }
        std::cout<<"IOU is : "<<IOUs[i]<<std::endl;
    }
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