#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>

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
float compute_IOU(cv::rectangle& box1, cv::rectangle& box2){
    // Define the variable to store the areas of intersection, union and the respective IoU.
    double areas_int;
    double areas_union;

    // Compute intersection union of boxes. 
    cv::Rect intersect = box1 & box2;
    areas_int = intersect.area();
    areas_union = box1.area() + box2.area() - areas_int[i];

    // Compute and return the IoU.
    IoU = areas_int / areas_union;
    return IoU;
}


// Function to parse the labels of the positions and emotion from the given textual file.
std::vector<std::vector<float>> parse_labels(const std::string& filename){
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<float>> faces;
    int count = 0;
    int line_count = 0;

    while(getline(file, line)){
        std::stringstream ss(line);
        std::string word;
        
        while(ss >> word && count < 4){
            faces[line_count][count] = std::stof(word);
            count++;
        }

        count = 0;
        line_count++;
    }

    return faces;
}

int main(){
    std::vector<std::vector<float>> faces = parse_labels("../labels/angry_1.txt")
    std::cout<<faces[0][0]<<std::endl;
    std::cout<<faces[0][1]<<std::endl;
    std::cout<<faces[0][2]<<std::endl;
    std::cout<<faces[0][3]<<std::endl;
}