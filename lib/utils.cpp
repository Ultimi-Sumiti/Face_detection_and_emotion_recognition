#include <iostream>
#include <unistd.h>

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



// FUNCTION MEMBERS
void PerformanceMetrics:: compute_IoU(){

     // Parser for predicted labels
    parser(this->path_pred_labels, this->sugar_p, this->mustard_p, this->power_drill_p);
    // Parser for true labels
    parser(this->path_true_labels, this->sugar_t, this->mustard_t, this->power_drill_t);

    // Initialize 2 vectors for the rectangle of our Algo dectector prediction, and for the true label of the dataset
    // In each cell of each vector we find the coordinates of the top left and bottom right corners
    std::vector<cv::Rect> rect_p(3) , rect_t(3);
    rect_p[0] = cv::Rect(sugar_p[0], sugar_p[1]);

    if(sugar_t[0].x == 0 && sugar_t[0].y == 0 && sugar_t[1].x == 0 && sugar_t[1].y == 0) this->miss[0] = true;
    rect_t[0] = cv::Rect(sugar_t[0], sugar_t[1]);

    rect_p[1] = cv::Rect(mustard_p[0], mustard_p[1]);
    if(mustard_t[0].x == 0 && mustard_t[0].y == 0 && mustard_t[1].x == 0 && mustard_t[1].y == 0) this->miss[1] = true;
    rect_t[1] = cv::Rect(mustard_t[0], mustard_t[1]);

    rect_p[2] = cv::Rect(power_drill_p[0], power_drill_p[1]);
    if(power_drill_t[0].x == 0 && power_drill_t[0].y == 0 && power_drill_t[1].x == 0 && power_drill_t[1].y == 0) this->miss[2] = true;
    rect_t[2] = cv::Rect(power_drill_t[0], power_drill_t[1]);

    // Compute and storage IoU array in the data member: IoU =  Area of overlap / Area of union
    // Define the variable to store the areas of intersection, union and the respective IoU
    double areas_int[3];
    double areas_union[3];

    for (int i = 0; i < rect_p.size(); ++i)
    {
        if (!miss[i])
        {
            cv::Rect intersect = rect_p[i] & rect_t[i];
            areas_int[i] = intersect.area();
            areas_union[i] = rect_p[i].area() + rect_t[i].area() - areas_int[i];

            this->IoU[i] = areas_int[i] / areas_union[i];
        }
    }
}

// Function to parse the labels of the positions and emotion from the given textual file.
std::vector<std::vector<float>> parse_labels(const std::string& filename){

}
