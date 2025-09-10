#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <cstdio>  // popen, pclose
#include "../include/utils.h"
#include "../include/performance_metrics.h"


int main(void){

    std::string img_name = "sad_2";
    // Detect and save the faces in a specific folder.
    std::vector<cv::Rect> faces = face_detect(img_name);

    // Call the python pipeline to classify the faces
    /*FILE* pipe = popen("python3 sender.py", "r"); // "r" to read
    if (!pipe) {
        std::cerr << "Error in opening the pipe" << std::endl;
        return -1;
    }

    std::vector<int> vals;
    int val;
s file
    if (fscanf(pipe, "%d", &val) != 1){
        std::cerr << "Error in opening the pipe" << std::endl;
        pclose(pipe);
        return -1;
    }

    while (fscanf(pipe, "%d", &val) == 1) {  // Read a int in the pipe
        vals.push_back(val);
    }
    // Close the pipe
    pclose(pipe);  

    if (val == 0) {
        std::cerr << "Recivied 0 possible default" << std::endl;
    } else {
        for (size_t i = 0; i < vals.size(); i++)
        {
            std::cout << "Recivied val:" << vals[i] << std::endl;
        }
         
    }
*/
    return 0;
}
