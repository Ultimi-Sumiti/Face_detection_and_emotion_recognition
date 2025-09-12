#include <iostream>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <dirent.h>
#include <numeric>
#include <sys/stat.h>
#include <filesystem>

#include "../include/utils.h"
#include <filesystem>
#include "../include/performance_metrics.h"

namespace fs = std::filesystem;

void parse_command_line(
        int argc,
        char **argv,
        std::string& imgs_dir_path,
        std::string& labels_dir_path
) {
    int opt;
    while ((opt = getopt(argc, argv, "i:l:")) != -1) {
        switch (opt) {
            case 'i':
                imgs_dir_path = optarg;
                break;
            case 'l':
                labels_dir_path = optarg;
                break;
            case '?':
                std::cerr << "Usage: " << argv[0] 
                    << " -i <path> -l <path>\n"
                    << "  Where:\n"
                    << "    -i is the input image directory path\n"
                    << "    -l is the label directory path (OPTIONAL)\n";
                break;
        }
    }
}

std::vector<std::string> get_all_filenames(const std::string& dir_path) {
    DIR* dir;
    struct dirent* ent;
    std::vector<std::string> filenames;

    // Process all the files insider the directory.
    if ((dir = opendir(dir_path.c_str())) != NULL) {

        while ((ent = readdir (dir)) != NULL) {

            // Get filename.
            std::string file_name = ent->d_name;

            // Skip current and parent.
            if (file_name == "." || file_name == "..") continue;

            if (*(dir_path.end() - 1) == '/')
                filenames.push_back(dir_path + file_name);
            else
                filenames.push_back(dir_path + "/" + file_name);
        }

        closedir(dir); // Close the directory.
    }

    return filenames;
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

std::vector<std::string> crop_images(cv::Mat img, std::vector<cv::Rect> faces, const std::string& folder_path) {
    // Vector of paths to return 
    std::vector<std::string> cropped_paths(faces.size());

    for (size_t i = 0; i < faces.size(); i++){
        // Cropping the detected faces.
        cv::Mat faceROI = img(faces[i]);
        // Saving the cropped images.
        std::string path = folder_path + "cut_" + std::to_string(i) + ".png";
        cv::imwrite(path,faceROI);
        cropped_paths[i] = path;
    }

    return cropped_paths;
}

void remove_images(const std::vector<std::string>& cropped_paths){
    for (const auto &cropped : cropped_paths) {
        try {
            // The remove function returns true if a file was deleted, false otherwise
            fs::remove(cropped);
        } catch (const fs::filesystem_error &e) {
            // This catch block handles errors like permission issues
            std::cerr << "Error deleting file: " << e.what() << std::endl;
            return;
        }
    }
}
    
void fifo_creation(const char *fifo_name){

    if (access(fifo_name, F_OK) != 0){ 
        // If fifo doesn't exist.
        if (mkfifo(fifo_name, 0666) == -1){
             // If creation doesn't work
            std::cerr << "Error in the creation of "
                      << fifo_name << ": "
                      << std::strerror(errno) << std::endl;
        }
        else{
            std::cout << "Creation of the fifo: " << fifo_name << std::endl;
        }
    }else{
        std::cout << "fifo already exist" << std::endl;
    }
}
