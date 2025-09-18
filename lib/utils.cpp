#include "../include/utils.h"

#include <iostream>

#include <unistd.h>

#include <fstream>

#include <dirent.h>

#include <sys/stat.h>

#include <filesystem>

#include <opencv2/imgcodecs.hpp>

#include <getopt.h>

#include "../include/face_detector.h"

namespace fs = std::filesystem;

// This fucntion is used to parse command line in input.
int parse_command_line(
    int argc,
    char ** argv,
    std::string & imgs_dir_path,
    std::string & labels_dir_path,
    std::string & video,
    int & webcam
) {
    while (1) {
        static struct option long_options[] = {
            {
                "imgsdir", required_argument, 0, 1
            },
            {
                "labelsdir", required_argument, 0, 2
            },
            {
                "webcam", no_argument, 0, 3
            },
            {
                "video",required_argument, 0, 4
            },
            {
                0,0,0,0
            }
        };

        int c = getopt_long(argc, argv, "", long_options, NULL);

        if (c == -1)
            break;

        switch (c) {
        case 1:
            imgs_dir_path = optarg;
            break;

        case 2:
            labels_dir_path = optarg;
            break;

        case 3:
            webcam = 1;
            break;

        case 4:
            video = optarg;
            break;

        case '?':
            return 1;
            break;

        default:
            return 1;
        }
    }

    return 0;
}

// This function returns all file names given the directory path.
std::vector < std::string > get_all_filenames(const std::string & dir_path) {
    DIR * dir;
    struct dirent * ent;
    std::vector < std::string > filenames;

    // Process all the files insider the directory.
    if ((dir = opendir(dir_path.c_str())) != NULL) {

        while ((ent = readdir(dir)) != NULL) {

            // Get filename.
            std::string file_name = ent -> d_name;

            // Skip current and parent.
            if (file_name == "." || file_name == "..") continue;

            if ( * (dir_path.end() - 1) == '/')
                filenames.push_back(dir_path + file_name);
            else
                filenames.push_back(dir_path + "/" + file_name);
        }

        closedir(dir); // Close the directory.
    }

    return filenames;
}

// This function is used to parse label rectangle positions.
std::vector < std::vector < float >> parse_labels(const std::string & filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector < std::vector < float >> faces;
    int count = 0;
    int line_count = 0;
    std::vector < float > current_face;

    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string word;

        while (ss >> word && count < 5) {
            current_face.push_back(std::stof(word));
            count++;
        }

        if (!current_face.empty()) {
            faces.push_back(current_face);
            current_face.clear();
        }

        count = 0;
        line_count++;
    }

    return faces;
}

// This function is used to crop images from the original image and the corresponding rectangles.
std::vector < std::string > crop_images(
    const cv::Mat & img,
        const std::vector < cv::Rect > & faces,
            const std::string & folder_path
) {
    // Vector of paths to return 
    std::vector < std::string > cropped_paths(faces.size());

    for (size_t i = 0; i < faces.size(); i++) {
        // Cropping the detected faces.
        cv::Mat faceROI = img(faces[i]);
        // Saving the cropped images.
        std::string path = folder_path + "cut_" + std::to_string(i) + ".png";
        cv::imwrite(path, faceROI);
        cropped_paths[i] = path;
    }

    return cropped_paths;
}

// This function is used to remove images given their complete path.
void remove_images(const std::vector < std::string > & cropped_paths) {
    for (const std::string & cropped: cropped_paths) {
        try {
            // The remove function returns true if a file was deleted, false otherwise
            fs::remove(cropped);
        } catch (const fs::filesystem_error & e) {
            // This catch block handles errors like permission issues
            std::cerr << "Error deleting file: " << e.what() << std::endl;
            return;
        }
    }
}

// This function is used to create a fifo.
int fifo_creation(const std::string & fifo_name) {
    // Check if it already exists.
    if (!access(fifo_name.c_str(), F_OK)) return 0;
    // Try to create the fifo.
    return mkfifo(fifo_name.c_str(), 0666);
}

// This function is used to compute emotions from the string description.
std::vector < int > parse_emotions(std::vector < std::string > & emotions) {
    std::vector < int > emotions_val;
    for (int i = 0; i < emotions.size(); i++) {
        std::stringstream ss(emotions[i]);
        std::string word;
        if (ss >> word) {
            if (word == "angry") {
                emotions_val.push_back(0);
            } else if (word == "disgust") {
                emotions_val.push_back(1);
            } else if (word == "fear") {
                emotions_val.push_back(2);
            } else if (word == "happy") {
                emotions_val.push_back(3);
            } else if (word == "sad") {
                emotions_val.push_back(4);
            } else if (word == "surprise") {
                emotions_val.push_back(5);
            } else if (word == "neutral") {
                emotions_val.push_back(6);
            }
        } else {
            std::cerr << "Emotion not detected!" << std::endl;
        }
    }
    return emotions_val;
}