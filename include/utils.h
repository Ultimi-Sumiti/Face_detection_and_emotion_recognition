// Alessandro Chinello
#ifndef UTILS_H
#define UTILS_H

#include <string>

#include <vector>

#include <opencv2/core/mat.hpp>

// Help message.
static
const char help_msg[] =
    "Usage:\n"
" ./out --imgsdir <path> --labelsdir <path> --video <path> --webcam\n"
"  Where:\n"
"    --imgsdir   ./inputs/dir ->  process all images in ./input/dir\n"
"    --labelsdir ./labels/dir ->  labels associated to each img in ./inputs/dir (OPTIONAL)\n"
"    --video     ./video.mp4  ->  process video.mp4\n"
"    --webcam                 ->  process wecam (default device=0)";

// Function used to parse the command line arguments.
// Returns 0 on succes, 1 on failure.
int parse_command_line(
    int argc,
    char ** argv,
    std::string & imgs_dir_path,
    std::string & labels_dir_path,
    std::string & video,
    int & webcam
);

// Function used to store all the file names inside 'dir_path' in 'filenames'.
std::vector < std::string > get_all_filenames(const std::string & dir_path);

// Function to parse the labels of the positions and emotion from the given textual file.
std::vector < std::vector < float >> parse_labels(const std::string & filename);

// Function to crop images, save it in a specific folder and returns the paths of saving.
std::vector < std::string > crop_images(
    const cv::Mat & img,
        const std::vector < cv::Rect > & faces,
            const std::string & folder_path
);

// Function to remove imgs/files from a certain vector of paths.
void remove_images(const std::vector < std::string > & cropped_paths);

// Function used to create a fifo file.
// Returns 0 on succes, 1 on failure.
int fifo_creation(const std::string & fifo_name);

// Function used to comnvert emotions string into correspondent int.
std::vector < int > parse_emotions(std::vector < std::string > & emotions);

#endif