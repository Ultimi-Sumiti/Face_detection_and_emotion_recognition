#include <iostream>
#include <string>

#include "../include/utils.h"

int main(int argc, char* argv[]) {

    // Parse command line.
    std::string input_path{}, label_path{};
    parse_command_line(argc, argv, input_path, label_path);

    if (input_path.empty()) {
        std::cerr << "Error in parsing the command line...\n";
        return 1;
    }

    if (label_path.empty())
        std::cerr << "Info: label file not provided, IoU will not be computed.\n";

    // Print args found.
    std::cout << "INPUT FILE PATH " << input_path << "\n";
    if (!label_path.empty())
        std::cout << "LABEL FILE PATH " << label_path << "\n";

    return 0;
}
