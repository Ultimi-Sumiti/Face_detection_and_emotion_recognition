#include <iostream>
#include <string>

#include "../include/utils.h"

int main(int argc, char* argv[]) {

    std::string input_path{};
    parse_command_line(argc, argv, input_path);

    if (input_path.empty()) {
        std::cerr << "Error in parsing the command line...\n";
        return 1;
    }

    std::cout << input_path << "\n";

    return 0;
}
