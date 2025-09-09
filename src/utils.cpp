#include <iostream>
#include <unistd.h>

#include "../include/utils.h"

void parse_command_line(int argc, char **argv, std::string& input_path) {
    int opt;
    while ((opt = getopt(argc, argv, "i:")) != -1) {
        switch (opt) {
            case 'i':
                input_path = optarg;
                break;
            case '?':
                std::cerr << "Usage: " << argv[0] 
                    << " -i <path>\n"
                    << "  Where:\n"
                    << "    -i is the input image path\n";
                break;
        }
    }
}
