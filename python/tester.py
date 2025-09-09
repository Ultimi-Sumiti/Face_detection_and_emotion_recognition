import os


# Path of the test directory.
TEST_DIR_PATH = "./dataset_detection/images/"

# Path of the binary (executable) file.
BIN_PATH = "./build/out"


def main():
    # Test directory path.
    print("Test directory:", TEST_DIR_PATH)

    if not os.path.isdir(TEST_DIR_PATH):
        print("Error:", TEST_DIR_PATH, "not found.")
        return

    # Retreive all file path.
    for filename in os.listdir(TEST_DIR_PATH):

        # Image file path.
        filepath = os.path.join(TEST_DIR_PATH, filename)

        # Create the command to be executed and execute it.
        command = f"{BIN_PATH} -i {filepath}"
        os.system(command)


if __name__ == "__main__":
    main()
