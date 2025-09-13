############################ Import Modules ##################################
print("INFO: Loading Keras modules...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.saving import load_model
from keras.preprocessing import image
import numpy as np
print("INFO: Modules loaded.") 


############################ Define Params ####################################
# Pre-trained model path.
CHKP_PATH = "../data/trained_models/model_efficientnetB0.keras"

# Input image size. Specify the size of the image that the model takes as input
# (i.e. the input is IMG_SIZE x IMG_SIZE). (Does NOT reflect the 
# actual size of the image).
IMG_SIZE = 224

# Define all the possible classes.
# Note: the order cannot be changed, it was # specified during model training.
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Path where the C++ process stores the cropped faces.
CROPPED_PATH = "../tmp/cropped_imgs/"

# Fifo files paths for inter process communication.
RECEIVE_FIFO = "../tmp/cpp_to_py.fifo"
SEND_FIFO = "../tmp/py_to_cpp.fifo"
##############################################################################


# Returns the class index predicted by the model.
def class_idx(model, img_path):
    # Load and transform image.
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict.
    predictions = model.predict(img, verbose=0)
    
    # Get class index.
    idx = np.argmax(predictions, axis=1)[0]

    return idx


def main():
    print("INFO: Loading pre-trained model...")
    model = load_model(CHKP_PATH)
    print("INFO: Pre-trained model loaded.")

    # Manage inter process communication.
    while True: # Quit when the cpp process send an exit message.

        # Waiting the signal from image detection.
        print("Python: waiting c++ instructions.")
        with open(RECEIVE_FIFO, "r") as fifo:

            # Read the message.
            msg = fifo.readline()
            print(f"[Python] Recived from C++: {msg}")
            #fifo.flush()

            if msg == "exit":
                return
            elif msg == "start":
                pass # Do nothing.
            #else: 
            #    print("PYTHON ERROR! MESSAGE IS", msg) # TODO: Remove this.
            #    return
                
        # Load all images in the directory.
        imgs = [
            os.path.join(CROPPED_PATH, filename) 
            for filename in os.listdir(CROPPED_PATH) if filename != ".gitkeep"
        ]

        # Classify each image and send the results.
        print("Python: sending classes.")
        with open(SEND_FIFO, "w") as fifo:
            for img in imgs:
                idx = class_idx(model, img)
                fifo.write(f"{CLASSES[idx]}\n")
                fifo.flush()
  
if __name__ == "__main__":
    main()
