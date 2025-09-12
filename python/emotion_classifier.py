print("INFO: Loading Keras modules...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.saving import load_model
from keras.preprocessing import image
import numpy as np
print("INFO: Modules loaded.") 

### PARAMETERS ###
CHKP_PATH = "../python/model_efficientnetB0.keras"
IMG_SIZE = 224
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
IMGS_DIR = "../cropped_imgs/"


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

    while True:
        print("Python: waiting c++ instructions.")
        # Waiting the signal from image detection.
        with open("cpp_to_py.fifo", "r") as fifo:
            msg = fifo.readline().strip()
            print(f"[Python] Recived from C++: {msg}")
            fifo.flush()
            # If nothing is detected the program terminates
            if msg == "exit":
                return -1
            if msg == "continue":
                continue
                
        # Load all images in the directory.
        imgs = [
            os.path.join(IMGS_DIR, filename) 
            for filename in os.listdir(IMGS_DIR) if filename != ".gitkeep"
        ]


        print("Python: sending classes.")
        # Classify each image.
        with open("py_to_cpp.fifo", "w") as fifo:
            for img in imgs:
                idx = class_idx(model, img)
                fifo.write(f"{CLASSES[idx]}\n")
                fifo.flush()
  
if __name__ == "__main__":
    main()
