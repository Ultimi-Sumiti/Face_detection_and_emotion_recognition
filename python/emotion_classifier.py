import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.saving import load_model
from keras.preprocessing import image
import numpy as np

### PARAMETERS ###
CHKP_PATH = "./model.keras"
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

    model = load_model(CHKP_PATH)

    imgs = [
        os.path.join(IMGS_DIR, filename) 
        for filename in os.listdir(IMGS_DIR) if filename != ".gitkeep"
    ]

    for img in imgs:
        idx = class_idx(model, img)
        print(idx)


if __name__ == "__main__":
    main()
