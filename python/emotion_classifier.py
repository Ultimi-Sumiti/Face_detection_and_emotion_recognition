import os

os.environ["KERAS_BACKEND"] = "torch"
import keras

from keras.applications import VGG16
from keras import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

NUM_CLASSES = 7
BATCH_SIZE = 64

TRAIN_DIR = "../dataset_classification/train/"
TEST_DIR = "../dataset_classification/test/"
IMG_SHAPE = (48, 48)


def create_dataset():
    # Define the two generators (train and test).
    train_gen = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_gen = ImageDataGenerator(rescale = 1./255)

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
