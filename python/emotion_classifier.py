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

    # Define the two datastets.
    train_ds = train_gen.flow_from_directory(
        directory=TRAIN_DIR,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        target_size=IMG_SHAPE
    )
    test_ds = test_gen.flow_from_directory(
        directory=TEST_DIR,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        target_size=IMG_SHAPE
    )
    return train_ds, test_ds


def main():
    train_ds, test_ds = create_dataset()

    # Load pre-trained model and remove output layer.
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(48, 48, 3) # TODO: must be IMG_SHAPE
    )

    # Add the output layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    # Define the pre-trained model.
    pretrain_model = Model(inputs=base_model.input, outputs=predictions)

    # Plot the model summary.
    pretrain_model.summary()

    # Freeze layers except the last one.
    for layer in base_model.layers[:-2]:
        layer.trainable = False

    # Compile the model.
    pretrain_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fit the pretraining model.
    history_pretrain = pretrain_model.fit(
        train_ds,
        epochs=10,
        validation_data=test_ds,
    )

    # Fine-tuning the model
    for layer in pretrain_model.layers:
        layer.trainable = True
    
    # Compile the model with a lower learning rate
    pretrain_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fit the fine-tuning model
    history_finetune = pretrain_model.fit(
        train_ds,
        epochs=5,
        validation_data=test_ds
    )
    
    # Evaluate the model on validation data
    val_loss, val_accuracy = pretrain_model.evaluate(test_ds, verbose=2)
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    
    # Plot training & validation accuracy values
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_pretrain.history['accuracy'] + history_finetune.history['accuracy'])
    plt.plot(history_pretrain.history['val_accuracy'] + history_finetune.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history_pretrain.history['loss'] + history_finetune.history['loss'])
    plt.plot(history_pretrain.history['val_loss'] + history_finetune.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.show()

    
if __name__ == "__main__":
    main()
