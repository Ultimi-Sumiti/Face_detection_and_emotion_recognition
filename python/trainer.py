import os

# Define environment variables.
os.environ["KERAS_BACKEND"] = "torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.applications import EfficientNetV2B0, ConvNeXtTiny
#from keras.applications.efficientnet import preprocess_input
from keras.applications.convnext import preprocess_input

from keras import Model, Sequential, Input
from keras.layers import Dense, GlobalAveragePooling2D, RandomZoom, RandomFlip, RandomRotation, BatchNormalization
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


### PARAMETERS ###
EPOCHS_PRETRAIN = 0
EPOCHS_FINETUNE = 0

BATCH_SIZE = 32
NUM_CLASSES = 7

IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 224, 224, 3

TRAIN_DIR = "../dataset_classification/train/"
TEST_DIR = "../dataset_classification/test/"

CHKP_PATH = "./model.keras"


def create_datasets():
    train_ds, val_ds = image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        validation_split=0.2,
        seed=123,
        subset="both"
    )

    test_ds = image_dataset_from_directory(
        directory=TEST_DIR,
        labels='inferred',
        label_mode='categorical',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH)
    )

    return train_ds, val_ds, test_ds


def evalutate_model(model, test_ds, history_pretrain, history_finetune):
    # Evaluate the model on validation data.
    val_loss, val_accuracy = model.evaluate(test_ds, verbose=2)
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    
    # Plot training & validation accuracy values.
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


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def build_model(data_augmentation):
    # Load pre-trained model and remove output layer.
    base_model = ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    )

    # Freeze base model.
    base_model.trainable = False

    # Define the model.
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    # Define the pre-trained model.
    return Model(inputs, outputs)


def main():
    train_ds, val_ds, test_ds = create_datasets()

    class_names = train_ds.class_names
    print("Class names:", class_names)

    # Define transformations.
    data_augmentation = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ])

    # Define the model.
    model = build_model(data_augmentation)
    model.summary()


    #base_model = model.get_layer("efficientnetv2-b0")
    #for layer in base_model.layers[-121:]:
    #    print(layer.name)
    #    #if not isinstance(layer, BatchNormalization):
    #return

    # Define optimizer, loss and metrics used.
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks.
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            filepath=CHKP_PATH,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        )
    ]

    # Fit the pretraining model.
    history_pretrain = model.fit(
        train_ds,
        epochs=EPOCHS_PRETRAIN,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # Fine-tuning the model.
    #for layer in model.layers:
    #    layer.trainable = True

    base_model = model.get_layer("convnext_tiny")
    for layer in base_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    # Compile the model with a lower learning rate.
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fit the fine-tuning model.
    history_finetune = model.fit(
        train_ds,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # Plot accuracy/loss.
    #evalutate_model(model, test_ds, history_pretrain, history_finetune)

    # Plot confusion matrix.
    Y_pred = model.predict(test_ds)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)
    plot_confusion_matrix(y_true, y_pred, class_names)

    # Print classification report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
if __name__ == "__main__":
    main()
