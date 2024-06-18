# Description: This file contains the code for the Flask API that will be used to train the model. 
# The API will accept a JSON object containing the model configuration and hyperparameters, train 
# the model on the CIFAR-10 dataset, and return the test accuracy and loss of the model.
# 
# Flask Imports
from flask import Flask, request, jsonify
from flask_cors import CORS


#preprocessing imports
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#model building and training imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#visualization imports
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app, resources = {r"/*": {"origins": "http://localhost:5173"}})

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    layers = data['layers']
    units = data['units']
    epochs = data['epochs']
    batch_size = data['batch_size']
    optimizer = data['optimizer']

    #Loading CIFAR-10 in and splitting dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    #Normalizing pixel values
    x_train, x_test = x_train/255.0, x_test/255.0

    #Mean subtraction
    mean = np.mean(x_train, axis = (0, 1, 2) )

    #Mean subtraction
    mean = np.mean(x_train, axis = (0, 1, 2) )
    x_train = x_train - mean
    x_test = x_test - mean

    #Extracting validation set
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, stratify = y_train)

    #Data augmentation via shifts and horizontal flips

    datagen = ImageDataGenerator(
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        horizontal_flip = True
    )

    datagen.fit(x_train)

    train_generator = datagen.flow(x_train, y_train, batch_size = 64)
    val_generator = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(64)
    test_generator = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

    # Build model
    model = Sequential()
    model.add(Conv2D(units[0], (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))
    for i in range(1, layers):
        model.add(Conv2D(units[i], (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units[-1], activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    # model = Sequential([
    #     Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(64, (3, 3), activation='relu'),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(128, (3, 3), activation='relu'),
    #     Flatten(),
    #     Dense(128, activation='relu'),
    #     Dropout(0.5),
    #     Dense(10, activation='softmax')
    # ])

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator
    )
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator)
    response = {'test_accuracy': test_accuracy, 'test_loss': test_loss}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port = 5000)