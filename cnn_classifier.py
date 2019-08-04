import numpy as np

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *

from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt


def data_preparation():
    #################
    # Prepare data
    #################

    train_path = 'dataset/train'
    validate_path = 'dataset/validate'
    test_path = 'dataset/test'

    target_size = (224, 224)
    classes = ['manutd', 'chelsea']

    image_gen = ImageDataGenerator()

    train_batches = image_gen.flow_from_directory(
        directory=train_path,
        target_size=target_size,
        classes=classes,
        batch_size=10,
    )

    validate_batches = image_gen.flow_from_directory(
        directory=validate_path,
        target_size=target_size,
        classes=classes,
        batch_size=10
    )

    test_batches = image_gen.flow_from_directory(
        directory=test_path,
        target_size=target_size,
        classes=classes,
        batch_size=10
    )

    return train_batches, validate_batches, test_batches
    

def train_model(model, train_batches, validate_batches):
    #################
    # Train model
    #################
    
    # Set compile parameters
    learning_rate = 0.0001
    loss_function = 'categorical_crossentropy'
    validation_metrics = ['accuracy']
    optimizer = Adam(lr=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=validation_metrics
    )

    # Set fit parameters
    epochs = 30
    model.fit_generator(
        generator=train_batches,
        steps_per_epoch=4,
        validation_data=validate_batches,
        validation_steps=4,
        epochs=epochs,
        verbose=2
    )

def predict_model(model, test_batches):
    #################
    # Predict model
    #################

    test_imgs, test_labels = next(test_batches)
    test_labels = test_labels[:,0]
    print(test_labels)

    predictions = model.predict_generator(
        generator=test_batches,
        steps=1,
        verbose=1
    )

    print(predictions)

def build_custom_model_1():
    #################
    # Build model 1
    #################
    model = Sequential()
    
    layer_1 = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(224, 224, 3)
    )
    layer_2 = Flatten()
    output_layer = Dense(
        units=2,
        activation='softmax'
    )
    
    layers = [layer_1, layer_2, output_layer]

    for layer in layers:
        model.add(layer)

    model.summary()
    
    return model

def build_vgg16_model():
    #################
    # Load vgg16 model
    #################
    vgg16_model = keras.applications.vgg16.VGG16()
    #vgg16_model.summary()
    vgg16_model.layers.pop()

    #print(type(vgg16_model))
    model = Sequential()
    for layers in vgg16_model.layers:
        model.add(layers)

    for layer in model.layers:
        layer.trainable = False

    new_output_layer = Dense(
        units=2,
        activation='softmax'
    )
    model.add(new_output_layer)
    model.summary()
    
    return model


if __name__ == '__main__':
    train_data, validate_data, test_data = data_preparation()

    model_1 = build_custom_model_1()
    train_model(model_1, train_data, validate_data)
    predict_model(model_1, test_data)
    
    model_2 = build_vgg16_model()
    train_model(model_2, train_data, validate_data)
    predict_model(model_2, test_data)
