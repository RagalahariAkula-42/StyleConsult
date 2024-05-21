import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras import regularizers, optimizers
from gender_classifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config):
        self.config = config

    def get_base_model(self):
        num_classes = self.config.params_classes
        input_shape = self.config.params_input_shape
        
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                                kernel_regularizer=regularizers.l2(0.001), padding="valid"))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

        self.save_model(path=self.config.base_model_path, model=model)
        model.summary()

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)