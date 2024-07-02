import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from src.gender_classifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            rotation_range=30,
            shear_range=0.3,
            zoom_range=0.3
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size,
            batch_size=self.config.params_batch_size
        )

        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            class_mode='binary',
            **dataflow_kwargs
        )

        self.train_generator =  train_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            class_mode='binary',
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(str(path)) 

    def train(self):
        
        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.config.params_steps_per_epoch,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator,
            validation_steps=self.config.params_validation_steps
        )

        self.config.trained_model_path.parent.mkdir(parents=True, exist_ok=True)

        self.save_model(self.config.trained_model_path, self.model)
        self.save_model(self.config.trained_model_path_in_use, self.model)