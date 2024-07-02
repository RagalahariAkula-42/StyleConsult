import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from src.gender_classifier.entity.config_entity import EvaluationConfig
from src.gender_classifier.utils.common import read_yaml, create_directories,save_json
import dagshub

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale=1./255,
            rotation_range=30,
            shear_range=0.3,
            zoom_range=0.3
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size,
            batch_size=self.config.params_batch_size,
            seed=self.config.params_seed
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.validation_data,
            class_mode='binary',
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()
        self.log_into_mlflow()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        dagshub.init(repo_owner='RagalahariAkula-42', repo_name='StyleConsult', mlflow=True)

        with mlflow.start_run() as run:
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            mlflow.keras.log_model(self.model, "model", registered_model_name="gender_classifier")
