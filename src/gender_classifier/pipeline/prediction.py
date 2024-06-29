import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import ImageFile
from faceshape import detect_face_shape
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load the pre-trained model
        model = load_model(os.path.join("model", "model.h5"))
        
        # Preprocess the image for gender prediction
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(250, 250))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        # Predict gender
        gender_prob = model.predict(test_image)[0][0]
        # Preprocess the image for face shape detection
        input_image = image.load_img(imagename, target_size=(500, 500))
        input_image = image.img_to_array(input_image)
        input_image = input_image.astype(np.uint8)
        
        # Detect face shape
        face_shape = detect_face_shape(input_image)
        
        # Determine gender prediction result
        gender_pred = 'Male' if gender_prob >= 0.5 else 'Female'
        
        # Mapping face shape and gender to recommendations
        recommendations = {
            "Female": {
                "Round": {
                    "styling_tips": "Angular or geometric earrings and necklaces can help create contrast.Hairstyles with volume on top or side-swept bangs can balance the roundness."
                },
                "Square": {
                    "styling_tips": "Choose round or oval-shaped earrings and necklaces to contrast with the angular jaw. Soft curls or waves and side-swept bangs soften the face shape."
                },
                "Heart": {
                    "styling_tips": "Statement earrings and necklaces draw attention away from the forehead. Hairstyles with volume at the chin or soft waves help balance the face shape."
                },
                "Rectangle": {
                    "styling_tips": "Statement earrings and necklaces draw attention to the collarbone area. Hairstyles with waves or curls add volume and width."
                },
                "Oval": {
                    "styling_tips": "Experiment with various glasses shapes, from round to rectangular. Versatile hairstyles like layers or a medium-length bob complement the balanced proportions."
                },
                "Mixed face shape":{
                    "styling_tips": "Opt for statement earrings or necklaces that draw attention away from specific facial features and add a focal point to outfits. Try versatile hairstyles like layers, soft waves, or updos that can be adapted to enhance different face shapes and features."
                }
            },
            "Male": {
                "Round": {
                    "styling_tips": "Opt for square or rectangular glasses frames to add definition and angles. Hairstyles with height on top, like a faux hawk or pompadour, can elongate the face."
                },
                "Square": {
                    "styling_tips": "Round or oval glasses frames soften angular features. Hairstyles with soft layers or a side part help to soften the jawline."
                },
                "Heart": {
                    "styling_tips": "Bottom-heavy glasses frames or aviators balance the broad forehead. Hairstyles with volume at the chin or textured layers soften the forehead."
                },
                "Rectangle": {
                    "styling_tips": "Wide or round glasses frames add width to balance the face. Hairstyles with layers or side-swept styles help create the illusion of width."
                },
                "Oval": {
                    "styling_tips": "Almost any glasses shape works well, but try aviators or rectangular frames for a classic look. Hairstyles can vary, but avoid excessive volume on top."
                },
                "Mixed face shape":{
                    "styling_tips": "Consider wearing classic watches or subtle bracelets to complement outfits without overwhelming facial features. Experiment with hairstyles that can be tailored to complement different face shapes, such as textured cuts or styles with side parts to add dimension."
                }
            }
        }
        
        # Generate the result
        if face_shape in recommendations[gender_pred]:
            result = [{
                "image": [{
                    "gender": gender_pred,
                    "face shape": face_shape,
                    "styling tips": recommendations[gender_pred][face_shape]["styling_tips"],
                }]
            }]
        else:
            result = [{
                "image": face_shape
            }]
        
        return result