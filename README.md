# StyleConsult

StyleConsult is an AI-powered application that provides personalized styling tips based on gender and face shape analysis using computer vision techniques. This project integrates Dagshub and MLflow for enhanced collaboration and model tracking, with a CI/CD pipeline for continuous integration and continuous deployment.

## Overview

StyleConsult utilizes deep learning models to determine the gender of a person from their photograph and employs facial recognition techniques to identify their face shape. Based on these analyses, the application recommends styling tips tailored to the user's gender and face shape.

## Features

- **Gender Determination**: Uses Convolutional Neural Networks (CNN) implemented with TensorFlow to classify the gender of the user.
- **Face Shape Recognition**: Utilizes dlib and OpenCV to detect and classify the user's face shape based on facial landmarks.
- **Personalized Styling Tips**: Provides fashion and grooming advice specific to the user's identified gender and face shape.
- **Model Tracking with MLflow**: Tracks experiments, parameters, and metrics with MLflow to monitor model performance and facilitate reproducibility.
- **Collaboration with Dagshub**: Utilizes Dagshub for version control, collaboration, and continuous integration/deployment (CI/CD) of machine learning models.
- **CI/CD Pipeline**: Implements a CI/CD pipeline for automated testing, building, and deployment of the application.

## Technologies Used

- **Python**: Programming language used for backend development.
- **TensorFlow**: Deep learning framework for gender classification.
- **dlib**: Library for face shape detection using facial landmarks.
- **OpenCV**: Computer vision library for image processing tasks.
- **Flask**: Web framework used for creating the application backend.
- **MLflow**: Open-source platform for managing the end-to-end machine learning lifecycle.
- **Dagshub**: Git-based platform for collaborative data science and machine learning.

## Installation

To run StyleConsult locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/StyleConsult.git
   cd StyleConsult
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

4. Access the application in your web browser at `http://localhost:8080`.

## Usage

1. Upload a photograph containing a clear view of the face.
2. The application will analyze the image to determine the gender and identify the face shape.
3. Receive personalized styling tips based on the analysis results.

## CI/CD Pipeline

The CI/CD pipeline automates the following processes:

- **Continuous Integration**: Runs automated tests and checks on every code push to ensure code quality and functionality.
- **Continuous Deployment**: Automatically builds and deploys new versions of the application to production or staging environments based on predefined triggers or schedules.
