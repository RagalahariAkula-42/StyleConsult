import numpy as np
import cv2
import dlib
from sklearn.cluster import KMeans
import math
import bz2
import os
from src.gender_classifier import logger
from math import floor

def download_file(url, extract_to):
    import urllib
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(extract_to), exist_ok=True)
    # Download the file
    urllib.request.urlretrieve(url, extract_to)

def detect_face_shape(image):
    face_cascade_path = r".\faceshape_req\haarcascade_frontalface_default.xml"
    predictor_path = r".\faceshape_req\shape_predictor_68_face_landmarks.dat"

    # If you haven't downloaded and extracted the predictor file yet:
    if not os.path.exists(predictor_path):
        download_file('https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat', predictor_path)

    if not os.path.exists(face_cascade_path):
        download_file('https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml', face_cascade_path)
    # Create the haar cascade for detecting face
    faceCascade = cv2.CascadeClassifier(face_cascade_path)

    # Create the landmark predictor
    predictor = dlib.shape_predictor(predictor_path)
    original = image.copy()

    # Convert the image to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        logger.info("Error: Invalid number of channels in input image i.e ",image.shape)

    # Apply a Gaussian blur with a 3 x 3 kernel to help remove high frequency noise
    gauss = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gauss,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    logger.info("Found {0} faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        detected_landmarks = predictor(image, dlib_rect).parts()
        landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

        results = original.copy()
        cv2.rectangle(results, (x, y), (x + w, y + h), (0, 255, 0), 2)
        temp = original.copy()
        forehead = temp[y:y + int(0.25 * h), x:x + w]
        rows, cols, bands = forehead.shape
        X = forehead.reshape(rows * cols, bands)

        kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
        y_kmeans = kmeans.fit_predict(X)

        for i in range(rows):
            for j in range(cols):
                if y_kmeans[i * cols + j] == True:
                    forehead[i][j] = [255, 255, 255]
                if y_kmeans[i * cols + j] == False:
                    forehead[i][j] = [0, 0, 0]

        forehead_mid = [int(cols / 2), int(rows / 2)]
        lef = 0
        pixel_value = forehead[forehead_mid[1], forehead_mid[0]]
        for i in range(cols):
            if forehead[forehead_mid[1], forehead_mid[0] - i].all() != pixel_value.all():
                lef = forehead_mid[0] - i
                break
        left = [lef, forehead_mid[1]]
        rig = 0
        for i in range(cols):
            if forehead[forehead_mid[1], forehead_mid[0] + i].all() != pixel_value.all():
                rig = forehead_mid[0] + i
                break
        right = [rig, forehead_mid[1]]

        line1 = np.subtract(right + y, left + x)[0]
        cv2.line(results, tuple(x + left), tuple(y + right), color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 1', tuple(x + left), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, tuple(x + left), 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, tuple(y + right), 5, color=(255, 0, 0), thickness=-1)

        linepointleft = (landmarks[1, 0], landmarks[1, 1])
        linepointright = (landmarks[15, 0], landmarks[15, 1])
        line2 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 2', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        linepointleft = (landmarks[3, 0], landmarks[3, 1])
        linepointright = (landmarks[13, 0], landmarks[13, 1])
        line3 = np.subtract(linepointright, linepointleft)[0]
        cv2.line(results, linepointleft, linepointright, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 3', linepointleft, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointleft, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointright, 5, color=(255, 0, 0), thickness=-1)

        linepointbottom = (landmarks[8, 0], landmarks[8, 1])
        linepointtop = (landmarks[8, 0], y)
        line4 = np.subtract(linepointbottom, linepointtop)[1]
        cv2.line(results, linepointtop, linepointbottom, color=(0, 255, 0), thickness=2)
        cv2.putText(results, ' Line 4', linepointbottom, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.circle(results, linepointtop, 5, color=(255, 0, 0), thickness=-1)
        cv2.circle(results, linepointbottom, 5, color=(255, 0, 0), thickness=-1)

        face_width = np.max([line1, line2, line3])
        face_width_length_similarity = np.std([face_width, line4])
        similarity = np.std([line1, line2, line3])
        ovalsimilarity = np.std([line2, line4])

        angle1 = math.atan2(landmarks[5, 1] - landmarks[3, 1], landmarks[5, 0] - landmarks[3, 0])
        angle2 = math.atan2(landmarks[13, 1] - landmarks[15, 1], landmarks[13, 0] - landmarks[15, 0])
        angle_difference = abs(math.degrees(angle1 - angle2))
        
        for i in range(1):
            if ovalsimilarity <= 30 and similarity <= 40:
                if similarity < 20 and angle_difference > 20:
                    return "Square"
                return "Round"
            elif line1 > line2 and line2 > line3 and line1 > line3 and ovalsimilarity <= 40 and similarity <= 40:
                return "Heart"
            elif line4 > line2 and ovalsimilarity <= 50 and similarity < 50:
                if similarity < 20 and angle_difference > 20:
                    return "Rectangle"
                return "Oval"
            else:
                return "Mixed face shape"
