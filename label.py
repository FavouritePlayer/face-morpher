"""
Labeling task
Too time consuming to do a cv thing, just import standard libraries
pip install dlib, use 
shape_predictor_68_face_landmarks.dat

use opencv to analyze images
"""
import dlib
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

"""
According to spec, detections are
0–16: jawline
17–21: right eyebrow
22–26: left eyebrow
27–35: nose
36–41: right eye
42–47: left eye
48–67: mouth outline/lips
"""

def label_faces(image_path):
    img = cv2.imread(image_path)
    #Now use the model to get points on face

    #according to spec, dlib uses rgb
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #find the face
    #can return multiple? If multiple people in an image, find points for both and crop to show it.
    faces = detector(rgb_img)

    #All points in one image
    all_points = []

    for face in faces:
        #Predict 68 facial landmarks
        shape = predictor(rgb_img, face)

        #Convert dlib shape object into numpy array
        points = np.array([
            [point.x, point.y]
            for point in shape.parts()
        ])
        #Need forehead points ASAP

        #Use eye-to-brow distance as a face-relative "unit" so forehead points scale with
        #face size/proportions instead of being sensitive to how flat or arched the brows are
        eye_y = points[36:48, 1].mean()
        brow_y = points[17:27, 1].mean()
        unit = eye_y - brow_y
        third_eye = [points[27][0], brow_y - 1.5 * unit]
        left_temple = [points[19][0], brow_y - unit]
        right_temple = [points[24][0], brow_y - unit]
        points = np.vstack([points, [left_temple, right_temple, third_eye]]).astype(int)

        all_points.append(points)

    return all_points