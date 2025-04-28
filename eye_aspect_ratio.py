from math import sqrt

def euclidean_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_ear(eye):
    horizontal = euclidean_distance(eye[0], eye[1])
    vertical = euclidean_distance(eye[2], eye[3])
    return vertical / horizontal

def get_eye_landmarks(landmarks, width, height):
    left_eye_indices = [33, 133, 159, 145]
    right_eye_indices = [362, 263, 386, 374]

    left_eye = [(int(landmarks.landmark[i].x * width), int(landmarks.landmark[i].y * height)) for i in left_eye_indices]
    right_eye = [(int(landmarks.landmark[i].x * width), int(landmarks.landmark[i].y * height)) for i in right_eye_indices]

    return left_eye, right_eye
