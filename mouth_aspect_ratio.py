from math import sqrt

# Indices MediaPipe pour la bouche
# Haut - Bas - Gauche - Droit
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

def euclidean_distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def get_mouth_landmarks(landmarks, width, height):
    top = (int(landmarks.landmark[MOUTH_TOP].x * width), int(landmarks.landmark[MOUTH_TOP].y * height))
    bottom = (int(landmarks.landmark[MOUTH_BOTTOM].x * width), int(landmarks.landmark[MOUTH_BOTTOM].y * height))
    left = (int(landmarks.landmark[MOUTH_LEFT].x * width), int(landmarks.landmark[MOUTH_LEFT].y * height))
    right = (int(landmarks.landmark[MOUTH_RIGHT].x * width), int(landmarks.landmark[MOUTH_RIGHT].y * height))
    
    return top, bottom, left, right

def calculate_mar(mouth):
    vertical = euclidean_distance(mouth[0], mouth[1])  # top-bottom
    horizontal = euclidean_distance(mouth[2], mouth[3])  # left-right
    return vertical / horizontal
