import cv2
import numpy as np

# Indices utilisés pour estimer la pose de la tête
FACE_LANDMARKS_IDXS = [33, 263, 1, 61, 291, 199]  # Yeux, nez, bouche, menton

def get_head_pose_angles(landmarks, width, height):
    image_points = np.array([
        (landmarks.landmark[idx].x * width, landmarks.landmark[idx].y * height)
        for idx in FACE_LANDMARKS_IDXS
    ], dtype="double")

    # Modèle 3D de la tête (repère approximatif)
    model_points = np.array([
        (0.0, 0.0, 0.0),         # Nez
        (0.0, -330.0, -65.0),    # Menton
        (-225.0, 170.0, -135.0), # Coin oeil gauche
        (225.0, 170.0, -135.0),  # Coin oeil droit
        (-150.0, -150.0, -125.0),# Coin bouche gauche
        (150.0, -150.0, -125.0)  # Coin bouche droit
    ])

    # Paramètres caméra simulés
    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Pas de distorsion optique

    # Résolution de PnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Calcul de la matrice de rotation
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = np.hstack((rotation_mat, translation_vector))

    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    # Retourne les angles pitch, yaw, roll
    pitch, yaw, roll = euler_angles.flatten()

    return pitch, yaw, roll
