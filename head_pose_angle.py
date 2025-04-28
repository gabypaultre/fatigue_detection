import cv2
import numpy as np

FACE_LANDMARKS_IDXS = {
    'nose_tip': 1,
    'chin': 152,
    'left_eye_outer': 33,
    'right_eye_outer': 263,
    'left_mouth': 61,
    'right_mouth': 291
}

def get_head_pose_angles(landmarks, width, height):
    image_points = np.array([
        (landmarks.landmark[FACE_LANDMARKS_IDXS['nose_tip']].x * width,
         landmarks.landmark[FACE_LANDMARKS_IDXS['nose_tip']].y * height),
        (landmarks.landmark[FACE_LANDMARKS_IDXS['chin']].x * width,
         landmarks.landmark[FACE_LANDMARKS_IDXS['chin']].y * height),
        (landmarks.landmark[FACE_LANDMARKS_IDXS['left_eye_outer']].x * width,
         landmarks.landmark[FACE_LANDMARKS_IDXS['left_eye_outer']].y * height),
        (landmarks.landmark[FACE_LANDMARKS_IDXS['right_eye_outer']].x * width,
         landmarks.landmark[FACE_LANDMARKS_IDXS['right_eye_outer']].y * height),
        (landmarks.landmark[FACE_LANDMARKS_IDXS['left_mouth']].x * width,
         landmarks.landmark[FACE_LANDMARKS_IDXS['left_mouth']].y * height),
        (landmarks.landmark[FACE_LANDMARKS_IDXS['right_mouth']].x * width,
         landmarks.landmark[FACE_LANDMARKS_IDXS['right_mouth']].y * height)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ])

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None, None

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = np.hstack((rotation_mat, translation_vector))

    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()

    return pitch, yaw, roll
