import cv2
import mediapipe as mp

# Constants for visuals and thresholds
EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.6
HEAD_PITCH_THRESHOLD = -40  # Variation seuil (par rapport au pitch de référence)
OVERLAY_ALPHA = 0.6
RECT_COLOR = (20, 20, 20)
LANDMARK_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)
RECT_START = (10, 10)
RECT_END = (260, 190)

# Variable interne pour stocker le pitch initial
pitch_reference = None

def draw_face_landmarks(overlay, face_landmarks):
    essential_connections = frozenset().union(
        mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
        mp.solutions.face_mesh.FACEMESH_LIPS,
        mp.solutions.face_mesh.FACEMESH_LEFT_EYE,
        mp.solutions.face_mesh.FACEMESH_RIGHT_EYE,
        mp.solutions.face_mesh.FACEMESH_NOSE
    )
    mp.solutions.drawing_utils.draw_landmarks(
        overlay, face_landmarks,
        connections=essential_connections,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
            color=LANDMARK_COLOR, thickness=1)
    )

def display_status(frame, overlay, avg_ear, mar, pitch):
    global pitch_reference

    # Initialisation du pitch de référence
    if pitch_reference is None and pitch is not None:
        pitch_reference = pitch

    # Calcul de la variation de pitch
    pitch_variation = pitch - pitch_reference if (pitch_reference is not None and pitch is not None) else 0

    # Déterminer head tilt en fonction de la variation
    head_tilt_detected = pitch_variation < HEAD_PITCH_THRESHOLD

    # EAR status
    fatigue_status_color = (0, 255, 0) if avg_ear > EAR_THRESHOLD else (0, 0, 255)
    fatigue_text = "Normal State" if avg_ear > EAR_THRESHOLD else "Fatigue Alert!"

    # MAR status
    yawning_status_color = (0, 255, 0) if mar < MAR_THRESHOLD else (0, 140, 255)
    yawning_text = "No Yawn" if mar < MAR_THRESHOLD else "Yawning!"

    # Head tilt status
    head_status_color = (0, 255, 0) if not head_tilt_detected else (0, 0, 255)
    head_text = "Head Normal" if not head_tilt_detected else "Head Tilt!"

    # Draw semi-transparent rectangle
    cv2.rectangle(overlay, RECT_START, RECT_END, RECT_COLOR, -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    # Text display
    start_x, start_y = 20, 40
    line_spacing = 30

    cv2.putText(frame, f'EAR: {avg_ear:.2f}', (start_x, start_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, fatigue_text, (start_x, start_y + line_spacing),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, fatigue_status_color, 2)
    cv2.putText(frame, f'MAR: {mar:.2f}', (start_x, start_y + 2 * line_spacing),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, TEXT_COLOR, 2)
    cv2.putText(frame, yawning_text, (start_x, start_y + 3 * line_spacing),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, yawning_status_color, 2)
    cv2.putText(frame, head_text, (start_x, start_y + 4 * line_spacing),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, head_status_color, 2)
