import cv2
from face_mesh_utils import initialize_face_mesh
from eye_aspect_ratio import get_eye_landmarks, calculate_ear
from mouth_aspect_ratio import get_mouth_landmarks, calculate_mar
from head_pose_angle import get_head_pose_angles
from visual_utils import draw_face_landmarks, display_status

def process_frame(frame, face_mesh):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        overlay = frame.copy()
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = frame.shape
            left_eye, right_eye = get_eye_landmarks(face_landmarks, width, height)
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            mouth = get_mouth_landmarks(face_landmarks, width, height)
            mar = calculate_mar(mouth)
            pitch, yaw, roll = get_head_pose_angles(face_landmarks, width, height)
            draw_face_landmarks(overlay, face_landmarks)
            display_status(frame, overlay, avg_ear, mar, pitch)
            return avg_ear, mar, pitch
    return None, None, None

def main():
    face_mesh = initialize_face_mesh()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame from webcam.")
            break
        process_frame(frame, face_mesh)
        cv2.imshow('Fatigue Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
