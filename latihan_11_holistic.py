'''
MediaPipe Holistic menggabungkan 3 model utama:

Face Mesh → mendeteksi 468 titik di wajah

Pose → 33 titik tubuh

Hands → 21 titik di setiap tangan

Model ini bisa digunakan untuk:

Virtual avatar / body tracking

Kontrol gesture tubuh penuh

Yoga posture analyzer

AI fitness / exercise coach

Full-body motion recognition
'''

import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        # Gambar landmark tangan kiri dan kanan
        # mp_drawing.draw_landmarks(
        #     frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        #     mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        # )
        # mp_drawing.draw_landmarks(
        #     frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        #     mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
        #     mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
        # )

        # gambar landmark pose tubuh
        mp_drawing.draw_landmarks(
            frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
        )

        # Gambar landmark wajah
        # mp_drawing.draw_landmarks(
        #     frame, result.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        #     mp_drawing.DrawingSpec(color=(255, 200, 200), thickness=1, circle_radius=1),
        #     mp_drawing.DrawingSpec(color=(255, 150, 150), thickness=1)
        # )

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark
            nose = lm[0]
            right_wrist = lm[16]

            dist = math.sqrt((nose.x - right_wrist.x)**2 + (nose.y - right_wrist.y)**2)
            if dist < 0.1:
                cv2.putText(frame, "TANGAN DEKAT WAJAH", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Holistic Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()