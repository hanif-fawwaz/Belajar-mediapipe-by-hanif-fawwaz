import cv2
import mediapipe as mp
import math
import time
import os

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
os.makedirs("blinks", exist_ok=True)

with mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                h, w, c = frame.shape
                # Landmark iris kiri (474-475), kanan(469-470)
                left_iris = face_landmarks.landmark[474]
                right_iris = face_landmarks.landmark[469]

                xL, yL = int(left_iris.x * w), int(left_iris.y * h)
                xR, yR = int(right_iris.x * w), int(right_iris.y * h)

                cv2.circle(frame, (xL, yL), 3, (255, 0, 0), -1)
                cv2.circle(frame, (xR, yR), 3, (0, 255, 0), -1)

                # Hitung jarak antar pupil
                distance = math.dist((xL, yL), (xR, yR))
                cv2.putText(frame, f"Dist: {int(distance)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 0), 2)
                
                # Deteksi berkedip (jika jarak < ambang tertentu)
                if distance < 90:
                    cv2.putText(frame, "Berkedip!", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imwrite(f"blinks/{int(time.time())}.jpg", frame)
                    
        cv2.imshow("Face Mesh - Blink Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()