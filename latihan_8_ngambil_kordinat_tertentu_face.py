import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    
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
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                nose = face_landmarks.landmark[1]

                x_left, y_left = int(left_eye.x * w), int(left_eye.y * h)
                x_right, y_right = int(right_eye.x * w), int(right_eye.y * h)
                x_nose, y_nose = int(nose.x * w), int(nose.y * h)

                # Hitung jarak antar titik wajah (contoh: Antar Mata)
                dist_eyes = math.sqrt((x_right - x_left)**2 + (y_right - y_left)**2)
                cv2.putText(frame, f"Eye Dist: {int(dist_eyes)}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0, 255, 255),2)
                
                # Logika jika wajah mendekat
                if dist_eyes > 80:
                    text, color = "wajah mendekat", (0, 0, 255)
                else:
                    text, color = "wajah normal", (0, 255, 0)
                
                cv2.putText(frame, text, (30, 200),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.circle(frame, (x_left, y_left), 3, (255, 0, 0), -1)
                cv2.circle(frame, (x_right, y_right), 3, (255, 0, 0), -1)
                # cv2.circle(frame, (x_nose, y_nose), 3, (0, 0, 255), -1)

        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

'''
refine_landmarks=True	        Mengaktifkan landmark iris (mata) untuk akurasi tinggi
FACEMESH_TESSELATION	        Menampilkan seluruh jaringan mesh wajah
DrawingSpec	                    Mengatur warna, ketebalan, dan ukuran titik

Koordinat
Mata kiri → id = 33

Mata kanan → id = 263

Ujung hidung → id = 1

Mulut atas → id = 13

Mulut bawah → id = 14
'''