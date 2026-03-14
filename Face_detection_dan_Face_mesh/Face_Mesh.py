import cv2
import mediapipe as mp

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
                mp_draw.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        cv2.imshow("Face Mesh", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

'''
refine_landmarks=True	        Mengaktifkan landmark iris (mata) untuk akurasi tinggi
FACEMESH_TESSELATION	        Menampilkan seluruh jaringan mesh wajah
DrawingSpec	                    Mengatur warna, ketebalan, dan ukuran titik
'''