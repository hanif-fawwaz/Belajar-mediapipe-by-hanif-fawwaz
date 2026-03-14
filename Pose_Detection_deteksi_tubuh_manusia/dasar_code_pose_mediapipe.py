import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,    # False agar realtime tracking
    model_complexity=1,         # 0, 1, 2 (semakin tinggi = jauh lebih akurat)
    enable_segmentation=False,  # True jika ingin segmentasi tubuh
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.pose_landmarks:
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_draw.DrawingSpec(color=(255,0,0), thickness=2,)
            )

        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

'''
Baris	Penjelasan
mp_pose = mp.solutions.pose	        Import modul pose dari MediaPipe
Pose()	                            Membuat objek deteksi pose
static_image_mode=False	            Jika True, deteksi tiap frame (lebih lambat tapi akurat untuk gambar diam)
model_complexity=1	                Nilai 0 = cepat, 2 = sangat akurat tapi berat
enable_segmentation=False	        Jika True, dapat memisahkan tubuh & background
results.pose_landmarks	            Menyimpan hasil landmark (koordinat tubuh)
mp_draw.draw_landmarks()	        Menggambar landmark & koneksi antar titik
'''