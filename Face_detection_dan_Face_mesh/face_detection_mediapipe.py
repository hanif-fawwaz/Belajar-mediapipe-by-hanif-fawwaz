import cv2
import mediapipe as mp

mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Ubah ke RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_detection.process(rgb)

        # Jika wajah terdeteksi
        if result.detections:
            for detection in result.detections:
                mp_draw.draw_detection(frame, detection)

                # Tampilkan nilai confidence
                score = detection.score[0]
                cv2.putText(frame, f"Confidence: {int(score * 100)}%", (10, 40),cv2
                            .FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        cv2.imshow('Face Detection Mediapipe', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

'''
Baris	                                            Penjelasan

mp_face.FaceDetection(model_selection=0)	        Model ringan untuk jarak dekat (0), model jarak jauh (1)
results.detections	                                List hasil deteksi wajah
mp_draw.draw_detection()	                        Menggambar kotak & titik fitur (mata, hidung, dll)
detection.score[0]	                                Confidence wajah terdeteksi
'''