import cv2
import mediapipe as mp

# Inisialisasi
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    # ubah warna ke rgb
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_detection.process(image_rgb)

    # Logika detection
    if result.detections:
        for detection in result.detections:
            # Ambil koordinar kotak wajah
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape

            # Hitung titik tengah wajah (center point)
            center_x = int((bbox.xmin + bbox.width / 2) * iw)
            center_y = int((bbox.ymin + bbox.height / 2) * ih)

            # Gambar kotak
            cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), -1)

            # Output untuk Robot
            if center_x < iw // 3:
                print("Robot: Belok Kiri")
            elif center_x > 2 * (iw // 3):
                print("Robot: Belok Kanan")
            else:
                print('Robot: Maju Lurus')

    cv2.imshow('AI Robot Vision', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
