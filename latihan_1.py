import cv2
import mediapipe as mp
import time

# Instalisasi utulitas menggambar dan modul hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Buka webcam
cap = cv2.VideoCapture(0)

# Konfigurasi deteksi tangan
hands = mp_hands.Hands(
    static_image_mode = False,
    max_num_hands = 2,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

pTime = 0 # untuk menampilkan FPS

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(" Tidak bisa membaca frame dari webcam")
        break

    # Konversi ke RGB (Mediapipe butuh RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Deteksi tangan
    result = hands.process(image)

    # Kembali ke BGR untuk tampilkan di opencv
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # jika tangan terdeteksi, gambar lanmark-nya
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks)

    
    # Hitung dan tampilkan FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    cv2.putText(image, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()