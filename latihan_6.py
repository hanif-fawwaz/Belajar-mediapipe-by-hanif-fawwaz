import cv2
import mediapipe as mp
import pandas as pd
import time

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Inisialisasi FaceMesh
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        flip = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        h, w, c = frame.shape

        if result.multi_hand_landmarks:
            for hand_idx, (hand_landmarks, hand_handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
                # tentukan jenis tangan
                hand_label = hand_handedness.classification[0].label
                color = (0, 255, 0) if hand_label == "Right" else (255, 0, 0)

                # Gambar skeleton tangan
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=3))
                
                # Loop semua landmark
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y, z = int(landmark.x * w), int(landmark.y * h), landmark.z

                    # Tambahkan text posisi tiap jari
                    cv2.putText(frame, f"{idx}", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    finger8_y = hand_landmarks.landmark[8].y # telunjuk
                    finger12_y = hand_landmarks.landmark[12].y # jari tengah
                    if finger8_y < finger12_y:
                        cv2.putText(frame, "Telunjuk di atas", (10, 40 + 40*hand_idx),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                    # Tampilkan tabel tangan
                    cv2.putText(frame, hand_label, (10, 80 + 40*hand_idx),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Mediapipe hands Advanced", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()         