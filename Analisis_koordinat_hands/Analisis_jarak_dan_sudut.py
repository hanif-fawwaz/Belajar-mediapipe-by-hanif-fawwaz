import cv2
import mediapipe as mp
import math
import time
import pandas as pd

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# DataFrame untuk menyimpan hasil
df = pd.DataFrame(columns=['Timestamp', 'Distance_index', 'angle_index'])

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        h, w, c = frame.shape

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Ambil koordinat jari jempol (id=4) dan telunjuk (id=8)
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]

                x1, y1 = int(thumb.x  * w), int(thumb. y * h)
                x2, y2 = int(index. x * w), int(index. y * h)

                # Hitung jarak Euclidean
                distance = math.hypot(x2 - x1, y2 - y1)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Jarak: {int(distance)}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                

                # Hitung sudut sendi jari
                # Gunakan 3 titik perghelangan (0), pangkal telunjuk (5), ujung telunjuk (8)
                p0 = hand_landmarks.landmark[0]
                p5 = hand_landmarks.landmark[5]
                p8 = hand_landmarks.landmark[8]

                # Buat vektor a dan b
                Ax, Ay = p0.x - p5.x, p0.y - p5.y
                Bx, By = p8.x - p5.x, p8.y - p5.y

                # Sudut antara vektor a dan b (radian → derajat)
                dot_product = Ax*Bx + Ay*By
                magnitude = math.sqrt(Ax**2 + Ay**2) * math.sqrt(Bx**2 + By**2)
                if magnitude != 0:
                    angle = math.degrees(math.acos(dot_product / magnitude))
                else:
                    angle = 0

                cv2.putText(frame, f"Sudut: {int(angle)}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                # Logika sederhana: jika jari lurus (sudut kecil):
                if angle < 30:
                    cv2.putText(frame, "Telunjuk Lurus!", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # df.loc[len(df)] = [time.time(), distance, angle]

            cv2.imshow("Analisis koordinat (jarak dan sudut)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()

# Simpan Hasil ke csv
# df.to_csv("Analisisi.csv", index=False)
# print("Data analisis tersimpan di Analisis.csv")

'''
Bagian	                            Penjelasan

math.hypot(dx, dy)	                Menghitung jarak 2D antara dua titik
dot_product / magnitude	            Rumus kosinus sudut antar dua vektor
math.degrees(math.acos(...))	    Mengubah hasil ke satuan derajat
angle < 30	                        Kondisi logika: jika jari lurus (karena sudut kecil)
df.to_csv()	                        Menyimpan hasil analisis ke CSV untuk plotting atau ML
'''