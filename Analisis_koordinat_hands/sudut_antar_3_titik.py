import cv2
import mediapipe as mp
import math

def calculate_angle(a, b, c):
    import numpy as np
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands() as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x1, y1 = int(hand_landmarks.landmark[6].x * w), int(hand_landmarks.landmark[6].y * h)
                x2, y2 = int(hand_landmarks.landmark[7].x * w), int(hand_landmarks.landmark[7].y * h)
                x3, y3 = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
                
                angle = calculate_angle((x1,y1), (x2, y2), (x3, y3))
                cv2.putText(frame, f"Angle: {int(angle)}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Angle", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()