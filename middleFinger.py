import cv2
import mediapipe as mp
import os

# MediaPipe ayarları
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Parmak yukarı mı?
def is_finger_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y

# Sadece orta parmak yukarıda mı?
def is_middle_only_up(landmarks):
    middle_up = is_finger_up(landmarks,
                             mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                             mp_hands.HandLandmark.MIDDLE_FINGER_PIP)

    index_down = not is_finger_up(landmarks,
                                  mp_hands.HandLandmark.INDEX_FINGER_TIP,
                                  mp_hands.HandLandmark.INDEX_FINGER_PIP)

    ring_down = not is_finger_up(landmarks,
                                 mp_hands.HandLandmark.RING_FINGER_TIP,
                                 mp_hands.HandLandmark.RING_FINGER_PIP)

    pinky_down = not is_finger_up(landmarks,
                                  mp_hands.HandLandmark.PINKY_TIP,
                                  mp_hands.HandLandmark.PINKY_PIP)

    # Başparmak x ekseninde kontrol edilir (sağ el için)
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_down = thumb_tip.x < thumb_ip.x

    return middle_up and index_down and ring_down and pinky_down and thumb_down

# Kamera başlat
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_middle_only_up(hand_landmarks.landmark):
                cv2.putText(frame, " ORTA PARMAK ALGILANDI!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                print(" Orta parmak algılandı, 2 saniye içinde kapanacak...")

                # Son 5 saniyelik kamera görüntüsü
                for i in range(5):  # 50 * 100ms = 2 saniye
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    cv2.imshow("Orta Parmak Algilama", frame)
                    cv2.waitKey(100)

                cap.release()
                cv2.destroyAllWindows()
                os.system("shutdown /s /t 1")  # Gerçek kapatma
                break
            else:
                cv2.putText(frame, "El Takibi: Devam", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Orta Parmak Algilama", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
