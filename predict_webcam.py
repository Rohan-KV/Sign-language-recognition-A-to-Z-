import cv2
import numpy as np
import mediapipe as mp
import joblib

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# ==============================
# INIT MEDIAPIPE
# ==============================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ==============================
# START WEBCAM
# ==============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ❌ DO NOT flip (important)
    # frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ==============================
            # NORMALIZATION (FIXED)
            # ==============================
            base = hand_landmarks.landmark[0]   # wrist
            ref = hand_landmarks.landmark[12]   # middle finger tip

            scale = ((ref.x - base.x)**2 + (ref.y - base.y)**2)**0.5

            data = []
            for lm in hand_landmarks.landmark:
                data.extend([
                    (lm.x - base.x) / scale,
                    (lm.y - base.y) / scale,
                    (lm.z - base.z) / scale
                ])

            # ==============================
            # PREDICTION
            # ==============================
            data = np.array(data).reshape(1, -1)
            data = scaler.transform(data)

            pred = model.predict(data)
            probs = model.predict_proba(data)

            label = label_encoder.inverse_transform(pred)[0]
            confidence = np.max(probs)

            # ==============================
            # DISPLAY
            # ==============================
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Sign Language Recognition", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()