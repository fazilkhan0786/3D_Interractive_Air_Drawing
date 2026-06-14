# Air Draw (fixed stroke recording + palm gestures)
# Requirements: pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

all_strokes = []
current_stroke = []

recording = False  

cap = cv2.VideoCapture(0)

def is_pointing(lm):
    """Index finger extended, others folded."""
    idx_tip, idx_pip = lm[8].y, lm[6].y
    mid_tip, mid_pip = lm[12].y, lm[10].y
    ring_tip, ring_pip = lm[16].y, lm[14].y
    pinky_tip, pinky_pip = lm[20].y, lm[18].y
    return (idx_tip < idx_pip and
            mid_tip > mid_pip and
            ring_tip > ring_pip and
            pinky_tip > pinky_pip)

def is_open_palm(lm):
    """All fingers extended."""
    return (lm[8].y < lm[6].y and
            lm[12].y < lm[10].y and
            lm[16].y < lm[14].y and
            lm[20].y < lm[18].y)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0].landmark

        # Palm gesture to start recording
        if not recording and is_open_palm(lm):
            recording = True
            cv2.putText(frame, "Recording Strokes....", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow("Air Draw", frame)
            cv2.waitKey(2000)  # pause for 2s

        elif recording and is_open_palm(lm) and all_strokes:
            # Save strokes
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
            for stroke in all_strokes:
                for i in range(1, len(stroke)):
                    cv2.line(canvas, stroke[i - 1], stroke[i], (0, 0, 0), 2)
            cv2.imwrite("drawing.png", canvas)

            cv2.putText(frame, "Recording Stopped", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.imshow("Air Draw", frame)
            cv2.waitKey(2000)

            # Show saved PNG
            drawing_img = cv2.imread("drawing.png")
            cv2.imshow("Your Drawing", drawing_img)
            cv2.waitKey(0)
            break

        # Stroke recording (only when recording ON + pointing)
        if recording and is_pointing(lm):
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            current_stroke.append((ix, iy))
            cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)
        else:
            if len(current_stroke) > 1:
                all_strokes.append(current_stroke)
            current_stroke = []

    # Draw saved strokes into canvas
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            cv2.line(frame, stroke[i - 1], stroke[i], (255, 0, 0), 2)

    # Draw current stroke into canvas
    for i in range(1, len(current_stroke)):
        cv2.line(frame, current_stroke[i - 1], current_stroke[i], (0, 0, 255), 2)

    cv2.imshow("Air Draw", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit anytime
        break

cap.release()
cv2.destroyAllWindows()
