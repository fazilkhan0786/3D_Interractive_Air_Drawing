#Iteration-2
# Air Draw v2
#using palm gesture to stop recording and close the camera and save the strokes as a png image

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

all_strokes = []
current_stroke = []
drawing = False
recorded = False

cap = cv2.VideoCapture(0)

def is_pointing(lm):
    """
    Return True if index finger is extended and others are folded.
    Simple rule: index_tip higher (y smaller) than PIP, while
    middle, ring, pinky tips are below their PIP joints.
    """
    idx_tip = lm[8].y
    idx_pip = lm[6].y
    mid_tip = lm[12].y; mid_pip = lm[10].y
    ring_tip = lm[16].y; ring_pip = lm[14].y
    pinky_tip = lm[20].y; pinky_pip = lm[18].y

    pointing = idx_tip < idx_pip and \
               mid_tip > mid_pip and \
               ring_tip > ring_pip and \
               pinky_tip > pinky_pip
    return pointing

def is_open_palm(lm):
    """
    Return True if all fingers are extended (open palm).
    """
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

        if is_pointing(lm):
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            current_stroke.append((ix, iy))
            drawing = True
            recorded = True

            # Draw fingertip marker
            cv2.circle(frame, (ix, iy), 6, (0, 255, 0), -1)

        else:
            if drawing and len(current_stroke) > 1:
                all_strokes.append(current_stroke)
            drawing = False
            current_stroke = []

        # Check for exit gesture (open palm AFTER drawing done)
        if recorded and is_open_palm(lm):
            print("Open palm detected → saving strokes as PNG & exiting.")
            # Save strokes as PNG
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
            for stroke in all_strokes:
                for i in range(1, len(stroke)):
                    cv2.line(canvas, stroke[i - 1], stroke[i], (0, 0, 0), 2)
            cv2.imwrite("drawing.png", canvas)
            break

    # Draw live strokes on video feed
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            cv2.line(frame, stroke[i - 1], stroke[i], (255, 0, 0), 2)
    for i in range(1, len(current_stroke)):
        cv2.line(frame, current_stroke[i - 1], current_stroke[i], (0, 0, 255), 2)

    cv2.imshow("Air Draw", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
