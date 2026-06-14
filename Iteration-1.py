#Iteration-1

#import cv2 for image processing, mediapipe for gesture tracking and numpy 
import cv2
import mediapipe as mp
import numpy as np

#import hands and drawing_utils functions to recognize hands and respective gestures
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

#set parameters and thresholds for Mediapipe to track hand gestures
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

#store recorded strokes
all_strokes = []
current_stroke = []
drawing = False  #flag 

cap = cv2.VideoCapture(0)  # your mobile cam should show up here

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark

        # Index fingertip and thumb tip
        index_tip = lm[8]
        thumb_tip = lm[4]

        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

        # Distance between thumb and index → drawing toggle
        dist = np.hypot(ix - tx, iy - ty)

        if dist < 40:  # pinch detected → pen down
            drawing = True
            current_stroke.append((ix, iy))
        else:
            if drawing and len(current_stroke) > 1:
                all_strokes.append(current_stroke)
            drawing = False
            current_stroke = []

        # Draw fingertip
        cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)

    # Draw all strokes
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            cv2.line(frame, stroke[i - 1], stroke[i], (255, 0, 0), 2)

    # Draw current stroke (live)
    for i in range(1, len(current_stroke)):
        cv2.line(frame, current_stroke[i - 1], current_stroke[i], (0, 0, 255), 2)

    cv2.imshow("Air Draw", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

