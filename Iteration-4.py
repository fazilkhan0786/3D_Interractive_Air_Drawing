# Air Draw (stable fingertip + palm gestures with landmarks)
# Requirements: pip install opencv-python mediapipe numpy

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

recording = False
all_strokes = []
current_stroke = []

def is_open_palm(hand_landmarks):
    """
    Detect open palm: all fingertips higher (y smaller) than their pip joints.
    Finger tips: 8, 12, 16, 20 ; Thumb: 4 relative to wrist (0).
    """
    landmarks = hand_landmarks.landmark

    # Index, Middle, Ring, Pinky finger check
    fingers_extended = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_extended.append(landmarks[tip].y < landmarks[pip].y)

    # Thumb check: compare x (for left/right hand difference you can refine later)
    thumb_extended = landmarks[4].x < landmarks[3].x

    return all(fingers_extended) and thumb_extended


def save_strokes(h, w):
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            cv2.line(canvas, stroke[i - 1], stroke[i], (0, 0, 0), 2)
    cv2.imwrite("drawing.png", canvas)
    print("✅ Saved drawing.png")
    cv2.imshow("Your Drawing", canvas)
    cv2.waitKey(0)


def main():
    global recording, all_strokes, current_stroke

    cap = cv2.VideoCapture(0)
    h, w = None, None

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            if h is None or w is None:
                h, w, _ = frame.shape

            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    index_tip = hand_landmarks.landmark[8]
                    x, y = int(index_tip.x * w), int(index_tip.y * h)

                    if is_open_palm(hand_landmarks):
                        if not recording:
                            recording = True
                            print("✋ Palm detected → Recording Started")
                            cv2.putText(frame, "Recording Started", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            cv2.imshow("Live Feed", frame)
                            cv2.waitKey(1500)
                        elif recording and all_strokes:
                            recording = False
                            print("✋ Palm detected again → Recording Stopped")
                            cv2.putText(frame, "Recording Stopped", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            cv2.imshow("Live Feed", frame)
                            cv2.waitKey(1500)
                            save_strokes(h, w)
                            cap.release()
                            cv2.destroyAllWindows()
                            return

                    if recording:
                        current_stroke.append((x, y))
                        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                    else:
                        if len(current_stroke) > 1:
                            all_strokes.append(current_stroke)
                        current_stroke = []

                    # Draw strokes
                    for stroke in all_strokes:
                        for i in range(1, len(stroke)):
                            cv2.line(frame, stroke[i - 1], stroke[i], (255, 0, 0), 2)
                    for i in range(1, len(current_stroke)):
                        cv2.line(frame, current_stroke[i - 1], current_stroke[i], (0, 0, 255), 2)

            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
