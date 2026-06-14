# Air Draw  (MediaPipe with proper start/stop palm gestures)

import cv2
import mediapipe as mp
import numpy as np

import math
import time

class OneEuroFilter:
    def __init__(self, freq=120, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.last_time = None
        self.x_prev = None
        self.dx_prev = None

    def smooth(self, x):
        now = time.time()
        
        if self.last_time is None:
            self.last_time = now
            self.x_prev = x
            self.dx_prev = 0
            return x

        dt = now - self.last_time
        self.last_time = now

        freq = 1.0 / dt if dt > 0 else self.freq

        # derivative smoothing
        dx = (x - self.x_prev) * freq
        edx = self._exp_smooth(dx, self.dx_prev, self._alpha(freq, self.dcutoff))
        self.dx_prev = edx

        # variable cutoff that adapts based on the hand speed
        cutoff = self.mincutoff + self.beta * abs(edx)
        alpha = self._alpha(freq, cutoff)

        # main smoothing part
        x_hat = self._exp_smooth(x, self.x_prev, alpha)
        self.x_prev = x_hat

        return x_hat

    def _exp_smooth(self, x, x_prev, alpha):
        return alpha * x + (1 - alpha) * x_prev

    def _alpha(self, freq, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

recording = False
all_strokes = []
current_stroke = []


def is_open_palm(hand_landmarks):
    """Detect open palm: all fingertips higher than pip joints + thumb extended"""
    landmarks = hand_landmarks.landmark

    #extended fingers
    fingers_extended = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_extended.append(landmarks[tip].y < landmarks[pip].y)

    # Thumb reference check
    thumb_extended = landmarks[4].x < landmarks[3].x

    return all(fingers_extended) and thumb_extended


def save_strokes(h, w):
    """Save strokes as PNG and display"""
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            cv2.line(canvas, stroke[i - 1], stroke[i], (0, 0, 0), 2)
    cv2.imwrite("drawing.png", canvas)
    print("Saved drawing.png")

    # popup
    cv2.imshow("Your Drawing", canvas)
    cv2.waitKey(2000)  # display for 2 sec


def main():
    global recording, all_strokes, current_stroke

    cap = cv2.VideoCapture(0)
    # One Euro smoothing for index fingertip x and y
    x_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)
    y_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)

    h, w = None, None
    palm_prev_state = False  # for gesture toggling

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    index_tip = hand_landmarks.landmark[8]

                    raw_x = index_tip.x * w
                    raw_y = index_tip.y * h

                    smooth_x = int(x_filter.smooth(raw_x))
                    smooth_y = int(y_filter.smooth(raw_y))

                    x, y = smooth_x, smooth_y


                    # gesture detection and recognition
                    palm_now = is_open_palm(hand_landmarks)

                    if palm_now and not palm_prev_state:   
                        if not recording:
                            recording = True
                            all_strokes = []
                            current_stroke = []
                            print("Palm → Recording Started")
                            cv2.putText(frame, "Recording Started", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            cv2.imshow("Live Feed", frame)
                            cv2.waitKey(1000)

                        elif recording:
                            recording = False 
                            if current_stroke:
                                all_strokes.append(current_stroke)
                                current_stroke = []
                            print("Palm → Recording Stopped & Saved")
                            cv2.putText(frame, "Recording Stopped", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            cv2.imshow("Live Feed", frame)
                            cv2.waitKey(1000)
                            save_strokes(h, w)

                    palm_prev_state = palm_now

                    # stroke recording and drawing
                    if recording:
                        current_stroke.append((x, y))
                        cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                    else:
                        if len(current_stroke) > 1:
                            all_strokes.append(current_stroke)
                        current_stroke = []

                    # Draw strokes live on the screen
                    for stroke in all_strokes:
                        for i in range(1, len(stroke)):
                            cv2.line(frame, stroke[i - 1], stroke[i], (255, 0, 0), 2)
                    for i in range(1, len(current_stroke)):
                        cv2.line(frame, current_stroke[i - 1], current_stroke[i], (0, 0, 255), 2)

            cv2.imshow("Live Feed", frame)
            if cv2.waitKey(1) & 0xFF == 27:  #esc to exit
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
