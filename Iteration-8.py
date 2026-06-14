import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import re
import shutil
from datetime import datetime
from tkinter import Tk, simpledialog

# smoothing filter for fingertip coordinates
class OneEuroFilter:
    def __init__(self, freq=120, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.last_time = None
        self.prev_value = None
        self.prev_derivative = None

    def smooth(self, value):
        now = time.time()
        if self.last_time is None:
            self.last_time = now
            self.prev_value = value
            self.prev_derivative = 0.0
            return value
        dt = now - self.last_time
        if dt <= 0:
            dt = 1e-6
        self.last_time = now
        freq = 1.0 / dt if dt > 0 else self.freq
        derivative = (value - self.prev_value) * freq
        smooth_derivative = self._exp_smooth(derivative, self.prev_derivative, self._alpha(freq, self.dcutoff))
        self.prev_derivative = smooth_derivative
        cutoff = self.mincutoff + self.beta * abs(smooth_derivative)
        alpha = self._alpha(freq, cutoff)
        smooth_value = self._exp_smooth(value, self.prev_value, alpha)
        self.prev_value = smooth_value
        return smooth_value

    def _exp_smooth(self, x, x_prev, alpha):
        return alpha * x + (1 - alpha) * x_prev

    def _alpha(self, freq, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * freq)

# mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# global stroke state
is_recording = False
recorded_strokes = []
current_stroke = []

# palm gesture detection
def is_open_palm(hand_landmarks, handedness_hint=None):
    landmarks = hand_landmarks.landmark
    fingers_extended = []
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_extended.append(landmarks[tip_idx].y < landmarks[pip_idx].y)
    thumb_left_test = landmarks[4].x < landmarks[3].x
    thumb_right_test = landmarks[4].x > landmarks[3].x
    # handedness_hint not always provided; accept either thumb test if unsure
    if handedness_hint == "Left":
        thumb_ok = thumb_left_test
    elif handedness_hint == "Right":
        thumb_ok = thumb_right_test
    else:
        thumb_ok = thumb_left_test or thumb_right_test
    return all(fingers_extended) and thumb_ok

# save strokes to PNG
def save_strokes_image(canvas_height, canvas_width, out_name="drawing.png", preview_ms=1000):
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    for stroke in recorded_strokes:
        for i in range(1, len(stroke)):
            cv2.line(canvas, stroke[i - 1], stroke[i], (0, 0, 0), 2)
    out_path = out_name
    cv2.imwrite(out_path, canvas)
    print("saved image.")
    try:
        cv2.imshow("drawing", canvas)
        cv2.waitKey(preview_ms)
        cv2.destroyWindow("drawing")
    except Exception:
        pass
    return out_path

# label helpers
def sanitize_label(text):
    if not text:
        return "unknown"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", text.strip().replace(" ", "_"))
    return cleaned[:64]

def ask_for_label(initial_text=""):
    try:
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        answer = simpledialog.askstring("Label", "name (optional):", initialvalue=initial_text, parent=root)
        root.destroy()
        return answer.strip() if answer else ""
    except Exception:
        print("label window failed.")
        return ""

# tubular OBJ exporter
def write_tubular_obj(path_points, out_path="stroke.obj", radius=10.0, radial_segments=12):
    if len(path_points) < 2:
        print("not enough points for 3d model.")
        return None
    try:
        pts = np.array(path_points, dtype=float)
        n = len(pts)
        seg = int(max(3, radial_segments))
        tangents = []
        for i in range(n):
            if i == 0:
                t = pts[1] - pts[0]
            elif i == n - 1:
                t = pts[-1] - pts[-2]
            else:
                t = pts[i + 1] - pts[i - 1]
            norm = np.linalg.norm(t)
            tangents.append(t / norm if norm != 0 else np.array([1.0, 0.0]))
        up = np.array([0.0, 0.0, 1.0])
        vertices = []
        faces = []
        for i, p in enumerate(pts):
            tangent3 = np.array([tangents[i][0], tangents[i][1], 0.0])
            normal = np.cross(up, tangent3)
            if np.linalg.norm(normal) < 1e-6:
                normal = np.array([1.0, 0.0, 0.0])
            normal = normal / (np.linalg.norm(normal) + 1e-12)
            binormal = np.cross(tangent3, normal)
            binormal = binormal / (np.linalg.norm(binormal) + 1e-12)
            center = np.array([p[0], p[1], 0.0])
            for j in range(seg):
                theta = 2.0 * math.pi * j / seg
                pos = center + radius * (normal * math.cos(theta) + binormal * math.sin(theta))
                vertices.append(tuple(pos.tolist()))
        for i in range(n - 1):
            for j in range(seg):
                a = i * seg + j
                b = i * seg + (j + 1) % seg
                c = (i + 1) * seg + (j + 1) % seg
                d = (i + 1) * seg + j
                faces.append((a, b, c))
                faces.append((a, c, d))
        with open(out_path, "w") as f:
            f.write("# obj file\n")
            for v in vertices:
                f.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
            for fa in faces:
                f.write("f {} {} {}\n".format(fa[0] + 1, fa[1] + 1, fa[2] + 1))
        print("3d model saved.")
        return out_path
    except Exception:
        print("3d model save failed.")
        return None

# convert pixel coords into centered coordinate system
def pixel_to_centered_space(px, py, canvas_width, canvas_height, scale=1.0):
    cx = (px - canvas_width / 2.0) * (scale / max(canvas_width, canvas_height))
    cy = (canvas_height / 2.0 - py) * (scale / max(canvas_width, canvas_height))
    return cx, cy

# main drawing loop
def main():
    global is_recording, recorded_strokes, current_stroke
    camera = None
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("camera open failed.")
            return

        x_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)
        y_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)
        canvas_height, canvas_width = None, None

        palm_previous_state = False
        palm_stable_count = 0
        PALM_REQUIRED = 6
        last_toggle_time = 0.0
        TOGGLE_COOLDOWN = 0.4

        print("camera ready.")

        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("camera feed ended.")
                    break
                frame = cv2.flip(frame, 1)
                if canvas_height is None or canvas_width is None:
                    canvas_height, canvas_width, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                palm_open_now = False
                handedness_hint = None

                if results.multi_handedness:
                    try:
                        handedness_hint = results.multi_handedness[0].classification[0].label
                    except Exception:
                        handedness_hint = None

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        index_tip = hand_landmarks.landmark[8]
                        raw_x = index_tip.x * canvas_width
                        raw_y = index_tip.y * canvas_height
                        smoothed_x = int(x_filter.smooth(raw_x))
                        smoothed_y = int(y_filter.smooth(raw_y))
                        palm_open_now = is_open_palm(hand_landmarks, handedness_hint)
                        palm_stable_count = palm_stable_count + 1 if palm_open_now else 0

                        now = time.time()
                        if palm_stable_count >= PALM_REQUIRED and palm_open_now and not palm_previous_state:
                            if now - last_toggle_time < TOGGLE_COOLDOWN:
                                pass
                            else:
                                last_toggle_time = now
                                if not is_recording:
                                    is_recording = True
                                    recorded_strokes = []
                                    current_stroke = []
                                    print("recording started.")
                                else:
                                    is_recording = False
                                    if current_stroke:
                                        recorded_strokes.append(current_stroke)
                                        current_stroke = []
                                    print("recording stopped.")
                                    saved_image = save_strokes_image(canvas_height, canvas_width)
                                    label_text = ask_for_label("")
                                    if label_text:
                                        print(f"label added: {label_text}")
                                    else:
                                        print("no label given.")
                                    try:
                                        os.makedirs("dataset", exist_ok=True)
                                        timestamp_ms = int(time.time() * 1000)
                                        safe_label = sanitize_label(label_text)
                                        file_name = f"{safe_label}_{timestamp_ms}.png"
                                        shutil.copy(saved_image, os.path.join("dataset", file_name))
                                        print("stored copy in dataset.")
                                    except Exception:
                                        print("couldn't store a copy.")
                                    last_stroke = recorded_strokes[-1] if recorded_strokes else None
                                    if last_stroke and len(last_stroke) >= 2:
                                        path_points = []
                                        for px, py in last_stroke:
                                            cx, cy = pixel_to_centered_space(px, py, canvas_width, canvas_height)
                                            path_points.append((cx * max(canvas_width, canvas_height),
                                                                cy * max(canvas_width, canvas_height)))
                                        os.makedirs("generated", exist_ok=True)
                                        obj_name = f"{sanitize_label(label_text)}_{timestamp_ms}.obj"
                                        write_tubular_obj(path_points, out_path=os.path.join("generated", obj_name),
                                                         radius=max(canvas_width, canvas_height) * 0.01,
                                                         radial_segments=16)
                                    else:
                                        print("not enough points for 3d model.")

                        palm_previous_state = palm_stable_count >= PALM_REQUIRED

                        if is_recording:
                            current_stroke.append((smoothed_x, smoothed_y))
                            cv2.circle(frame, (smoothed_x, smoothed_y), 6, (0, 255, 0), -1)
                        else:
                            if len(current_stroke) > 1:
                                recorded_strokes.append(current_stroke)
                            current_stroke = []

                        for s in recorded_strokes:
                            for i in range(1, len(s)):
                                cv2.line(frame, s[i - 1], s[i], (255, 0, 0), 2)
                        for i in range(1, len(current_stroke)):
                            cv2.line(frame, current_stroke[i - 1], current_stroke[i], (0, 0, 255), 2)

                cv2.imshow("Live Feed", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("closing program.")
                    break

    except KeyboardInterrupt:
        print("interrupted by user.")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
