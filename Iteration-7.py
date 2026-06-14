# Air Draw  (MediaPipe with proper start/stop palm gestures)
# Integrated: OneEuro smoothing + popup label input + simple OBJ 3D exporter
# Save this as your script and run locally.

import cv2
import mediapipe as mp
import numpy as np

import math
import time
import os
from collections import deque
from tkinter import Tk, simpledialog

# ------------------------
# OneEuroFilter (unchanged)
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
        # avoid dt==0
        if dt <= 0:
            dt = 1e-6
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
# ------------------------

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

recording = False
all_strokes = []
current_stroke = []

# ------------------------
# helper: open palm check (unchanged)
def is_open_palm(hand_landmarks):
    """Detect open palm: all fingertips higher than pip joints + thumb extended"""
    landmarks = hand_landmarks.landmark

    #extended fingers
    fingers_extended = []
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_extended.append(landmarks[tip].y < landmarks[pip].y)

    # Thumb reference check (this assumes right-hand facing camera; might need adaptation)
    thumb_extended = landmarks[4].x < landmarks[3].x

    return all(fingers_extended) and thumb_extended
# ------------------------

# ------------------------
# save strokes as PNG (unchanged except small path)
def save_strokes(h, w):
    """Save strokes as PNG and display"""
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    for stroke in all_strokes:
        for i in range(1, len(stroke)):
            cv2.line(canvas, stroke[i - 1], stroke[i], (0, 0, 0), 2)
    out_path = "drawing.png"
    cv2.imwrite(out_path, canvas)
    print("Saved", out_path)

    # popup (show image for 2 seconds)
    cv2.imshow("Your Drawing", canvas)
    cv2.waitKey(2000)  # display for 2 sec
    cv2.destroyWindow("Your Drawing")
    return out_path
# ------------------------

# ------------------------
# Simple Tk popup to ask user for label
def ask_label_popup(initial_text=""):
    """
    Blocks until user enters text or cancels. Returns string (possibly empty).
    Uses tkinter.simpledialog.
    """
    try:
        root = Tk()
        root.withdraw()  # hide main window
        root.attributes("-topmost", True)
        answer = simpledialog.askstring("Label", "Enter label for this object (optional):", initialvalue=initial_text, parent=root)
        root.destroy()
        if answer is None:
            return ""
        return answer.strip()
    except Exception as e:
        print("Popup failed (tkinter):", e)
        return ""
# ------------------------

# ------------------------
# Simple OBJ writer: create a tubular mesh (rings along stroke) and write .obj (no external deps)
def write_tubular_obj(path_pts, out_path="stroke.obj", radius=10.0, radial_segments=12):
    """
    path_pts: list of (x, y) in pixel coordinates (or arbitrary units). We'll treat them as 2D and create tubes in a 3D space.
    radius: tube radius in same units as path_pts (if path_pts are pixels, radius in pixels)
    radial_segments: number of segments around tube
    This writes a basic OBJ with vertices and faces. Normals/uv omitted for brevity.
    """
    if len(path_pts) < 2:
        print("Not enough points to create OBJ.")
        return None

    pts = np.array(path_pts, dtype=float)
    n = len(pts)
    seg = radial_segments

    # Create tangents
    tangents = []
    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n-1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i+1] - pts[i-1]
        norm = np.linalg.norm(t)
        if norm == 0:
            t = np.array([1.0, 0.0])
        else:
            t = t / norm
        tangents.append(t)

    # choose arbitrary up vector
    up = np.array([0.0, 0.0, 1.0])

    vertices = []
    faces = []

    # For each point create a ring in 3D: x->X, y->Y, and Z along 0 (we'll treat 2D stroke living in X-Y plane, sweep along it)
    for i, p in enumerate(pts):
        # tangent in XY
        t2 = np.array([tangents[i][0], tangents[i][1], 0.0])
        # pick normal via cross with up
        nvec = np.cross(up, t2)
        nlen = np.linalg.norm(nvec)
        if nlen < 1e-6:
            # choose arbitrary
            nvec = np.array([1.0, 0.0, 0.0])
            nlen = 1.0
        nvec = nvec / nlen
        bvec = np.cross(t2, nvec)
        bvec = bvec / (np.linalg.norm(bvec) + 1e-8)

        center = np.array([p[0], p[1], 0.0])
        for j in range(seg):
            theta = 2.0 * math.pi * j / seg
            pos = center + radius * (nvec * math.cos(theta) + bvec * math.sin(theta))
            vertices.append(tuple(pos.tolist()))

    # connect faces between rings
    for i in range(n-1):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            c = (i + 1) * seg + (j + 1) % seg
            d = (i + 1) * seg + j
            faces.append((a, b, c))
            faces.append((a, c, d))

    # write OBJ
    try:
        with open(out_path, "w") as f:
            f.write("# Simple tubular OBJ exported by Air Draw\n")
            for v in vertices:
                f.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
            for face in faces:
                # OBJ is 1-indexed
                f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))
        print("Wrote OBJ:", out_path)
        return out_path
    except Exception as e:
        print("Failed to write OBJ:", e)
        return None
# ------------------------

# ------------------------
# Helper: convert pixel stroke to approx "camera space" for nicer 3D scaling (optional)
def pixel_to_working_space(px, py, w, h, scale=1.0):
    """
    Convert pixel coords to centered coordinate system (for OBJ export).
    Returns (x,y) in same units as pixels but centered on canvas.
    """
    cx = (px - w / 2.0) * (scale / max(w, h))
    cy = (h / 2.0 - py) * (scale / max(w, h))  # flip y so up is positive
    # we keep units small and centered; tubular writer will add thickness in pixel-like units
    return cx, cy
# ------------------------

def main():
    global recording, all_strokes, current_stroke

    cap = cv2.VideoCapture(0)
    # One Euro smoothing for index fingertip x and y (good defaults for drawing)
    x_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)
    y_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)

    h, w = None, None
    palm_prev_state = False  # for gesture toggling
    palm_stable_counter = 0  # require palm stable for N frames before toggling
    PALM_STABLE_REQUIRED = 6

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

                    # gesture detection and recognition with stability hysteresis
                    palm_now = is_open_palm(hand_landmarks)
                    if palm_now:
                        palm_stable_counter += 1
                    else:
                        palm_stable_counter = 0

                    if palm_stable_counter >= PALM_STABLE_REQUIRED and palm_now and not palm_prev_state:
                        # toggle
                        if not recording:
                            recording = True
                            all_strokes = []
                            current_stroke = []
                            print("Palm → Recording Started")
                            cv2.putText(frame, "Recording Started", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            cv2.imshow("Live Feed", frame)
                            cv2.waitKey(500)
                        elif recording:
                            # stop recording
                            recording = False
                            if current_stroke:
                                all_strokes.append(current_stroke)
                                current_stroke = []
                            print("Palm → Recording Stopped & Saved")
                            cv2.putText(frame, "Recording Stopped", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                            cv2.imshow("Live Feed", frame)
                            cv2.waitKey(500)

                            # --- SAVE PNG ---
                            png_path = save_strokes(h, w)

                            # --- ASK USER FOR LABEL ---
                            user_label = ask_label_popup(initial_text="")
                            if user_label:
                                print("User label:", user_label)
                            else:
                                print("User did not enter a label (left blank)")

                            # --- Save labeled example to dataset/ (copy drawing.png)
                            try:
                                os.makedirs("dataset", exist_ok=True)
                                base_ts = int(time.time() * 1000)
                                label_safe = user_label.replace(" ", "_") if user_label else "unknown"
                                dest_name = f"{label_safe}_{base_ts}.png"
                                dest_path = os.path.join("dataset", dest_name)
                                # copy file
                                import shutil
                                shutil.copy(png_path, dest_path)
                                print("Saved labeled example to", dest_path)
                            except Exception as e:
                                print("Failed to save labeled example:", e)

                            # --- Generate a very simple 3D OBJ from the last stroke ---
                            # We'll take the most recent stroke in all_strokes (last stroke)
                            last_stroke = None
                            if all_strokes:
                                last_stroke = all_strokes[-1]
                            if last_stroke and len(last_stroke) >= 2:
                                # Convert stroke (pixel coords) to centered working space for OBJ
                                # Choose radius relative to canvas size
                                # Create path_pts as 2D points in working space (units similar to pixels but centered)
                                path_pts = []
                                for (px, py) in last_stroke:
                                    cx, cy = pixel_to_working_space(px, py, w, h, scale=1.0)
                                    # Scale up so tubular radius is visible; convert to pixel-like units for OBJ writer
                                    # We'll re-scale to a nominal unit space
                                    path_pts.append((cx * max(w,h), cy * max(w,h)))
                                # radius (in these units) - tuned to look reasonable
                                tube_radius = max(w, h) * 0.01  # ~1% of max dimension
                                obj_out = os.path.join("generated", f"{label_safe}_{base_ts}.obj")
                                os.makedirs("generated", exist_ok=True)
                                written = write_tubular_obj(path_pts, out_path=obj_out, radius=tube_radius, radial_segments=16)
                                if written:
                                    print("Generated OBJ:", written)
                                else:
                                    print("No OBJ generated.")
                            else:
                                print("No stroke points available to generate 3D.")

                    # update palm_prev_state only when palm went stable low->high or high->low
                    if palm_stable_counter == 0:
                        palm_prev_state = False
                    elif palm_stable_counter >= PALM_STABLE_REQUIRED:
                        palm_prev_state = True

                    # stroke recording and drawing
                    if recording:
                        # append smoothed fingertip to current stroke
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
            if cv2.waitKey(1) & 0xFF == 27:  # esc to exit
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
