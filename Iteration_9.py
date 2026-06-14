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

# Drawing states
class DrawingState:
    IDLE = 0
    RECORDING = 1
    AR_DISPLAY = 2

# mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# NEW: Pointer gesture (index + thumb extended, others closed)
def is_pointer_gesture(hand_landmarks):
    """Index finger and thumb extended, other fingers closed"""
    if not hand_landmarks:
        return False
    
    lm = hand_landmarks.landmark
    
    # Index finger extended (tip above PIP)
    index_extended = lm[8].y < lm[6].y
    
    # Thumb extended (tip away from palm)
    thumb_extended = (
        abs(lm[4].x - lm[2].x) > 0.05 or  # Thumb is spread out
        lm[4].y < lm[3].y  # Or thumb tip is above IP joint
    )
    
    # Middle, ring, pinky are curled (tips below PIP)
    middle_curled = lm[12].y > lm[10].y
    ring_curled = lm[16].y > lm[14].y
    pinky_curled = lm[20].y > lm[18].y
    
    return (index_extended and thumb_extended and 
            middle_curled and ring_curled and pinky_curled)

# NEW: Open palm (all fingers extended) - FIXED for any orientation
def is_open_palm(hand_landmarks):
    """All five fingers extended - works regardless of hand orientation"""
    if not hand_landmarks:
        return False
    
    lm = hand_landmarks.landmark
    
    # Check all four fingers extended (tip above PIP joint in Y)
    fingers_extended = []
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_extended.append(lm[tip_idx].y < lm[pip_idx].y)
    
    # Thumb extended - check distance from palm center
    thumb_tip = lm[4]
    palm_center = lm[0]  # Wrist as reference
    thumb_distance = abs(thumb_tip.x - palm_center.x)
    thumb_extended = thumb_distance > 0.08  # Thumb is spread away from palm
    
    return all(fingers_extended) and thumb_extended

# Fist gesture detection for reset
def is_fist(hand_landmarks):
    """All fingers curled into fist"""
    if not hand_landmarks:
        return False
    
    lm = hand_landmarks.landmark
    
    # All fingers curled (tips below PIP)
    fingers_curled = []
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_curled.append(lm[tip_idx].y > lm[pip_idx].y)
    
    # Thumb curled close to palm
    thumb_curled = abs(lm[4].x - lm[2].x) < 0.06
    
    return all(fingers_curled) and thumb_curled

# IMPROVED: Simpler, less dense mesh generation
def generate_tube_mesh_fast(path_points, radius=15.0, segments=6):
    """Generate tube mesh with fewer vertices - segments reduced from 8 to 6"""
    if len(path_points) < 2:
        return None, None
    
    pts = np.array(path_points, dtype=float)
    n = len(pts)
    
    vertices = []
    faces = []
    
    for i in range(n):
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == n-1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i+1] - pts[i-1]
        
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-6:
            tangent = tangent / tangent_norm
        else:
            tangent = np.array([1.0, 0.0, 0.0])
        
        if abs(tangent[0]) < 0.9:
            perp = np.cross(tangent, np.array([1, 0, 0]))
        else:
            perp = np.cross(tangent, np.array([0, 1, 0]))
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 1e-6:
            perp = perp / perp_norm
        else:
            perp = np.array([0, 1, 0])
        
        binormal = np.cross(tangent, perp)
        binormal_norm = np.linalg.norm(binormal)
        if binormal_norm > 1e-6:
            binormal = binormal / binormal_norm
        
        center = np.array([pts[i][0], pts[i][1], 0.0])
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            rot = np.cos(angle) * perp + np.sin(angle) * binormal
            v = center + radius * rot
            vertices.append(v)
    
    for i in range(n-1):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j+1) % segments
            c = (i+1) * segments + (j+1) % segments
            d = (i+1) * segments + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    
    return np.array(vertices), np.array(faces)

# IMPROVED: Anchor to middle finger base (MCP joint)
def get_hand_transform(hand_landmarks, frame_width, frame_height):
    """Get transform anchored to middle finger MCP (knuckle)"""
    # Anchor point: Middle finger MCP (landmark 9 - base of middle finger)
    anchor = hand_landmarks.landmark[9]
    
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    index_mcp = hand_landmarks.landmark[5]
    
    pos_x = anchor.x * frame_width
    pos_y = anchor.y * frame_height
    pos_z = anchor.z
    
    # Calculate hand size for scaling
    hand_size = np.linalg.norm([
        (middle_mcp.x - wrist.x) * frame_width,
        (middle_mcp.y - wrist.y) * frame_height
    ])
    
    # INCREASED scale factor for bigger AR display (was / 80.0, now / 50.0)
    scale = hand_size / 50.0
    
    forward = np.array([
        middle_mcp.x - wrist.x,
        middle_mcp.y - wrist.y,
        middle_mcp.z - wrist.z
    ])
    forward_norm = np.linalg.norm(forward)
    if forward_norm > 1e-6:
        forward = forward / forward_norm
    
    return {
        'position': (pos_x, pos_y, pos_z),
        'scale': scale,
        'forward': forward
    }

# transform and project mesh to 2D
def transform_and_project_mesh(vertices, hand_transform, frame_width, frame_height):
    pos = hand_transform['position']
    scale = hand_transform['scale']
    
    projected = []
    for v in vertices:
        x = pos[0] + v[0] * scale
        y = pos[1] - v[1] * scale
        
        depth_scale = 1.0 - pos[2] * 0.5
        x = int(x * depth_scale)
        y = int(y * depth_scale)
        
        x = max(0, min(frame_width-1, x))
        y = max(0, min(frame_height-1, y))
        
        projected.append((x, y))
    
    return projected

# render mesh as wireframe overlay
def render_mesh_wireframe(frame, vertices, faces, hand_transform, frame_width, frame_height):
    if vertices is None or faces is None:
        return frame
    
    projected = transform_and_project_mesh(vertices, hand_transform, frame_width, frame_height)
    
    # Draw with thicker lines for better visibility
    for face in faces:
        try:
            pts = np.array([projected[face[0]], projected[face[1]], projected[face[2]]], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        except:
            pass
    
    return frame

# draw UI overlays with gesture indicators
def draw_ui(frame, state, stroke_count=0, is_pointer=False, is_palm=False, is_fist_detected=False):
    h, w = frame.shape[:2]
    
    if state == DrawingState.IDLE:
        cv2.rectangle(frame, (5, 5), (450, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (450, 85), (0, 255, 0), 2)
        cv2.putText(frame, "OPEN PALM to start drawing", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "(All 5 fingers extended)", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        
        # Gesture indicator
        if is_palm:
            cv2.circle(frame, (420, 45), 15, (0, 255, 0), -1)
    
    elif state == DrawingState.RECORDING:
        cv2.circle(frame, (25, 25), 12, (0, 0, 255), -1)
        cv2.rectangle(frame, (5, 5), (450, 85), (0, 0, 0), -1)
        cv2.putText(frame, "RECORDING - Use pointer gesture", (50, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "(Index+Thumb out, others closed)", (50, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        cv2.putText(frame, f"Points: {stroke_count}", (w - 150, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Gesture indicators
        if is_pointer:
            cv2.circle(frame, (420, 45), 15, (0, 255, 0), -1)
        if is_palm:
            cv2.circle(frame, (420, 70), 10, (255, 255, 0), -1)
            cv2.putText(frame, "Palm to stop", (270, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    elif state == DrawingState.AR_DISPLAY:
        cv2.rectangle(frame, (5, 5), (450, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (450, 85), (255, 0, 255), 2)
        cv2.putText(frame, "AR MODE - Move your hand!", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, "FIST to reset", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 255), 1)
        
        # Gesture indicator
        if is_fist_detected:
            cv2.circle(frame, (420, 45), 15, (255, 100, 100), -1)
    
    cv2.putText(frame, "ESC to quit", (w - 150, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

# convert pixel coords into centered coordinate system
def pixel_to_centered_space(px, py, canvas_width, canvas_height, scale=1.0):
    cx = (px - canvas_width / 2.0) * (scale / max(canvas_width, canvas_height))
    cy = (canvas_height / 2.0 - py) * (scale / max(canvas_width, canvas_height))
    return cx, cy

# save mesh to OBJ file
def save_mesh_obj(vertices, faces, out_path="output.obj"):
    try:
        with open(out_path, "w") as f:
            f.write("# AR Hand Drawing Mesh\n")
            for v in vertices:
                f.write("v {:.6f} {:.6f} {:.6f}\n".format(v[0], v[1], v[2]))
            for face in faces:
                f.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
        print(f"Mesh saved to {out_path}")
        return out_path
    except Exception as e:
        print(f"Failed to save mesh: {e}")
        return None

# main drawing loop
def main():
    camera = None
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Camera failed to open.")
            return

        x_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)
        y_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)
        canvas_height, canvas_width = None, None

        drawing_state = DrawingState.IDLE
        recorded_strokes = []
        current_stroke = []
        ar_mesh_vertices = None
        ar_mesh_faces = None

        palm_stable_count = 0
        pointer_stable_count = 0
        fist_stable_count = 0
        GESTURE_REQUIRED = 8
        last_gesture_time = 0.0
        GESTURE_COOLDOWN = 0.5

        print("Camera ready. Waiting for hand...")
        print("Gestures:")
        print("  - OPEN PALM (all 5 fingers): Start/Stop recording")
        print("  - POINTER (index+thumb): Draw while recording")
        print("  - FIST (all closed): Reset from AR mode")
        print("\nAR mesh will be anchored to middle finger knuckle")

        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("Camera feed ended.")
                    break
                
                frame = cv2.flip(frame, 1)
                if canvas_height is None or canvas_width is None:
                    canvas_height, canvas_width, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                palm_open_now = False
                pointer_active_now = False
                fist_closed_now = False

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Draw hand skeleton
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
                    )
                    
                    # Highlight anchor point (middle finger MCP) in AR mode
                    if drawing_state == DrawingState.AR_DISPLAY:
                        anchor = hand_landmarks.landmark[9]
                        anchor_x = int(anchor.x * canvas_width)
                        anchor_y = int(anchor.y * canvas_height)
                        cv2.circle(frame, (anchor_x, anchor_y), 8, (255, 0, 255), -1)
                        cv2.circle(frame, (anchor_x, anchor_y), 10, (255, 255, 255), 2)
                    
                    # Get index finger tip position
                    index_tip = hand_landmarks.landmark[8]
                    raw_x = index_tip.x * canvas_width
                    raw_y = index_tip.y * canvas_height
                    smoothed_x = int(x_filter.smooth(raw_x))
                    smoothed_y = int(y_filter.smooth(raw_y))
                    
                    # Detect gestures
                    palm_open_now = is_open_palm(hand_landmarks)
                    pointer_active_now = is_pointer_gesture(hand_landmarks)
                    fist_closed_now = is_fist(hand_landmarks)
                    
                    # Update stable counters
                    palm_stable_count = palm_stable_count + 1 if palm_open_now else 0
                    pointer_stable_count = pointer_stable_count + 1 if pointer_active_now else 0
                    fist_stable_count = fist_stable_count + 1 if fist_closed_now else 0

                    now = time.time()
                    
                    # STATE: IDLE -> RECORDING
                    if drawing_state == DrawingState.IDLE:
                        if palm_stable_count >= GESTURE_REQUIRED and now - last_gesture_time > GESTURE_COOLDOWN:
                            drawing_state = DrawingState.RECORDING
                            recorded_strokes = []
                            current_stroke = []
                            last_gesture_time = now
                            print("▶ RECORDING STARTED - Use pointer gesture to draw")
                    
                    # STATE: RECORDING
                    elif drawing_state == DrawingState.RECORDING:
                        # Only add points when pointer gesture is active
                        if pointer_active_now:
                            current_stroke.append((smoothed_x, smoothed_y))
                            cv2.circle(frame, (smoothed_x, smoothed_y), 6, (0, 255, 0), -1)
                        
                        # RECORDING -> AR_DISPLAY (palm to stop)
                        if palm_stable_count >= GESTURE_REQUIRED and now - last_gesture_time > GESTURE_COOLDOWN:
                            if current_stroke:
                                recorded_strokes.append(current_stroke)
                            
                            # Convert all strokes to 3D mesh with BIGGER scale
                            all_points = []
                            for stroke in recorded_strokes:
                                for px, py in stroke:
                                    cx, cy = pixel_to_centered_space(px, py, canvas_width, canvas_height)
                                    # INCREASED from 150 to 250 for bigger AR display
                                    all_points.append((cx * 250, cy * 250))
                            
                            if len(all_points) >= 2:
                                # REDUCED segments from 8 to 6 for less complexity
                                ar_mesh_vertices, ar_mesh_faces = generate_tube_mesh_fast(
                                    all_points, radius=12.0, segments=6
                                )
                                if ar_mesh_vertices is not None:
                                    drawing_state = DrawingState.AR_DISPLAY
                                    last_gesture_time = now
                                    print("✓ AR MODE ACTIVATED - Anchored to middle finger!")
                                    print(f"  Mesh: {len(ar_mesh_vertices)} vertices, {len(ar_mesh_faces)} faces")
                                    
                                    # Save mesh to file
                                    os.makedirs("generated", exist_ok=True)
                                    timestamp = int(time.time() * 1000)
                                    save_mesh_obj(ar_mesh_vertices, ar_mesh_faces, 
                                                f"generated/ar_mesh_{timestamp}.obj")
                                else:
                                    drawing_state = DrawingState.IDLE
                                    print("⚠ Mesh generation failed")
                            else:
                                drawing_state = DrawingState.IDLE
                                print("⚠ Not enough points - need at least 2 points")
                            
                            current_stroke = []
                    
                    # STATE: AR_DISPLAY
                    elif drawing_state == DrawingState.AR_DISPLAY:
                        if ar_mesh_vertices is not None:
                            hand_transform = get_hand_transform(hand_landmarks, canvas_width, canvas_height)
                            frame = render_mesh_wireframe(
                                frame, ar_mesh_vertices, ar_mesh_faces, 
                                hand_transform, canvas_width, canvas_height
                            )
                        
                        # AR_DISPLAY -> IDLE (fist to reset)
                        if fist_stable_count >= GESTURE_REQUIRED and now - last_gesture_time > GESTURE_COOLDOWN:
                            drawing_state = DrawingState.IDLE
                            ar_mesh_vertices = None
                            ar_mesh_faces = None
                            recorded_strokes = []
                            last_gesture_time = now
                            print("↻ RESET TO IDLE")

                else:
                    palm_stable_count = 0
                    pointer_stable_count = 0
                    fist_stable_count = 0

                # Draw previous strokes in recording mode
                if drawing_state == DrawingState.RECORDING:
                    for stroke in recorded_strokes:
                        for i in range(1, len(stroke)):
                            cv2.line(frame, stroke[i-1], stroke[i], (255, 0, 0), 2)
                    for i in range(1, len(current_stroke)):
                        cv2.line(frame, current_stroke[i-1], current_stroke[i], (0, 0, 255), 3)

                # Draw UI with gesture feedback
                stroke_count = len(current_stroke) if drawing_state == DrawingState.RECORDING else 0
                frame = draw_ui(frame, drawing_state, stroke_count, 
                               pointer_active_now, palm_open_now, fist_closed_now)

                cv2.imshow("AR Hand Drawing", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print("Exiting...")
                    break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
