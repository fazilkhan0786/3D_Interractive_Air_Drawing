import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import pygame
from pygame.locals import *

class MetricsDashboard:
    def __init__(self):
        self.total_frames = 0
        self.hand_detected = 0
        self.mesh_gen_times = []
        self.fps_samples = []
        self.start_time = time.time()
        
    def update(self, fps, hand_detected, mesh_time=None):
        self.total_frames += 1
        if hand_detected:
            self.hand_detected += 1
        if mesh_time:
            self.mesh_gen_times.append(mesh_time)
        self.fps_samples.append(fps)
        
    def get_report(self):
        uptime = time.time() - self.start_time
        avg_fps = np.mean(self.fps_samples) if self.fps_samples else 0
        max_fps = np.max(self.fps_samples) if self.fps_samples else 0
        min_fps = np.min(self.fps_samples) if self.fps_samples else 0
        
        return {
            'avg_fps': avg_fps,
            'max_fps': max_fps,
            'min_fps': min_fps,
            'detection_rate': (self.hand_detected / self.total_frames * 100) if self.total_frames else 0,
            'avg_mesh_gen': np.mean(self.mesh_gen_times) if self.mesh_gen_times else 0,
            'best_mesh_gen': np.min(self.mesh_gen_times) if self.mesh_gen_times else 0,
            'total_meshes': len(self.mesh_gen_times),
            'uptime': uptime,
            'total_frames': self.total_frames
        }
    
    def print_report(self):
        report = self.get_report()
        print("\n" + "="*50)
        print("         PERFORMANCE METRICS REPORT")
        print("="*50)
        print(f"\n⏱️  Session Duration: {report['uptime']:.1f}s")
        print(f"🎞️  Total Frames: {report['total_frames']}")
        print(f"\n FPS Performance:")
        print(f"   Average: {report['avg_fps']:.1f} FPS")
        print(f"   Best:    {report['max_fps']:.1f} FPS")
        print(f"\n Hand Detection Rate: {report['detection_rate']:.1f}%")
        
        if report['total_meshes'] > 0:
            print(f"\n Mesh Generation ({report['total_meshes']} meshes):")
            print(f"   Average: {report['avg_mesh_gen']:.2f}ms")
            print(f"   Best:    {report['best_mesh_gen']:.2f}ms")
        
        print("\n" + "="*50 + "\n")

FIXED_ROTATION = None

def get_fixed_rotation():
    """Pre-computed rotation matrix - calculate once"""
    global FIXED_ROTATION
    if FIXED_ROTATION is None:
        angle_y = np.pi / 2
        rot_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])
        
        angle_x = -np.pi / 2
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])
        
        angle_z = np.pi / 2
        rot_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])
        
        # Add 180° flip around X axis
        flip_x = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        FIXED_ROTATION = flip_x @ rot_z @ rot_x @ rot_y
    return FIXED_ROTATION

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

# Pointer gesture (index + thumb extended, others closed)
def is_pointer_gesture(hand_landmarks):
    if not hand_landmarks:
        return False
    lm = hand_landmarks.landmark
    index_extended = lm[8].y < lm[6].y
    thumb_extended = (abs(lm[4].x - lm[2].x) > 0.05 or lm[4].y < lm[3].y)
    middle_curled = lm[12].y > lm[10].y
    ring_curled = lm[16].y > lm[14].y
    pinky_curled = lm[20].y > lm[18].y
    return (index_extended and thumb_extended and middle_curled and ring_curled and pinky_curled)

# Open palm
def is_open_palm(hand_landmarks):
    if not hand_landmarks:
        return False
    lm = hand_landmarks.landmark
    fingers_extended = []
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_extended.append(lm[tip_idx].y < lm[pip_idx].y)
    thumb_tip = lm[4]
    palm_center = lm[0]
    thumb_distance = abs(thumb_tip.x - palm_center.x)
    thumb_extended = thumb_distance > 0.08
    return all(fingers_extended) and thumb_extended

# Fist gesture
def is_fist(hand_landmarks):
    if not hand_landmarks:
        return False
    lm = hand_landmarks.landmark
    fingers_curled = []
    for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers_curled.append(lm[tip_idx].y > lm[pip_idx].y)
    thumb_curled = abs(lm[4].x - lm[2].x) < 0.06
    return all(fingers_curled) and thumb_curled

# Generate 3D tube mesh - simpler version
def generate_tube_mesh_fast(path_points, radius=15.0, segments=12):
    if len(path_points) < 2:
        return None, None
    
    pts = np.array(path_points, dtype=float)
    n = len(pts)
    seg = int(max(3, segments))
    
    # Calculate tangents
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
    
    # Generate vertices
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
            theta = 2.0 * np.pi * j / seg
            pos = center + radius * (normal * np.cos(theta) + binormal * np.sin(theta))
            vertices.append(pos)
    
    # Generate faces
    for i in range(n - 1):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            c = (i + 1) * seg + (j + 1) % seg
            d = (i + 1) * seg + j
            faces.append([a, b, c])
            faces.append([a, c, d])
    
    # Convert to numpy array and apply cached rotation
    vertices = np.array(vertices)
    rotation = get_fixed_rotation()
    vertices = vertices @ rotation.T  # Faster matrix multiplication
    
    return vertices, np.array(faces)

# Get hand 3D transform
def get_hand_transform_3d(hand_landmarks, frame_width, frame_height):
    """Extract 3D transformation from hand pose"""
    anchor = hand_landmarks.landmark[9]  # Middle finger MCP
    wrist = hand_landmarks.landmark[0]
    middle_mcp = hand_landmarks.landmark[9]
    index_mcp = hand_landmarks.landmark[5]
    pinky_mcp = hand_landmarks.landmark[17]
    
    float_offset=70
    # Position in pixel space
    pos = np.array([
        anchor.x * frame_width,
        anchor.y * frame_height - float_offset,
        anchor.z * frame_width
    ])
    
    # Build coordinate system from hand geometry
    forward = np.array([
        middle_mcp.x - wrist.x,
        middle_mcp.y - wrist.y,
        middle_mcp.z - wrist.z
    ])
    forward_norm = np.linalg.norm(forward)
    if forward_norm > 1e-6:
        forward = forward / forward_norm
    else:
        forward = np.array([0, 1, 0])
    
    right = np.array([
        pinky_mcp.x - index_mcp.x,
        pinky_mcp.y - index_mcp.y,
        pinky_mcp.z - index_mcp.z
    ])
    right_norm = np.linalg.norm(right)
    if right_norm > 1e-6:
        right = right / right_norm
    else:
        right = np.array([1, 0, 0])
    
    up = np.cross(forward, right)
    up_norm = np.linalg.norm(up)
    if up_norm > 1e-6:
        up = up / up_norm
    else:
        up = np.array([0, 0, 1])
    
    right = np.cross(up, forward)
    right = right / (np.linalg.norm(right) + 1e-6)
    
    # Scale
    hand_size = np.linalg.norm([
        (middle_mcp.x - wrist.x) * frame_width,
        (middle_mcp.y - wrist.y) * frame_height
    ])
    scale = hand_size / 50.0
    
    # Build rotation matrix
    rotation = np.column_stack([right, up, forward])
    
    return {
        'position': pos,
        'rotation': rotation,
        'scale': scale
    }

def render_mesh_solid(frame, vertices, faces, hand_transform, frame_width, frame_height):
    """Solid rendering with depth sorting and shading - OPTIMIZED"""
    if vertices is None or faces is None:
        return frame
    
    try:
        pos = hand_transform['position']
        rotation = hand_transform['rotation']
        scale = hand_transform['scale']
        
        # VECTORIZED: Transform all vertices at once
        vertices_scaled = vertices * scale
        transformed_3d = (rotation @ vertices_scaled.T).T
        
        # VECTORIZED: Project to 2D
        vertices_world = transformed_3d + pos
        projected_2d = vertices_world[:, :2].astype(np.int32)
        
        # Clamp to frame bounds
        projected_2d[:, 0] = np.clip(projected_2d[:, 0], 0, frame_width - 1)
        projected_2d[:, 1] = np.clip(projected_2d[:, 1], 0, frame_height - 1)
        
        # Calculate face data with depth and lighting
        light_dir = np.array([0.3, -0.5, -0.8])
        light_dir = light_dir / np.linalg.norm(light_dir)
        
        face_data = []
        for face in faces:
            v0 = transformed_3d[face[0]]
            v1 = transformed_3d[face[1]]
            v2 = transformed_3d[face[2]]
            
            # Face depth (average Z)
            depth = (v0[2] + v1[2] + v2[2]) / 3.0
            
            # Calculate normal for lighting
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-6:
                normal = normal / normal_norm
            else:
                continue
            
            # Brightness based on angle to light
            brightness = max(0.5, abs(np.dot(normal, light_dir)))
            
            face_data.append({
                'depth': depth,
                'brightness': brightness,
                'points': projected_2d[face].tolist()
            })
        
        # Sort back-to-front (painter's algorithm)
        face_data.sort(key=lambda x: x['depth'], reverse=True)
        
        # Create overlay for blending
        overlay = frame.copy()
        
        # Draw solid faces
        for fd in face_data:
            pts = np.array(fd['points'], dtype=np.int32)
            
            # Cyan color with brightness
            base_color = np.array([255, 180, 0])  # BGR: cyan
            color = (base_color * fd['brightness']).astype(np.int32)
            color = tuple(int(c) for c in color)
            
            # Fill triangle
            cv2.fillPoly(overlay, [pts], color)
            
            # Thin edge outline
            cv2.polylines(overlay, [pts], True, (0, 120, 120), 1, cv2.LINE_AA)
        
        # Blend with frame (semi-transparent)
        alpha = 1.0
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
    except Exception as e:
        print(f"Render error: {e}")
    
    return frame

# Draw UI overlays
def draw_ui(frame, state, stroke_count=0, is_pointer=False, is_palm=False, is_fist_detected=False, fps=0):
    h, w = frame.shape[:2]
    
    if state == DrawingState.IDLE:
        cv2.rectangle(frame, (5, 5), (450, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (450, 85), (0, 255, 0), 2)
        cv2.putText(frame, "OPEN PALM to start drawing", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "(All 5 fingers extended)", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
        if is_palm:
            cv2.circle(frame, (420, 45), 15, (0, 255, 0), -1)
            cv2.putText(frame, "READY", (360, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    elif state == DrawingState.RECORDING:
        # Animated recording indicator
        cv2.circle(frame, (25, 25), 12, (0, 0, 255), -1)
        cv2.rectangle(frame, (5, 5), (550, 85), (0, 0, 0), -1)
        cv2.putText(frame, "RECORDING - Use ☝️ to start drawing", (50, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "(Index+Thumb out, others closed)", (50, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # Show point count with color coding
        if stroke_count < 10:
            count_color = (0, 100, 255)  # Orange - need more points
        else:
            count_color = (0, 255, 0)  # Green - good
        cv2.putText(frame, f"Points: {stroke_count}", (w - 180, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, count_color, 2)
        
        if is_pointer:
            cv2.circle(frame, (520, 45), 15, (0, 255, 0), -1)
            cv2.putText(frame, "DRAWING", (440, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        if is_palm:
            cv2.circle(frame, (520, 70), 10, (255, 255, 0), -1)
            cv2.putText(frame, "Palm 🖐️ to stop", (420, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    elif state == DrawingState.AR_DISPLAY:
        cv2.rectangle(frame, (5, 5), (500, 85), (0, 0, 0), -1)
        cv2.rectangle(frame, (5, 5), (500, 85), (255, 0, 255), 2)
        cv2.putText(frame, "3D MODE - Rotate your hand!", (15, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.putText(frame, "FIST ✊ to reset | PALM 🖐️ to draw again", (15, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 150, 255), 1)
        if is_fist_detected:
            cv2.circle(frame, (470, 45), 15, (255, 100, 100), -1)
            cv2.putText(frame, "RESET!", (410, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)
    
    # FPS counter with color coding
    fps_color = (0, 255, 0) if fps > 25 else (0, 165, 255) if fps > 15 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, h - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
    cv2.putText(frame, "ESC to quit", (w - 150, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return frame

def pixel_to_centered_space(px, py, canvas_width, canvas_height, scale=1.0):
    cx = (px - canvas_width / 2.0) * (scale / max(canvas_width, canvas_height))
    cy = (canvas_height / 2.0 - py) * (scale / max(canvas_width, canvas_height))
    return cx, cy

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

# Main loop
def main():
    # Initialize camera first
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Camera failed to open.")
        return
    
    # Get camera resolution
    WIDTH = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize Pygame window
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("AI-Powered 3D Drawing in Air")

    x_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)
    y_filter = OneEuroFilter(mincutoff=0.7, beta=0.3)

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

    gesture_check_interval = 3  # Check gestures every 2 frames
    frame_count = 0

# In the main loop, update gesture detection (around line 435):
    frame_count += 1

# Only update gesture counters every N frames
    if frame_count % gesture_check_interval == 0:
        palm_open_now = is_open_palm(hand_landmarks)
        pointer_active_now = is_pointer_gesture(hand_landmarks)
        fist_closed_now = is_fist(hand_landmarks)
    
        palm_stable_count = palm_stable_count + 1 if palm_open_now else 0
        pointer_stable_count = pointer_stable_count + 1 if pointer_active_now else 0
        fist_stable_count = fist_stable_count + 1 if fist_closed_now else 0
    else:
    # Use previous gesture state
        pass

    # FPS tracking
    fps = 0
    frame_times = []
    metrics = MetricsDashboard()
    
    print("3D AR Hand Drawing Ready!")
    print(f"Resolution: {WIDTH}x{HEIGHT}")
    print("Gestures:")
    print("  - OPEN PALM: Start/Stop recording")
    print("  - POINTER (index+thumb): Draw")
    print("  - FIST: Reset from AR mode")

    clock = pygame.time.Clock()

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.6) as hands:
        running = True
        while running:
            frame_start = time.time()
            
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False

            ret, frame = camera.read()
            if not ret:
                print("Camera read failed!")
                break
            
            frame = cv2.flip(frame, 1)
            canvas_height, canvas_width = frame.shape[:2]

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
                
                # Highlight anchor in AR mode
                if drawing_state == DrawingState.AR_DISPLAY:
                    anchor = hand_landmarks.landmark[9]
                    anchor_x = int(anchor.x * canvas_width)
                    anchor_y = int(anchor.y * canvas_height)
                    cv2.circle(frame, (anchor_x, anchor_y), 8, (255, 0, 255), -1)
                    cv2.circle(frame, (anchor_x, anchor_y), 10, (255, 255, 255), 2)
                
                index_tip = hand_landmarks.landmark[8]
                raw_x = index_tip.x * canvas_width
                raw_y = index_tip.y * canvas_height
                smoothed_x = int(x_filter.smooth(raw_x))
                smoothed_y = int(y_filter.smooth(raw_y))
                
                palm_open_now = is_open_palm(hand_landmarks)
                pointer_active_now = is_pointer_gesture(hand_landmarks)
                fist_closed_now = is_fist(hand_landmarks)
                
                palm_stable_count = palm_stable_count + 1 if palm_open_now else 0
                pointer_stable_count = pointer_stable_count + 1 if pointer_active_now else 0
                fist_stable_count = fist_stable_count + 1 if fist_closed_now else 0

                now = time.time()
                
                if drawing_state == DrawingState.IDLE:
                    if palm_stable_count >= GESTURE_REQUIRED and now - last_gesture_time > GESTURE_COOLDOWN:
                        drawing_state = DrawingState.RECORDING
                        recorded_strokes = []
                        current_stroke = []
                        last_gesture_time = now
                        print("▶ RECORDING")
                
                elif drawing_state == DrawingState.RECORDING:
                    if pointer_active_now:
                        current_stroke.append((smoothed_x, smoothed_y))
                        cv2.circle(frame, (smoothed_x, smoothed_y), 6, (0, 255, 0), -1)
                    
                    if palm_stable_count >= GESTURE_REQUIRED and now - last_gesture_time > GESTURE_COOLDOWN:
                        if current_stroke:
                            recorded_strokes.append(current_stroke)
                        
                        all_points = []
                        for stroke in recorded_strokes:
                            sampled_stroke = stroke[::3]
                            for px, py in sampled_stroke:
                                cx, cy = pixel_to_centered_space(px, py, canvas_width, canvas_height)
                                all_points.append((cx * 300, cy * 300))
                        
                        if len(all_points) >= 2:
                            print(f"Generating mesh from {len(all_points)} points...")
                            mesh_start = time.time()
                            ar_mesh_vertices, ar_mesh_faces = generate_tube_mesh_fast(
                                all_points, radius=12.0, segments=6
                            )
                            
                            mesh_time = (time.time() - mesh_start)*1000
                            
                            if ar_mesh_vertices is not None:
                                drawing_state = DrawingState.AR_DISPLAY
                                last_gesture_time = now
                                print("✓ 3D AR MODE")
                                print(f"  Mesh: {len(ar_mesh_vertices)} vertices, {len(ar_mesh_faces)} faces")
                                
                                os.makedirs("generated", exist_ok=True)
                                timestamp = int(time.time() * 1000)
                                save_mesh_obj(ar_mesh_vertices, ar_mesh_faces, 
                                            f"generated/ar_mesh_{timestamp}.obj")
                            else:
                                print("Mesh generation failed")
                        
                        current_stroke = []
                
                elif drawing_state == DrawingState.AR_DISPLAY:
                    # Render 3D mesh
                    if ar_mesh_vertices is not None:
                        hand_transform = get_hand_transform_3d(hand_landmarks, canvas_width, canvas_height)
                        frame = render_mesh_solid(frame, ar_mesh_vertices, ar_mesh_faces, 
                                hand_transform, canvas_width, canvas_height)
                    
                    if fist_stable_count >= GESTURE_REQUIRED and now - last_gesture_time > GESTURE_COOLDOWN:
                        drawing_state = DrawingState.IDLE
                        ar_mesh_vertices = None
                        ar_mesh_faces = None
                        last_gesture_time = now
                        print("↻ RESET")

            else:
                palm_stable_count = 0
                pointer_stable_count = 0
                fist_stable_count = 0

            # Draw strokes in recording mode
            if drawing_state == DrawingState.RECORDING:
                for stroke in recorded_strokes:
                    for i in range(1, len(stroke)):
                        cv2.line(frame, stroke[i-1], stroke[i], (255, 0, 0), 2)
                for i in range(1, len(current_stroke)):
                    cv2.line(frame, current_stroke[i-1], current_stroke[i], (0, 0, 255), 3)

            # Calculate FPS
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))
            metrics.update (fps, results.multi_hand_landmarks is not None)

            # Draw UI
            stroke_count = len(current_stroke) if drawing_state == DrawingState.RECORDING else 0
            frame = draw_ui(frame, drawing_state, stroke_count, 
                           pointer_active_now, palm_open_now, fist_closed_now, fps)
            
            # Display frame using Pygame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(frame_surface, (0, 0))
            
            pygame.display.flip()
            clock.tick(30)

    metrics.print_report()
    camera.release()
    pygame.quit()

if __name__ == "__main__":
    main()
