"""
Autonomous Obstacle Avoidance & ArUco Chaser Robot
Author: Rover & Robotics Engineering Team
Platform: Host (Laptop / Jetson Orin Nano) -> Serial -> ESP32 Motor Controller
Sensors: Intel RealSense Depth Camera (D435/D415/D455)
"""

import time
import math
import sys
import numpy as np
import cv2

# RealSense SDK
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[WARNING] pyrealsense2 not installed. Falling back to webcam simulation if needed.")

# YOLO Object Detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Depth-only collision avoidance will be active.")

# Serial for ESP32
import serial
import serial.tools.list_ports


# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
TARGET_ARUCO_ID = 0               # ID of the ArUco marker to chase (or None to chase any)
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Threshold distances in meters
STOP_AND_SPIN_DISTANCE = 0.12     # Target reached distance (<= 0.10 - 0.12m triggers 360 spin)
OBSTACLE_DANGER_DISTANCE = 0.45   # Obstacle avoidance triggers when an object is closer than 45cm
CRITICAL_STOP_DISTANCE = 0.20     # Emergency stop if an obstacle is too close (20cm)

# Image frame settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30
YOLO_INPUT_SIZE = 320             # 320px tensor cuts CPU compute by 4x for 30+ FPS
YOLO_CONFIDENCE = 0.30

# Serial configuration
SERIAL_PORT = "COM3"              # Replace with your ESP32 COM port (e.g. COM3 on Windows, /dev/ttyUSB0 on Jetson)
BAUD_RATE = 115200

# Classes of small objects to track with YOLO (bottle, cup, book, etc.)
TARGET_OBSTACLE_CLASSES = ["bottle", "cup", "book", "cell phone", "mouse", "keyboard", "bowl"]


class ESP32Interface:
    """Manages Serial communication with ESP32 motor controller."""
    def __init__(self, port=SERIAL_PORT, baudrate=BAUD_RATE):
        self.ser = None
        self.port = port
        self.baudrate = baudrate
        self.last_sent_cmd = None
        self.connect()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            time.sleep(2)  # Wait for ESP32 boot/reset
            print(f"[SERIAL] Connected to ESP32 on {self.port}")
        except Exception as e:
            print(f"[SERIAL WARNING] Could not connect to ESP32 on {self.port}: {e}")
            print("[SERIAL] Operating in MOCK mode (Commands will print to terminal).")
            self.ser = None

    def send_command(self, cmd: str):
        if cmd == self.last_sent_cmd and cmd != "SPIN":
            return  # Avoid flooding identical commands
        self.last_sent_cmd = cmd
        
        msg = f"{cmd}\n"
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(msg.encode())
            except Exception as e:
                print(f"[SERIAL ERROR] Send failed: {e}")
        else:
            print(f"[MOCK MOTOR COMMAND] -> {cmd}")


class RealSenseCamera:
    """Handles RealSense pipeline with aligned Color and Depth frames."""
    def __init__(self, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FRAME_FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.align = None
        self.is_running = False
        self.init_realsense()

    def init_realsense(self):
        if not REALSENSE_AVAILABLE:
            return
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Align depth to color frame
            self.align = rs.align(rs.stream.color)
            self.is_running = True
            print("[REALSENSE] Camera stream started successfully.")
        except Exception as e:
            print(f"[REALSENSE ERROR] Failed to initialize RealSense camera: {e}")
            self.is_running = False

    def get_frames(self):
        if not self.is_running:
            return None, None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=3000)
            aligned_frames = self.align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
            
            color_image = np.asanyarray(color_frame.get_data())
            return color_image, depth_frame
        except Exception as e:
            print(f"[REALSENSE] Frame acquisition error: {e}")
            return None, None

    def get_median_depth(self, depth_frame, center_x, center_y, window=5):
        """Extracts median depth around (center_x, center_y) in meters to eliminate single-pixel noise."""
        if depth_frame is None:
            return 0.0
        
        half = window // 2
        x_min = max(0, int(center_x - half))
        x_max = min(self.width, int(center_x + half + 1))
        y_min = max(0, int(center_y - half))
        y_max = min(self.height, int(center_y + half + 1))
        
        depth_values = []
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                d = depth_frame.get_distance(x, y)
                if 0.05 < d < 6.0:  # Valid sensor range filter
                    depth_values.append(d)
        
        if depth_values:
            return float(np.median(depth_values))
        return 0.0

    def stop(self):
        if self.pipeline and self.is_running:
            self.pipeline.stop()
            self.is_running = False


class ArUcoTracker:
    """Detects ArUco markers and calculates target relative bearing & depth."""
    def __init__(self, dict_type=ARUCO_DICT_TYPE, target_id=TARGET_ARUCO_ID):
        self.target_id = target_id
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        
        # Support both OpenCV 4.7+ and legacy versions
        if hasattr(cv2.aruco, "ArucoDetector"):
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, parameters)
            self.use_new_api = True
        else:
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.use_new_api = False

    def detect(self, color_image, depth_frame, camera_handler):
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        if self.use_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        target_info = None

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            for i in range(len(ids)):
                marker_id = ids[i][0]
                if self.target_id is None or marker_id == self.target_id:
                    # Calculate center pixel
                    c = corners[i][0]
                    cx = int(c[:, 0].mean())
                    cy = int(c[:, 1].mean())
                    
                    # Fetch depth in meters
                    dist = camera_handler.get_median_depth(depth_frame, cx, cy)
                    target_info = {
                        "id": marker_id,
                        "center": (cx, cy),
                        "distance": dist,
                        "corners": c
                    }
                    break

        return target_info


class ObstacleDetector:
    """
    Dual-layer Obstacle Detection:
    Layer 1: YOLOv8 for small object recognition (bottle, pen, book, mug, etc.)
    Layer 2: RealSense 3-Sector Depth Corridor (Left, Center, Right) for ground objects that YOLO might miss.
    """
    def __init__(self):
        self.yolo_model = None
        if YOLO_AVAILABLE:
            try:
                print("[YOLO] Loading YOLO-World v2 open-vocabulary model...")
                self.yolo_model = YOLO("yolov8s-worldv2.pt")
                self.classes = ["bottle", "pen", "pencil", "watch", "wristwatch", "mug", "glass", "cup", "book", "cell phone", "obstacle"]
                if hasattr(self.yolo_model, "set_classes"):
                    self.yolo_model.set_classes(self.classes)
                    print("[YOLO] Model loaded with custom obstacle classes.")
                else:
                    print("[YOLO] Standard YOLO detection active.")
            except Exception as e:
                print(f"[YOLO WARNING] Could not load model: {e}")

    def analyze(self, color_image, depth_frame, camera_handler):
        obstacles = []
        h, w = color_image.shape[:2]

        # 1. YOLO Object Detection for small objects
        if self.yolo_model:
            results = self.yolo_model(color_image, imgsz=416, conf=0.18, verbose=False)
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                if hasattr(self.yolo_model, "names") and cls_id in self.yolo_model.names:
                    cls_name = self.yolo_model.names[cls_id]
                elif hasattr(self, "classes") and cls_id < len(self.classes):
                    cls_name = self.classes[cls_id]
                else:
                    cls_name = "obstacle"
                conf = float(box.conf[0])
                
                # All custom classes are valid obstacles
                if True:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    dist = camera_handler.get_median_depth(depth_frame, cx, cy)
                    
                    if 0.05 < dist < OBSTACLE_DANGER_DISTANCE + 0.3:
                        obstacles.append({
                            "type": cls_name,
                            "bbox": (x1, y1, x2, y2),
                            "center": (cx, cy),
                            "distance": dist
                        })
                        # Draw detection
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(color_image, f"{cls_name} {dist:.2f}m", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # 2. Sector Depth Corridor Analysis (Covers ground zone in front of rover)
        # Slices lower 40% of depth image into 3 vertical zones: Left, Center, Right
        y_ground_start = int(h * 0.50)
        y_ground_end = int(h * 0.90)
        
        sector_w = w // 3
        sectors = {"LEFT": (0, sector_w), "CENTER": (sector_w, 2 * sector_w), "RIGHT": (2 * sector_w, w)}
        sector_depths = {}

        for name, (sx1, sx2) in sectors.items():
            scx = (sx1 + sx2) // 2
            scy = (y_ground_start + y_ground_end) // 2
            s_dist = camera_handler.get_median_depth(depth_frame, scx, scy, window=15)
            sector_depths[name] = s_dist

        return obstacles, sector_depths


class AutonomousNavigator:
    """Decision Engine: Fuses ArUco target chasing and reactive obstacle avoidance."""
    def __init__(self, esp32_client):
        self.esp32 = esp32_client
        self.state = "SEARCHING_TARGET"
        self.completed_spin = False

    def update(self, target_info, obstacles, sector_depths, frame_width):
        """
        State Machine & Reactive Navigation Logic:
        1. Check if Target reached (distance <= STOP_AND_SPIN_DISTANCE): Execute SPIN.
        2. Check for immediate obstacles in collision path. If blocked, steer around.
        3. If path is clear and target is visible, track and move towards ArUco.
        4. If target is lost, search or idle safely.
        """
        center_x = frame_width // 2

        # 1. Target Arrival Condition (Victory Spin!)
        if target_info and target_info["distance"] > 0:
            if target_info["distance"] <= STOP_AND_SPIN_DISTANCE and not self.completed_spin:
                self.state = "TARGET_REACHED_SPINNING"
                print(f"[MISSION SUCCESS] Target distance: {target_info['distance']:.2f}m! Triggering 360 Spin!")
                self.esp32.send_command("SPIN")
                self.completed_spin = True
                return self.state

        if self.completed_spin:
            self.state = "MISSION_COMPLETED"
            self.esp32.send_command("S")
            return self.state

        # 2. Collision Risk Evaluation
        center_depth = sector_depths.get("CENTER", 0.0)
        left_depth = sector_depths.get("LEFT", 0.0)
        right_depth = sector_depths.get("RIGHT", 0.0)

        # Check if any detected obstacle is in the critical center corridor
        center_obstacle = any(
            (obs["center"][0] > frame_width * 0.3) and 
            (obs["center"][0] < frame_width * 0.7) and 
            (0.05 < obs["distance"] < OBSTACLE_DANGER_DISTANCE)
            for obs in obstacles
        ) or (0.05 < center_depth < OBSTACLE_DANGER_DISTANCE)

        if center_obstacle:
            # Immediate obstacle avoidance required!
            # Decide avoidance direction based on sector clearance
            if left_depth > right_depth:
                self.state = "AVOIDING_OBSTACLE (STEER LEFT)"
                self.esp32.send_command("L")
            else:
                self.state = "AVOIDING_OBSTACLE (STEER RIGHT)"
                self.esp32.send_command("R")
            return self.state

        # 3. Path is clear: Chase ArUco Tag
        if target_info:
            tag_cx = target_info["center"][0]
            tag_dist = target_info["distance"]
            err_x = tag_cx - center_x  # Horizontal pixel error from camera center

            # Deadband of +- 45 pixels to avoid oscillation
            if err_x < -50:
                self.state = "CHASING_TAG (ALIGN LEFT)"
                self.esp32.send_command("L")
            elif err_x > 50:
                self.state = "CHASING_TAG (ALIGN RIGHT)"
                self.esp32.send_command("R")
            else:
                self.state = f"CHASING_TAG (FORWARD - {tag_dist:.2f}m)"
                self.esp32.send_command("F")
        else:
            # ArUco tag not in view
            self.state = "SEARCHING_FOR_TAG (SCANNING)"
            self.esp32.send_command("S")

        return self.state


def main():
    print("==========================================================")
    print("  Mars Rover Prototype: RealSense Obstacle Avoidance + ArUco")
    print("==========================================================")
    
    # Initialize hardware interfaces
    esp32 = ESP32Interface(port=SERIAL_PORT, baudrate=BAUD_RATE)
    camera = RealSenseCamera(width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FRAME_FPS)
    aruco = ArUcoTracker(target_id=TARGET_ARUCO_ID)
    obstacle_detector = ObstacleDetector()
    navigator = AutonomousNavigator(esp32)

    if not camera.is_running:
        print("[CRITICAL] RealSense camera not detected or failed to start.")
        print("Please connect Intel RealSense USB 3.0 port and restart.")
        return

    print("[SYSTEM] Starting autonomous navigation loop. Press 'q' to quit.")

    try:
        while True:
            color_img, depth_frame = camera.get_frames()
            if color_img is None:
                continue

            # 1. Detect ArUco Marker
            target_info = aruco.detect(color_img, depth_frame, camera)

            # 2. Detect Small Obstacles & Corridor Depths
            obstacles, sector_depths = obstacle_detector.analyze(color_img, depth_frame, camera)

            # 3. Execute Navigation & Avoidance Decision Engine
            nav_state = navigator.update(target_info, obstacles, sector_depths, FRAME_WIDTH)

            # 4. HUD / Visual Feedback Display
            cv2.putText(color_img, f"STATE: {nav_state}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            if target_info:
                cv2.putText(color_img, f"Target ID {target_info['id']}: {target_info['distance']:.2f}m",
                            (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.circle(color_img, target_info["center"], 6, (0, 255, 0), -1)

            # Display sector clearances at bottom of screen
            c_info = f"L: {sector_depths.get('LEFT', 0):.2f}m | C: {sector_depths.get('CENTER', 0):.2f}m | R: {sector_depths.get('RIGHT', 0):.2f}m"
            cv2.putText(color_img, c_info, (15, FRAME_HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            # Show live camera feed
            cv2.imshow("Rover RealSense AI Navigation", color_img)

            # Exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[USER] Quitting...")
                break

    except KeyboardInterrupt:
        print("[USER] Interrupted.")
    finally:
        esp32.send_command("S")
        camera.stop()
        cv2.destroyAllWindows()
        print("[SYSTEM] Safe shutdown complete.")


if __name__ == "__main__":
    main()
