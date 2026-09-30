# Autonomous ArUco Chaser & Obstacle Avoidance Robot

## Architecture Overview

```
 [Intel RealSense Depth Camera]
             │ (USB 3.0 / Depth + RGB)
             ▼
┌────────────────────────────────────────────────────────┐
│  Host Brain (Laptop / PC or Jetson Orin Nano)          │
│  - Python 3.10+ (main_robot_controller.py)             │
│  - RealSense SDK (pyrealsense2)                        │
│  - Dual Obstacle Detection:                            │
│     * YOLOv8 (Detects bottle, cup, mug, pen, book, etc)│
│     * Depth Corridor Grid (Ground clearance check)     │
│  - ArUco Tag Tracker + Distance Filter                 │
│  - State Machine & Reactive Navigation Engine          │
│  - Distance <= 0.1m trigger -> Victory Spin            │
└────────────────────────────────────────────────────────┘
             │ (USB Serial @ 115200 Baud: 'F', 'L', 'R', 'S', 'SPIN')
             ▼
┌────────────────────────────────────────────────────────┐
│  Low-Level Actuator Controller (ESP32)                 │
│  - Arduino Sketch (esp32_motor_controller.ino)         │
│  - PWM Speed & Direction Control                       │
│  - Fail-safe Watchdog (Auto-stops if connection lost)  │
│  - 360° Victory Spin Routine                           │
└────────────────────────────────────────────────────────┘
             │
             ▼
    [Motor Driver (L298N / TB6612FNG)]
             │
             ▼
      [DC Gear Motors / Rover Wheels]
```

---

## 1. Hardware Pinout (ESP32 to Motor Driver)

Default pins configured in [esp32_motor_controller.ino](file:///c:/Users/Surface%20laptop%204/OneDrive/Desktop/objectdetection/esp32_motor_controller.ino):

| Function | ESP32 GPIO | Motor Driver (e.g., L298N) |
|---|---|---|
| Left Motor PWM (ENA) | GPIO 13 | ENA |
| Left Motor Dir 1 (IN1) | GPIO 12 | IN1 |
| Left Motor Dir 2 (IN2) | GPIO 14 | IN2 |
| Right Motor Dir 1 (IN3) | GPIO 27 | IN3 |
| Right Motor Dir 2 (IN4) | GPIO 26 | IN4 |
| Right Motor PWM (ENB) | GPIO 25 | ENB |
| Ground | GND | GND (Common ground with battery & driver) |

---

## 2. Software Installation

1. Open your terminal / command prompt in this directory:
   ```bash
   pip install -r requirements.txt
   ```

2. Upload [esp32_motor_controller.ino](file:///c:/Users/Surface%20laptop%204/OneDrive/Desktop/objectdetection/esp32_motor_controller.ino) to your ESP32 board using Arduino IDE.

3. Verify your ESP32 COM port in Device Manager (e.g., `COM3` on Windows, or `/dev/ttyUSB0` on Linux/Jetson). Update `SERIAL_PORT` in [main_robot_controller.py](file:///c:/Users/Surface%20laptop%204/OneDrive/Desktop/objectdetection/main_robot_controller.py) if different.

4. Connect your **Intel RealSense** camera to a USB 3.0 port (blue port).

---

## 3. Running the Robot

Run the main navigation program:
```bash
python main_robot_controller.py
```

### Features:
- **Dual Obstacle Avoidance**:
  - **YOLOv8** identifies small everyday objects (bottles, mugs, cups, books, electronics).
  - **RealSense 3-Sector Ground Depth**: Even for tiny objects like pens lying on the floor that neural networks might miss, the RealSense 3D depth grid directly measures ground obstruction and detours the rover safely.
- **ArUco Tracking**:
  - Tracks the target marker (default ID 0).
  - Centers marker in the camera view using proportional steering.
- **Target Reached & 360° Spin**:
  - Continuous millimeter-accurate depth measurement.
  - When distance reaches **<= 0.10m - 0.12m**, sends the `SPIN` command to perform a 360-degree victory spin and halt!

---

## 4. Migrating to Jetson Orin Nano for Mars Rover

When moving from Laptop to Jetson Orin Nano:
1. Simply install `pyrealsense2` and Jetson-compatible PyTorch/Ultralytics on the Orin Nano.
2. Change `SERIAL_PORT = "/dev/ttyUSB0"` or `/dev/ttyTHS1`.
3. The exact same Python brain runs on the Orin Nano with hardware-accelerated TensorRT for 60+ FPS inference!
