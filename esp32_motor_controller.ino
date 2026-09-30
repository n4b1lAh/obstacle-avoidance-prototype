/*
 * ESP32 Robot Motor Controller
 * Receives navigation commands over Serial (from Laptop / Jetson Orin Nano running RealSense AI)
 * Controls 2-wheel differential drive or 4-wheel skid steer via standard H-Bridge (L298N / TB6612 / BTS7960)
 *
 * Commands received via Serial (at 115200 baud):
 *   "F"      - Move Forward
 *   "B"      - Move Backward
 *   "L"      - Steer / Turn Left
 *   "R"      - Steer / Turn Right
 *   "S"      - Stop
 *   "SPIN"   - Execute 360-degree victory spin when ArUco target is reached (< 0.1m)
 */

// Define Pin Connections (Adjust to match your ESP32 pinout)
const int PIN_ENA = 13;   // Left Motor Speed (PWM)
const int PIN_IN1 = 12;   // Left Motor Direction 1
const int PIN_IN2 = 14;   // Left Motor Direction 2

const int PIN_IN3 = 27;   // Right Motor Direction 1
const int PIN_IN4 = 26;   // Right Motor Direction 2
const int PIN_ENB = 25;   // Right Motor Speed (PWM)

// PWM Configuration for ESP32
const int PWM_FREQ = 1000;      // 1 kHz
const int PWM_RES = 8;          // 8-bit resolution (0-255)
const int PWM_CHAN_A = 0;
const int PWM_CHAN_B = 1;

// Default speeds (0-255)
const int BASE_SPEED = 180;
const int TURN_SPEED = 160;
const int SPIN_SPEED = 210;

// Safety timeout: stop if no command is received within 800ms
unsigned long lastCommandTime = 0;
const unsigned long TIMEOUT_MS = 800;

void setup() {
  Serial.begin(115200);
  
  // Set control pins as outputs
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  pinMode(PIN_IN3, OUTPUT);
  pinMode(PIN_IN4, OUTPUT);

  // Setup PWM Channels
  ledcAttachChannel(PIN_ENA, PWM_FREQ, PWM_RES, PWM_CHAN_A);
  ledcAttachChannel(PIN_ENB, PWM_FREQ, PWM_RES, PWM_CHAN_B);

  stopMotors();
  Serial.println("ESP32 Motor Controller Ready. Awaiting commands...");
}

void loop() {
  // Check for incoming serial commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    cmd.toUpperCase();

    if (cmd.length() > 0) {
      lastCommandTime = millis();
      executeCommand(cmd);
    }
  }

  // Watchdog timeout to prevent runaway robot if connection drops
  if (millis() - lastCommandTime > TIMEOUT_MS) {
    stopMotors();
  }
}

void executeCommand(String cmd) {
  if (cmd == "F") {
    moveForward(BASE_SPEED);
  } else if (cmd == "B") {
    moveBackward(BASE_SPEED);
  } else if (cmd == "L") {
    turnLeft(TURN_SPEED);
  } else if (cmd == "R") {
    turnRight(TURN_SPEED);
  } else if (cmd == "S") {
    stopMotors();
  } else if (cmd == "SPIN") {
    celebrationSpin();
  } else {
    // Unrecognized command
    stopMotors();
  }
}

void moveForward(int speed) {
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  ledcWriteChannel(PWM_CHAN_A, speed);
  ledcWriteChannel(PWM_CHAN_B, speed);
}

void moveBackward(int speed) {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
  ledcWriteChannel(PWM_CHAN_A, speed);
  ledcWriteChannel(PWM_CHAN_B, speed);
}

void turnLeft(int speed) {
  // Pivot left (Left backward, Right forward)
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, HIGH);
  digitalWrite(PIN_IN3, HIGH);
  digitalWrite(PIN_IN4, LOW);
  ledcWriteChannel(PWM_CHAN_A, speed);
  ledcWriteChannel(PWM_CHAN_B, speed);
}

void turnRight(int speed) {
  // Pivot right (Left forward, Right backward)
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, HIGH);
  ledcWriteChannel(PWM_CHAN_A, speed);
  ledcWriteChannel(PWM_CHAN_B, speed);
}

void stopMotors() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  digitalWrite(PIN_IN3, LOW);
  digitalWrite(PIN_IN4, LOW);
  ledcWriteChannel(PWM_CHAN_A, 0);
  ledcWriteChannel(PWM_CHAN_B, 0);
}

// 360-degree victory spin triggered when target ArUco is within 0.1m
void celebrationSpin() {
  Serial.println("Executing ArUco Target Reached Celebration Spin!");
  turnRight(SPIN_SPEED);
  // Calibrate this delay according to your robot's chassis & wheel traction (~1.8 - 2.5s for 360 deg)
  delay(2200);
  stopMotors();
  Serial.println("Spin Completed. Target Mission Success.");
  lastCommandTime = millis();
}
