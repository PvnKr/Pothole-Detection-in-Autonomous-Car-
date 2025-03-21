from ultralytics import YOLO
import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
import threading
import queue

# Load YOLO model
model = YOLO("best21.pt")
class_names = model.names

# GPIO Pin Definitions
IN1, IN2, IN3, IN4, ENA, ENB = 17, 27, 22, 23, 18, 19
BUZZER_PIN = 24  # Buzzer Pin

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB, BUZZER_PIN], GPIO.OUT)

# PWM setup
pwmA = GPIO.PWM(ENA, 100)  # 100 Hz frequency for Motor A
pwmB = GPIO.PWM(ENB, 100)  # 100 Hz frequency for Motor B
pwmA.start(0)
pwmB.start(0)

# Motor control functions
def set_motor(motor, direction, speed):
    if motor == "A":
        GPIO.output(IN1, direction[0])
        GPIO.output(IN2, direction[1])
        pwmA.ChangeDutyCycle(speed)
    elif motor == "B":
        GPIO.output(IN3, direction[0])
        GPIO.output(IN4, direction[1])
        pwmB.ChangeDutyCycle(speed)

def move_forward(speed=53):
    set_motor("A", (GPIO.HIGH, GPIO.LOW), speed)
    set_motor("B", (GPIO.HIGH, GPIO.LOW), speed)

def turn_left(speed=53):
    set_motor("A", (GPIO.LOW, GPIO.LOW), 0)
    set_motor("B", (GPIO.HIGH, GPIO.LOW), speed)

def turn_right(speed=53):
    set_motor("A", (GPIO.HIGH, GPIO.LOW), speed)
    set_motor("B", (GPIO.LOW, GPIO.LOW), 0)

def stop():
    set_motor("A", (GPIO.LOW, GPIO.LOW), 0)
    set_motor("B", (GPIO.LOW, GPIO.LOW), 0)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce internal buffer to 1 frame for low latency

frame_queue = queue.Queue(maxsize=1)  # Only hold the latest frame
result_queue = queue.Queue(maxsize=1)

# Frame capture thread
def capture_frames():
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_queue.mutex:
                frame_queue.queue.clear()  # Clear old frames
            frame_queue.put(frame)

# YOLO detection thread - Process every 3rd frame
def yolo_detection():
    frame_count = 0
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            frame_count += 1

            if frame_count % 3 == 0:  # Process only every 3rd frame
                frame = cv2.resize(frame, (480, 270))
                results = model.predict(frame, device="cpu", imgsz=320, verbose=False)
                with result_queue.mutex:
                    result_queue.queue.clear()
                result_queue.put((frame, results))

# Start threads
capture_thread = threading.Thread(target=capture_frames, daemon=True)
detection_thread = threading.Thread(target=yolo_detection, daemon=True)
capture_thread.start()
detection_thread.start()

try:
    while True:
        if not result_queue.empty():
            frame, results = result_queue.get()
            h, w, _ = frame.shape
            pothole_detected = False

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x, y, x1, y1 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    c = class_names[cls]

                    if c == "pothole":
                        pothole_detected = True
                        center_x = (x + x1) // 2

                        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn on buzzer
                        time.sleep(0.5)
                        GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn off buzzer

                        if w // 3 < center_x < 2 * w // 3:
                            print("Pothole in center. Overtaking.")
                            
                            turn_left()
                            time.sleep(1.25)
                            move_forward()
                            time.sleep(1.75)
                            turn_right()
                            time.sleep(1.25)
                            move_forward()
                            time.sleep(2)
                            turn_right()
                            time.sleep(1.25)
                            move_forward()
                            time.sleep(1.75)
                            turn_left()
                            time.sleep(1.25)
                            move_forward()
                            time.sleep(1.75)
                        else:
                            print("Pothole detected but not in center. Slowing Down.")
                            move_forward(40)

                        cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)
                        cv2.putText(frame, f"{c}", (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if not pothole_detected:
                print("No pothole detected. Moving forward.")
                move_forward()

            cv2.imshow('Pothole Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    GPIO.output(BUZZER_PIN, GPIO.LOW)  # Ensure buzzer is off
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()
