#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import threading
import lgpio
import signal
import sys

# =============== CAI DAT THONG SO ===============
FRAME_W, FRAME_H = 1280, 720
NEAR_THRESHOLD = 0.08      # nguong gan co canh bao
CRITICAL_PERSON = 0.10     # nguoi >10% => cuc nguy hiem (D4)
CRITICAL_OBJECT = 0.15     # vat nguy hiem >15% => cuc nguy hiem (D4)

MODEL_PATH = "yolov8n.pt"

DANGEROUS_OBJECTS = {
    "car": "xe hoi",
    "truck": "xe tai",
    "bus": "xe buyt",
    "motorcycle": "xe may",
    "bicycle": "xe dap",
    "dog": "cho",
    "knife": "dao",
    "scissors": "keo",
    "fire": "lua"
}

# LED PINS (BCM)
LED_RED = 17
LED_GREEN = 27
LED_BLUE = 22

# LGPIO
CHIP = 0
handle = lgpio.gpiochip_open(CHIP)

# =============== HAM DIEU KHIEN LED ===============
def led_off():
    lgpio.gpio_write(handle, LED_RED, 0)
    lgpio.gpio_write(handle, LED_GREEN, 0)
    lgpio.gpio_write(handle, LED_BLUE, 0)

def led_green():
    led_off()
    lgpio.gpio_write(handle, LED_GREEN, 1)

def led_yellow():
    led_off()
    lgpio.gpio_write(handle, LED_RED, 1)
    lgpio.gpio_write(handle, LED_GREEN, 1)

def led_red():
    led_off()
    lgpio.gpio_write(handle, LED_RED, 1)

def led_red_blink():
    # nhap nhay 3 lan khi cuc nguy hiem
    for _ in range(3):
        lgpio.gpio_write(handle, LED_RED, 1)
        time.sleep(0.15)
        lgpio.gpio_write(handle, LED_RED, 0)
        time.sleep(0.15)

# =============== CAMERA ===============
def init_camera():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_W, FRAME_H)}
    )
    picam2.configure(config)
    picam2.start()
    print("[CAM] Camera da san sang")
    return picam2

# =============== VE BOX LEN KHUNG HINH ===============
def draw_boxes(frame, boxes, labels, confs):
    for (x1, y1, x2, y2), name, conf in zip(boxes, labels, confs):
        color = (0, 0, 255) if name in DANGEROUS_OBJECTS else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# =============== THOAT CHUONG TRINH AN TOAN ===============
def cleanup(*args):
    print("\n[EXIT] Dang tat LED va giai phong GPIO...")
    led_off()
    lgpio.gpiochip_close(handle)
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# =============== MAIN ===============
def main():
    print("[FTSCENE] He thong dang khoi dong...")

    # claim GPIO
    lgpio.gpio_claim_output(handle, LED_RED, 0)
    lgpio.gpio_claim_output(handle, LED_GREEN, 0)
    lgpio.gpio_claim_output(handle, LED_BLUE, 0)

    model = YOLO(MODEL_PATH)
    picam2 = init_camera()

    cv2.namedWindow("FTSCENE - LED RGB", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FTSCENE - LED RGB", 1000, 600)

    while True:
        frame = picam2.capture_array()
        if frame is None:
            print("[CAM] Khong doc duoc frame")
            time.sleep(0.1)
            continue

        results = model(frame, imgsz=640, conf=0.5, verbose=False)
        r = results[0]
        boxes, labels, confs = [], [], []
        h, w = frame.shape[:2]

        status = "safe"  # safe / warning / critical

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = r.names.get(cls, str(cls))
            boxes.append((x1, y1, x2, y2))
            labels.append(name)
            confs.append(conf)

            area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

            # D4 logic
            if name == "person" and area_ratio >= CRITICAL_PERSON:
                status = "critical"
            elif name in DANGEROUS_OBJECTS and area_ratio >= CRITICAL_OBJECT:
                status = "critical"
            elif name in DANGEROUS_OBJECTS and area_ratio >= NEAR_THRESHOLD:
                status = "warning"

        # Dieu khien LED
        if status == "safe":
            led_green()
        elif status == "warning":
            led_yellow()
        elif status == "critical":
            threading.Thread(target=led_red_blink, daemon=True).start()

        draw_boxes(frame, boxes, labels, confs)

        cv2.putText(frame, "Nhan dien dang hoat dong...",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        cv2.imshow("FTSCENE - LED RGB", frame)
        if cv2.waitKey(1) == ord('q'):
            cleanup()

if __name__ == "__main__":
    main()
