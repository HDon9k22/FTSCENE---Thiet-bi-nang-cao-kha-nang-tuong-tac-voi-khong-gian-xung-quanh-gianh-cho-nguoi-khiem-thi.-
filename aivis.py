#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aivis - Thiet bi nang cao kha nang tuong tac voi khong gian xung quanh gianh cho nguoi khiem thi.
Tac gia: hoang thi bao chau - ptdtnt thpt dak glei
Dong tac gia: le thien thinh - DHSP*DHDN.
Ngon ngu: python
LED: Common Cathode (Chung GND)
Update: 02/12/2025
    - Chi su dung 2 LED (Xanh/Do)
    - Them Loa (Buzzer) canh bao 3 nhip khi chuyen sang Do
    - Su dung 4 luong (Camera, AI, LED, Loa) de giam giat lag
    - LED Do nhay nhanh dan khi vat the den gan (area_ratio lon)
    - Tang FPS xu ly AI bang cach giam imgsz=416
    - Luu anh khi phat hien vat the nguy hiem vao thu muc rieng.
    - Phat am thanh qua thiet bi bluetooth/.mp3 khi o trang thai nguy hiem
"""

from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import threading
import lgpio
import signal
import sys
import os
import pygame 

# =============== CAI DAT THONG SO ===============
FRAME_W, FRAME_H = 1280, 720
MODEL_PATH = "yolov8n.pt"

# [MOI] Duong dan den file am thanh
# Da doi ten thu muc tu FTSENCE thanh aivis
SOUND_FILE_PATH = "/home/huudong/Desktop/aivis/mp3 dangerous.mp3" 

# --- Nguong canh bao ---
# Bat dau canh bao (nhay cham) khi vat > 8% man hinh
START_DANGER_RATIO = 0.08
# Dat canh bao toi da (nhay nhanh nhat) khi vat > 30% man hinh
MAX_DANGER_RATIO = 0.30 

# --- Toc do nhay LED (nua chu ky, tinh bang giay) ---
MIN_BLINK_DELAY = 0.05 # Nhanh nhat
MAX_BLINK_DELAY = 0.4  # Cham nhat

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

# --- PINS  ---
LED_RED = 17
LED_GREEN = 27
BUZZER_PIN = 18

# LGPIO
CHIP = 0
try:
    handle = lgpio.gpiochip_open(CHIP)
except Exception as e:
    print(f"[ERROR] Khong the mo lgpio chip {CHIP}. {e}")
    print("Hay dam bao ban da chay 'sudo pigpiod' (neu dung pigpio) hoac co quyen truy cap.")
    sys.exit(1)

# --- Thu muc luu anh nguy hiem --- 
DANGER_CAPTURE_DIR = "dangerous_captures"
os.makedirs(DANGER_CAPTURE_DIR, exist_ok=True) # Tao thu muc neu chua ton tai

# --- Globals cho cac luong  ---

# LED
led_state = "safe" # 'safe' (green) or 'danger' (red blink)
blink_delay = MAX_BLINK_DELAY  
stop_led_thread = threading.Event()
led_thread = None # Se gan sau

# Camera
latest_frame = None
frame_lock = threading.Lock()
camera_running = True
cam_thread = None # Se gan sau
# =============== HAM DIEU KHIEN LOA (BUZZER) ===============
def beep_worker(count, on_delay, off_delay):
    """
    Worker chay trong luong rieng de tao tieng beep.
    Se tu dong dung neu co lenh thoat (stop_led_thread).
    """
    for _ in range(count):
        if stop_led_thread.is_set(): 
            break
        try:
            lgpio.gpio_write(handle, BUZZER_PIN, 1)
            time.sleep(on_delay)
            lgpio.gpio_write(handle, BUZZER_PIN, 0)
            time.sleep(off_delay)
        except Exception:
            break # Stop beeping if GPIO fails
    
    # Dam bao loa da tat sau khi keu xong
    try:
        lgpio.gpio_write(handle, BUZZER_PIN, 0)
    except Exception:
        pass


def beep_buzzer_non_blocking(beep_count=3, on_delay=0.15, off_delay=0.1):
    """
    Kich hoat loa keu 'beep_count' lan ma khong lam chan luong chinh.
    """
    # Chay worker trong 1 luong rieng de khong bi chan
    beep_thread = threading.Thread(target=beep_worker, args=(beep_count, on_delay, off_delay), daemon=True)
    beep_thread.start()

# =============== HAM DIEU KHIEN LED ===============
def led_off():
    """Tat ca 2 den LED."""
    try:
        lgpio.gpio_write(handle, LED_RED, 0)
        lgpio.gpio_write(handle, LED_GREEN, 0)
    except Exception as e:
        pass # Loi xay ra khi da cleanup

def led_green():
    led_off()
    lgpio.gpio_write(handle, LED_GREEN, 1)

def led_red():
    led_off()
    lgpio.gpio_write(handle, LED_RED, 1)

def led_controller_thread():
    """Luan rieng de dieu khien LED khong chan luong chinh."""
    global led_state, blink_delay
    
    print("[LED] LED controller thread bat dau.")
    while not stop_led_thread.is_set():
        try:
            if led_state == "safe":
                led_green() # Da bao gom tat do
                # Sleep de giam tai CPU
                time.sleep(0.1) 
            elif led_state == "danger":
                lgpio.gpio_write(handle, LED_GREEN, 0) # Dam bao xanh tat
                
                # Kiem tra truoc khi bat den
                if stop_led_thread.is_set(): break
                lgpio.gpio_write(handle, LED_RED, 1)
                time.sleep(blink_delay)
                
                # Kiem tra truoc khi tat den
                if stop_led_thread.is_set(): break
                lgpio.gpio_write(handle, LED_RED, 0)
                time.sleep(blink_delay)
                
        except Exception as e:
            if not stop_led_thread.is_set():
                print(f"[LED ERROR] Loi trong luong LED: {e}")
                time.sleep(1) # Neu co loi thi cho
    
    print("[LED] LED controller thread dung.")
    led_off() # Tat den khi thoat

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

def camera_capture_thread(picam2):
    """Luan rieng de lien tuc lay khung hinh tu camera."""
    global latest_frame, frame_lock, camera_running
    print("[CAM] Luong lay khung hinh bat dau.")
    while camera_running:
        try:
            new_frame = picam2.capture_array()
            with frame_lock:
                latest_frame = new_frame
        except Exception as e:
            print(f"[CAM ERROR] Khong the capture frame: {e}")
            time.sleep(0.1) # Doi mot chut neu co loi
    
    print("[CAM] Luong lay khung hinh dung.")
    picam2.stop()
    print("[CAM] Camera da stop.")

# =============== VE BOX LEN KHUNG HINH ===============
def draw_boxes(frame, boxes, labels, confs):
    for (x1, y1, x2, y2), name, conf in zip(boxes, labels, confs):
        color = (0, 0, 255) if (name in DANGEROUS_OBJECTS or name == "person") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# =============== THOAT CHUONG TRINH AN TOAN ===============
def cleanup(*args):
    global camera_running, stop_led_thread, cam_thread, led_thread
    
    print("\n[EXIT] Dang dung cac luong...")
    
    # 1. Dung luong camera
    camera_running = False
    if cam_thread and cam_thread.is_alive():
        cam_thread.join(timeout=1.0)
        
    # 2. Dung luong LED (va loa)
    stop_led_thread.set()
    if led_thread and led_thread.is_alive():
        led_thread.join(timeout=1.0)

    print("[EXIT] Dang tat LED, tat Loa va giai phong GPIO...")
    # Tat loa ngay lap tuc
    try:
        lgpio.gpio_write(handle, BUZZER_PIN, 0)
    except Exception:
        pass
        
    led_off() # Tat den lan cuoi
    lgpio.gpiochip_close(handle)
    
    # [MOI] Dung am thanh va thoat pygame
    print("[EXIT] Dang tat am thanh...")
    try:
        pygame.mixer.music.stop()
        pygame.quit()
    except Exception as e:
        print(f"[EXIT] Loi khi tat pygame: {e}")

    print("[EXIT] Dong cua so...")
    cv2.destroyAllWindows()
    print("[EXIT] Thoat chuong trinh.")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# =============== MAIN ===============
def main():
    global cam_thread, led_thread 
    global led_state, blink_delay
    
    print("[aivis] He thong dang khoi dong...")

    # claim GPIO
    try:
        lgpio.gpio_claim_output(handle, LED_RED, 0)
        lgpio.gpio_claim_output(handle, LED_GREEN, 0)
        lgpio.gpio_claim_output(handle, BUZZER_PIN, 0)
    except Exception as e:
        print(f"[ERROR] Khong the claim GPIO: {e}")
        lgpio.gpiochip_close(handle)
        sys.exit(1)

    # [MOI] Khoi tao bo phat am thanh
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(SOUND_FILE_PATH)
        print(f"[AUDIO] Da tai file am thanh: {SOUND_FILE_PATH}")
    except Exception as e:
        print(f"[AUDIO ERROR] Khong the khoi tao pygame hoac tai file am thanh: {e}")
        print("[AUDIO] Chuong trinh se tiep tuc chay ma khong co am thanh.")
        
    model = YOLO(MODEL_PATH)
    picam2 = init_camera()

    # Bat dau luong LED
    led_thread = threading.Thread(target=led_controller_thread, daemon=True)
    led_thread.start()
    
    # Bat dau luong Camera
    cam_thread = threading.Thread(target=camera_capture_thread, args=(picam2,), daemon=True)
    cam_thread.start()
    
    print("[MAIN] Da khoi dong luong Camera va LED.")

    # [CHANGE] Doi ten cua so thanh aivis
    window_name = "aivis - LED CANH BAO (Xanh/Do)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000, 600)
    
    last_print_time = time.time()
    
    # Bien theo doi trang thai truoc do de kich hoat loa va luu anh
    previous_led_state = "safe" 
    is_sound_playing = False # Bien theo doi trang thai am thanh
    while True:
        # Lay frame moi nhat tu luong camera
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
        
        if frame is None:
            # Cho camera khoi dong va cung cap frame dau tien
            if time.time() - last_print_time > 1.0:
                print("[MAIN] Dang cho frame dau tien tu camera...")
                last_print_time = time.time()
            time.sleep(0.05)
            continue
        
        # Thuc hien xu ly AI
        results = model(frame, imgsz=416, conf=0.5, verbose=False) # imgsz=416 de tang toc do
        
        r = results[0]
        boxes, labels, confs = [], [], []
        h, w = frame.shape[:2]
        max_danger_ratio = 0.0 # Tim vat nguy hiem lon nhat
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = r.names.get(cls, str(cls))
            
            boxes.append((x1, y1, x2, y2))
            labels.append(name)
            confs.append(conf)

            area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

            # Logic: Tim vat the nguy hiem co area_ratio lon nhat
            if name == "person" or name in DANGEROUS_OBJECTS:
                max_danger_ratio = max(max_danger_ratio, area_ratio)
                # Dieu khien LED dua tren vat nguy hiem lon nhat
        if max_danger_ratio < START_DANGER_RATIO:
            # An toan
            led_state = "safe"
        else:
            # Nguy hiem
            led_state = "danger"
            
            # Tinh toan toc do nhay
            if max_danger_ratio >= MAX_DANGER_RATIO:
                blink_delay = MIN_BLINK_DELAY
            else:
                # Noi suy tuyen tinh
                progress = (max_danger_ratio - START_DANGER_RATIO) / (MAX_DANGER_RATIO - START_DANGER_RATIO)
                blink_delay = MAX_BLINK_DELAY - progress * (MAX_BLINK_DELAY - MIN_BLINK_DELAY)
                
            # Clamp de dam bao an toan
            blink_delay = max(MIN_BLINK_DELAY, min(MAX_BLINK_DELAY, blink_delay))

        # --- KICH HOAT LOA, AM THANH VA LUU ANH KHI CHUYEN TRANG THAI ---
        
        #  Chuyen tu AN TOAN -> NGUY HIEM
        if led_state == "danger" and previous_led_state == "safe":
            print("[ALARM] Phat hien nguy hiem! Kich hoat loa 3 nhip, phat am thanh va luu anh.")
            beep_buzzer_non_blocking(3) # Keu 3 nhip

            # Phat am thanh (lap lai lien tuc)
            if not is_sound_playing:
                try:
                    pygame.mixer.music.play(loops=-1) # loops=-1 de lap vo han
                    is_sound_playing = True
                    print("[AUDIO] Bat dau phat am thanh canh bao.")
                except Exception as e:
                    print(f"[AUDIO ERROR] Khong the phat am thanh: {e}")

            # Luu anh
            if frame is not None:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                # Ten file se co them chuoi "danger" de de nhan biet
                filename = os.path.join(DANGER_CAPTURE_DIR, f"danger_{timestamp}.jpg")
                try:
                    # Ve box len anh truoc khi luu de nhin ro hon vat the nguy hiem
                    draw_boxes(frame, boxes, labels, confs) 
                    cv2.imwrite(filename, frame)
                    print(f"[SAVE] Da luu anh nguy hiem: {filename}")
                except Exception as e:
                    print(f"[SAVE ERROR] Khong the luu anh: {e}")
        
        # Chuyen tu NGUY HIEM -> AN TOAN
        elif led_state == "safe" and previous_led_state == "danger":
            print("[ALARM] Tro ve trang thai an toan.")
            
            # [MOI] Tat am thanh
            if is_sound_playing:
                try:
                    pygame.mixer.music.stop()
                    is_sound_playing = False
                    print("[AUDIO] Da tat am thanh canh bao.")
                except Exception as e:
                    print(f"[AUDIO ERROR] Khong the dung am thanh: {e}")

        # Cap nhat trang thai truoc do
        previous_led_state = led_state
        # ------------------------------------------------------------------

        # Ve box va hien thi len cua so
        draw_boxes(frame, boxes, labels, confs)

        cv2.putText(frame, "Nhan dien dang hoat dong...",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)

        # [CHANGE] Su dung dung ten bien window_name (aivis) de cua so hien thi dung kich thuoc
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) == ord('q'):
            cleanup()

if __name__ == "__main__":
    main()