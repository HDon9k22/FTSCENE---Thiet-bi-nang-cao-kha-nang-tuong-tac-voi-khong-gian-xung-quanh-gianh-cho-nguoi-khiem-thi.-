#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FTSCENE - Thiet bi nang cao kha nang tuong tac voi khong gian xung quanh gianh cho nguoi khiem thi.
Tac gia: SV HOANG HUU DONG - 22SPT - DHSP DN
Update: 02/11/2025
    - Chi su dung 2 LED (Xanh/Do)
    - Them Loa (Buzzer) canh bao 3 nhip khi chuyen sang Do
    - LED Do nhay nhanh dan khi vat the den gan (area_ratio lon)
    - Tang FPS xu ly AI bang cach giam imgsz=416
    - Luu anh khi phat hien vat the nguy hiem vao thu muc rieng.
    - Phat am thanh qua thiet bi bluetooth/.mp3 khi o trang thai nguy hiem
    - Chuc nang dieu chinh am luong truc tiep (+/-)
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

# Duong dan den file am thanh
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

# --- [MOI] Cau hinh Am thanh ---
DEFAULT_VOLUME = 1.0 # 1.0 la 100%, 0.5 la 50%

# --- [MOI] Cau hinh MOG2 frame gating de tang FPS ---
ENABLE_MOG2_FILTER = True
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 25
MOTION_THRESHOLD = 0.01
FORCE_YOLO_INTERVAL = 10
MAX_REUSE_DETECTIONS = 5
KERNEL_SIZE = 3

# --- [MOI] Cau hinh YOLO va debug ---
YOLO_IMGSZ = 416
YOLO_CONF = 0.5
FPS_SMOOTHING = 0.9

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

def init_mog2():
    """Khoi tao bo tach nen MOG2 de loc frame it thay doi."""
    if not ENABLE_MOG2_FILTER:
        return None

    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False
    )

def calculate_motion_ratio(frame, mog2, morph_kernel):
    """
    Tra ve foreground mask va ti le chuyen dong tren frame hien tai.
    MOG2 chi dung de loc frame, khong thay the YOLO.
    """
    if mog2 is None:
        return None, 1.0

    fg_mask = mog2.apply(frame)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, morph_kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, morph_kernel)
    fg_mask = cv2.dilate(fg_mask, morph_kernel, iterations=1)

    motion_pixels = cv2.countNonZero(fg_mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    motion_ratio = motion_pixels / float(total_pixels)
    return fg_mask, motion_ratio

def extract_yolo_detections(result, frame_shape):
    """Tach ket qua YOLO ra boxes/labels/confs va tinh max_danger_ratio."""
    boxes, labels, confs = [], [], []
    h, w = frame_shape[:2]
    max_danger_ratio = 0.0

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        name = result.names.get(cls, str(cls))

        boxes.append((x1, y1, x2, y2))
        labels.append(name)
        confs.append(conf)

        area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
        if name == "person" or name in DANGEROUS_OBJECTS:
            max_danger_ratio = max(max_danger_ratio, area_ratio)

    return boxes, labels, confs, max_danger_ratio

def update_led_from_danger_ratio(max_danger_ratio):
    """Giu nguyen logic safe/danger va blink_delay theo area_ratio."""
    global led_state, blink_delay

    if max_danger_ratio < START_DANGER_RATIO:
        led_state = "safe"
        return

    led_state = "danger"
    if max_danger_ratio >= MAX_DANGER_RATIO:
        blink_delay = MIN_BLINK_DELAY
    else:
        progress = (max_danger_ratio - START_DANGER_RATIO) / (MAX_DANGER_RATIO - START_DANGER_RATIO)
        blink_delay = MAX_BLINK_DELAY - progress * (MAX_BLINK_DELAY - MIN_BLINK_DELAY)

    blink_delay = max(MIN_BLINK_DELAY, min(MAX_BLINK_DELAY, blink_delay))

def draw_debug_overlay(frame, motion_ratio, yolo_status, skip_count, fps_value):
    """Ve thong tin debug de theo doi gating va toc do xu ly."""
    overlay_lines = [
        f"Motion: {motion_ratio:.4f}",
        f"AI: {yolo_status}",
        f"Skip count: {skip_count}",
        f"FPS: {fps_value:.1f}"
    ]

    y = 30
    for line in overlay_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += 28

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

    # [MOI] Khoi tao bo phat am thanh va dat am luong mac dinh
    current_volume = DEFAULT_VOLUME
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(SOUND_FILE_PATH)
        pygame.mixer.music.set_volume(current_volume) # Dat volume ban dau
        print(f"[AUDIO] Da tai file am thanh: {SOUND_FILE_PATH}")
        print(f"[AUDIO] Am luong ban dau: {int(current_volume*100)}%")
    except Exception as e:
        print(f"[AUDIO ERROR] Khong the khoi tao pygame hoac tai file am thanh: {e}")
        print("[AUDIO] Chuong trinh se tiep tuc chay ma khong co am thanh.")
        
    model = YOLO(MODEL_PATH)
    mog2 = init_mog2()
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
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
    motion_ratio = 1.0
    yolo_status = "YOLO RUN"
    consecutive_skip_count = 0
    frames_since_last_yolo = FORCE_YOLO_INTERVAL
    last_boxes, last_labels, last_confs = [], [], []
    last_max_danger_ratio = 0.0
    last_yolo_time = 0.0
    fps_value = 0.0
    prev_loop_time = time.time()
    print("\n------------------------------------------------")
    print("HUONG DAN DIEU KHIEN:")
    print("  'q' : Thoat chuong trinh")
    print("  '+' : Tang am luong")
    print("  '-' : Giam am luong")
    print("------------------------------------------------\n")

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

        h, w = frame.shape[:2]
        _, motion_ratio = calculate_motion_ratio(frame, mog2, morph_kernel)
        frames_since_last_yolo += 1

        should_force_yolo = frames_since_last_yolo >= FORCE_YOLO_INTERVAL
        can_reuse_detections = (
            last_boxes and
            consecutive_skip_count < MAX_REUSE_DETECTIONS and
            (time.time() - last_yolo_time) < 2.0
        )
        should_skip_yolo = (
            ENABLE_MOG2_FILTER and
            motion_ratio < MOTION_THRESHOLD and
            not should_force_yolo and
            can_reuse_detections
        )

        if should_skip_yolo:
            boxes = list(last_boxes)
            labels = list(last_labels)
            confs = list(last_confs)
            max_danger_ratio = last_max_danger_ratio
            consecutive_skip_count += 1
            yolo_status = "YOLO SKIP"
        else:
            results = model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
            r = results[0]
            boxes, labels, confs, max_danger_ratio = extract_yolo_detections(r, frame.shape)
            last_boxes = list(boxes)
            last_labels = list(labels)
            last_confs = list(confs)
            last_max_danger_ratio = max_danger_ratio
            last_yolo_time = time.time()
            frames_since_last_yolo = 0
            consecutive_skip_count = 0
            yolo_status = "YOLO RUN"

        update_led_from_danger_ratio(max_danger_ratio)

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
        current_time = time.time()
        loop_delta = max(current_time - prev_loop_time, 1e-6)
        instant_fps = 1.0 / loop_delta
        fps_value = instant_fps if fps_value == 0.0 else (FPS_SMOOTHING * fps_value + (1.0 - FPS_SMOOTHING) * instant_fps)
        prev_loop_time = current_time

        draw_debug_overlay(frame, motion_ratio, yolo_status, consecutive_skip_count, fps_value)
        
        # Hien thi muc Volume tren man hinh de de theo doi
        cv2.putText(frame, f"Vol: {int(current_volume*100)}%", 
                    (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # [CHANGE] Su dung dung ten bien window_name (aivis) de cua so hien thi dung kich thuoc
        cv2.imshow(window_name, frame)
        
        # [MOI] Xu ly phim bam: q=thoat, +=tang volume, -=giam volume
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            cleanup()
        elif key == ord('+') or key == ord('='): # '='/'+' key
            current_volume = min(1.0, current_volume + 0.1) # Tang 10%, toi da 100%
            try:
                pygame.mixer.music.set_volume(current_volume)
                print(f"[AUDIO] Da tang am luong: {int(current_volume*100)}%")
            except Exception:
                pass
        elif key == ord('-') or key == ord('_'): # '-'/'_' key
            current_volume = max(0.0, current_volume - 0.1) # Giam 10%, toi thieu 0%
            try:
                pygame.mixer.music.set_volume(current_volume)
                print(f"[AUDIO] Da giam am luong: {int(current_volume*100)}%")
            except Exception:
                pass

if __name__ == "__main__":
    main()
