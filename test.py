from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import threading
import queue
import lgpio
import signal
import sys
import os
import pygame 
import subprocess

# =============== CAI DAT THONG SO ===============
FRAME_W, FRAME_H = 1280, 720

# [TOI UU] Cau hinh AI can bang toc do/chinh xac
# 416 thuong cho ket qua chinh xac hon 320.
AI_IMG_SIZE = 416
YOLO_CONF = 0.4

# [TOI UU] Uu tien tim model da duoc export sang NCNN (nhanh gap 3 lan .pt)
# Neu khong co, he thong se tu dong dung file .pt nhu cu.
MODEL_NAME = "yolov8n" 
MODEL_PT = f"{MODEL_NAME}.pt"
MODEL_NCNN = f"{MODEL_NAME}_ncnn_model" # Folder chua model ncnn

SOUND_FILE_PATH = "/home/huudong/Desktop/aivis/mp3 dangerous.mp3" 

# --- Nguong canh bao ---
START_DANGER_RATIO = 0.08
MAX_DANGER_RATIO = 0.30 

# --- Toc do nhay LED ---
MIN_BLINK_DELAY = 0.05
MAX_BLINK_DELAY = 0.4 

# --- Hien thi ---
DISPLAY_W = 1280
DISPLAY_H = 720
BOX_THICKNESS = 3
LABEL_FONT_SCALE = 0.75
LABEL_THICKNESS = 2

# --- Cau hinh am thanh ---
DEFAULT_VOLUME = 1.0

# --- TTS thong bao vat nguy hiem (offline) ---
# Cai dat:
#   sudo apt update && sudo apt install -y espeak-ng
ENABLE_DANGER_TTS = True
TTS_ENGINE_CMD = "espeak-ng"
TTS_VOICE = "vi"
TTS_SPEED = 155
TTS_COOLDOWN_SECONDS = 2.5
TTS_PREFIX_TEXT = "Phia truoc co"
PERSON_VI_LABEL = "nguoi"
TTS_AFTER_DANGER_DELAY = 0.8

# --- MOG2 frame gating de tang FPS ---
ENABLE_MOG2_FILTER = True
MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 25
MOTION_THRESHOLD = 0.01
FORCE_YOLO_INTERVAL = 10
MAX_REUSE_DETECTIONS = 5
REUSE_VALID_SECONDS = 2.0
KERNEL_SIZE = 3

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

# --- PINS (BCM) ---
LED_RED = 17
LED_GREEN = 27
BUZZER_PIN = 18

# LGPIO
CHIP = 0
try:
    handle = lgpio.gpiochip_open(CHIP)
except Exception as e:
    print(f"[ERROR] Khong the mo lgpio chip {CHIP}. {e}")
    sys.exit(1)

# --- Thu muc luu anh nguy hiem ---
DANGER_CAPTURE_DIR = "dangerous_captures"
os.makedirs(DANGER_CAPTURE_DIR, exist_ok=True)

# --- Globals ---
led_state = "safe"
blink_delay = MAX_BLINK_DELAY  
stop_led_thread = threading.Event()
led_thread = None 

# Camera Globals
latest_frame = None
frame_lock = threading.Lock()
camera_running = True
cam_thread = None 

# TTS thread
stop_tts_thread = threading.Event()
tts_queue = queue.Queue()
tts_thread = None

# Audio shared state
audio_state_lock = threading.Lock()
is_sound_playing = False
is_currently_danger = False

# =============== HAM DIEU KHIEN LOA (BUZZER) ===============
def beep_worker(count, on_delay, off_delay):
    for _ in range(count):
        if stop_led_thread.is_set(): 
            break
        try:
            lgpio.gpio_write(handle, BUZZER_PIN, 1)
            time.sleep(on_delay)
            lgpio.gpio_write(handle, BUZZER_PIN, 0)
            time.sleep(off_delay)
        except Exception:
            break
    
    try:
        lgpio.gpio_write(handle, BUZZER_PIN, 0)
    except Exception:
        pass

def beep_buzzer_non_blocking(beep_count=3, on_delay=0.15, off_delay=0.1):
    beep_thread = threading.Thread(target=beep_worker, args=(beep_count, on_delay, off_delay), daemon=True)
    beep_thread.start()

# =============== HAM DIEU KHIEN LED ===============
def led_off():
    try:
        lgpio.gpio_write(handle, LED_RED, 0)
        lgpio.gpio_write(handle, LED_GREEN, 0)
    except Exception:
        pass

def led_green():
    led_off()
    lgpio.gpio_write(handle, LED_GREEN, 1)

def stop_warning_outputs():
    global led_state, blink_delay
    try:
        lgpio.gpio_write(handle, BUZZER_PIN, 0)
    except Exception:
        pass
    try:
        pygame.mixer.music.stop()
    except Exception:
        pass
    led_state = "safe"
    blink_delay = MAX_BLINK_DELAY

def init_mog2():
    if not ENABLE_MOG2_FILTER:
        return None
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=False
    )

def calculate_motion_ratio(frame, mog2, morph_kernel):
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

def danger_label_to_vietnamese(label):
    if label == "person":
        return PERSON_VI_LABEL
    return DANGEROUS_OBJECTS.get(label, label)

def get_primary_dangerous_label(boxes, labels):
    primary_label = None
    primary_area = -1
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        if label == "person" or label in DANGEROUS_OBJECTS:
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > primary_area:
                primary_area = area
                primary_label = label
    return primary_label

def tts_worker():
    global is_sound_playing
    while not stop_tts_thread.is_set():
        try:
            text = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if text is None:
            break

        try:
            with audio_state_lock:
                should_resume = is_currently_danger and is_sound_playing
                if should_resume:
                    pygame.mixer.music.pause()
            subprocess.run(
                [TTS_ENGINE_CMD, "-v", TTS_VOICE, "-s", str(TTS_SPEED), text],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False
            )
            with audio_state_lock:
                if is_currently_danger and should_resume:
                    pygame.mixer.music.unpause()
        except Exception as e:
            print(f"[TTS ERROR] {e}")

def enqueue_danger_tts(label):
    if not ENABLE_DANGER_TTS:
        return
    vi_label = danger_label_to_vietnamese(label)
    tts_queue.put(f"{TTS_PREFIX_TEXT} {vi_label}")

def draw_debug_overlay(frame, motion_ratio, yolo_status, skip_count, dangerous_label, cooldown_sec, fps):
    lines = [
        f"Motion: {motion_ratio:.4f}",
        f"AI: {yolo_status}",
        f"Skip count: {skip_count}",
        f"Danger label: {dangerous_label}",
        f"TTS cooldown: {cooldown_sec:.1f}s",
        f"FPS: {fps:.1f}"
    ]
    y = 30
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 26

def led_controller_thread():
    global led_state, blink_delay
    print("[LED] LED controller thread bat dau.")
    applied_mode = None
    while not stop_led_thread.is_set():
        try:
            if led_state == "safe":
                if applied_mode != "safe":
                    # Safe: xanh sang on dinh, khong nhap nhay
                    lgpio.gpio_write(handle, LED_RED, 0)
                    lgpio.gpio_write(handle, LED_GREEN, 1)
                    applied_mode = "safe"
                time.sleep(0.1) 
            elif led_state == "danger":
                applied_mode = "danger"
                lgpio.gpio_write(handle, LED_GREEN, 0)
                
                if stop_led_thread.is_set(): break
                lgpio.gpio_write(handle, LED_RED, 1)
                time.sleep(blink_delay)
                
                if stop_led_thread.is_set(): break
                lgpio.gpio_write(handle, LED_RED, 0)
                time.sleep(blink_delay)
        except Exception as e:
            if not stop_led_thread.is_set():
                print(f"[LED ERROR] {e}")
                time.sleep(1)
    led_off()

# =============== CAMERA ===============
def init_camera():
    picam2 = Picamera2()
    # [TOI UU] Cau hinh buffer count=2 de tranh xe hinh
    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_W, FRAME_H)},
        buffer_count=2
    )
    picam2.configure(config)
    picam2.start()
    print("[CAM] Camera da san sang")
    return picam2
def camera_capture_thread(picam2):
    global latest_frame, frame_lock, camera_running
    print("[CAM] Luong lay khung hinh bat dau.")
    while camera_running:
        try:
            # capture_array block cho den khi co frame moi
            new_frame = picam2.capture_array()
            with frame_lock:
                latest_frame = new_frame
        except Exception as e:
            print(f"[CAM ERROR] {e}")
            time.sleep(0.1)
    picam2.stop()

# =============== VE BOX ===============
def draw_boxes(frame, boxes, labels, confs):
    # [TOI UU] Khong ve gi neu list rong
    if not boxes: return
    for (x1, y1, x2, y2), name, conf in zip(boxes, labels, confs):
        color = (0, 0, 255) if (name in DANGEROUS_OBJECTS or name == "person") else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, color, LABEL_THICKNESS)

# =============== CLEANUP ===============
def cleanup(*args):
    global camera_running, stop_led_thread, cam_thread, led_thread
    global tts_thread
    print("\n[EXIT] Dang dung he thong...")
    
    camera_running = False
    if cam_thread and cam_thread.is_alive():
        cam_thread.join(timeout=1.0)
        
    stop_led_thread.set()
    if led_thread and led_thread.is_alive():
        led_thread.join(timeout=1.0)

    stop_tts_thread.set()
    try:
        tts_queue.put_nowait(None)
    except Exception:
        pass
    if tts_thread and tts_thread.is_alive():
        tts_thread.join(timeout=1.0)

    try:
        lgpio.gpio_write(handle, BUZZER_PIN, 0)
        led_off()
        lgpio.gpiochip_close(handle)
    except Exception:
        pass
    
    try:
        pygame.mixer.music.stop()
        pygame.quit()
    except Exception:
        pass

    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup) 
                    # =============== MAIN ===============
def main():
    global cam_thread, led_thread, tts_thread
    global led_state, blink_delay, is_sound_playing, is_currently_danger
    
    print("[FTSCENE] He thong dang khoi dong (Che do High FPS)...")

    # GPIO Setup
    try:
        lgpio.gpio_claim_output(handle, LED_RED, 0)
        lgpio.gpio_claim_output(handle, LED_GREEN, 0)
        lgpio.gpio_claim_output(handle, BUZZER_PIN, 0)
    except Exception as e:
        print(f"[ERROR] GPIO Claim failed: {e}")
        sys.exit(1)

    # Audio Setup
    current_volume = DEFAULT_VOLUME
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(SOUND_FILE_PATH)
        pygame.mixer.music.set_volume(current_volume)
        print("[AUDIO] OK")
    except Exception as e:
        print(f"[AUDIO ERROR] {e}")

    # [TOI UU] LOAD MODEL THONG MINH
    # Kiem tra xem co ban NCNN (sieu nhanh) chua, neu chua thi dung .pt
    if os.path.exists(MODEL_NCNN) and os.path.isdir(MODEL_NCNN):
        print(f"[AI] Phat hien model NCNN toi uu: {MODEL_NCNN}")
        print("[AI] Dang tai model NCNN (Tang toc do xu ly)...")
        model = YOLO(MODEL_NCNN, task='detect')
    else:
        print(f"[AI] Khong tim thay model NCNN, su dung model chuan: {MODEL_PT}")
        print("[TIP] De he thong nhanh hon 300%, hay chay lenh: yolo export model=yolov8n.pt format=ncnn")
        model = YOLO(MODEL_PT)

    mog2 = init_mog2()
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
    picam2 = init_camera()
    stop_led_thread.clear()
    stop_tts_thread.clear()

    # Start Threads
    led_thread = threading.Thread(target=led_controller_thread, daemon=True)
    led_thread.start()
    
    cam_thread = threading.Thread(target=camera_capture_thread, args=(picam2,), daemon=True)
    cam_thread.start()

    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    
    cv2.namedWindow("FTSCENE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FTSCENE", DISPLAY_W, DISPLAY_H) 
    
    previous_led_state = "safe" 
    is_sound_playing = False
    is_currently_danger = False
    last_announced_label = None
    last_tts_time = 0.0
    danger_started_time = 0.0
    debug_danger_label = "none"
    debug_cooldown_remaining = 0.0
    yolo_status = "YOLO RUN"
    skip_count = 0
    frames_since_last_yolo = FORCE_YOLO_INTERVAL
    last_boxes, last_labels, last_confs = [], [], []
    last_max_danger_ratio = 0.0
    last_yolo_time = 0.0

    # [TOI UU] Pre-calculate FPS variables
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0.0

    print("[MAIN] Bat dau vong lap chinh...")
    print("Dieu khien: q=thoat | +=volume | -=volume")

    while True:
        # 1. Lay frame (Chi copy khi co frame moi de tranh race condition)
        frame_process = None
        with frame_lock:
            if latest_frame is not None:
                # [TOI UU] Copy la bat buoc, nhung se nhanh vi dung memory view
                frame_process = latest_frame.copy()
        
        if frame_process is None:
            time.sleep(0.01)
            continue

        # 2. MOG2 frame gating + YOLO inference
        boxes, labels, confs = [], [], []
        max_danger_ratio = 0.0
        h, w = frame_process.shape[:2]
        _, motion_ratio = calculate_motion_ratio(frame_process, mog2, morph_kernel)
        frames_since_last_yolo += 1

        should_force_yolo = frames_since_last_yolo >= FORCE_YOLO_INTERVAL
        can_reuse = (
            last_boxes and
            skip_count < MAX_REUSE_DETECTIONS and
            (time.time() - last_yolo_time) < REUSE_VALID_SECONDS
        )
        should_skip_yolo = (
            ENABLE_MOG2_FILTER and
            motion_ratio < MOTION_THRESHOLD and
            not should_force_yolo and
            can_reuse
        )

        if should_skip_yolo:
            boxes = list(last_boxes)
            labels = list(last_labels)
            confs = list(last_confs)
            max_danger_ratio = last_max_danger_ratio
            skip_count += 1
            yolo_status = "YOLO SKIP"
        else:
            results = model(frame_process, imgsz=AI_IMG_SIZE, conf=YOLO_CONF, verbose=False, stream=True)
            for r in results:
                if r.boxes:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        name = r.names.get(cls, str(cls))

                        if name in DANGEROUS_OBJECTS or name == "person":
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])

                            boxes.append((x1, y1, x2, y2))
                            labels.append(name)
                            confs.append(conf)

                            area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)
                            max_danger_ratio = max(max_danger_ratio, area_ratio)

            last_boxes = list(boxes)
            last_labels = list(labels)
            last_confs = list(confs)
            last_max_danger_ratio = max_danger_ratio
            last_yolo_time = time.time()
            frames_since_last_yolo = 0
            skip_count = 0
            yolo_status = "YOLO RUN"

        # 3. Logic dieu khien (giu nguyen)
        if max_danger_ratio < START_DANGER_RATIO:
            led_state = "safe"
        else:
            led_state = "danger"
            if max_danger_ratio >= MAX_DANGER_RATIO:
                blink_delay = MIN_BLINK_DELAY
            else:
                progress = (max_danger_ratio - START_DANGER_RATIO) / (MAX_DANGER_RATIO - START_DANGER_RATIO)
                blink_delay = MAX_BLINK_DELAY - progress * (MAX_BLINK_DELAY - MIN_BLINK_DELAY)
            blink_delay = max(MIN_BLINK_DELAY, min(MAX_BLINK_DELAY, blink_delay))

        # 4. Xu ly Am thanh & Luu anh (Trang thai thay doi)
        if led_state == "danger" and previous_led_state == "safe":
            # print("[ALARM] NGUY HIEM!") # Comment bot print de tang toc
            beep_buzzer_non_blocking(3)
            is_currently_danger = True
            danger_started_time = time.time()

            with audio_state_lock:
                if not is_sound_playing:
                    try:
                        pygame.mixer.music.play(loops=-1)
                        is_sound_playing = True
                    except Exception:
                        pass

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(DANGER_CAPTURE_DIR, f"danger_{timestamp}.jpg")
            # Chay luu anh trong thread rieng biet de khong lag hinh? 
            # Hien tai giu nguyen theo yeu cau user, nhung luu y day la diem gay lag nhe.
            try:
                draw_boxes(frame_process, boxes, labels, confs) 
                cv2.imwrite(filename, frame_process)
            except Exception: pass

        elif led_state == "safe" and previous_led_state == "danger":
            is_currently_danger = False
            with audio_state_lock:
                if is_sound_playing:
                    try:
                        pygame.mixer.music.stop()
                        is_sound_playing = False
                    except Exception:
                        pass

        # 4.1 Thong bao bang giong noi ten vat nguy hiem (co cooldown, chong spam)
        danger_label = get_primary_dangerous_label(boxes, labels)
        if danger_label:
            debug_danger_label = danger_label_to_vietnamese(danger_label)
        else:
            debug_danger_label = "none"

        now = time.time()
        if led_state == "danger" and danger_label:
            debug_cooldown_remaining = max(0.0, TTS_COOLDOWN_SECONDS - (now - last_tts_time))
            is_new_label = danger_label != last_announced_label
            if is_new_label and debug_cooldown_remaining <= 0.0:
                is_waiting_after_danger = previous_led_state == "safe" and (now - danger_started_time) < TTS_AFTER_DANGER_DELAY
                if not is_waiting_after_danger:
                    enqueue_danger_tts(danger_label)
                    last_announced_label = danger_label
                    last_tts_time = now
                    debug_cooldown_remaining = TTS_COOLDOWN_SECONDS
        else:
            debug_cooldown_remaining = 0.0
            last_announced_label = None

        previous_led_state = led_state

        # 5. Hien thi
        # [TOI UU] Chi ve box neu can thiet
        draw_boxes(frame_process, boxes, labels, confs)
        
        # Tinh FPS de theo doi
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count / (time.time() - fps_start_time)
            # Reset
            fps_frame_count = 0
            fps_start_time = time.time()

        draw_debug_overlay(
            frame_process,
            motion_ratio,
            yolo_status,
            skip_count,
            debug_danger_label,
            debug_cooldown_remaining,
            fps
        )
        cv2.putText(frame_process, f"Vol: {int(current_volume*100)}%", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("FTSCENE", frame_process)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cleanup()
        elif key == ord('+') or key == ord('='):
            current_volume = min(1.0, current_volume + 0.1)
            try:
                pygame.mixer.music.set_volume(current_volume)
                print(f"[AUDIO] Tang volume: {int(current_volume*100)}%")
            except Exception:
                pass
        elif key == ord('-') or key == ord('_'):
            current_volume = max(0.0, current_volume - 0.1)
            try:
                pygame.mixer.music.set_volume(current_volume)
                print(f"[AUDIO] Giam volume: {int(current_volume*100)}%")
            except Exception:
                pass

if __name__ == "__main__":
    main()
