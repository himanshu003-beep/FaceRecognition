import os
import sys
import time
import signal
import threading
import requests
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/attendance/process")
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS_LAPLACIAN", "85.0"))
MOVEMENT_THRESHOLD_PX = int(os.getenv("STATIONARY_MOVEMENT_PIXELS", "25"))
RECRAW_COOLDOWN = float(os.getenv("RECRAW_COOLDOWN_SECONDS", "10.0"))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CAM_LOGIN_SRC = 0
CAM_LOGOUT_SRC = "rtsp://nidhin:Nidhin123@192.168.2.178:554"

mtcnn = MTCNN(
    keep_all=True,
    min_face_size=60,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    device=DEVICE
)

running = True

def signal_handler(sig, frame):
    global running
    print("\n[*] Shutting down streams cleanly...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class FaceTrackerState:
    def __init__(self):
        self.last_centroid = None
        self.last_capture_time = 0.0
        self.last_box_size = None
        self.is_stationary = False

    def evaluate_movement(self, box, current_time):
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        w = x2 - x1
        h = y2 - y1

        if self.last_centroid is None:
            self.last_centroid = (cx, cy)
            self.last_box_size = (w, h)
            self.last_capture_time = current_time
            self.is_stationary = False
            return True, "new_presence"

        dist = np.sqrt((cx - self.last_centroid[0])**2 + (cy - self.last_centroid[1])**2)
        size_diff = abs(w - self.last_box_size[0]) + abs(h - self.last_box_size[1])

        time_since_last = current_time - self.last_capture_time

        # Agar banda stationary hai aur cooldown pass nahi hua, skip upload
        if dist < MOVEMENT_THRESHOLD_PX and size_diff < (MOVEMENT_THRESHOLD_PX * 1.5):
            self.is_stationary = True
            if time_since_last < RECRAW_COOLDOWN:
                return False, "stationary_suppressed"
            else:
                # Periodic verification capture for stationary presence
                self.last_capture_time = current_time
                return True, "stationary_cooldown_expired"
        else:
            # Significant movement detected
            self.is_stationary = False
            self.last_centroid = (cx, cy)
            self.last_box_size = (w, h)
            if time_since_last >= 1.5:  # Debounce small tremors
                self.last_capture_time = current_time
                return True, "movement_recapture"
            return False, "movement_debounced"

gate_trackers = {
    "LOGIN": FaceTrackerState(),
    "LOGOUT": FaceTrackerState()
}

class ThreadedCamera:
    def __init__(self, src, is_rtsp=False):
        self.src = src
        self.is_rtsp = is_rtsp
        if self.is_rtsp:
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(self.src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()

        self.t = threading.Thread(target=self.update, args=(), daemon=True)
        self.t.start()

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(1)
                try:
                    if self.is_rtsp: self.cap.open(self.src, cv2.CAP_FFMPEG)
                    else: self.cap.open(self.src)
                except Exception:
                    pass
                continue

            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.01)
                continue

            with self.lock:
                self.ret = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

def calculate_sharpness(image_bgr):
    """Laplacian variance se focus aur motion blur measure karta hai"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_safe_face_crop(clean_frame, box, target_size=(240, 240)):
    h_img, w_img, _ = clean_frame.shape
    x1, y1, x2, y2 = [int(b) for b in box]

    w = x2 - x1
    h = y2 - y1

    if w < 50 or h < 50:
        return None, 0.0

    pad_w = int(w * 0.28)
    pad_h = int(h * 0.28)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w_img, x2 + pad_w)
    ny2 = min(h_img, y2 + pad_h)

    crop = clean_frame[ny1:ny2, nx1:nx2]
    if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
        return None, 0.0

    sharpness = calculate_sharpness(crop)
    if sharpness < MIN_SHARPNESS:
        return None, sharpness

    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)
    return resized, sharpness

def send_frame_to_backend(cropped_face, camera_type: str, sharpness: float):
    def _worker():
        try:
            start_t = time.time()
            success, encoded_img = cv2.imencode('.jpg', cropped_face, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
            if not success: return

            files = {'file': ('face.jpg', encoded_img.tobytes(), 'image/jpeg')}
            data = {'camera_type': camera_type.upper()}

            response = requests.post(BACKEND_URL, files=files, data=data, timeout=3.5)
            elapsed = (time.time() - start_t) * 1000
            if response.status_code == 200:
                res = response.json()
                print(f"[{camera_type} {elapsed:.0f}ms | Sharpness: {sharpness:.1f}] Status: {res.get('status')} | User: {res.get('user_id')} | Score: {res.get('similarity_score')}")
        except Exception as e:
            pass

    threading.Thread(target=_worker, daemon=True).start()

def run_hybrid_streams():
    global running
    print("[*] Starting Movement-Aware & Sharpness-Gated Attendance Stream")
    print(f"[*] Strict Similarity Threshold: {os.getenv('RECOGNITION_THRESHOLD', '0.90')}")
    print(f"[*] Min Sharpness: {MIN_SHARPNESS} | Movement Threshold: {MOVEMENT_THRESHOLD_PX}px")

    stream_login = ThreadedCamera(CAM_LOGIN_SRC, is_rtsp=False)
    stream_logout = ThreadedCamera(CAM_LOGOUT_SRC, is_rtsp=True)

    while running:
        ret_in, frame_in = stream_login.read()
        ret_out, frame_out = stream_logout.read()
        now = time.time()

        # Gate 1: System Cam (LOGIN)
        if ret_in and frame_in is not None:
            raw_clean_in = frame_in.copy()
            rgb_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
            try:
                boxes_in, _ = mtcnn.detect(rgb_in)
            except Exception:
                boxes_in = None

            if boxes_in is not None and len(boxes_in) > 0:
                primary_box = boxes_in[0]
                should_capture, reason = gate_trackers["LOGIN"].evaluate_movement(primary_box, now)
                
                if should_capture:
                    face_crop, score_sharp = extract_safe_face_crop(raw_clean_in, primary_box)
                    if face_crop is not None:
                        send_frame_to_backend(face_crop, "LOGIN", score_sharp)
                
                x1, y1, x2, y2 = [int(b) for b in primary_box]
                color = (0, 255, 0) if not gate_trackers["LOGIN"].is_stationary else (255, 165, 0)
                tag = f"LOGIN ({'MOVING' if not gate_trackers['LOGIN'].is_stationary else 'STATIONARY'})"
                cv2.rectangle(frame_in, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_in, tag, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

            cv2.imshow("Gate 1 - LOGIN", cv2.resize(frame_in, (640, 360)))

        # Gate 2: Office RTSP (LOGOUT)
        if ret_out and frame_out is not None:
            raw_clean_out = frame_out.copy()
            rgb_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            try:
                boxes_out, _ = mtcnn.detect(rgb_out)
            except Exception:
                boxes_out = None

            if boxes_out is not None and len(boxes_out) > 0:
                primary_box = boxes_out[0]
                should_capture, reason = gate_trackers["LOGOUT"].evaluate_movement(primary_box, now)
                
                if should_capture:
                    face_crop, score_sharp = extract_safe_face_crop(raw_clean_out, primary_box)
                    if face_crop is not None:
                        send_frame_to_backend(face_crop, "LOGOUT", score_sharp)

                x1, y1, x2, y2 = [int(b) for b in primary_box]
                color = (0, 0, 255) if not gate_trackers["LOGOUT"].is_stationary else (255, 165, 0)
                tag = f"LOGOUT ({'MOVING' if not gate_trackers['LOGOUT'].is_stationary else 'STATIONARY'})"
                cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_out, tag, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

            cv2.imshow("Gate 2 - LOGOUT", cv2.resize(frame_out, (640, 360)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream_login.stop()
    stream_logout.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hybrid_streams()