"""AI camera gateway for AIFace attendance.

Camera sources and credentials should be configured in .env, not hard-coded.
"""
import os
import signal
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from facenet_pytorch import MTCNN

load_dotenv(Path(__file__).resolve().parent / "backend" / ".env")

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/api/attendance/process")
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS_LAPLACIAN", "80.0"))
MOVEMENT_THRESHOLD_PX = int(os.getenv("STATIONARY_MOVEMENT_PIXELS", "25"))
RECAPTURE_COOLDOWN = float(os.getenv("RECAPTURE_COOLDOWN_SECONDS", "10.0"))
MOTION_DEBOUNCE_SECONDS = float(os.getenv("MOTION_DEBOUNCE_SECONDS", "1.2"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("BACKEND_REQUEST_TIMEOUT_SECONDS", "3.5"))
CAM_LOGIN_SRC = os.getenv("CAM_LOGIN_SRC", "0")
CAM_LOGOUT_SRC = os.getenv("CAM_LOGOUT_SRC", os.getenv("CAM2_RTSP_URL", ""))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, min_face_size=60, thresholds=[0.6, 0.7, 0.7], factor=0.709, device=DEVICE)
running = True


def signal_handler(sig, frame):
    global running
    print("\n[*] Shutting down edge video pipelines...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class FaceTrackerState:
    """Movement tracker whose cooldown is updated only after a usable crop is queued."""
    def __init__(self):
        self.last_centroid = None
        self.last_box_size = None
        self.last_successful_capture_time = 0.0
        self.is_stationary = False
        self.lock = threading.Lock()

    def evaluate_movement(self, box, current_time):
        x1, y1, x2, y2 = [float(v) for v in box]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        size = (x2 - x1, y2 - y1)
        with self.lock:
            if self.last_centroid is None:
                self.last_centroid, self.last_box_size = (cx, cy), size
                self.is_stationary = False
                return True, "initial_detection"

            dist = float(np.hypot(cx - self.last_centroid[0], cy - self.last_centroid[1]))
            size_diff = abs(size[0] - self.last_box_size[0]) + abs(size[1] - self.last_box_size[1])
            moved = dist >= MOVEMENT_THRESHOLD_PX or size_diff >= MOVEMENT_THRESHOLD_PX * 1.5
            since_capture = current_time - self.last_successful_capture_time

            if moved:
                self.is_stationary = False
                self.last_centroid, self.last_box_size = (cx, cy), size
                if since_capture >= MOTION_DEBOUNCE_SECONDS:
                    return True, "motion_recapture"
                return False, "motion_debounce"

            self.is_stationary = True
            if since_capture >= RECAPTURE_COOLDOWN:
                return True, "stationary_refresh"
            return False, "stationary_suppression"

    def mark_capture_queued(self, current_time, box=None):
        with self.lock:
            self.last_successful_capture_time = current_time
            if box is not None:
                x1, y1, x2, y2 = [float(v) for v in box]
                self.last_centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                self.last_box_size = (x2 - x1, y2 - y1)


gate_trackers = {"LOGIN": FaceTrackerState(), "LOGOUT": FaceTrackerState()}


class ThreadedCamera:
    def __init__(self, src, is_rtsp=False):
        self.src = src
        self.is_rtsp = is_rtsp
        self.cap = None
        self.frame = None
        self.ret = False
        self.stopped = False
        self.lock = threading.Lock()
        self._open_capture()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def _open_capture(self):
        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG) if self.is_rtsp else cv2.VideoCapture(int(self.src) if str(self.src).isdigit() else self.src)
        if not self.is_rtsp:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap

    def update(self):
        while not self.stopped:
            if self.cap is None or not self.cap.isOpened():
                self._release_capture()
                time.sleep(1.0)
                if not self.stopped:
                    self._open_capture()
                continue

            grabbed, frame = self.cap.read()
            if not grabbed or frame is None:
                with self.lock:
                    self.ret, self.frame = False, None
                # Recreate the capture object; read() can fail while isOpened() remains true.
                self._release_capture()
                time.sleep(0.5)
                if not self.stopped:
                    self._open_capture()
                continue

            with self.lock:
                self.ret, self.frame = True, frame

    def _release_capture(self):
        cap = self.cap
        self.cap = None
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:
                print(f"[CAMERA] release error: {exc}")

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.stopped = True
        self._release_capture()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


def calculate_sharpness(image_bgr):
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_canonical_face_crop(clean_frame, box, target_size=(240, 240)):
    if clean_frame is None or clean_frame.size == 0:
        return None, 0.0
    h_img, w_img = clean_frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 50 or h < 50:
        return None, 0.0

    pad_w, pad_h = int(w * 0.35), int(h * 0.35)
    nx1, ny1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
    nx2, ny2 = min(w_img, x2 + pad_w), min(h_img, y2 + pad_h)
    crop = clean_frame[ny1:ny2, nx1:nx2]
    if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
        return None, 0.0

    sharpness = calculate_sharpness(crop)
    if sharpness < MIN_SHARPNESS:
        return None, sharpness
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4), sharpness


def send_frame_to_backend(cropped_face, camera_id: str, camera_type: str, sharpness: float):
    """Send one validated crop; log HTTP and application-level status separately."""
    ok, encoded_img = cv2.imencode(".jpg", cropped_face, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    if not ok:
        print(f"[{camera_id}] JPEG encoding failed")
        return False

    files = {"file": ("face.jpg", encoded_img.tobytes(), "image/jpeg")}
    data = {"camera_id": camera_id.upper(), "camera_type": camera_type.upper()}
    started = time.time()
    try:
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=REQUEST_TIMEOUT_SECONDS)
        elapsed_ms = (time.time() - started) * 1000
        try:
            result = response.json()
        except ValueError:
            result = {}
        status = result.get("status", "unknown")
        print(f"[{camera_id} -> {camera_type} {elapsed_ms:.0f}ms | Sharpness: {sharpness:.1f}] HTTP {response.status_code} | Status: {status} | User: {result.get('user_id')} | Score: {result.get('similarity_score')}")
        if not response.ok:
            return False
        # A 200 response may represent a suppression/rejection, not a persisted attendance event.
        non_success = {"uncertain_evaluating", "already_logged_in_suppressed", "cooldown_throttled", "illegal_gate_operation", "unknown_suppressed", "rejected", "error"}
        return status not in non_success
    except requests.RequestException as exc:
        print(f"[{camera_id}] Backend request failed: {exc}")
        return False


def process_camera(camera, camera_key, camera_id, camera_type, window_name, color):
    ret, frame = camera.read()
    if not ret or frame is None:
        return
    display_frame = frame.copy()
    try:
        boxes, _ = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception as exc:
        print(f"[{camera_id}] Face detection failed: {exc}")
        boxes = None

    if boxes is not None and len(boxes) > 0:
        # Current design intentionally processes the largest/first detector result only.
        primary_box = max(boxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
        tracker = gate_trackers[camera_key]
        should_capture, reason = tracker.evaluate_movement(primary_box, time.time())
        if should_capture:
            crop, sharpness = extract_canonical_face_crop(frame, primary_box)
            if crop is None:
                print(f"[{camera_id}] Crop rejected (sharpness={sharpness:.1f}); retry remains eligible")
            else:
                # Only update the cooldown after a usable crop has been accepted for dispatch.
                tracker.mark_capture_queued(time.time(), primary_box)
                threading.Thread(target=send_frame_to_backend, args=(crop, camera_id, camera_type, sharpness), daemon=True).start()

        x1, y1, x2, y2 = [int(v) for v in primary_box]
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{camera_type} ({'STATIONARY' if tracker.is_stationary else 'MOVING'}) - {reason}"
        cv2.putText(display_frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)

    cv2.imshow(window_name, cv2.resize(display_frame, (640, 360)))


def run_hybrid_streams():
    global running
    if not CAM_LOGOUT_SRC:
        raise RuntimeError("CAM_LOGOUT_SRC is missing. Set it in your .env file; do not hard-code RTSP credentials.")

    login_source = int(CAM_LOGIN_SRC) if CAM_LOGIN_SRC.isdigit() else CAM_LOGIN_SRC
    login = ThreadedCamera(login_source, is_rtsp=False)
    logout = ThreadedCamera(CAM_LOGOUT_SRC, is_rtsp=True)
    print("[*] Edge Streaming Gateway initialized.")
    print("    - Login camera -> SYSTEM_CAM / LOGIN")
    print("    - Office camera -> OFFICE_RTSP / LOGOUT")
    try:
        while running:
            process_camera(login, "LOGIN", "SYSTEM_CAM", "LOGIN", "Gate 1 - LOGIN", (0, 255, 0))
            process_camera(logout, "LOGOUT", "OFFICE_RTSP", "LOGOUT", "Gate 2 - LOGOUT", (0, 0, 255))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        running = False
        login.stop()
        logout.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_hybrid_streams()