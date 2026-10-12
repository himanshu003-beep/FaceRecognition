import os
import sys
import time
import threading
import requests
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

# Force X11 backend and system fonts
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["FONTCONFIG_PATH"] = "/etc/fonts"

BACKEND_URL = "http://127.0.0.1:8000/api/attendance/process"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LOCAL_CAM_INDEX = 0

mtcnn = MTCNN(
    keep_all=True,
    min_face_size=75,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    device=DEVICE
)

class LocalWebcamThread:
    def __init__(self, src=0):
        self.src = src
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
                time.sleep(0.5)
                self.cap.open(self.src)
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

def is_frame_sharp(crop: np.ndarray, min_variance: float = 40.0) -> bool:
    """Laplacian blur gate: Ignores motion-blurred frames"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var >= min_variance

def extract_ultra_hd_face(clean_frame, box, target_size=(360, 360)):
    h_img, w_img, _ = clean_frame.shape
    x1, y1, x2, y2 = [int(b) for b in box]

    w = x2 - x1
    h = y2 - y1

    # Quality Gate 1: Face must be at least 75x75 pixels (no distant blurred faces)
    if w < 75 or h < 75:
        return None

    pad_w = int(w * 0.28)
    pad_h = int(h * 0.28)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w_img, x2 + pad_w)
    ny2 = min(h_img, y2 + pad_h)

    crop = clean_frame[ny1:ny2, nx1:nx2]
    if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
        return None

    # Quality Gate 2: Sharpness check
    if not is_frame_sharp(crop, min_variance=35.0):
        return None

    # High quality Lanczos interpolation
    hd_crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Adaptive Histogram Equalization
    lab = cv2.cvtColor(hd_crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    # Edge Sharpening
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.8)
    crisp_face = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    return crisp_face

def send_frame_to_backend(cropped_face, camera_type: str):
    def _worker():
        try:
            success, encoded_img = cv2.imencode('.jpg', cropped_face, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            if not success:
                return

            files = {'file': ('face.jpg', encoded_img.tobytes(), 'image/jpeg')}
            data = {'camera_type': camera_type}
            
            response = requests.post(BACKEND_URL, files=files, data=data, timeout=4)
            if response.status_code == 200:
                res = response.json()
                status = res.get('status')
                user_id = res.get('user_id', 'None')
                score = res.get('similarity_score', 'None')
                print(f"[{camera_type}] Status: {status} | User: {user_id} | Score: {score}")
            else:
                print(f"[{camera_type} HTTP {response.status_code}] -> {response.text}")
        except Exception as e:
            print(f"[{camera_type} Network Error] -> {e}")

    threading.Thread(target=_worker, daemon=True).start()

def run_local_stream():
    print("[*] Starting Local Laptop Webcam Stream (Testing Mode)...")
    print("[*] CONTROLS:")
    print("    - Press 't' to TOGGLE between LOGIN and LOGOUT gate mode.")
    print("    - Press 'q' to QUIT stream.")

    webcam = LocalWebcamThread(LOCAL_CAM_INDEX)
    last_process_time = 0.0
    COOLDOWN = 1.8
    current_mode = "LOGIN"

    while True:
        ret, frame = webcam.read()
        current_time = time.time()

        if ret and frame is not None:
            raw_clean = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)

            mode_color = (0, 255, 0) if current_mode == "LOGIN" else (0, 0, 255)

            if boxes is not None:
                for box in boxes:
                    if current_time - last_process_time > COOLDOWN:
                        hd_face = extract_ultra_hd_face(raw_clean, box)
                        if hd_face is not None:
                            send_frame_to_backend(hd_face, current_mode)
                            last_process_time = current_time

                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), mode_color, 2)
                    cv2.putText(
                        frame,
                        f"{current_mode} Mode",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.6,
                        mode_color,
                        1
                    )

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (15, 23, 42), -1)
            cv2.putText(
                frame,
                f"Active Mode: [{current_mode}] | Press 't' to toggle Gate | 'q' to quit",
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                mode_color,
                2
            )

            cv2.imshow("System Camera - AI Face Attendance", frame)
        else:
            time.sleep(0.02)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            current_mode = "LOGOUT" if current_mode == "LOGIN" else "LOGIN"
            print(f"[*] Switched Gate Mode to: {current_mode}")

    webcam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_local_stream()