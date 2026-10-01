import os
import sys
import time
import threading
import cv2
import torch
import requests
import numpy as np
from facenet_pytorch import MTCNN

# GUI Warnings ko suppress karein
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.font*=false"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

# ----------------- CONFIGURATION -----------------
BACKEND_URL = "http://127.0.0.1:8000/api/attendance/process"
COOLDOWN_SECONDS = 3.0

CAM1_CONFIG = {
    "name": "Office Entry Feed (Cam1 - LOGIN)",
    "source": 0,  # Built-in Webcam
    "type": "LOGIN",
    "box_color": (0, 255, 0)
}

CAM2_CONFIG = {
    "name": "Office Exit Feed (Cam2 - LOGOUT)",
    "source": "rtsp://nidhin:Nidhin123@192.168.2.178:554/Streaming/Channels/101",  # RTSP Camera
    "type": "LOGOUT",
    "box_color": (0, 0, 255)
}

# ----------------- MODEL SETUP -----------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[*] Initializing MTCNN Detector on device: {device}...")
detector = MTCNN(keep_all=False, select_largest=True, post_process=False, device=device)

# ----------------- THREAD-SAFE CAPTURE CLASS -----------------
class VideoCaptureAsync:
    def __init__(self, src, is_rtsp=False):
        self.src = src
        self.is_rtsp = is_rtsp
        if is_rtsp:
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)
            
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        if not self.cap.isOpened():
            print(f"[WARN] Cannot open video source: {src}")
            self.running = False
            return

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

# ----------------- CAMERA STATE OBJECT -----------------
class CameraWorker:
    def __init__(self, config, is_rtsp=False):
        self.config = config
        self.stream = VideoCaptureAsync(config["source"], is_rtsp=is_rtsp)
        self.last_sent = 0
        self.status = "Monitoring..."
        self.frame_count = 0

    def process_frame(self, frame):
        self.frame_count += 1
        h, w, _ = frame.shape

        # Har 4th frame par detect karein taaki smooth rahe
        if self.frame_count % 4 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = detector.detect(rgb)

            if boxes is not None and len(boxes) > 0 and probs[0] > 0.90:
                box = boxes[0].astype(int)
                x1, y1 = max(0, box[0]), max(0, box[1])
                x2, y2 = min(w, box[2]), min(h, box[3])
                
                face_crop = frame[y1:y2, x1:x2]
                curr_time = time.time()

                if face_crop.size > 0 and (curr_time - self.last_sent > COOLDOWN_SECONDS):
                    self.last_sent = curr_time
                    self.status = f"Sending {self.config['type']}..."

                    # API call in a quick daemon thread so video does not stutter
                    threading.Thread(
                        target=self._send_api, 
                        args=(face_crop, self.config['type']),
                        daemon=True
                    ).start()

                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config["box_color"], 2)
                cv2.putText(frame, f"{self.config['type']} DETECTED", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config["box_color"], 2)

        cv2.putText(frame, self.status, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    def _send_api(self, face_crop, cam_type):
        try:
            _, img_encoded = cv2.imencode('.jpg', face_crop)
            files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
            data = {'camera_type': cam_type}
            res = requests.post(BACKEND_URL, files=files, data=data, timeout=3)
            if res.status_code == 200:
                res_json = res.json()
                self.status = f"{res_json.get('status')} ({res_json.get('user_id')})"
                print(f"[{cam_type} SUCCESS] -> {res_json}")
            else:
                self.status = f"HTTP {res.status_code}"
        except Exception as e:
            self.status = "Backend Err"
            print(f"[{cam_type} ERROR] -> {e}")

# ----------------- MAIN RUNNER -----------------
if __name__ == "__main__":
    print("[*] Launching Dual Stream Attendance System...")
    print("[*] Press 'q' on any active window to exit.")

    cam1 = CameraWorker(CAM1_CONFIG, is_rtsp=False)
    cam2 = CameraWorker(CAM2_CONFIG, is_rtsp=True)

    try:
        while True:
            # Cam1 Frame Render (Main Thread)
            f1 = cam1.stream.read()
            if f1 is not None:
                f1_processed = cam1.process_frame(f1)
                cv2.imshow(CAM1_CONFIG["name"], f1_processed)

            # Cam2 Frame Render (Main Thread)
            f2 = cam2.stream.read()
            if f2 is not None:
                f2_processed = cam2.process_frame(f2)
                cv2.imshow(CAM2_CONFIG["name"], f2_processed)

            # GUI events must be checked on main thread
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[*] 'q' pressed. Closing all streams...")
                break

    except KeyboardInterrupt:
        print("\n[*] Stopping streams...")
    finally:
        cam1.stream.stop()
        cam2.stream.stop()
        cv2.destroyAllWindows()
        print("[*] All cameras stopped cleanly.")