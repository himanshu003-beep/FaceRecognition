import cv2
import time
import queue
import threading
import requests
import torch
from facenet_pytorch import MTCNN

# ----------------- CONFIGURATION -----------------
BACKEND_URL = "http://127.0.0.1:8000/api/attendance/process"
COOLDOWN_SECONDS = 4.0

# CAM1: Entry RTSP (LOGIN)
CAM1_CONFIG = {
    "name": "Office Entry Feed (Cam1 - LOGIN)",
    "source": "rtsp://nidhin:Nidhin123@192.168.2.178:554/Streaming/Channels/101",
    "type": "LOGIN",
    "box_color": (0, 255, 0)
}

# CAM2: Exit RTSP (LOGOUT)
CAM2_CONFIG = {
    "name": "Office Exit Feed (Cam2 - LOGOUT)",
    "source": "rtsp://nidhin:Nidhin123@192.168.2.179:554/Streaming/Channels/102",
    "type": "LOGOUT",
    "box_color": (0, 0, 255)
}

# ----------------- DETECTOR INITIALIZATION -----------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"[*] Initializing MTCNN Detector on device: {device}...")
detector = MTCNN(keep_all=True, device=device)

# ----------------- RTSP BUFFERLESS VIDEO CAPTURE -----------------
class RTSPStreamReader:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.q = queue.Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()

# ----------------- CAMERA PROCESSOR WORKER -----------------
class CameraWorker:
    def __init__(self, config):
        self.config = config
        self.stream = RTSPStreamReader(config["source"])
        self.last_sent_time = 0.0

    def send_attendance(self, cropped_face):
        success, encoded = cv2.imencode('.jpg', cropped_face)
        if not success:
            return
        files = {'file': ('face.jpg', encoded.tobytes(), 'image/jpeg')}
        data = {'camera_type': self.config['type']}
        try:
            res = requests.post(BACKEND_URL, files=files, data=data, timeout=3)
            print(f"[{self.config['type']} SUCCESS] -> {res.json()}")
        except Exception as e:
            print(f"[{self.config['type']} ERROR] -> {e}")

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        small_frame = cv2.resize(frame, (640, 360))
        rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        detection = detector.detect(rgb)
        boxes = detection[0]

        if boxes is not None:
            scale_x = w / 640.0
            scale_y = h / 360.0

            for box in boxes:
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)

                # Clamp boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.config["box_color"], 2)
                label = f"{self.config['type']}"
                cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config["box_color"], 2)

                # Cooldown check & Async API Dispatch
                now = time.time()
                if (now - self.last_sent_time) > COOLDOWN_SECONDS:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        self.last_sent_time = now
                        threading.Thread(target=self.send_attendance, args=(face_crop,), daemon=True).start()

        return cv2.resize(frame, (800, 480))

if __name__ == "__main__":
    print("[*] Launching Dual Stream Attendance System...")
    print("[*] Press 'q' on any active window to exit.")

    cam1 = CameraWorker(CAM1_CONFIG)
    cam2 = CameraWorker(CAM2_CONFIG)

    try:
        while True:
            f1 = cam1.stream.read()
            if f1 is not None:
                cv2.imshow(CAM1_CONFIG["name"], cam1.process_frame(f1))

            f2 = cam2.stream.read()
            if f2 is not None:
                cv2.imshow(CAM2_CONFIG["name"], cam2.process_frame(f2))

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        cam1.stream.stop()
        cam2.stream.stop()
        cv2.destroyAllWindows()