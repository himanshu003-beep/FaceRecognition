import os
import time
import requests
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

# RTSP TCP Transport Enable Karein taaki frame drop aur packet loss na ho
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ----------------- CONFIGURATION -----------------
BACKEND_URL = "http://127.0.0.1:8000/api/attendance/process"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Office RTSP Camera URLs
CAM_LOGIN_SRC = "rtsp://nidhin:Nidhin123@192.168.2.179:554"
CAM_LOGOUT_SRC = "rtsp://nidhin:Nidhin123@192.168.2.178:554"

# MTCNN Face Detector
mtcnn = MTCNN(
    keep_all=True,
    min_face_size=60,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    device=DEVICE
)

def extract_hd_face(frame, box, target_size=(300, 300)):
    """Face ke chaaron taraf 25% padding lekar clear HD crop extract karta hai"""
    h_img, w_img, _ = frame.shape
    x1, y1, x2, y2 = [int(b) for b in box]

    w = x2 - x1
    h = y2 - y1

    pad_w = int(w * 0.25)
    pad_h = int(h * 0.25)

    nx1 = max(0, x1 - pad_w)
    ny1 = max(0, y1 - pad_h)
    nx2 = min(w_img, x2 + pad_w)
    ny2 = min(h_img, y2 + pad_h)

    crop = frame[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None

    hd_crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_CUBIC)

    # Mild Sharpening
    sharpen_kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3.0, -0.5],
        [0, -0.5, 0]
    ])
    hd_sharp = cv2.filter2D(hd_crop, -1, sharpen_kernel)
    return hd_sharp

def send_frame_to_backend(cropped_face, camera_type: str):
    try:
        success, encoded_img = cv2.imencode('.jpg', cropped_face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            return

        files = {'file': ('face.jpg', encoded_img.tobytes(), 'image/jpeg')}
        data = {'camera_type': camera_type}
        
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=3)
        if response.status_code == 200:
            res = response.json()
            print(f"[{camera_type}] Status: {res.get('status')} | ID: {res.get('user_id')} | Score: {res.get('similarity_score')}")
    except Exception as e:
        print(f"[{camera_type} Network Error] -> {e}")

def create_capture(source):
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def run_streams():
    print("[*] Connecting to Office RTSP Streams...")
    print(f"    - Login Cam:  {CAM_LOGIN_SRC}")
    print(f"    - Logout Cam: {CAM_LOGOUT_SRC}")

    cap_login = create_capture(CAM_LOGIN_SRC)
    cap_logout = create_capture(CAM_LOGOUT_SRC)

    last_login_time = 0
    last_logout_time = 0
    COOLDOWN = 3

    print("[*] Streams active. Press 'q' on video window to quit.")

    while True:
        ret_in, frame_in = cap_login.read()
        ret_out, frame_out = cap_logout.read()

        current_time = time.time()

        # Office Login RTSP Stream Process
        if ret_in and frame_in is not None:
            rgb_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
            boxes_in, _ = mtcnn.detect(rgb_in)
            if boxes_in is not None:
                for box in boxes_in:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame_in, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_in, "Office In Cam", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if current_time - last_login_time > COOLDOWN:
                        hd_face = extract_hd_face(frame_in, box)
                        if hd_face is not None:
                            send_frame_to_backend(hd_face, "LOGIN")
                            last_login_time = current_time

            # Display window size normal rakhne ke liye resize
            display_in = cv2.resize(frame_in, (640, 360))
            cv2.imshow("Gate In - RTSP (192.168.2.179)", display_in)
        else:
            time.sleep(0.05)

        # Office Logout RTSP Stream Process
        if ret_out and frame_out is not None:
            rgb_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            boxes_out, _ = mtcnn.detect(rgb_out)
            if boxes_out is not None:
                for box in boxes_out:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_out, "Office Out Cam", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if current_time - last_logout_time > COOLDOWN:
                        hd_face = extract_hd_face(frame_out, box)
                        if hd_face is not None:
                            send_frame_to_backend(hd_face, "LOGOUT")
                            last_logout_time = current_time

            display_out = cv2.resize(frame_out, (640, 360))
            cv2.imshow("Gate Out - RTSP (192.168.2.178)", display_out)
        else:
            time.sleep(0.05)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_login.release()
    cap_logout.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_streams()