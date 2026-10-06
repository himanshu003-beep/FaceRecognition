import time
import requests
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

# ----------------- CONFIGURATION -----------------
BACKEND_URL = "http://127.0.0.1:8000/api/attendance/process"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MTCNN face detector setup
mtcnn = MTCNN(
    keep_all=True,
    min_face_size=50,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    device=DEVICE
)

# Camera Sources (0 for built-in/USB webcams, or replace with RTSP URL)
CAM_LOGIN_SRC = 0
CAM_LOGOUT_SRC = 2  # Agar doosra camera nahi hai toh test ke liye isko 0 bhi rakh sakte hain

def extract_hd_face(frame, box, target_size=(300, 300)):
    """
    Face box ke around 25% padding lekar clear bicubic HD crop aur mild sharpening apply karta hai.
    """
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

    sharpen_kernel = np.array([
        [0, -0.5, 0],
        [-0.5, 3.0, -0.5],
        [0, -0.5, 0]
    ])
    hd_sharp = cv2.filter2D(hd_crop, -1, sharpen_kernel)
    return hd_sharp

def send_frame_to_backend(cropped_face, camera_type: str):
    """Cropped image ko backend API me post karta hai"""
    try:
        success, encoded_img = cv2.imencode('.jpg', cropped_face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success:
            return

        files = {'file': ('face.jpg', encoded_img.tobytes(), 'image/jpeg')}
        data = {'camera_type': camera_type}
        
        response = requests.post(BACKEND_URL, files=files, data=data, timeout=3)
        if response.status_code == 200:
            res_json = response.json()
            print(f"[{camera_type}] Result: {res_json.get('status')} | User: {res_json.get('user_id')} | Score: {res_json.get('similarity_score')}")
    except Exception as e:
        print(f"[{camera_type} Network Error] -> {e}")

def run_streams():
    cap_login = cv2.VideoCapture(CAM_LOGIN_SRC)
    cap_logout = cv2.VideoCapture(CAM_LOGOUT_SRC)

    last_login_time = 0
    last_logout_time = 0
    COOLDOWN = 3  # seconds cooldown between hits

    print("[*] Starting dual camera streams... Press 'q' to stop.")

    while True:
        ret_in, frame_in = cap_login.read()
        ret_out, frame_out = cap_logout.read()

        current_time = time.time()

        # Login Camera Process
        if ret_in:
            rgb_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)
            boxes_in, _ = mtcnn.detect(rgb_in)
            if boxes_in is not None:
                for box in boxes_in:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame_in, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_in, "Login Cam", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if current_time - last_login_time > COOLDOWN:
                        hd_face = extract_hd_face(frame_in, box)
                        if hd_face is not None:
                            send_frame_to_backend(hd_face, "LOGIN")
                            last_login_time = current_time

            cv2.imshow("Gate 1 - LOGIN CAMERA", frame_in)

        # Logout Camera Process (if second camera connected)
        if ret_out:
            rgb_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            boxes_out, _ = mtcnn.detect(rgb_out)
            if boxes_out is not None:
                for box in boxes_out:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    cv2.rectangle(frame_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_out, "Logout Cam", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    if current_time - last_logout_time > COOLDOWN:
                        hd_face = extract_hd_face(frame_out, box)
                        if hd_face is not None:
                            send_frame_to_backend(hd_face, "LOGOUT")
                            last_logout_time = current_time

            cv2.imshow("Gate 2 - LOGOUT CAMERA", frame_out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_login.release()
    cap_logout.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_streams()