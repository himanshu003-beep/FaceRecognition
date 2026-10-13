import os
import io
import json
import base64
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pymysql
from PIL import Image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
import pytz

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CAPTURES_DIR = BASE_DIR / "captures"
UNKNOWN_DIR = CAPTURES_DIR / "unknown"
EMPLOYEE_PHOTOS_DIR = BASE_DIR / "employee_photos"

CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
EMPLOYEE_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

IST = pytz.timezone("Asia/Kolkata")

# Calibrated Threshold: 0.68 accounts for RTSP H.264 compression and local webcam variations
# Legitimate cross-camera matches score 0.70 - 0.88; strangers score < 0.50
RECOGNITION_THRESHOLD = 0.68

def get_current_ist() -> datetime:
    return datetime.now(IST)

def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def calculate_duration(start_str: str, end_dt: datetime) -> str:
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        start_dt = IST.localize(start_dt)
        delta = end_dt - start_dt
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            total_seconds = 0
        hrs, rem = divmod(total_seconds, 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        return "-"

def get_mysql_connection():
    try:
        return pymysql.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "ai_attendance"),
            password=os.getenv("DB_PASSWORD", "2003"),
            database=os.getenv("DB_NAME", "Ai_attendance"),
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
            connect_timeout=5
        )
    except Exception as err:
        print(f"[DB ERROR] Connection failed: {err}")
        return None

app = FastAPI(title="Office Attendance AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.mount("/employee_photos", StaticFiles(directory=str(EMPLOYEE_PHOTOS_DIR)), name="employee_photos")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")

# Deep Learning Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
registration_mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def clean_to_1d_flat_vector(raw_vector: Any) -> np.ndarray:
    arr = np.asarray(raw_vector, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm == 0 or np.isnan(norm):
        return arr
    return arr / norm

def check_image_blur(cv_img: np.ndarray, min_variance: float = 25.0) -> bool:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return bool(variance >= min_variance)

def extract_embedding_from_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        raw_embedding = resnet(tensor).squeeze().cpu().numpy()
    return clean_to_1d_flat_vector(raw_embedding)

def extract_face_and_embedding_for_registration(image_bytes: bytes) -> tuple[np.ndarray, Image.Image]:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_img is None:
        raise ValueError("Invalid image file format.")

    if not check_image_blur(cv_img, min_variance=25.0):
        raise ValueError("Image is too blurry. Please capture steady frontal photo.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face_tensor = registration_mtcnn(img)
    if face_tensor is None:
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_emb = resnet(tensor).squeeze().cpu().numpy()
        return clean_to_1d_flat_vector(raw_emb), img

    face_tensor = face_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        raw_emb = resnet(face_tensor).squeeze().cpu().numpy()
    return clean_to_1d_flat_vector(raw_emb), img

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    v1 = clean_to_1d_flat_vector(vec1)
    v2 = clean_to_1d_flat_vector(vec2)
    if v1.shape != (512,) or v2.shape != (512,):
        if len(v1) != len(v2) or len(v1) == 0:
            return 0.0
    dot_val = float(np.dot(v1, v2))
    return float(max(0.0, min(1.0, dot_val)))

REGISTERED_USERS: Dict[str, Dict[str, Any]] = {}
attendance_records: List[Dict[str, Any]] = []
recent_unknown_vectors: List[Dict[str, Any]] = []
employee_attendance_cooldown: Dict[str, float] = {}

def ensure_schema_compatibility(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL,
                name VARCHAR(100) NOT NULL,
                photo_url LONGTEXT,
                similarity_score VARCHAR(20) DEFAULT '-',
                login_time VARCHAR(50),
                logout_time VARCHAR(50) DEFAULT '-',
                duration VARCHAR(50) DEFAULT '-',
                status VARCHAR(50) DEFAULT 'IN OFFICE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INT AUTO_INCREMENT PRIMARY KEY,
                emp_id VARCHAR(50) UNIQUE NOT NULL,
                name VARCHAR(100) NOT NULL,
                photo_url VARCHAR(255) DEFAULT '',
                embedding_json LONGTEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)

def parse_stored_embeddings(json_str: str) -> List[np.ndarray]:
    try:
        data = json.loads(json_str)
        vectors = []
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], (int, float)):
                vectors.append(clean_to_1d_flat_vector(data))
            else:
                for item in data:
                    flat = clean_to_1d_flat_vector(item)
                    if len(flat) == 512:
                        vectors.append(flat)
        return vectors
    except Exception as e:
        print(f"[EMBEDDING PARSE ERROR] -> {e}")
        return []

def init_db_and_load_state():
    global attendance_records, REGISTERED_USERS
    conn = get_mysql_connection()
    if not conn:
        print("[!] DB Connection offline. Retrying...")
        return

    try:
        ensure_schema_compatibility(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT emp_id, name, photo_url, embedding_json FROM employees WHERE photo_url != '' AND photo_url IS NOT NULL")
            emp_rows = cur.fetchall()
            REGISTERED_USERS.clear()
            for er in emp_rows:
                vec_list = parse_stored_embeddings(er["embedding_json"])
                if vec_list:
                    REGISTERED_USERS[er["emp_id"]] = {
                        "name": er["name"],
                        "photo_url": er.get("photo_url", ""),
                        "embeddings": vec_list
                    }
            print(f"[*] Loaded {len(REGISTERED_USERS)} verified employees into memory.")

            cur.execute("SELECT id, user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status FROM attendance ORDER BY id DESC LIMIT 500")
            rows = cur.fetchall()
            attendance_records = list(rows) if rows else []
            print(f"[*] Restored {len(attendance_records)} attendance logs from database.")
    except Exception as e:
        print(f"[DB INIT ERROR] -> {traceback.format_exc()}")
    finally:
        conn.close()

init_db_and_load_state()

def identify_face(incoming_embedding: np.ndarray, threshold: float = RECOGNITION_THRESHOLD) -> tuple[str, str, float, bool]:
    best_user_id = None
    best_name = None
    highest_score = 0.0

    in_vec = clean_to_1d_flat_vector(incoming_embedding)

    for uid, data in REGISTERED_USERS.items():
        for reg_vec in data.get("embeddings", []):
            score = calculate_cosine_similarity(in_vec, reg_vec)
            if score > highest_score:
                highest_score = score
                best_user_id = uid
                best_name = data["name"]

    print(f"[DEBUG INFERENCE] Top Match: {best_user_id} ({best_name}) | Score: {highest_score:.4f} | Threshold: {threshold}")

    if highest_score >= threshold and best_user_id:
        return best_user_id, best_name, highest_score, False

    return "UNKNOWN", "Unknown Visitor", highest_score, True

def check_is_new_unknown(incoming_embedding: np.ndarray) -> bool:
    global recent_unknown_vectors
    current_time = time.time()
    recent_unknown_vectors = [u for u in recent_unknown_vectors if current_time - u["time"] < 600.0]

    for item in recent_unknown_vectors:
        score = calculate_cosine_similarity(incoming_embedding, item["vec"])
        if score >= 0.70:
            return False

    recent_unknown_vectors.append({"vec": incoming_embedding, "time": current_time})
    return True

def convert_bytes_to_base64_data_url(image_bytes: bytes) -> str:
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

def save_unknown_photo(image_bytes: bytes) -> str:
    timestamp = int(time.time() * 1000)
    filename = f"unknown_{timestamp}.jpg"
    filepath = UNKNOWN_DIR / filename
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.save(filepath, "JPEG", quality=95)
    return f"http://127.0.0.1:8000/captures/unknown/{filename}"

def record_attendance(user_id: str, name: str, camera_type: str, photo_url: str, score: float, is_unknown: bool):
    global attendance_records, employee_attendance_cooldown
    now_ist = get_current_ist()
    time_str = format_timestamp(now_ist)
    current_ts = time.time()
    score_display = f"{score:.3f}"

    conn = get_mysql_connection()

    if is_unknown:
        new_entry = {
            "id": len(attendance_records) + 1,
            "user_id": "UNKNOWN",
            "name": "Unknown Visitor",
            "photo_url": photo_url,
            "similarity_score": score_display,
            "login_time": time_str if camera_type == "LOGIN" else "-",
            "logout_time": time_str if camera_type == "LOGOUT" else "-",
            "duration": "-",
            "status": "UNAUTHORIZED"
        }
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (new_entry["user_id"], new_entry["name"], photo_url, score_display, new_entry["login_time"], new_entry["logout_time"], "-", "UNAUTHORIZED")
                    )
                    new_entry["id"] = cur.lastrowid
            finally:
                conn.close()

        attendance_records.insert(0, new_entry)
        return "unknown_detected", attendance_records

    # Registered Employee Logic:
    last_marked = employee_attendance_cooldown.get(user_id, 0.0)
    active_record = None
    for r in attendance_records:
        if r["user_id"] == user_id and r["status"] == "IN OFFICE":
            active_record = r
            break

    if camera_type == "LOGIN":
        if active_record:
            # Already active in office: Update live score without adding duplicate row
            active_record["similarity_score"] = score_display
            if conn:
                conn.close()
            return "already_logged_in", attendance_records

        # Debounce: Do not insert if employee already marked within 30 seconds
        if current_ts - last_marked < 30.0:
            if conn:
                conn.close()
            return "cooldown_throttled", attendance_records

        employee_attendance_cooldown[user_id] = current_ts
        display_photo = REGISTERED_USERS.get(user_id, {}).get("photo_url") or photo_url

        new_entry = {
            "id": len(attendance_records) + 1,
            "user_id": user_id,
            "name": name,
            "photo_url": display_photo,
            "similarity_score": score_display,
            "login_time": time_str,
            "logout_time": "-",
            "duration": "-",
            "status": "IN OFFICE"
        }
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (user_id, name, display_photo, score_display, time_str, "-", "-", "IN OFFICE")
                    )
                    new_entry["id"] = cur.lastrowid
            finally:
                conn.close()

        attendance_records.insert(0, new_entry)
        return "login_recorded", attendance_records

    elif camera_type == "LOGOUT":
        if not active_record:
            # Fallback if employee was not logged in prior: mark independent logout
            display_photo = REGISTERED_USERS.get(user_id, {}).get("photo_url") or photo_url
            new_entry = {
                "id": len(attendance_records) + 1,
                "user_id": user_id,
                "name": name,
                "photo_url": display_photo,
                "similarity_score": score_display,
                "login_time": "-",
                "logout_time": time_str,
                "duration": "00:00:00",
                "status": "CHECKED OUT"
            }
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                            (user_id, name, display_photo, score_display, "-", time_str, "00:00:00", "CHECKED OUT")
                        )
                        new_entry["id"] = cur.lastrowid
                finally:
                    conn.close()

            attendance_records.insert(0, new_entry)
            return "logout_recorded", attendance_records

        # Complete Active Session
        duration_calc = calculate_duration(str(active_record["login_time"]), now_ist)
        active_record["logout_time"] = time_str
        active_record["duration"] = duration_calc
        active_record["status"] = "CHECKED OUT"
        active_record["similarity_score"] = score_display

        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE attendance SET logout_time=%s, duration=%s, status=%s, similarity_score=%s WHERE id=%s",
                        (time_str, duration_calc, "CHECKED OUT", score_display, active_record["id"])
                    )
            finally:
                conn.close()

        return "logout_recorded", attendance_records

    if conn:
        conn.close()
    return "invalid_camera", attendance_records

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        payload = json.dumps(message, default=str)
        for conn in list(self.active_connections):
            try:
                await conn.send_text(payload)
            except Exception:
                pass

manager = ConnectionManager()

@app.get("/")
def root():
    return {"status": "online", "system": "Office Attendance", "time_ist": format_timestamp(get_current_ist())}

@app.get("/attendance/logs")
def get_logs():
    return attendance_records

@app.get("/api/employees/list")
def list_employees():
    result = []
    for uid, data in REGISTERED_USERS.items():
        result.append({
            "emp_id": uid,
            "name": data["name"],
            "photo_url": data.get("photo_url", ""),
            "vector_count": len(data.get("embeddings", []))
        })
    return result

@app.delete("/api/employees/{emp_id}")
def delete_employee(emp_id: str):
    global REGISTERED_USERS
    if emp_id not in REGISTERED_USERS:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Employee {emp_id} not found."})

    del REGISTERED_USERS[emp_id]

    conn = get_mysql_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM employees WHERE emp_id = %s", (emp_id,))
        finally:
            conn.close()

    print(f"[*] Successfully purged employee: {emp_id}")
    return {"status": "success", "message": f"Employee {emp_id} successfully deleted."}

@app.post("/api/employees/register")
async def register_employee(
    emp_id: str = Form(...),
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        clean_emp_id = emp_id.strip()
        clean_name = name.strip()

        if not files or len(files) == 0:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Face photo is strictly required."})

        embeddings_list = []
        primary_photo_url = ""
        processing_errors = []

        for idx, file in enumerate(files):
            try:
                image_bytes = await file.read()
                if len(image_bytes) == 0:
                    continue
                emb, pil_img = extract_face_and_embedding_for_registration(image_bytes)
                embeddings_list.append(emb)

                if primary_photo_url == "":
                    filename = f"{clean_emp_id}_{int(time.time())}.jpg"
                    filepath = EMPLOYEE_PHOTOS_DIR / filename
                    pil_img.save(filepath, "JPEG", quality=95)
                    primary_photo_url = f"http://127.0.0.1:8000/employee_photos/{filename}"
            except ValueError as ve:
                processing_errors.append(f"Photo {idx + 1}: {str(ve)}")

        if not embeddings_list or not primary_photo_url:
            err_msg = " | ".join(processing_errors) if processing_errors else "No valid face photo could be verified."
            return JSONResponse(status_code=400, content={"status": "error", "message": err_msg})

        REGISTERED_USERS[clean_emp_id] = {
            "name": clean_name,
            "photo_url": primary_photo_url,
            "embeddings": embeddings_list
        }

        conn = get_mysql_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    json_payload = json.dumps([e.tolist() for e in embeddings_list])
                    cur.execute(
                        "REPLACE INTO employees (emp_id, name, photo_url, embedding_json) VALUES (%s, %s, %s, %s)",
                        (clean_emp_id, clean_name, primary_photo_url, json_payload)
                    )
            finally:
                conn.close()

        print(f"[SUCCESS ENROLLMENT] {clean_name} ({clean_emp_id}) stored.")
        return {
            "status": "success",
            "message": f"Successfully Registered: {clean_name} ({clean_emp_id})",
            "emp_id": clean_emp_id,
            "name": clean_name,
            "photo_url": primary_photo_url
        }
    except Exception as e:
        print(f"[REGISTER CRASH] -> {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Server Error: {str(e)}"})

@app.post("/api/attendance/process")
async def process_attendance(file: UploadFile = File(...), camera_type: str = Form(...)):
    try:
        image_bytes = await file.read()
        embedding = extract_embedding_from_bytes(image_bytes)

        user_id, name, score, is_unknown = identify_face(incoming_embedding=embedding, threshold=RECOGNITION_THRESHOLD)

        if is_unknown:
            is_new_unknown = check_is_new_unknown(embedding)
            if not is_new_unknown:
                await manager.broadcast({
                    "event": "LIVE_SCORE_UPDATE",
                    "user_id": "UNKNOWN",
                    "similarity_score": f"{score:.3f}",
                    "logs": attendance_records
                })
                return {
                    "status": "ignored_same_unknown_spam",
                    "user_id": "UNKNOWN",
                    "similarity_score": round(score, 3),
                    "is_unknown": True
                }

            photo_url = save_unknown_photo(image_bytes)
        else:
            photo_url = REGISTERED_USERS.get(user_id, {}).get("photo_url") or convert_bytes_to_base64_data_url(image_bytes)

        action, current_logs = record_attendance(user_id, name, camera_type.upper(), photo_url, score, is_unknown)

        await manager.broadcast({
            "event": "ATTENDANCE_UPDATE",
            "action": action,
            "user_id": user_id,
            "name": name,
            "photo_url": photo_url,
            "similarity_score": f"{score:.3f}",
            "is_unknown": is_unknown,
            "logs": current_logs
        })

        return {
            "status": action,
            "user_id": user_id,
            "name": name,
            "similarity_score": round(score, 3),
            "is_unknown": is_unknown,
            "timestamp": format_timestamp(get_current_ist())
        }
    except Exception as e:
        print(f"[PROCESS CRASH] -> {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "user_id": "None",
            "similarity_score": 0.0,
            "message": str(e)
        })

@app.websocket("/ws/attendance")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps({
            "event": "ATTENDANCE_UPDATE",
            "action": "initial_load",
            "logs": attendance_records
        }, default=str))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)