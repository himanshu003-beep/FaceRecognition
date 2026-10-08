import os
import io
import json
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pymysql
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from dotenv import load_dotenv
import pytz

load_dotenv()

# Directories
CAPTURES_DIR = Path("captures")
UNKNOWN_DIR = CAPTURES_DIR / "unknown"
EMPLOYEE_PHOTOS_DIR = Path("employee_photos")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
EMPLOYEE_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

IST = pytz.timezone("Asia/Kolkata")

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
            connect_timeout=3
        )
    except Exception as err:
        print(f"[DB WARNING] Connection error: {err}")
        return None

app = FastAPI(title="Office Attendance Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/employee_photos", StaticFiles(directory=EMPLOYEE_PHOTOS_DIR), name="employee_photos")
app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

# FaceNet AI Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def extract_embedding(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(tensor).squeeze(0).cpu().numpy()
    return embedding

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = float(np.dot(vec1, vec2))
    norm_a = float(np.linalg.norm(vec1))
    norm_b = float(np.linalg.norm(vec2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    val = dot_product / (norm_a * norm_b)
    return max(0.0, min(1.0, float(val)))

# Global In-Memory State
REGISTERED_USERS: Dict[str, Dict[str, Any]] = {}
attendance_records: List[Dict[str, Any]] = []
last_unknown_capture_time = 0.0
UNKNOWN_COOLDOWN = 600.0  # 10 minutes tak dubara capture nahi karega

def init_db_and_load_state():
    global attendance_records, REGISTERED_USERS
    conn = get_mysql_connection()
    if not conn:
        print("[!] DB Offline. Starting with empty memory state.")
        return

    try:
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
            conn.commit()

            cur.execute("SELECT emp_id, name, photo_url, embedding_json FROM employees")
            emp_rows = cur.fetchall()
            for er in emp_rows:
                emb_list = json.loads(er["embedding_json"])
                REGISTERED_USERS[er["emp_id"]] = {
                    "name": er["name"],
                    "photo_url": er.get("photo_url", ""),
                    "embeddings": [np.array(emb_list, dtype=np.float32)]
                }
            print(f"[*] Loaded {len(REGISTERED_USERS)} registered employees from DB.")

            cur.execute("SELECT id, user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status FROM attendance ORDER BY id DESC LIMIT 500")
            rows = cur.fetchall()
            if rows:
                attendance_records = rows
                print(f"[*] Restored {len(attendance_records)} attendance logs from DB.")
    except Exception as e:
        print(f"[DB INIT ERROR] -> {e}")
    finally:
        conn.close()

init_db_and_load_state()

def identify_face(incoming_embedding: np.ndarray, threshold: float = 0.82) -> tuple[str, str, float, bool]:
    best_user_id = None
    best_name = None
    highest_score = 0.0

    for uid, data in REGISTERED_USERS.items():
        for reg_vec in data.get("embeddings", []):
            score = calculate_cosine_similarity(incoming_embedding, reg_vec)
            if score > highest_score:
                highest_score = score
                best_user_id = uid
                best_name = data["name"]

    if highest_score >= threshold and best_user_id:
        return best_user_id, best_name, highest_score, False

    return "UNKNOWN", "Unknown Visitor", highest_score, True

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
    global attendance_records
    now_ist = get_current_ist()
    time_str = format_timestamp(now_ist)
    score_display = f"{score:.3f}"

    if is_unknown:
        new_entry = {
            "id": len(attendance_records) + 1,
            "user_id": "UNKNOWN",
            "name": "Unknown Visitor",
            "photo_url": photo_url,
            "similarity_score": score_display,
            "login_time": time_str,
            "logout_time": "-",
            "duration": "-",
            "status": "UNAUTHORIZED"
        }
        attendance_records.insert(0, new_entry)
        return "unknown_detected", attendance_records

    active_record = None
    for r in attendance_records:
        if r["user_id"] == user_id and r["status"] == "IN OFFICE":
            active_record = r
            break

    if camera_type == "LOGIN":
        if active_record:
            return "already_logged_in", attendance_records

        inserted_id = len(attendance_records) + 1
        conn = get_mysql_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (user_id, name, photo_url, score_display, time_str, "-", "-", "IN OFFICE")
                    )
                    conn.commit()
                    inserted_id = cur.lastrowid
            finally:
                conn.close()

        new_entry = {
            "id": inserted_id,
            "user_id": user_id,
            "name": name,
            "photo_url": photo_url,
            "similarity_score": score_display,
            "login_time": time_str,
            "logout_time": "-",
            "duration": "-",
            "status": "IN OFFICE"
        }
        attendance_records.insert(0, new_entry)
        return "login_recorded", attendance_records

    elif camera_type == "LOGOUT":
        if not active_record:
            inserted_id = len(attendance_records) + 1
            conn = get_mysql_connection()
            if conn:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                            (user_id, name, photo_url, score_display, "-", time_str, "00:00:00", "CHECKED OUT")
                        )
                        conn.commit()
                        inserted_id = cur.lastrowid
                finally:
                    conn.close()

            new_entry = {
                "id": inserted_id,
                "user_id": user_id,
                "name": name,
                "photo_url": photo_url,
                "similarity_score": score_display,
                "login_time": "-",
                "logout_time": time_str,
                "duration": "00:00:00",
                "status": "CHECKED OUT"
            }
            attendance_records.insert(0, new_entry)
            return "logout_recorded", attendance_records

        duration_calc = calculate_duration(str(active_record["login_time"]), now_ist)
        active_record["logout_time"] = time_str
        active_record["duration"] = duration_calc
        active_record["status"] = "CHECKED OUT"
        active_record["photo_url"] = photo_url
        active_record["similarity_score"] = score_display

        conn = get_mysql_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE attendance SET logout_time=%s, duration=%s, status=%s, photo_url=%s, similarity_score=%s WHERE id=%s",
                        (time_str, duration_calc, "CHECKED OUT", photo_url, score_display, active_record["id"])
                    )
                    conn.commit()
            finally:
                conn.close()

        return "logout_recorded", attendance_records

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

# ----------------- ENDPOINTS -----------------
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
            "photo_url": data.get("photo_url", "")
        })
    return result

@app.post("/api/employees/register")
async def register_employee(emp_id: str = Form(...), name: str = Form(...), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        embedding = extract_embedding(image_bytes)

        # Save profile photo
        filename = f"{emp_id}_{int(time.time())}.jpg"
        filepath = EMPLOYEE_PHOTOS_DIR / filename
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(filepath, "JPEG", quality=95)
        photo_url = f"http://127.0.0.1:8000/employee_photos/{filename}"

        REGISTERED_USERS[emp_id] = {
            "name": name,
            "photo_url": photo_url,
            "embeddings": [embedding]
        }

        conn = get_mysql_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "REPLACE INTO employees (emp_id, name, photo_url, embedding_json) VALUES (%s, %s, %s, %s)",
                        (emp_id, name, photo_url, json.dumps(embedding.tolist()))
                    )
                    conn.commit()
            finally:
                conn.close()

        print(f"[REGISTERED] {emp_id} - {name} saved with photo.")
        return {"status": "success", "message": f"Successfully Saved: {name} ({emp_id})"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/attendance/process")
async def process_attendance(file: UploadFile = File(...), camera_type: str = Form(...)):
    global last_unknown_capture_time
    try:
        image_bytes = await file.read()
        embedding = extract_embedding(image_bytes)

        user_id, name, score, is_unknown = identify_face(incoming_embedding=embedding, threshold=0.82)

        # UNKNOWN 1-TIME CAPTURE FILTER: Do not spam capture if unknown repeats within 10 min
        current_time = time.time()
        if is_unknown:
            if current_time - last_unknown_capture_time < UNKNOWN_COOLDOWN:
                return {"status": "ignored_unknown_cooldown", "score": round(score, 3)}
            last_unknown_capture_time = current_time
            # Save strictly inside unknown/ folder
            photo_url = save_unknown_photo(image_bytes)
        else:
            photo_url = convert_bytes_to_base64_data_url(image_bytes)

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
        print(f"[PROCESS ERROR] -> {e}")
        return {"status": "error", "message": str(e)}

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