import os
import io
import json
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

# ----------------- LOCAL CAPTURES DIRECTORY -----------------
CAPTURES_DIR = Path("captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- TIMEZONE CONFIGURATION -----------------
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

# ----------------- DATABASE CONNECTION -----------------
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

# ----------------- APP INITIALIZATION -----------------
app = FastAPI(title="AI Face Attendance Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/captures", StaticFiles(directory=CAPTURES_DIR), name="captures")

# ----------------- AI MODEL SETUP -----------------
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
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

# ----------------- PERSISTENT STATE & MATCH ENGINE -----------------
REGISTERED_USERS: Dict[str, Dict[str, Any]] = {}
user_counter = 1
attendance_records: List[Dict[str, Any]] = []

def init_db_and_load_state():
    global attendance_records, user_counter
    conn = get_mysql_connection()
    if not conn:
        print("[!] DB Offline. Starting with in-memory state.")
        return

    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(50) NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    photo_url VARCHAR(255) DEFAULT '',
                    similarity_score VARCHAR(20) DEFAULT '-',
                    login_time VARCHAR(50),
                    logout_time VARCHAR(50) DEFAULT '-',
                    duration VARCHAR(50) DEFAULT '-',
                    status VARCHAR(50) DEFAULT 'IN OFFICE',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """)
            conn.commit()

            cur.execute("SELECT id, user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status FROM attendance ORDER BY id DESC LIMIT 100")
            rows = cur.fetchall()
            if rows:
                attendance_records = rows
                print(f"[*] Successfully restored {len(attendance_records)} previous records from MySQL.")

            cur.execute("SELECT user_id FROM attendance WHERE user_id LIKE 'EMP_%'")
            emp_rows = cur.fetchall()
            max_num = 0
            for r in emp_rows:
                try:
                    num = int(r["user_id"].split("_")[1])
                    if num > max_num:
                        max_num = num
                except Exception:
                    pass
            user_counter = max_num + 1

    except Exception as e:
        print(f"[DB INIT ERROR] -> {e}")
    finally:
        conn.close()

init_db_and_load_state()

# THRESHOLD UPDATED TO 0.82 (Strict face matching)
def match_or_create_user(incoming_embedding: np.ndarray, threshold: float = 0.82) -> tuple[str, str, float, bool]:
    global user_counter, REGISTERED_USERS

    best_user_id = None
    best_name = None
    highest_score = -1.0

    for uid, data in REGISTERED_USERS.items():
        for reg_vec in data.get("embeddings", []):
            score = calculate_cosine_similarity(incoming_embedding, reg_vec)
            if score > highest_score:
                highest_score = score
                best_user_id = uid
                best_name = data["name"]

    # Agar match score 0.82 se zyada hai: Existing User Reuse
    if highest_score >= threshold and best_user_id:
        if len(REGISTERED_USERS[best_user_id]["embeddings"]) < 5:
            REGISTERED_USERS[best_user_id]["embeddings"].append(incoming_embedding)
        return best_user_id, best_name, highest_score, False

    # Agar naya chehra hai: New User ID generate karein
    new_uid = f"EMP_{user_counter:03d}"
    new_name = f"Person_{user_counter}"
    user_counter += 1

    REGISTERED_USERS[new_uid] = {
        "name": new_name,
        "embeddings": [incoming_embedding]
    }
    # New user ke liye base score 1.0 ya captured highest store karein
    final_score = 1.0 if highest_score < 0 else highest_score
    print(f"[AUTO-REGISTERED] Created {new_uid} for {new_name} (Threshold check passed)")
    return new_uid, new_name, final_score, True

# ----------------- LOCAL PHOTO SAVE -----------------
def save_photo_and_get_url(image_bytes: bytes, user_id: str, camera_type: str) -> str:
    timestamp = int(time.time() * 1000)
    filename = f"{user_id}_{camera_type.lower()}_{timestamp}.jpg"
    file_path = CAPTURES_DIR / filename
    with open(file_path, "wb") as f:
        f.write(image_bytes)
    return f"http://127.0.0.1:8000/captures/{filename}"

# ----------------- ATTENDANCE ENGINE -----------------
def record_attendance(user_id: str, name: str, camera_type: str, photo_url: str, score: float):
    global attendance_records
    now_ist = get_current_ist()
    time_str = format_timestamp(now_ist)
    score_display = f"{score:.3f}"

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
            except Exception as e:
                print(f"[DB INSERT WARN] -> {e}")
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
                except Exception as e:
                    print(f"[DB INSERT WARN] -> {e}")
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
            except Exception as e:
                print(f"[DB UPDATE WARN] -> {e}")
            finally:
                conn.close()

        return "logout_recorded", attendance_records

    return "invalid_camera", attendance_records

# ----------------- WEBSOCKET BROADCASTER -----------------
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

# ----------------- API ENDPOINTS -----------------
@app.get("/")
def root():
    return {"status": "online", "time_ist": format_timestamp(get_current_ist())}

@app.get("/attendance/logs")
def get_logs():
    return attendance_records

@app.post("/api/attendance/process")
async def process_attendance(file: UploadFile = File(...), camera_type: str = Form(...)):
    try:
        image_bytes = await file.read()
        embedding = extract_embedding(image_bytes)

        # Strict Threshold >= 0.82 matching
        user_id, name, score, is_new = match_or_create_user(incoming_embedding=embedding, threshold=0.82)

        # Photo save & URL generate
        photo_url = save_photo_and_get_url(image_bytes, user_id, camera_type)

        # Attendance record with similarity score
        action, current_logs = record_attendance(user_id, name, camera_type.upper(), photo_url, score)

        # WebSocket live broadcast
        await manager.broadcast({
            "event": "ATTENDANCE_UPDATE",
            "action": action,
            "user_id": user_id,
            "name": name,
            "photo_url": photo_url,
            "similarity_score": f"{score:.3f}",
            "is_new_user": is_new,
            "logs": current_logs
        })

        return {
            "status": action,
            "user_id": user_id,
            "name": name,
            "similarity_score": round(score, 3),
            "is_new_user": is_new,
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