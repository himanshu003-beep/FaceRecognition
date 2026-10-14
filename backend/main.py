import os
import io
import json
import base64
import time
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
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
BACKUPS_DIR = BASE_DIR / "backups"

CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
EMPLOYEE_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

IST = pytz.timezone("Asia/Kolkata")

# Calibrated Thresholds for Dual-Camera Hybrid System
RECOGNITION_THRESHOLD = 0.58
UNCERTAIN_LOWER_THRESHOLD = 0.45

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
        return "00:00:00"

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

app = FastAPI(title="Enterprise Office Attendance AI")

class WebSocketOriginBypassMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] == "websocket":
            new_headers = []
            for name, value in scope.get("headers", []):
                if name.lower() == b"origin":
                    new_headers.append((b"origin", b"http://127.0.0.1:8000"))
                else:
                    new_headers.append((name, value))
            scope["headers"] = new_headers
        await self.app(scope, receive, send)

app.add_middleware(WebSocketOriginBypassMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/employee_photos", StaticFiles(directory=str(EMPLOYEE_PHOTOS_DIR)), name="employee_photos")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")

@app.get("/captures/unknown/{filename}")
async def get_unknown_photo_fallback(filename: str):
    file_path = UNKNOWN_DIR / filename
    if file_path.exists():
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="image/jpeg")
    dummy_jpg = cv2.imencode('.jpg', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()
    return Response(content=dummy_jpg, media_type="image/jpeg")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mtcnn_aligner = MTCNN(
    image_size=160,
    margin=10,
    keep_all=False,
    post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def l2_normalize(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm == 0 or np.isnan(norm):
        return arr
    return arr / norm

def extract_canonical_face_embedding(image_bytes: bytes, require_face: bool = True) -> tuple[np.ndarray, Optional[Image.Image]]:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_img is None:
        raise ValueError("Invalid image file format.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    face_tensor = mtcnn_aligner(img)

    if face_tensor is None:
        if require_face:
            raise ValueError("No distinct face detected. Center your face directly towards camera.")
        else:
            transform_direct = transforms.Compose([
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            face_tensor = transform_direct(img)

    with torch.no_grad():
        face_tensor = face_tensor.unsqueeze(0).to(device)
        raw_emb = resnet(face_tensor).squeeze().cpu().numpy()

    aligned_pil = None
    try:
        aligned_np = face_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        aligned_np = ((aligned_np * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        aligned_pil = Image.fromarray(aligned_np)
    except Exception:
        aligned_pil = img

    return l2_normalize(raw_emb), aligned_pil

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    v1 = l2_normalize(vec1)
    v2 = l2_normalize(vec2)
    if v1.shape != (512,) or v2.shape != (512,):
        return 0.0
    return float(max(0.0, min(1.0, float(np.dot(v1, v2)))))

REGISTERED_USERS: Dict[str, Dict[str, Any]] = {}
attendance_records: List[Dict[str, Any]] = []
employee_attendance_cooldown: Dict[str, float] = {}

TEMPORAL_IDENTITY_LOCK: Dict[str, Dict[str, Any]] = {
    "LOGIN": {},
    "LOGOUT": {}
}

UNKNOWN_CONSECUTIVE_BUFFER: Dict[str, int] = {
    "LOGIN": 0,
    "LOGOUT": 0
}

def parse_stored_embeddings(json_str: str) -> List[np.ndarray]:
    try:
        data = json.loads(json_str)
        vectors = []
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], (int, float)):
                flat = l2_normalize(data)
                if len(flat) == 512:
                    vectors.append(flat)
            else:
                for item in data:
                    flat = l2_normalize(item)
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
        print("[!] DB Offline. Retrying...")
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

            cur.execute("SELECT emp_id, name, photo_url, embedding_json FROM employees")
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
            print(f"[*] LOADED {len(REGISTERED_USERS)} VERIFIED EMPLOYEES: {list(REGISTERED_USERS.keys())}")

            cur.execute("SELECT id, user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status FROM attendance ORDER BY id DESC LIMIT 500")
            rows = cur.fetchall()
            attendance_records = list(rows) if rows else []
            print(f"[*] Restored {len(attendance_records)} attendance logs from database.")
    except Exception as e:
        print(f"[DB INIT ERROR] -> {traceback.format_exc()}")
    finally:
        conn.close()

init_db_and_load_state()

def identify_face_stabilized(incoming_embedding: np.ndarray, camera_type: str) -> tuple[str, str, float, str]:
    global TEMPORAL_IDENTITY_LOCK, UNKNOWN_CONSECUTIVE_BUFFER

    best_user_id = None
    best_name = None
    highest_score = 0.0

    in_vec = l2_normalize(incoming_embedding)
    now_ts = time.time()

    for uid, data in REGISTERED_USERS.items():
        for reg_vec in data.get("embeddings", []):
            score = calculate_cosine_similarity(in_vec, reg_vec)
            if score > highest_score:
                highest_score = score
                best_user_id = uid
                best_name = data["name"]

    print(f"[{camera_type} TELEMETRY] Match: {best_user_id} ({best_name}) | Cosine: {highest_score:.4f}")

    if highest_score >= RECOGNITION_THRESHOLD and best_user_id:
        TEMPORAL_IDENTITY_LOCK[camera_type] = {
            "user_id": best_user_id,
            "name": best_name,
            "last_seen": now_ts,
            "confidence": highest_score
        }
        UNKNOWN_CONSECUTIVE_BUFFER[camera_type] = 0
        return best_user_id, best_name, highest_score, "REGISTERED"

    locked = TEMPORAL_IDENTITY_LOCK.get(camera_type, {})
    if locked and (now_ts - locked.get("last_seen", 0.0) < 5.0):
        preserved_id = locked["user_id"]
        preserved_name = locked["name"]
        print(f"[{camera_type} STABILIZER] Sticky Lock Preserving: {preserved_id} ({preserved_name})")
        return preserved_id, preserved_name, highest_score, "REGISTERED"

    if highest_score >= UNCERTAIN_LOWER_THRESHOLD:
        return "UNCERTAIN", "Evaluating...", highest_score, "UNCERTAIN_PRESERVED"

    UNKNOWN_CONSECUTIVE_BUFFER[camera_type] = UNKNOWN_CONSECUTIVE_BUFFER.get(camera_type, 0) + 1
    if UNKNOWN_CONSECUTIVE_BUFFER[camera_type] < 3:
        return "EVALUATING", "Evaluating...", highest_score, "UNCERTAIN_PRESERVED"

    TEMPORAL_IDENTITY_LOCK[camera_type] = {}
    return "UNKNOWN", "Unknown Visitor", highest_score, "CONFIRMED_UNKNOWN"

def save_unknown_photo(image_bytes: bytes) -> str:
    timestamp = int(time.time() * 1000)
    filename = f"unknown_{timestamp}.jpg"
    filepath = UNKNOWN_DIR / filename
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(filepath, "JPEG", quality=90)
        return f"http://127.0.0.1:8000/captures/unknown/{filename}"
    except Exception:
        return ""

def record_attendance(user_id: str, name: str, camera_type: str, photo_url: str, score: float, match_state: str):
    global attendance_records, employee_attendance_cooldown
    now_ist = get_current_ist()
    time_str = format_timestamp(now_ist)
    current_ts = time.time()
    score_display = f"{score:.3f}"

    conn = get_mysql_connection()

    if match_state == "UNCERTAIN_PRESERVED" and user_id in ["UNCERTAIN", "EVALUATING"]:
        if conn: conn.close()
        return "uncertain_suppressed", attendance_records

    # 1. UNKNOWN VISITOR
    if match_state == "CONFIRMED_UNKNOWN" or user_id == "UNKNOWN":
        last_unknown = employee_attendance_cooldown.get("UNKNOWN", 0.0)
        if current_ts - last_unknown < 15.0:
            if conn: conn.close()
            return "ignored_same_unknown_spam", attendance_records

        employee_attendance_cooldown["UNKNOWN"] = current_ts

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

    # 2. REGISTERED EMPLOYEE (LOGIN & LOGOUT FLOW)
    active_record = None
    for r in attendance_records:
        if r["user_id"] == user_id and r["status"] == "IN OFFICE":
            active_record = r
            break

    last_marked = employee_attendance_cooldown.get(user_id, 0.0)
    display_photo = REGISTERED_USERS.get(user_id, {}).get("photo_url") or photo_url

    if camera_type == "LOGIN":
        if active_record:
            active_record["similarity_score"] = score_display
            if conn: conn.close()
            return "already_logged_in", attendance_records

        if current_ts - last_marked < 10.0:
            if conn: conn.close()
            return "cooldown_throttled", attendance_records

        employee_attendance_cooldown[user_id] = current_ts

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
            if current_ts - last_marked < 10.0:
                if conn: conn.close()
                return "cooldown_throttled", attendance_records

            employee_attendance_cooldown[user_id] = current_ts
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

        duration_calc = calculate_duration(str(active_record["login_time"]), now_ist)
        active_record["logout_time"] = time_str
        active_record["duration"] = duration_calc
        active_record["status"] = "CHECKED OUT"
        active_record["similarity_score"] = score_display
        employee_attendance_cooldown[user_id] = current_ts

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

    if conn: conn.close()
    return "invalid_camera", attendance_records

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        if websocket not in self.active_connections:
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
    return {"status": "online", "system": "AI Office Attendance", "time_ist": format_timestamp(get_current_ist())}

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
async def delete_employee(emp_id: str):
    global REGISTERED_USERS
    clean_emp = emp_id.strip()
    if clean_emp not in REGISTERED_USERS:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Employee {clean_emp} not found."})

    photo_url = REGISTERED_USERS[clean_emp].get("photo_url", "")
    if photo_url:
        try:
            filename = photo_url.split("/")[-1]
            p = EMPLOYEE_PHOTOS_DIR / filename
            if p.exists():
                p.unlink()
        except Exception:
            pass

    del REGISTERED_USERS[clean_emp]

    conn = get_mysql_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM employees WHERE emp_id = %s", (clean_emp,))
        finally:
            conn.close()

    print(f"[*] DELETED EMPLOYEE: {clean_emp}")
    return {"status": "success", "message": f"Employee {clean_emp} successfully deleted."}

@app.post("/api/attendance/clear-dashboard")
async def clear_dashboard():
    global attendance_records
    conn = get_mysql_connection()
    backup_filename = f"attendance_backup_{int(time.time())}.json"
    backup_path = BACKUPS_DIR / backup_filename

    try:
        with open(backup_path, "w") as f:
            json.dump(attendance_records, f, indent=2, default=str)
        print(f"[*] Backup created: {backup_path}")

        if conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE attendance;")
            conn.close()

        attendance_records.clear()

        await manager.broadcast({
            "event": "ATTENDANCE_UPDATE",
            "action": "dashboard_cleared",
            "logs": []
        })

        return {
            "status": "success",
            "message": "Dashboard cleared successfully. Full backup preserved.",
            "backup_file": backup_filename
        }
    except Exception as e:
        if conn: conn.close()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/attendance/restore-latest-backup")
async def restore_latest_backup():
    global attendance_records
    backup_files = sorted(list(BACKUPS_DIR.glob("attendance_backup_*.json")), reverse=True)
    if not backup_files:
        return JSONResponse(status_code=404, content={"status": "error", "message": "No backups found."})

    latest_file = backup_files[0]
    conn = get_mysql_connection()

    try:
        with open(latest_file, "r") as f:
            restored_data = json.load(f)

        if conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE attendance;")
                for r in reversed(restored_data):
                    cur.execute(
                        "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        (r["user_id"], r["name"], r.get("photo_url", ""), r.get("similarity_score", "-"), r.get("login_time", "-"), r.get("logout_time", "-"), r.get("duration", "-"), r.get("status", "CHECKED OUT"))
                    )
            conn.close()

        attendance_records = restored_data

        await manager.broadcast({
            "event": "ATTENDANCE_UPDATE",
            "action": "backup_restored",
            "logs": attendance_records
        })

        return {
            "status": "success",
            "message": f"Successfully restored {len(restored_data)} logs from {latest_file.name}."
        }
    except Exception as e:
        if conn: conn.close()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

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
            return JSONResponse(status_code=400, content={"status": "error", "message": "Face photo is required."})

        embeddings_list = []
        primary_photo_url = ""

        for file in files:
            image_bytes = await file.read()
            if len(image_bytes) == 0: continue
            emb, aligned_face_pil = extract_canonical_face_embedding(image_bytes, require_face=True)
            embeddings_list.append(emb)

            if primary_photo_url == "" and aligned_face_pil is not None:
                filename = f"{clean_emp_id}_{int(time.time())}.jpg"
                filepath = EMPLOYEE_PHOTOS_DIR / filename
                aligned_face_pil.save(filepath, "JPEG", quality=95)
                primary_photo_url = f"http://127.0.0.1:8000/employee_photos/{filename}"

        if not embeddings_list or not primary_photo_url:
            return JSONResponse(status_code=400, content={"status": "error", "message": "Face verification failed. Keep face straight."})

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

        print(f"[REGISTER SUCCESS] Registered {clean_name} ({clean_emp_id})")
        return {
            "status": "success",
            "message": f"Successfully Registered: {clean_name} ({clean_emp_id})",
            "emp_id": clean_emp_id,
            "name": clean_name,
            "photo_url": primary_photo_url
        }
    except Exception as e:
        print(f"[REGISTER CRASH] -> {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/api/attendance/process")
@app.post("/api/attendance/process/")
async def process_attendance(file: UploadFile = File(...), camera_type: str = Form(...)):
    try:
        image_bytes = await file.read()
        embedding, _ = extract_canonical_face_embedding(image_bytes, require_face=False)

        user_id, name, score, match_state = identify_face_stabilized(
            incoming_embedding=embedding,
            camera_type=camera_type.upper()
        )

        if match_state == "UNCERTAIN_PRESERVED" and user_id in ["UNCERTAIN", "EVALUATING"]:
            return {
                "status": "uncertain_evaluating",
                "user_id": user_id,
                "similarity_score": round(score, 3),
                "is_unknown": False
            }

        is_unknown = (match_state == "CONFIRMED_UNKNOWN")

        if is_unknown:
            photo_url = save_unknown_photo(image_bytes)
        else:
            photo_url = REGISTERED_USERS.get(user_id, {}).get("photo_url", "")

        action, current_logs = record_attendance(user_id, name, camera_type.upper(), photo_url, score, match_state)

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
        return JSONResponse(status_code=500, content={"status": "error", "user_id": "None", "message": str(e)})

@app.websocket("/ws/attendance")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print(f"[*] WebSocket Connected: {websocket.client}")

    try:
        payload = json.dumps({
            "event": "ATTENDANCE_UPDATE",
            "action": "initial_load",
            "logs": attendance_records
        }, default=str)
        await websocket.send_text(payload)

        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)