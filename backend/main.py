import os
import io
import json
import time
import shutil
import asyncio
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response, FileResponse, RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send
import pymysql
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB
from PIL import Image
import numpy as np
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
import pytz

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = BASE_DIR / "captures"
UNKNOWN_DIR = CAPTURES_DIR / "unknown"
LIVE_PUNCHES_DIR = CAPTURES_DIR / "live_punches"
EMPLOYEE_PHOTOS_DIR = BASE_DIR / "employee_photos"
BACKUPS_DIR = BASE_DIR / "backups"

FRONTEND_DIR = BASE_DIR / "frontend" if (BASE_DIR / "frontend" / "index.html").exists() else BASE_DIR
INDEX_HTML = FRONTEND_DIR / "index.html"

CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
LIVE_PUNCHES_DIR.mkdir(parents=True, exist_ok=True)
EMPLOYEE_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
BACKUPS_DIR.mkdir(parents=True, exist_ok=True)

IST = pytz.timezone("Asia/Kolkata")

# Strict Threshold: Exactly 0.90 as requested
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.90"))

# Configurable Refresh Interval
try:
    DASHBOARD_REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "15"))
    if DASHBOARD_REFRESH_INTERVAL < 3:
        DASHBOARD_REFRESH_INTERVAL = 3
except ValueError:
    DASHBOARD_REFRESH_INTERVAL = 15

GPU_MODEL_LOCK = threading.Lock()
ATTENDANCE_PROCESS_LOCK = asyncio.Lock()

DB_POOL = PooledDB(
    creator=pymysql,
    maxconnections=20,
    mincached=3,
    maxcached=10,
    maxshared=0,
    blocking=True,
    host=os.getenv("DB_HOST", "127.0.0.1"),
    port=int(os.getenv("DB_PORT", 3306)),
    user=os.getenv("DB_USER", "ai_attendance"),
    password=os.getenv("DB_PASSWORD", "2003"),
    database=os.getenv("DB_NAME", "Ai_attendance"),
    cursorclass=DictCursor,
    autocommit=True,
    charset="utf8mb4"
)

def get_db_connection():
    return DB_POOL.connection()

def get_current_ist() -> datetime:
    return datetime.now(IST)

def format_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def calculate_duration(start_str: str, end_dt: datetime) -> str:
    try:
        start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        start_dt = IST.localize(start_dt)
        delta = end_dt - start_dt
        total_seconds = max(0, int(delta.total_seconds()))
        hrs, rem = divmod(total_seconds, 3600)
        mins, secs = divmod(rem, 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    except Exception:
        return "00:00:00"

app = FastAPI(title="Enterprise Office Attendance AI Engine")

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

DUMMY_JPEG_BYTES = cv2.imencode('.jpg', np.zeros((48, 48, 3), dtype=np.uint8))[1].tobytes()

@app.get("/captures/unknown/{filename}")
async def get_unknown_photo_fallback(filename: str):
    sanitized = os.path.basename(filename)
    p = (UNKNOWN_DIR / sanitized).resolve()
    if p.is_file() and str(p).startswith(str(UNKNOWN_DIR.resolve())):
        return FileResponse(p)
    return Response(content=DUMMY_JPEG_BYTES, media_type="image/jpeg", status_code=200)

@app.get("/captures/live_punches/{filename}")
async def get_live_punch_photo(filename: str):
    sanitized = os.path.basename(filename)
    p = (LIVE_PUNCHES_DIR / sanitized).resolve()
    if p.is_file() and str(p).startswith(str(LIVE_PUNCHES_DIR.resolve())):
        return FileResponse(p)
    return Response(content=DUMMY_JPEG_BYTES, media_type="image/jpeg", status_code=200)

app.mount("/employee_photos", StaticFiles(directory=str(EMPLOYEE_PHOTOS_DIR)), name="employee_photos")
app.mount("/captures", StaticFiles(directory=str(CAPTURES_DIR)), name="captures")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn_aligner = MTCNN(image_size=160, margin=14, keep_all=False, post_process=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def l2_normalize(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm == 0 or np.isnan(norm):
        return arr
    return arr / norm

def extract_canonical_face_embedding_sync(image_bytes: bytes, require_face: bool = True) -> Tuple[Optional[np.ndarray], Optional[Image.Image]]:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_img is None:
        raise ValueError("Invalid image buffer.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with GPU_MODEL_LOCK:
        face_tensor = mtcnn_aligner(img)
        if face_tensor is None:
            if require_face:
                raise ValueError("No aligned face detected.")
            return None, None

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

REGISTERED_USERS: Dict[str, Dict[str, Any]] = {}
VECTOR_INDEX_MATRIX: np.ndarray = np.empty((0, 512), dtype=np.float32)
VECTOR_INDEX_MAP: List[Tuple[str, str]] = []
attendance_records: List[Dict[str, Any]] = []

ATTENDANCE_PUNCH_COOLDOWN: Dict[str, float] = {}

def rebuild_vector_cache():
    global VECTOR_INDEX_MATRIX, VECTOR_INDEX_MAP
    all_vecs = []
    all_maps = []
    for uid, data in REGISTERED_USERS.items():
        name = data["name"]
        for vec in data.get("embeddings", []):
            all_vecs.append(vec)
            all_maps.append((uid, name))

    if all_vecs:
        VECTOR_INDEX_MATRIX = np.vstack(all_vecs).astype(np.float32)
        VECTOR_INDEX_MAP = all_maps
    else:
        VECTOR_INDEX_MATRIX = np.empty((0, 512), dtype=np.float32)
        VECTOR_INDEX_MAP = []

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
        return []

def init_db_and_load_state():
    global attendance_records, REGISTERED_USERS
    conn = None
    try:
        conn = get_db_connection()
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_status (user_id, status),
                    INDEX idx_created (created_at)
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
            rebuild_vector_cache()
            print(f"[*] Verified Employees Loaded: {len(REGISTERED_USERS)}")

            cur.execute("SELECT id, user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status FROM attendance ORDER BY id DESC LIMIT 500")
            rows = cur.fetchall()
            attendance_records = list(rows) if rows else []
    except Exception as e:
        print(f"[DB INIT ERROR] -> {traceback.format_exc()}")
    finally:
        if conn: conn.close()

init_db_and_load_state()

def save_live_captured_photo(image_bytes: bytes, user_id: str) -> str:
    timestamp = int(time.time() * 1000)
    filename = f"{user_id}_live_{timestamp}.jpg"
    dest_dir = UNKNOWN_DIR if user_id == "UNKNOWN" else LIVE_PUNCHES_DIR
    filepath = dest_dir / filename
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(filepath, "JPEG", quality=95)
        rel_sub = "unknown" if user_id == "UNKNOWN" else "live_punches"
        return f"/captures/{rel_sub}/{filename}"
    except Exception:
        return ""

def identify_face_strict(incoming_embedding: np.ndarray) -> Tuple[str, str, float]:
    """Strictly applies threshold 0.90 without fuzzy candidate overrides"""
    global VECTOR_INDEX_MATRIX, VECTOR_INDEX_MAP

    in_vec = l2_normalize(incoming_embedding)
    best_user_id = "UNKNOWN"
    best_name = "Unknown Visitor"
    highest_score = 0.0

    if VECTOR_INDEX_MATRIX.shape[0] > 0:
        scores = np.dot(VECTOR_INDEX_MATRIX, in_vec)
        best_idx = int(np.argmax(scores))
        highest_score = float(scores[best_idx])

        # Strict 0.90 threshold enforcement
        if highest_score >= RECOGNITION_THRESHOLD:
            best_user_id, best_name = VECTOR_INDEX_MAP[best_idx]

    return best_user_id, best_name, highest_score

def record_attendance_strict(user_id: str, name: str, camera_type: str, live_photo_url: str, score: float):
    global attendance_records, ATTENDANCE_PUNCH_COOLDOWN
    now_ist = get_current_ist()
    time_str = format_timestamp(now_ist)
    today_str = now_ist.strftime("%Y-%m-%d")
    current_ts = time.time()
    score_display = f"{score:.4f}"

    # Cooldown check per user/camera
    cooldown_key = f"{camera_type}_{user_id}"
    if current_ts - ATTENDANCE_PUNCH_COOLDOWN.get(cooldown_key, 0.0) < 12.0:
        return "cooldown_throttled", attendance_records, None

    # Handle UNKNOWN visitor
    if user_id == "UNKNOWN":
        ATTENDANCE_PUNCH_COOLDOWN[cooldown_key] = current_ts

        if camera_type == "LOGOUT":
            # Close active unknown session from today
            active_unknown = None
            for r in attendance_records:
                if r["user_id"] == "UNKNOWN" and r["status"] == "UNAUTHORIZED":
                    if str(r.get("login_time", "")).startswith(today_str):
                        active_unknown = r
                        break

            if active_unknown:
                duration_calc = calculate_duration(str(active_unknown["login_time"]), now_ist)
                conn = None
                try:
                    conn = get_db_connection()
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE attendance SET logout_time=%s, duration=%s, status=%s WHERE id=%s",
                            (time_str, duration_calc, "CHECKED OUT", active_unknown["id"])
                        )
                    active_unknown["logout_time"] = time_str
                    active_unknown["duration"] = duration_calc
                    active_unknown["status"] = "CHECKED OUT"
                finally:
                    if conn: conn.close()
                return "logout_recorded", attendance_records, active_unknown

            # Direct checkout for unknown
            new_entry = {
                "id": None, "user_id": "UNKNOWN", "name": "Unknown Visitor",
                "photo_url": live_photo_url, "similarity_score": score_display,
                "login_time": "-", "logout_time": time_str, "duration": "00:00:00",
                "status": "CHECKED OUT"
            }
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        ("UNKNOWN", "Unknown Visitor", live_photo_url, score_display, "-", time_str, "00:00:00", "CHECKED OUT")
                    )
                    new_entry["id"] = cur.lastrowid
            finally:
                if conn: conn.close()
            attendance_records.insert(0, new_entry)
            return "unknown_detected", attendance_records, new_entry

        elif camera_type == "LOGIN":
            new_entry = {
                "id": None, "user_id": "UNKNOWN", "name": "Unknown Visitor",
                "photo_url": live_photo_url, "similarity_score": score_display,
                "login_time": time_str, "logout_time": "-", "duration": "-",
                "status": "UNAUTHORIZED"
            }
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                        ("UNKNOWN", "Unknown Visitor", live_photo_url, score_display, time_str, "-", "-", "UNAUTHORIZED")
                    )
                    new_entry["id"] = cur.lastrowid
            finally:
                if conn: conn.close()
            attendance_records.insert(0, new_entry)
            return "unknown_detected", attendance_records, new_entry

    # Handle REGISTERED employee (Score >= 0.90)
    active_record = None
    for r in attendance_records:
        if r["user_id"] == user_id and r["status"] == "IN OFFICE":
            if str(r.get("login_time", "")).startswith(today_str):
                active_record = r
                break

    if camera_type == "LOGIN":
        if active_record:
            # Person already inside for today; suppress redundant broadcast and DB writes
            return "already_logged_in_suppressed", attendance_records, None

        ATTENDANCE_PUNCH_COOLDOWN[cooldown_key] = current_ts
        new_entry = {
            "id": None, "user_id": user_id, "name": name,
            "photo_url": live_photo_url, "similarity_score": score_display,
            "login_time": time_str, "logout_time": "-", "duration": "-",
            "status": "IN OFFICE"
        }
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (user_id, name, live_photo_url, score_display, time_str, "-", "-", "IN OFFICE")
                )
                new_entry["id"] = cur.lastrowid
        finally:
            if conn: conn.close()
        attendance_records.insert(0, new_entry)
        return "login_recorded", attendance_records, new_entry

    elif camera_type == "LOGOUT":
        ATTENDANCE_PUNCH_COOLDOWN[cooldown_key] = current_ts

        if active_record:
            duration_calc = calculate_duration(str(active_record["login_time"]), now_ist)
            conn = None
            try:
                conn = get_db_connection()
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE attendance SET logout_time=%s, duration=%s, status=%s WHERE id=%s",
                        (time_str, duration_calc, "CHECKED OUT", active_record["id"])
                    )
                active_record["logout_time"] = time_str
                active_record["duration"] = duration_calc
                active_record["status"] = "CHECKED OUT"
            finally:
                if conn: conn.close()
            return "logout_recorded", attendance_records, active_record

        # Direct logout without login for today
        new_entry = {
            "id": None, "user_id": user_id, "name": name,
            "photo_url": live_photo_url, "similarity_score": score_display,
            "login_time": "-", "logout_time": time_str, "duration": "00:00:00",
            "status": "CHECKED OUT"
        }
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO attendance (user_id, name, photo_url, similarity_score, login_time, logout_time, duration, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (user_id, name, live_photo_url, score_display, "-", time_str, "00:00:00", "CHECKED OUT")
                )
                new_entry["id"] = cur.lastrowid
        finally:
            if conn: conn.close()
        attendance_records.insert(0, new_entry)
        return "logout_recorded", attendance_records, new_entry

    return "invalid_camera", attendance_records, None

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
        dead = []
        for conn in list(self.active_connections):
            try:
                await conn.send_text(payload)
            except Exception:
                dead.append(conn)
        for d in dead:
            self.disconnect(d)

manager = ConnectionManager()

# Config Route for Frontend
@app.get("/api/config/client")
def get_client_config():
    return {
        "refresh_interval_seconds": DASHBOARD_REFRESH_INTERVAL,
        "recognition_threshold": RECOGNITION_THRESHOLD
    }

# Employee Photos Comparison Route (Registered vs Live)
@app.get("/api/employees/{emp_id}/photos")
def get_employee_photos_comparison(emp_id: str):
    clean_id = emp_id.strip()
    if clean_id not in REGISTERED_USERS:
        return JSONResponse(status_code=404, content={"message": "Employee not found."})

    registered_photo = REGISTERED_USERS[clean_id].get("photo_url", "")
    latest_live_photo = ""

    # Find the latest attendance record with a live photo
    for r in attendance_records:
        if r["user_id"] == clean_id and r.get("photo_url"):
            latest_live_photo = r["photo_url"]
            break

    return {
        "emp_id": clean_id,
        "name": REGISTERED_USERS[clean_id]["name"],
        "registered_photo": registered_photo,
        "latest_live_photo": latest_live_photo or registered_photo
    }

@app.get("/")
async def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {"status": "online", "system": "AI Office Attendance Engine"}

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

@app.post("/api/employees/register")
async def register_employee(
    emp_id: str = Form(...),
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        clean_emp_id = emp_id.strip()
        clean_name = name.strip()

        if not files:
            return JSONResponse(status_code=400, content={"message": "Photo is required."})

        embeddings_list = []
        primary_photo_url = ""

        for file in files:
            image_bytes = await file.read()
            if not image_bytes: continue

            emb, aligned_pil = await asyncio.to_thread(extract_canonical_face_embedding_sync, image_bytes, True)
            if emb is not None:
                embeddings_list.append(emb)

            if primary_photo_url == "" and aligned_pil is not None:
                filename = f"{clean_emp_id}_{int(time.time())}.jpg"
                filepath = EMPLOYEE_PHOTOS_DIR / filename
                aligned_pil.save(filepath, "JPEG", quality=95)
                primary_photo_url = f"/employee_photos/{filename}"

        if not embeddings_list or not primary_photo_url:
            return JSONResponse(status_code=400, content={"message": "Clear front face could not be aligned."})

        REGISTERED_USERS[clean_emp_id] = {
            "name": clean_name,
            "photo_url": primary_photo_url,
            "embeddings": embeddings_list
        }
        rebuild_vector_cache()

        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                payload = json.dumps([e.tolist() for e in embeddings_list])
                cur.execute(
                    "REPLACE INTO employees (emp_id, name, photo_url, embedding_json) VALUES (%s, %s, %s, %s)",
                    (clean_emp_id, clean_name, primary_photo_url, payload)
                )
        finally:
            if conn: conn.close()

        return {"status": "success", "message": f"Registered: {clean_name}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/api/attendance/process")
@app.post("/api/attendance/process/")
async def process_attendance(file: UploadFile = File(...), camera_type: str = Form(...)):
    try:
        image_bytes = await file.read()

        embedding, _ = await asyncio.to_thread(extract_canonical_face_embedding_sync, image_bytes, False)
        if embedding is None:
            return {"status": "no_face_detected", "user_id": None, "similarity_score": 0.0}

        # Apply strict 0.90 threshold
        user_id, name, score = identify_face_strict(embedding)

        # Separate Live Capture photo storage
        live_photo_url = save_live_captured_photo(image_bytes, user_id)

        async with ATTENDANCE_PROCESS_LOCK:
            action, current_logs, target_record = record_attendance_strict(
                user_id=user_id,
                name=name,
                camera_type=camera_type.upper(),
                live_photo_url=live_photo_url,
                score=score
            )

        # Single Broadcast only for actual new record states
        if action in ["login_recorded", "logout_recorded", "unknown_detected"]:
            await manager.broadcast({
                "event": "ATTENDANCE_UPDATE",
                "action": action,
                "record": target_record,
                "logs": current_logs
            })

        return {
            "status": action,
            "user_id": user_id,
            "name": name,
            "similarity_score": round(score, 4),
            "threshold_applied": RECOGNITION_THRESHOLD
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.websocket("/ws/attendance")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        payload = json.dumps({"event": "ATTENDANCE_UPDATE", "action": "initial_load", "logs": attendance_records}, default=str)
        await websocket.send_text(payload)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend_static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)