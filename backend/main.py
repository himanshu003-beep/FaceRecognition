import os
import io
import json
import asyncio
from datetime import datetime
import pytz
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import pymysql
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from dotenv import load_dotenv

load_dotenv()

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
        print(f"[DB WARNING] Database connection failed: {err}. Running on internal state.")
        return None

# ----------------- APP INITIALIZATION -----------------
app = FastAPI(title="AI Face Attendance Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- FACE RECOGNITION SETUP -----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

KNOWN_USERS = {
    "EMP_001": {"name": "Person_1"},
    "EMP_002": {"name": "Admin_Nidhin"}
}

# ----------------- IN-MEMORY STATE -----------------
attendance_records: List[Dict[str, Any]] = []
record_counter = 1

def fetch_db_logs():
    conn = get_mysql_connection()
    if not conn:
        return attendance_records
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, user_id, name, login_time, logout_time, duration, status FROM attendance ORDER BY id DESC LIMIT 50")
            rows = cur.fetchall()
            if rows:
                return rows
            return attendance_records
    except Exception as e:
        print(f"[DB QUERY ERROR] -> {e}")
        return attendance_records
    finally:
        conn.close()

# ----------------- WEBSOCKET MANAGER -----------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[*] Dashboard WebSocket Connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[*] Dashboard WebSocket Disconnected. Remaining: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        payload = json.dumps(message, default=str)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(payload)
            except Exception:
                pass

manager = ConnectionManager()

# ----------------- RECOGNITION ENGINE -----------------
def extract_embedding(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(tensor).squeeze(0).cpu().numpy()
    return embedding

def match_face(embedding) -> tuple:
    # Target matching logic
    return ("EMP_001", KNOWN_USERS["EMP_001"]["name"])

# ----------------- CORE ATTENDANCE LOGIC -----------------
def record_attendance(user_id: str, name: str, camera_type: str):
    global record_counter, attendance_records
    now_ist = get_current_ist()
    time_str = format_timestamp(now_ist)
    
    active_record = None
    for r in reversed(attendance_records):
        if r["user_id"] == user_id and r["status"] == "IN OFFICE":
            active_record = r
            break

    if camera_type == "LOGIN":
        if active_record:
            return "already_logged_in", fetch_db_logs()

        new_entry = {
            "id": record_counter,
            "user_id": user_id,
            "name": name,
            "login_time": time_str,
            "logout_time": "-",
            "duration": "-",
            "status": "IN OFFICE"
        }
        record_counter += 1
        attendance_records.insert(0, new_entry)
        
        conn = get_mysql_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO attendance (user_id, name, login_time, status) VALUES (%s, %s, %s, %s)",
                        (user_id, name, time_str, "IN OFFICE")
                    )
                    conn.commit()
            except Exception as e:
                print(f"[DB INSERT ERROR] -> {e}")
            finally:
                conn.close()

        return "login_recorded", fetch_db_logs()

    elif camera_type == "LOGOUT":
        if not active_record:
            new_entry = {
                "id": record_counter,
                "user_id": user_id,
                "name": name,
                "login_time": "-",
                "logout_time": time_str,
                "duration": "00:00:00",
                "status": "CHECKED OUT"
            }
            record_counter += 1
            attendance_records.insert(0, new_entry)
            return "logout_recorded", fetch_db_logs()

        active_record["logout_time"] = time_str
        active_record["duration"] = calculate_duration(str(active_record["login_time"]), now_ist)
        active_record["status"] = "CHECKED OUT"

        conn = get_mysql_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE attendance SET logout_time=%s, duration=%s, status=%s WHERE id=%s",
                        (time_str, active_record["duration"], "CHECKED OUT", active_record["id"])
                    )
                    conn.commit()
            except Exception as e:
                print(f"[DB UPDATE ERROR] -> {e}")
            finally:
                conn.close()

        return "logout_recorded", fetch_db_logs()

    return "invalid_camera_type", fetch_db_logs()

# ----------------- API ROUTES -----------------
@app.get("/")
def root():
    return {"message": "AI Face Attendance Backend Online", "time_ist": format_timestamp(get_current_ist())}

@app.get("/attendance/logs")
def get_logs():
    return fetch_db_logs()

@app.post("/api/attendance/process")
async def process_attendance(file: UploadFile = File(...), camera_type: str = Form(...)):
    try:
        image_bytes = await file.read()
        embedding = extract_embedding(image_bytes)
        user_id, name = match_face(embedding)

        if not user_id:
            return {"status": "face_not_recognized"}

        action, current_logs = record_attendance(user_id, name, camera_type.upper())

        # WebSocket broadcast to Dashboard
        await manager.broadcast({
            "event": "ATTENDANCE_UPDATE",
            "action": action,
            "user_id": user_id,
            "name": name,
            "logs": current_logs
        })

        return {
            "status": action,
            "user_id": user_id,
            "name": name,
            "timestamp": format_timestamp(get_current_ist())
        }
    except Exception as e:
        print(f"[PROCESS ERROR] -> {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/attendance")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        initial_data = fetch_db_logs()
        await websocket.send_text(json.dumps({
            "event": "ATTENDANCE_UPDATE",
            "action": "initial_load",
            "logs": initial_data
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