import io
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import pymysql
import typesense
import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np

# ----------------- TIMEZONE & DB CONFIG -----------------
IST = ZoneInfo("Asia/Kolkata")

def get_mysql_connection():
    return pymysql.connect(
        host="127.0.0.1",
        user="ai_attendance",
        password="password123",
        database="attendance_db",
        cursorclass=pymysql.cursors.DictCursor
    )

typesense_client = typesense.Client({
    'nodes': [{'host': '127.0.0.1', 'port': 8108, 'protocol': 'http'}],
    'api_key': 'xyz123',
    'connection_timeout_seconds': 2
})

# Model setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[BACKEND ONLINE] FastAPI ready on port 8000. Timezone synchronized to local IST.")
    yield

app = FastAPI(title="Face Attendance Backend", lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- WEBSOCKET CONNECTION MANAGER -----------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[*] Dashboard WebSocket Connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[*] Dashboard WebSocket Disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/attendance")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Client se ping sunne ke liye open rakhte hain
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# ----------------- HELPER: FETCH ALL LOGS -----------------
def fetch_current_logs():
    conn = get_mysql_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM attendance_logs ORDER BY id DESC")
            records = cursor.fetchall()
            for r in records:
                if isinstance(r.get("login_time"), datetime):
                    r["login_time"] = r["login_time"].isoformat()
                if isinstance(r.get("logout_time"), datetime):
                    r["logout_time"] = r["logout_time"].isoformat()
                if r.get("duration") is not None:
                    r["duration"] = str(r["duration"])
            return records
    finally:
        conn.close()

# ----------------- REST ROUTES -----------------
@app.get("/attendance/logs")
def get_logs():
    return fetch_current_logs()

@app.post("/api/attendance/process")
async def process_attendance(
    file: Annotated[UploadFile, File()], 
    camera_type: Annotated[str, Form()]
):
    # 1. Read image from upload stream
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # 2. Resize Image & Generate Embeddings
    img_resized = image.resize((160, 160))
    tensor = torch.from_numpy(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - 0.5) / 0.5
    embedding = resnet(tensor.unsqueeze(0).to(device)).detach().cpu().numpy()[0].tolist()

    # 3. Typesense Vector Search (Passing kwargs via **search_params)
    search_params = {
        'q': '*',
        'query_by': 'vec',
        'vector_query': f'vec:({embedding}, k:1, distance_threshold: 0.95)'
    }
    search_res = typesense_client.collections['faces'].documents.search(**search_params) # type: ignore
    
    conn = get_mysql_connection()
    now_ist = datetime.now(IST).replace(tzinfo=None)
    action_status = ""
    user_id = ""

    try:
        with conn.cursor() as cursor:
            if search_res.get('hits'):
                # Existing Registered User
                user_id = search_res['hits'][0]['document']['id']
                name = search_res['hits'][0]['document']['name']
            else:
                # Auto-Enroll New Person
                cursor.execute("SELECT COUNT(DISTINCT user_id) as count FROM attendance_logs")
                count_res = cursor.fetchone()
                count = (count_res['count'] if count_res else 0) + 1
                user_id = f"EMP_{count:03d}"
                name = f"Person_{count}"
                typesense_client.collections['faces'].documents.create({
                    'id': user_id,
                    'name': name,
                    'vec': embedding
                })

            if camera_type.upper() == "LOGIN":
                # Check agar already inside hai
                cursor.execute("SELECT id FROM attendance_logs WHERE user_id=%s AND logout_time IS NULL", (user_id,))
                active = cursor.fetchone()
                if active:
                    action_status = "already_logged_in"
                else:
                    cursor.execute(
                        "INSERT INTO attendance_logs (user_id, name, login_time) VALUES (%s, %s, %s)",
                        (user_id, name, now_ist)
                    )
                    conn.commit()
                    action_status = "login_recorded"

            elif camera_type.upper() == "LOGOUT":
                cursor.execute(
                    "SELECT id, login_time FROM attendance_logs WHERE user_id=%s AND logout_time IS NULL ORDER BY id DESC LIMIT 1",
                    (user_id,)
                )
                active = cursor.fetchone()
                if not active:
                    action_status = "not_logged_in"
                else:
                    duration_sec = int((now_ist - active['login_time']).total_seconds())
                    duration_str = f"{duration_sec // 3600:02d}:{(duration_sec % 3600) // 60:02d}:{duration_sec % 60:02d}"
                    cursor.execute(
                        "UPDATE attendance_logs SET logout_time=%s, duration=%s WHERE id=%s",
                        (now_ist, duration_str, active['id'])
                    )
                    conn.commit()
                    action_status = "logout_recorded"
    finally:
        conn.close()

    # ----------------- REAL-TIME BROADCAST VIA WEBSOCKET -----------------
    if action_status in ["login_recorded", "logout_recorded"]:
        updated_records = fetch_current_logs()
        await manager.broadcast({
            "event": "ATTENDANCE_UPDATE",
            "action": action_status,
            "user_id": user_id,
            "logs": updated_records
        })

    return {"status": action_status, "user_id": user_id}