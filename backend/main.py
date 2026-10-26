import asyncio
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

if __package__:
    from .attendance_core import (
        ATTENDANCE_PUNCH_COOLDOWN,
        BACKUPS_DIR,
        BASE_DIR,
        CAPTURES_DIR,
        DASHBOARD_REFRESH_INTERVAL,
        EMPLOYEE_PHOTOS_DIR,
        IST,
        LIVE_PUNCHES_DIR,
        RECOGNITION_THRESHOLD,
        REGISTERED_USERS,
        UNKNOWN_DIR,
        UNKNOWN_THRESHOLD,
        attendance_records,
        calculate_duration,
        extract_canonical_face_embedding_sync,
        format_timestamp,
        get_current_ist,
        get_db_connection,
        identify_face_strict,
        l2_normalize,
        rebuild_vector_cache,
        record_attendance_strict,
        save_live_punch_photo
    )
else:
    from attendance_core import (
        ATTENDANCE_PUNCH_COOLDOWN,
        BACKUPS_DIR,
        BASE_DIR,
        CAPTURES_DIR,
        DASHBOARD_REFRESH_INTERVAL,
        EMPLOYEE_PHOTOS_DIR,
        IST,
        LIVE_PUNCHES_DIR,
        RECOGNITION_THRESHOLD,
        REGISTERED_USERS,
        UNKNOWN_DIR,
        UNKNOWN_THRESHOLD,
        attendance_records,
        calculate_duration,
        extract_canonical_face_embedding_sync,
        format_timestamp,
        get_current_ist,
        get_db_connection,
        identify_face_strict,
        l2_normalize,
        rebuild_vector_cache,
        record_attendance_strict,
        save_live_punch_photo
    )

# In the usual project layout, backend/ and frontend/ are sibling directories.
SIBLING_FRONTEND_DIR = BASE_DIR.parent / "frontend"

FRONTEND_DIR = (
    SIBLING_FRONTEND_DIR
    if (SIBLING_FRONTEND_DIR / "index.html").exists()
    else (BASE_DIR / "frontend" if (BASE_DIR / "frontend" / "index.html").exists() else BASE_DIR)
)

INDEX_HTML = FRONTEND_DIR / "index.html"

ATTENDANCE_PROCESS_LOCK = asyncio.Lock()

app = FastAPI(title="Enterprise Office Attendance AI Engine - Hardened Dual Gate")

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

@app.get("/api/config/client")

def get_client_config():
    return {
        "refresh_interval_seconds": DASHBOARD_REFRESH_INTERVAL,
        "recognition_threshold": RECOGNITION_THRESHOLD,
        "unknown_threshold": UNKNOWN_THRESHOLD
    }

@app.get("/api/employees/{emp_id}/photos")

def get_employee_photos_comparison(emp_id: str):
    clean_id = emp_id.strip()

    if clean_id not in REGISTERED_USERS:
        raise HTTPException(status_code=404, detail="Employee not found")

    user_info = REGISTERED_USERS[clean_id]
    registered_photo = user_info.get("photo_url", "")
    latest_live_photo = user_info.get("latest_live_photo", "")
    latest_live_timestamp = user_info.get("latest_live_timestamp", "")
    if not latest_live_photo:
        for r in attendance_records:
            if r["user_id"] == clean_id and r.get("live_photo_url"):
                latest_live_photo = r["live_photo_url"]
                latest_live_timestamp = r.get("login_time") or r.get("logout_time") or ""
                break

    return {
        "emp_id": clean_id,
        "name": user_info["name"],
        "registered_photo": registered_photo,
        "latest_live_photo": latest_live_photo,
        "latest_live_timestamp": latest_live_timestamp
    }

@app.get("/")

async def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)

    return {"status": "online", "system": "AI Office Attendance Hardened Gateway"}

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
            "latest_live_photo": data.get("latest_live_photo", ""),
            "latest_live_timestamp": data.get("latest_live_timestamp", ""),
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
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", clean_emp_id):
            return JSONResponse(status_code=400, content={"message": "Employee ID must contain only letters, numbers, underscores, or hyphens (max 50 characters)."})

        if not clean_name or len(clean_name) > 100:
            return JSONResponse(status_code=400, content={"message": "Employee name is required and must be at most 100 characters."})

        if not files or len(files) == 0:
            return JSONResponse(status_code=400, content={"message": "Photo is required."})

        conn_check = None

        try:
            conn_check = get_db_connection()

            with conn_check.cursor() as cur:
                cur.execute("SELECT emp_id FROM employees WHERE emp_id = %s LIMIT 1", (clean_emp_id,))

                if cur.fetchone():
                    return JSONResponse(status_code=409, content={"message": "Employee ID already exists. Use a new ID; existing employee data was not changed."})
        finally:
            if conn_check:
                conn_check.close()

        embeddings_list = []
        primary_photo_url = ""

        for file in files:
            image_bytes = await file.read()
            if not image_bytes: continue

            emb, aligned_pil = await asyncio.to_thread(extract_canonical_face_embedding_sync, image_bytes, True)

            if emb is not None:
                embeddings_list.append(l2_normalize(emb))

            if primary_photo_url == "" and aligned_pil is not None:
                filename = f"{clean_emp_id}_registered_{int(time.time())}.jpg"
                filepath = EMPLOYEE_PHOTOS_DIR / filename
                aligned_pil.save(filepath, "JPEG", quality=95)
                primary_photo_url = f"/employee_photos/{filename}"

        if not embeddings_list or not primary_photo_url:
            return JSONResponse(status_code=400, content={"message": "Clear front face could not be aligned."})

        conn = None

        try:
            conn = get_db_connection()

            with conn.cursor() as cur:
                payload = json.dumps([e.tolist() for e in embeddings_list])
                cur.execute(
                    """INSERT INTO employees

                       (emp_id, name, photo_url, latest_live_photo, latest_live_timestamp, embedding_json)

                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (clean_emp_id, clean_name, primary_photo_url, "", "", payload)
                )
        finally:
            if conn:
                conn.close()

        # Publish to the in-memory recognition cache only after the DB insert succeeds.
        REGISTERED_USERS[clean_emp_id] = {
            "name": clean_name,
            "photo_url": primary_photo_url,
            "latest_live_photo": "",
            "latest_live_timestamp": "",
            "embeddings": embeddings_list
        }

        rebuild_vector_cache()

        print(f"[REGISTRATION SUCCESS] Enrolled {clean_name} ({clean_emp_id}) with {len(embeddings_list)} face descriptors.")
        return {"status": "success", "message": f"Successfully Registered: {clean_name}"}
    except Exception as e:
        print(f"[REGISTRATION CRASH] -> {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.delete("/api/employees/{emp_id}")

def delete_employee(emp_id: str):
    clean_id = emp_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", clean_id):
        raise HTTPException(status_code=400, detail="Invalid employee ID.")

    conn = None

    try:
        conn = get_db_connection()

        with conn.cursor() as cur:
            cur.execute("DELETE FROM employees WHERE emp_id = %s", (clean_id,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Employee not found.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not delete employee: {exc}")
    finally:
        if conn:
            conn.close()

    REGISTERED_USERS.pop(clean_id, None)

    rebuild_vector_cache()

    # Attendance history is intentionally retained for audit purposes.
    return {"status": "success", "message": f"Employee {clean_id} deleted. Attendance history was retained."}

@app.post("/api/attendance/clear-dashboard")

def clear_attendance_dashboard():
    """Back up all attendance rows to JSON, then clear attendance history."""

    conn = None
    backup_path = BACKUPS_DIR / f"attendance_{int(time.time())}.json"

    try:
        conn = get_db_connection()

        with conn.cursor() as cur:
            cur.execute("""

                SELECT id, user_id, name, camera_id, photo_url, live_photo_url,

                       registered_photo_url, similarity_score, login_time, logout_time,

                       duration, status

                FROM attendance ORDER BY id DESC

            """)

            rows = cur.fetchall() or []

            # Save backup before deleting anything.
            with backup_path.open("w", encoding="utf-8") as fh:
                json.dump(rows, fh, ensure_ascii=False, default=str, indent=2)

            cur.execute("DELETE FROM attendance")

        attendance_records.clear()
        return {"status": "success", "message": f"Dashboard cleared. Backup saved as {backup_path.name}.", "backup": backup_path.name, "count": len(rows)}
    except Exception as exc:
        if backup_path.exists():
            try:
                backup_path.unlink()
            except OSError:
                pass

        raise HTTPException(status_code=500, detail=f"Could not clear attendance: {exc}")
    finally:
        if conn:
            conn.close()

@app.post("/api/attendance/restore-latest-backup")

def restore_latest_attendance_backup():
    backups = sorted(BACKUPS_DIR.glob("attendance_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        raise HTTPException(status_code=404, detail="No attendance backup is available.")

    try:
        rows = json.loads(backups[0].read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("Backup format is invalid.")

        conn = get_db_connection()

        try:
            conn.begin()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM attendance")

                for row in reversed(rows):
                    cur.execute("""

                        INSERT INTO attendance

                        (id, user_id, name, camera_id, photo_url, live_photo_url,

                         registered_photo_url, similarity_score, login_time, logout_time,

                         duration, status)

                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)

                    """, (
                        row.get("id"), row.get("user_id", "UNKNOWN"), row.get("name", "Unknown"),
                        row.get("camera_id", "SYSTEM_CAM"), row.get("photo_url", ""),
                        row.get("live_photo_url", ""), row.get("registered_photo_url", ""),
                        row.get("similarity_score", "-"), row.get("login_time", ""),
                        row.get("logout_time", "-"), row.get("duration", "-"),
                        row.get("status", "UNKNOWN")
                    ))

                cur.execute("""

                    SELECT id, user_id, name, camera_id, photo_url, live_photo_url,

                           registered_photo_url, similarity_score, login_time, logout_time,

                           duration, status

                    FROM attendance ORDER BY id DESC

                """)

                restored = cur.fetchall() or []
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()

        attendance_records[:] = restored
        return {"status": "success", "message": f"Restored {len(restored)} attendance records from {backups[0].name}.", "count": len(restored)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not restore backup: {exc}")

@app.post("/api/attendance/process")

@app.post("/api/attendance/process/")

async def process_attendance(
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    camera_type: str = Form(...)
):
    try:
        image_bytes = await file.read()
        embedding, _ = await asyncio.to_thread(extract_canonical_face_embedding_sync, image_bytes, False)

        if embedding is None:
            return {"status": "no_face_detected", "user_id": None, "similarity_score": 0.0}

        user_id, name, score, classification = identify_face_strict(embedding)
        if classification == "UNCERTAIN":
            return {
                "status": "uncertain_evaluating",
                "user_id": None,
                "similarity_score": round(score, 4),
                "thresholds": {"registered": RECOGNITION_THRESHOLD, "unknown": UNKNOWN_THRESHOLD}
            }

        live_photo_url = save_live_punch_photo(image_bytes, user_id)

        async with ATTENDANCE_PROCESS_LOCK:
            action, current_logs, target_record = record_attendance_strict(
                user_id=user_id,
                name=name,
                camera_id=camera_id.upper(),
                camera_type=camera_type.upper(),
                live_photo_url=live_photo_url,
                score=score,
                classification=classification
            )

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
            "camera_id": camera_id.upper(),
            "camera_type": camera_type.upper()
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