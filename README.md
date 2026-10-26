# AIFace — AI-Based Face Recognition Attendance System

An enterprise-ready, dual-gate AI Face Recognition Attendance System built with **FastAPI**, **PyTorch (MTCNN + FaceNet)**, **OpenCV**, and a real-time **WebSocket** dashboard.

---

## 🌟 Key Features

- **Dual-Gate Architecture:**
  - **Gate 1 (LOGIN):** Web camera / USB camera (`SYSTEM_CAM`) for entrance tracking.
  - **Gate 2 (LOGOUT):** RTSP IP camera (`OFFICE_RTSP`) for exit tracking.
- **Smart Shift Handling (18-Hour Window):**
  - Seamlessly handles cross-midnight overnight shifts (e.g., login at 11:15 PM, logout at 1:45 AM).
  - Automatically calculates exact shift duration (`HH:MM:SS`).
- **Orphan Checkout Prevention:**
  - Prevents accidental or phantom checkout records if an employee does not have an active check-in session.
- **Edge Vision Quality Gating:**
  - **Laplacian Sharpness Filter:** Drops blurred frames before sending to the backend.
  - **Movement & Debounce Tracker:** Prevents duplicate punches and stationary person flood.
- **Real-Time Live Dashboard:**
  - Responsive web dashboard with live WebSocket updates for instant attendance logging and notifications.
- **Dockerized Storage:**
  - High-performance MySQL database pool (`PooledDB`) and Typesense vector indexing.

---

## 📁 Project Architecture

```
AIFace_Project/
├── backend/
│   ├── main.py              # FastAPI Web Server, WebSocket manager, API routes
│   ├── attendance_core.py   # AI inference (MTCNN/FaceNet), vector cache & DB logic
│   ├── .env                 # Database credentials, thresholds, and RTSP configs
│   └── captures/            # Live punches and unknown visitor captures
├── frontend/
│   ├── index.html           # Real-time attendance monitoring dashboard
│   ├── app.js               # WebSocket client & dashboard UI logic
│   └── styles.css           # Modern dark-mode UI styling
├── stream.py                # Edge camera stream gateway (MTCNN + OpenCV loop)
├── docker-compose.yml       # MySQL 8.0 and Typesense service definitions
└── requirements.txt         # Python project dependencies
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.12+
- Docker & Docker Compose
- CUDA-compatible GPU (Optional, CPU fallback supported)

### 2. Start Database Services
```bash
docker compose up -d
```
*(Starts MySQL on port `3306` and Typesense on port `8109`)*

### 3. Start Backend Server
```bash
cd backend
source .venv_repair/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- **Dashboard:** [http://localhost:8000](http://localhost:8000)
- **API Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Start AI Camera Gateway
In a separate terminal:
```bash
source backend/.venv_repair/bin/activate
python stream.py
```
- Press **`q`** on any camera window to safely shut down video streams.

---

## 🛠 Tech Stack

- **Deep Learning:** PyTorch, InceptionResnetV1 (`vggface2`), MTCNN (`facenet-pytorch`)
- **Computer Vision:** OpenCV (`cv2`), PIL
- **Backend Framework:** FastAPI, Starlette, Uvicorn, WebSockets
- **Database:** MySQL 8.0, DBUtils (`PooledDB`), PyMySQL
- **Frontend:** Vanilla HTML5, CSS3, Modern JavaScript (ES6+)

---

## 📄 License
This project is licensed under the MIT License.
