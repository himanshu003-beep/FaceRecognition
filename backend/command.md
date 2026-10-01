# AI Attendance Project — Commands & Operations Cheatsheet

---

## 1. Daily Workflow (Rozana use hone wali commands)

### Subah: Sab kuch Start karna
```bash
# 1. Project folder me jayein
cd ~/Music/AI_Face_recoginition/backend

# 2. Docker containers (MySQL + Typesense) background me chalu karein
docker compose up -d

# 3. Python virtual environment activate karein
source venv/bin/activate

# 4. FastAPI Server & AI Camera loop live run karein
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


Docker me check karne ki commands
Docker container me live data verify karne ke liye terminal me yeh simple command run karein:

docker exec -it attendance_mysql mysql -u ai_attendance -p2003 -e "USE Ai_attendance; SELECT id, user_id, name, login_time, logout_time, duration FROM attendance_logs ORDER BY id DESC LIMIT 10;"

Sab kuch Stop karna
# 1. Terminal me 'Ctrl + C' dabayein (FastAPI aur camera band hoga)

# 2. Docker containers safely stop karein
docker compose down



Face Enrollment & Verification
# Naye person ka face scan karke Typesense me register karna
python register_face.py

# Check karna ki specific user Typesense me save hua ya nahi
curl -H "x-typesense-api-key: xyz123" http://localhost:8108/collections/face_embeddings/documents/EMP01

# Browser me live attendance dekhna:
# http://localhost:8000/attendance/logs
# Interactive API documentation:
# http://localhost:8000/docs

Ubuntu System & Port Fixes (One-Time Setup)
# Local MySQL ko stop karna (taaki Docker port 3306 use kar sake)
sudo systemctl stop mysql

# Local MySQL ko system restart par auto-start hone se rokna
sudo systemctl disable mysql

# Port 8108 par atki kisi bhi lingering process ko kill karna
sudo fuser -k 8108/tcp

# Local Typesense service agar active ho toh band karna
sudo systemctl stop typesense-server

Docker Permissions & Monitoring

# User ko Docker group me add karna (sudo ke bina docker chalane ke liye)
sudo usermod -aG docker $USER

# Terminal session me group permissions ko turant refresh karna
newgrp docker

# Check karna kaun se containers running ('Up') hain
docker ps -a

# Typesense container ke internal logs inspect karna


docker logs attendance_typesense
# User ko Docker group me add karna (sudo ke bina docker chalane ke liye)
sudo usermod -aG docker $USER

# Terminal session me group permissions ko turant refresh karna
newgrp docker

# Check karna kaun se containers running ('Up') hain
docker ps -a

# Typesense container ke internal logs inspect karna
docker logs attendance_typesense

MySQL Database Inside Docker
# Running MySQL container ke interactive shell me login karna
docker exec -it attendance_mysql mysql -u ai_attendance -p Ai_attendance
# (Password: 2003)

# Container ke bahar se direct attendance records table print karna
docker exec -it attendance_mysql mysql -u ai_attendance -p2003 -e "SELECT * FROM Ai_attendance.attendance_logs;"
