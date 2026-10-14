const BACKEND_BASE = "http://127.0.0.1:8000";
const WS_URL = "ws://127.0.0.1:8000/ws/attendance";

let globalRecords = [];
let socket = null;

// Fallback dummy avatar generator (Prevents broken 404 images)
const DUMMY_AVATAR = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='42' height='42' viewBox='0 0 24 24' fill='%2364748b'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";

// Initialize WebSocket with Reconnect Mechanism
function initWebSocket() {
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
        console.log("[WS] Connected successfully to Attendance Engine.");
        const dot = document.getElementById("statusDot");
        if (dot) dot.classList.add("connected");
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.event === "ATTENDANCE_UPDATE") {
                globalRecords = data.logs || [];
                renderTable(globalRecords);
                updateStats(globalRecords);
            }
        } catch (e) {
            console.error("[WS] Error parsing payload:", e);
        }
    };

    socket.onclose = () => {
        console.warn("[WS] Connection dropped. Retrying in 2 seconds...");
        const dot = document.getElementById("statusDot");
        if (dot) dot.classList.remove("connected");
        setTimeout(initWebSocket, 2000);
    };

    socket.onerror = (err) => {
        console.error("[WS] Socket error:", err);
        socket.close();
    };
}

// Render Attendance Table Body
function renderTable(records) {
    const tbody = document.getElementById("attendanceTableBody");
    if (!tbody) return;

    if (!records || records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 40px;">No attendance records available.</td></tr>`;
        return;
    }

    tbody.innerHTML = records.map(r => {
        const photo = r.photo_url && r.photo_url.trim() !== "" ? r.photo_url : DUMMY_AVATAR;
        const status = r.status || "IN OFFICE";

        let badgeClass = "badge-success";
        if (status === "CHECKED OUT") badgeClass = "badge-checkedout";
        else if (status === "UNAUTHORIZED" || r.user_id === "UNKNOWN") badgeClass = "badge-danger";

        const score = r.similarity_score ? r.similarity_score : "-";

        return `
            <tr>
                <td>#${r.id}</td>
                <td>
                    <img src="${photo}" class="avatar-img" onerror="this.src='${DUMMY_AVATAR}'" alt="Face">
                </td>
                <td><strong>${escapeHtml(r.user_id)}</strong></td>
                <td>${escapeHtml(r.name)}</td>
                <td>${r.login_time || "-"}</td>
                <td>${r.logout_time || "-"}</td>
                <td>${r.duration || "-"}</td>
                <td><span class="badge ${badgeClass}">${escapeHtml(status)}</span></td>
                <td><span class="score-pill">${escapeHtml(score)}</span></td>
            </tr>
        `;
    }).join("");
}

// Update Top Dashboard Metric Cards
function updateStats(records) {
    const totalEl = document.getElementById("statTotal");
    const inOfficeEl = document.getElementById("statInOffice");
    const checkedOutEl = document.getElementById("statCheckedOut");
    const unauthEl = document.getElementById("statUnauthorized");

    if (totalEl) totalEl.innerText = records.length;

    let inOffice = 0;
    let checkedOut = 0;
    let unauthorized = 0;

    records.forEach(r => {
        if (r.status === "IN OFFICE") inOffice++;
        else if (r.status === "CHECKED OUT") checkedOut++;
        else if (r.status === "UNAUTHORIZED" || r.user_id === "UNKNOWN") unauthorized++;
    });

    if (inOfficeEl) inOfficeEl.innerText = inOffice;
    if (checkedOutEl) checkedOutEl.innerText = checkedOut;
    if (unauthEl) unauthEl.innerText = unauthorized;
}

// Filter Records Dropdown
function filterRecords() {
    const filter = document.getElementById("recordFilter").value;
    if (filter === "ALL") {
        renderTable(globalRecords);
    } else if (filter === "IN_OFFICE") {
        renderTable(globalRecords.filter(r => r.status === "IN OFFICE"));
    } else if (filter === "CHECKED_OUT") {
        renderTable(globalRecords.filter(r => r.status === "CHECKED OUT"));
    } else if (filter === "UNAUTHORIZED") {
        renderTable(globalRecords.filter(r => r.status === "UNAUTHORIZED" || r.user_id === "UNKNOWN"));
    }
}

// Modal Helpers
function openRegisterModal() {
    document.getElementById("registerModal").style.display = "flex";
}

function openEmployeeListModal() {
    document.getElementById("employeeModal").style.display = "flex";
    loadEmployeeList();
}

function closeModal(id) {
    document.getElementById(id).style.display = "none";
}

window.onclick = function(event) {
    if (event.target.classList.contains("modal")) {
        event.target.style.display = "none";
    }
};

// 1-Click Clear Dashboard with Backup
async function clearDashboard() {
    if (!confirm("Are you sure you want to clear the dashboard? A complete backup snapshot will be saved automatically in backend/backups.")) return;

    try {
        const res = await fetch(`${BACKEND_BASE}/api/attendance/clear-dashboard`, { method: "POST" });
        const data = await res.json();
        if (res.ok) {
            alert(data.message);
        } else {
            alert("Error: " + data.message);
        }
    } catch (err) {
        alert("Failed to clear dashboard: " + err);
    }
}

// 1-Click Restore Latest Backup
async function restoreBackup() {
    if (!confirm("Restore the most recent attendance snapshot backup?")) return;

    try {
        const res = await fetch(`${BACKEND_BASE}/api/attendance/restore-latest-backup`, { method: "POST" });
        const data = await res.json();
        if (res.ok) {
            alert(data.message);
        } else {
            alert("Error: " + data.message);
        }
    } catch (err) {
        alert("Failed to restore backup: " + err);
    }
}

// Employee Enrollment Form Submit
async function handleRegisterSubmit(event) {
    event.preventDefault();

    const empId = document.getElementById("regEmpId").value.trim();
    const name = document.getElementById("regName").value.trim();
    const fileInput = document.getElementById("regPhoto");

    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please select a valid face photo.");
        return;
    }

    const formData = new FormData();
    formData.append("emp_id", empId);
    formData.append("name", name);
    formData.append("files", fileInput.files[0]);

    try {
        const res = await fetch(`${BACKEND_BASE}/api/employees/register`, {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        if (res.ok) {
            alert("Success: " + data.message);
            closeModal("registerModal");
            document.getElementById("registerForm").reset();
        } else {
            alert("Failed: " + data.message);
        }
    } catch (err) {
        alert("Registration network error: " + err);
    }
}

// Load and Render Employees with Single-Click Delete
async function loadEmployeeList() {
    const container = document.getElementById("employeeListContainer");
    container.innerHTML = `<p style="color: var(--text-muted);">Loading employees...</p>`;

    try {
        const res = await fetch(`${BACKEND_BASE}/api/employees/list`);
        const employees = await res.json();

        if (!employees || employees.length === 0) {
            container.innerHTML = `<p style="color: var(--text-muted); padding: 10px 0;">No registered employees found.</p>`;
            return;
        }

        container.innerHTML = employees.map(emp => `
            <div class="emp-item">
                <div class="emp-info">
                    <img src="${emp.photo_url || DUMMY_AVATAR}" class="avatar-img" onerror="this.src='${DUMMY_AVATAR}'">
                    <div>
                        <div style="font-weight: 600;">${escapeHtml(emp.name)}</div>
                        <div style="font-size: 12px; color: var(--text-muted);">${escapeHtml(emp.emp_id)}</div>
                    </div>
                </div>
                <button onclick="deleteEmployee('${escapeHtml(emp.emp_id)}')" style="background-color: rgba(239, 68, 68, 0.2); color: #f87171; padding: 6px 12px; font-size: 12px;">
                    Delete
                </button>
            </div>
        `).join("");

    } catch (err) {
        container.innerHTML = `<p style="color: #f87171;">Failed to load employees: ${err}</p>`;
    }
}

// Delete Employee Instantly
async function deleteEmployee(empId) {
    if (!confirm(`Are you sure you want to permanently delete employee ${empId}?`)) return;

    try {
        const res = await fetch(`${BACKEND_BASE}/api/employees/${empId}`, {
            method: "DELETE"
        });
        const data = await res.json();
        if (res.ok) {
            alert(data.message);
            loadEmployeeList();
        } else {
            alert("Error: " + data.message);
        }
    } catch (err) {
        alert("Failed to delete employee: " + err);
    }
}

// Helper to escape HTML tags
function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// Start on DOM ready
document.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
});