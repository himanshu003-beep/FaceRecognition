const tableBody = document.getElementById("logsTableBody");
const totalRecordsCard = document.getElementById("totalRecords");
const currentlyInOfficeCard = document.getElementById("currentlyInOffice");
const checkedOutCard = document.getElementById("checkedOut");
const liveDot = document.getElementById("liveDot");

const registerModal = document.getElementById("registerModal");
const empListModal = document.getElementById("empListModal");
const detailModal = document.getElementById("detailModal");
const empListContainer = document.getElementById("empListContainer");

let socket = null;
let reconnectTimer = null;
let allAttendanceData = [];
let activeFilter = "ALL";

function setFilter(filterType) {
    activeFilter = filterType;
    document.getElementById("filterAll").classList.toggle("active", filterType === "ALL");
    document.getElementById("filterEmp").classList.toggle("active", filterType === "EMPLOYEES");
    document.getElementById("filterUnknown").classList.toggle("active", filterType === "UNKNOWN");
    renderDashboard(allAttendanceData);
}

function renderDashboard(logs) {
    if (!tableBody) return;
    tableBody.innerHTML = "";
    allAttendanceData = logs || [];

    let inOfficeCount = 0;
    let checkedOutCount = 0;

    let filtered = allAttendanceData.filter(log => {
        const isUnk = (log.user_id === "UNKNOWN" || log.status === "UNAUTHORIZED");
        if (!isUnk) {
            if (log.status === "IN OFFICE" || !log.logout_time || log.logout_time === "-") inOfficeCount++;
            else checkedOutCount++;
        }
        if (activeFilter === "EMPLOYEES") return !isUnk;
        if (activeFilter === "UNKNOWN") return isUnk;
        return true;
    });

    totalRecordsCard.innerText = allAttendanceData.length;
    currentlyInOfficeCard.innerText = inOfficeCount;
    checkedOutCard.innerText = checkedOutCount;

    if (!filtered || filtered.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: #94a3b8;">No records to display.</td></tr>`;
        return;
    }

    filtered.forEach(log => {
        const isUnknown = (log.user_id === "UNKNOWN" || log.status === "UNAUTHORIZED");
        const isInOffice = (log.status === "IN OFFICE" || !log.logout_time || log.logout_time === "-");

        let badgeHtml = isUnknown 
            ? `<span class="badge badge-unknown">⚠️ UNKNOWN</span>`
            : (isInOffice ? `<span class="badge badge-in">IN OFFICE</span>` : `<span class="badge badge-out">CHECKED OUT</span>`);

        const avatarBorderClass = isUnknown ? 'unknown-border' : '';
        const cardClass = isUnknown ? 'unknown-card' : '';

        // Restored WhatsApp Style Floating Hover Card Structure
        const avatarHtml = `
            <div class="avatar-box">
                <img src="${log.photo_url || ''}" class="avatar-img ${avatarBorderClass}" alt="face" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'40\\' height=\\'40\\' viewBox=\\'0 0 24 24\\' fill=\\'%23666\\'><circle cx=\\'12\\' cy=\\'12\\' r=\\'10\\'/></svg>'">
                <div class="whatsapp-preview ${cardClass}">
                    <img src="${log.photo_url || ''}" alt="preview">
                    <div class="name-tag" style="${isUnknown ? 'color: #f87171;' : 'color: #38bdf8;'}">${log.name}</div>
                    <div class="id-tag" style="${isUnknown ? 'color: #f87171;' : 'color: #94a3b8;'}">${log.user_id}</div>
                </div>
            </div>
        `;

        const row = document.createElement("tr");
        row.onclick = () => showDetailModal(log);

        row.innerHTML = `
            <td>#${log.id}</td>
            <td>${avatarHtml}</td>
            <td><strong style="${isUnknown ? 'color: #f87171;' : 'color: #38bdf8;'}">${log.user_id}</strong></td>
            <td>${log.name}</td>
            <td>${log.login_time || "-"}</td>
            <td>${log.logout_time || "-"}</td>
            <td>${log.duration || "-"}</td>
            <td>${badgeHtml}</td>
            <td><span class="score-pill">${log.similarity_score || "-"}</span></td>
        `;
        tableBody.appendChild(row);
    });
}

function showDetailModal(log) {
    document.getElementById("detailImg").src = log.photo_url || "";
    document.getElementById("dLogId").innerText = `#${log.id}`;
    document.getElementById("dUserId").innerText = log.user_id;
    document.getElementById("dName").innerText = log.name;
    document.getElementById("dStatus").innerText = log.status;
    document.getElementById("dLogin").innerText = log.login_time || "-";
    document.getElementById("dLogout").innerText = log.logout_time || "-";
    document.getElementById("dDuration").innerText = log.duration || "-";
    document.getElementById("dScore").innerText = log.similarity_score || "-";
    detailModal.style.display = "flex";
}

function closeDetailModal() { detailModal.style.display = "none"; }

async function loadInitialData() {
    try {
        const res = await fetch("http://127.0.0.1:8000/attendance/logs");
        if (res.ok) {
            const logs = await res.json();
            renderDashboard(logs);
        }
    } catch (e) {
        console.error("Initial load failed", e);
    }
}

function initWebSocket() {
    socket = new WebSocket("ws://127.0.0.1:8000/ws/attendance");
    socket.onopen = () => {
        if (liveDot) liveDot.classList.add("online");
        if (reconnectTimer) clearTimeout(reconnectTimer);
        loadInitialData();
    };
    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.event === "ATTENDANCE_UPDATE" && data.logs) {
                renderDashboard(data.logs);
            }
        } catch (e) {
            console.error(e);
        }
    };
    socket.onclose = () => {
        if (liveDot) liveDot.classList.remove("online");
        reconnectTimer = setTimeout(initWebSocket, 3000);
    };
}

function openRegisterModal() { registerModal.style.display = "flex"; }
function closeRegisterModal() { registerModal.style.display = "none"; }

async function openEmployeeListModal() {
    empListModal.style.display = "flex";
    try {
        const res = await fetch("http://127.0.0.1:8000/api/employees/list");
        const list = await res.json();
        if (list.length === 0) {
            empListContainer.innerHTML = "<p style='color: #94a3b8;'>No employees registered yet.</p>";
            return;
        }
        empListContainer.innerHTML = list.map(emp => `
            <div class="emp-list-item">
                <div style="display: flex; align-items: center;">
                    <img src="${emp.photo_url}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'40\\' height=\\'40\\' viewBox=\\'0 0 24 24\\' fill=\\'%23666\\'><circle cx=\\'12\\' cy=\\'12\\' r=\\'10\\'/></svg>'">
                    <div>
                        <strong style="color: #f1f5f9; display: block;">${emp.name}</strong>
                        <span style="color: #38bdf8; font-size: 11px; font-family: monospace;">${emp.emp_id} (${emp.vector_count || 1} face samples)</span>
                    </div>
                </div>
            </div>
        `).join("");
    } catch (e) {
        empListContainer.innerHTML = "<p style='color: #ef4444;'>Failed to load employee list.</p>";
    }
}
function closeEmployeeListModal() { empListModal.style.display = "none"; }

// Multiple Photo Registration Handler
async function handleRegister(e) {
    e.preventDefault();
    const empId = document.getElementById("regEmpId").value.trim();
    const name = document.getElementById("regName").value.trim();
    const fileInput = document.getElementById("regPhotos");
    const files = fileInput.files;

    if (!files || files.length === 0) {
        alert("Please select at least 1 photo.");
        return;
    }

    const formData = new FormData();
    formData.append("emp_id", empId);
    formData.append("name", name);
    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }

    try {
        const res = await fetch("http://127.0.0.1:8000/api/employees/register", {
            method: "POST",
            body: formData
        });
        const d = await res.json();
        if (d.status === "success") {
            alert(d.message);
            closeRegisterModal();
            document.getElementById("registerForm").reset();
        } else alert(d.message);
    } catch (err) { alert(err); }
}

window.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
});