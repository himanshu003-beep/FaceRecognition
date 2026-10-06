const tableBody = document.getElementById("logsTableBody");
const totalRecordsCard = document.getElementById("totalRecords");
const currentlyInOfficeCard = document.getElementById("currentlyInOffice");
const checkedOutCard = document.getElementById("checkedOut");
const liveDot = document.getElementById("liveDot");

let socket = null;
let reconnectTimer = null;
let currentAttendanceData = [];

function renderDashboard(logs) {
    if (!tableBody) return;
    tableBody.innerHTML = "";
    currentAttendanceData = logs || [];

    let inOfficeCount = 0;
    let checkedOutCount = 0;

    if (!logs || logs.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: #94a3b8;">No attendance records found yet.</td></tr>`;
        totalRecordsCard.innerText = "0";
        currentlyInOfficeCard.innerText = "0";
        checkedOutCard.innerText = "0";
        return;
    }

    logs.forEach(log => {
        const isInOffice = (log.status === "IN OFFICE" || !log.logout_time || log.logout_time === "-");
        if (isInOffice) {
            inOfficeCount++;
        } else {
            checkedOutCount++;
        }

        const avatarHtml = log.photo_url 
            ? `<div class="avatar-box">
                 <img src="${log.photo_url}" class="avatar-img" alt="face" onerror="this.parentElement.innerHTML='<div class=\\'avatar-fallback\\'>N/A</div>'">
                 <div class="whatsapp-preview">
                     <img src="${log.photo_url}" alt="preview">
                     <div class="name-tag">${log.name}</div>
                     <div class="id-tag">${log.user_id}</div>
                 </div>
               </div>`
            : `<div class="avatar-fallback">N/A</div>`;

        const row = document.createElement("tr");
        row.innerHTML = `
            <td>#${log.id}</td>
            <td>${avatarHtml}</td>
            <td><strong>${log.user_id}</strong></td>
            <td>${log.name}</td>
            <td>${log.login_time || "-"}</td>
            <td>${log.logout_time || "-"}</td>
            <td>${log.duration || "-"}</td>
            <td>
                <span class="badge ${isInOffice ? 'badge-in' : 'badge-out'}">
                    ${isInOffice ? 'IN OFFICE' : 'CHECKED OUT'}
                </span>
            </td>
            <td>
                <span class="score-pill">${log.similarity_score || "-"}</span>
            </td>
        `;
        tableBody.appendChild(row);
    });

    totalRecordsCard.innerText = logs.length;
    currentlyInOfficeCard.innerText = inOfficeCount;
    checkedOutCard.innerText = checkedOutCount;
}

async function loadInitialData() {
    try {
        const response = await fetch("http://127.0.0.1:8000/attendance/logs");
        if (response.ok) {
            const logs = await response.json();
            renderDashboard(logs);
        }
    } catch (err) {
        console.error("[REST API] Failed to fetch logs:", err);
    }
}

function initWebSocket() {
    socket = new WebSocket("ws://127.0.0.1:8000/ws/attendance");

    socket.onopen = () => {
        // Online: Green dot
        if (liveDot) {
            liveDot.classList.add("online");
        }
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        loadInitialData();
    };

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.event === "ATTENDANCE_UPDATE" && data.logs) {
                renderDashboard(data.logs);
            }
        } catch (e) {
            console.error("[WebSocket] Parse Error:", e);
        }
    };

    socket.onerror = (error) => {
        console.error("[WebSocket] Error occurred:", error);
    };

    socket.onclose = () => {
        // Offline: Red dot
        if (liveDot) {
            liveDot.classList.remove("online");
        }
        reconnectTimer = setTimeout(initWebSocket, 3000);
    };
}

// ----------------- EXCEL EXPORT FUNCTION -----------------
function exportToExcel() {
    if (!currentAttendanceData || currentAttendanceData.length === 0) {
        alert("No attendance records to export!");
        return;
    }

    // CSV Header row
    const headers = ["Log ID", "User ID", "Name", "Login Time (IST)", "Logout Time (IST)", "Duration", "Status", "Match Score"];
    
    const rows = currentAttendanceData.map(log => [
        log.id,
        `"${log.user_id}"`,
        `"${log.name}"`,
        `"${log.login_time || '-'}"`,
        `"${log.logout_time || '-'}"`,
        `"${log.duration || '-'}"`,
        `"${log.status}"`,
        `"${log.similarity_score || '-'}"`
    ]);

    const csvContent = "data:text/csv;charset=utf-8," 
        + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    const dateStr = new Date().toISOString().slice(0, 10);
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `Office_Attendance_${dateStr}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

window.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
});