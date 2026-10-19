const BACKEND_HOST = window.location.hostname || "127.0.0.1";

const BACKEND_BASE = `http://${BACKEND_HOST}:8000`;

const WS_URL = `ws://${BACKEND_HOST}:8000/ws/attendance`;



let globalRecords = [];

let socket = null;

let currentPhotoMethod = 'file';

let liveCameraStream = null;

let capturedBlob = null;

let activePreviewUrl = null;

let isWsConnecting = false;



// Dynamic refresh interval configuration (from .env via backend API)

let dashboardRefreshIntervalMs = 15000;

let autoRefreshTimer = null;



const DUMMY_AVATAR = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='44' height='44' viewBox='0 0 24 24' fill='%2364748b'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";



// Load dynamic dashboard refresh interval from .env via /api/config/client

async function loadClientConfig() {

    try {

        const res = await fetch(`${BACKEND_BASE}/api/config/client`);

        if (res.ok) {

            const cfg = await res.json();

            if (cfg.refresh_interval_seconds) {

                dashboardRefreshIntervalMs = Math.max(3, cfg.refresh_interval_seconds) * 1000;

                console.log(`[CONFIG] Dashboard Polling Interval set to: ${dashboardRefreshIntervalMs / 1000}s`);

            }

        }

    } catch (e) {

        console.warn("[CONFIG] Falling back to default interval 15s:", e);

    }

    setupAutoRefreshTimer();

}



function setupAutoRefreshTimer() {

    if (autoRefreshTimer) clearInterval(autoRefreshTimer);

    autoRefreshTimer = setInterval(() => {

        // Silent background sync only if the tab is currently visible

        if (document.visibilityState === 'visible') {

            fetchInitialLogs(true);

        }

    }, dashboardRefreshIntervalMs);

}



async function fetchInitialLogs(isSilent = false) {

    try {

        const res = await fetch(`${BACKEND_BASE}/attendance/logs`);

        if (res.ok) {

            const logs = await res.json();

            globalRecords = Array.isArray(logs) ? logs : [];

            renderFullTable(globalRecords);

            updateStats(globalRecords);

        }

    } catch (e) {

        if (!isSilent) console.warn("[REST] Initial logs load warning:", e);

    }

}



function initWebSocket() {

    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {

        return;

    }

    if (isWsConnecting) return;

    isWsConnecting = true;



    try {

        socket = new WebSocket(WS_URL);



        socket.onopen = () => {

            isWsConnecting = false;

            console.log("[WS] Connected to Attendance Engine.");

            const dot = document.getElementById("statusDot");

            if (dot) dot.classList.add("connected");

        };



        socket.onmessage = (event) => {

            try {

                const data = JSON.parse(event.data);

                if (data.event === "ATTENDANCE_UPDATE") {

                    if (data.action === "initial_load" || data.action === "dashboard_cleared" || data.action === "backup_restored") {

                        globalRecords = data.logs || [];

                        renderFullTable(globalRecords);

                        updateStats(globalRecords);

                    } else if (data.action === "login_recorded" || data.action === "unknown_detected") {

                        globalRecords = data.logs || [];

                        if (data.record) {

                            insertSingleRowOnTop(data.record);

                        } else {

                            renderFullTable(globalRecords);

                        }

                        updateStats(globalRecords);

                    } else if (data.action === "logout_recorded") {

                        globalRecords = data.logs || [];

                        if (data.record) {

                            updateExistingRowInPlace(data.record);

                        } else {

                            renderFullTable(globalRecords);

                        }

                        updateStats(globalRecords);

                    }

                }

            } catch (err) {

                console.error("[WS] Parse error:", err);

            }

        };



        socket.onclose = () => {

            isWsConnecting = false;

            const dot = document.getElementById("statusDot");

            if (dot) dot.classList.remove("connected");

            setTimeout(initWebSocket, 3000);

        };



        socket.onerror = (err) => {

            console.warn("[WS] Socket error:", err);

            try { socket.close(); } catch(e){}

        };

    } catch (e) {

        isWsConnecting = false;

        setTimeout(initWebSocket, 3000);

    }

}



// Full Table Render (Only on page start / filter change / clear)

function renderFullTable(records) {

    const tbody = document.getElementById("attendanceTableBody");

    if (!tbody) return;



    if (!records || records.length === 0) {

        tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 40px;">No attendance records found.</td></tr>`;

        return;

    }



    const currentFilter = document.getElementById("recordFilter") ? document.getElementById("recordFilter").value : "ALL";

    let filteredList = records;

    if (currentFilter === "IN_OFFICE") {

        filteredList = records.filter(r => r.status === "IN OFFICE");

    } else if (currentFilter === "CHECKED_OUT") {

        filteredList = records.filter(r => r.status === "CHECKED OUT");

    } else if (currentFilter === "UNAUTHORIZED") {

        filteredList = records.filter(r => r.status === "UNAUTHORIZED" || r.user_id === "UNKNOWN");

    }



    tbody.innerHTML = filteredList.map(r => generateRowHtml(r)).join("");

}



// Generate HTML for a single row (Includes link to open photo comparison modal)

function generateRowHtml(r) {

    let photo = DUMMY_AVATAR;

    if (r.photo_url && r.photo_url.trim() !== "") {

        photo = r.photo_url.startsWith("http") ? r.photo_url : `${BACKEND_BASE}${r.photo_url.startsWith("/") ? "" : "/"}${r.photo_url}`;

    }

    

    const status = r.status || "IN OFFICE";

    let badgeClass = "badge-success";

    if (status === "CHECKED OUT") badgeClass = "badge-checkedout";

    else if (status === "UNAUTHORIZED" || r.user_id === "UNKNOWN") badgeClass = "badge-danger";



    const score = r.similarity_score ? r.similarity_score : "-";



    return `

        <tr id="row-${r.id}">

            <td>#${r.id}</td>

            <td>

                <div class="photo-wrapper">

                    <img src="${photo}" class="avatar-img" loading="lazy" onerror="this.onerror=null;this.src='${DUMMY_AVATAR}'" alt="Face" title="Live Capture Photo">

                </div>

            </td>

            <td>

                <a href="javascript:void(0)" onclick="viewEmployeePhotoModal('${escapeHtml(r.user_id)}')" style="color: var(--accent-blue); font-weight: bold; text-decoration: none;">

                    ${escapeHtml(r.user_id)}

                </a>

            </td>

            <td>${escapeHtml(r.name)}</td>

            <td class="col-login">${r.login_time || "-"}</td>

            <td class="col-logout">${r.logout_time || "-"}</td>

            <td class="col-duration">${r.duration || "-"}</td>

            <td class="col-status"><span class="badge ${badgeClass}">${escapeHtml(status)}</span></td>

            <td><span class="score-pill">${escapeHtml(score)}</span></td>

        </tr>

    `;

}



// Anti-Flicker: New Login row added directly on top without touching other rows

function insertSingleRowOnTop(record) {

    const tbody = document.getElementById("attendanceTableBody");

    if (!tbody) return;



    // Check if duplicate row already exists

    const existing = document.getElementById(`row-${record.id}`);

    if (existing) {

        updateExistingRowInPlace(record);

        return;

    }



    // Remove empty state message if present

    if (tbody.children.length === 1 && tbody.children[0].innerText.includes("No attendance records")) {

        tbody.innerHTML = "";

    }



    const tempDiv = document.createElement("table");

    tempDiv.innerHTML = generateRowHtml(record);

    const newTr = tempDiv.querySelector("tr");



    if (newTr) {

        tbody.insertBefore(newTr, tbody.firstChild);

    }

}



// Anti-Flicker: Logout update changes only specific cells in the existing row

function updateExistingRowInPlace(record) {

    const row = document.getElementById(`row-${record.id}`);

    if (!row) {

        // If row not visible due to filters, perform full render

        renderFullTable(globalRecords);

        return;

    }



    const logoutCell = row.querySelector(".col-logout");

    const durationCell = row.querySelector(".col-duration");

    const statusCell = row.querySelector(".col-status");



    if (logoutCell) logoutCell.innerText = record.logout_time || "-";

    if (durationCell) durationCell.innerText = record.duration || "-";

    if (statusCell) {

        const status = record.status || "CHECKED OUT";

        let badgeClass = "badge-success";

        if (status === "CHECKED OUT") badgeClass = "badge-checkedout";

        else if (status === "UNAUTHORIZED" || record.user_id === "UNKNOWN") badgeClass = "badge-danger";

        statusCell.innerHTML = `<span class="badge ${badgeClass}">${escapeHtml(status)}</span>`;

    }

}



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



// Photo Comparison Modal: Displays Registered Profile Photo vs Latest Live Captured Photo

async function viewEmployeePhotoModal(empId) {

    if (empId === "UNKNOWN") {

        alert("This record belongs to an unregistered Unknown Visitor.");

        return;

    }



    try {

        const res = await fetch(`${BACKEND_BASE}/api/employees/${empId}/photos`);

        if (!res.ok) {

            alert("Could not fetch employee photo details.");

            return;

        }



        const data = await res.json();

        const regPhoto = data.registered_photo.startsWith("http") ? data.registered_photo : `${BACKEND_BASE}${data.registered_photo}`;

        const livePhoto = data.latest_live_photo.startsWith("http") ? data.latest_live_photo : `${BACKEND_BASE}${data.latest_live_photo}`;



        const modalTitle = document.getElementById("modalPhotoEmpId");

        if (modalTitle) modalTitle.innerText = `${data.name} (${data.emp_id})`;



        const regImg = document.getElementById("imgRegisteredPhoto");

        if (regImg) regImg.src = regPhoto;



        const liveImg = document.getElementById("imgLivePhoto");

        if (liveImg) liveImg.src = livePhoto;



        const compModal = document.getElementById("photoComparisonModal");

        if (compModal) compModal.style.display = "flex";

    } catch (e) {

        alert("Failed to load employee photo comparison: " + e);

    }

}



function closePhotoComparisonModal() {

    const compModal = document.getElementById("photoComparisonModal");

    if (compModal) compModal.style.display = "none";

}



function filterRecords() {

    renderFullTable(globalRecords);

}



// Modals Handlers

function openRegisterModal() {

    document.getElementById("registerModal").style.display = "flex";

    switchPhotoTab('file');

}



function closeRegisterModal() {

    document.getElementById("registerModal").style.display = "none";

    stopWebcamStream();

    if (activePreviewUrl) {

        URL.revokeObjectURL(activePreviewUrl);

        activePreviewUrl = null;

    }

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

        closeRegisterModal();

        closeModal("employeeModal");

        closePhotoComparisonModal();

    }

};



function switchPhotoTab(method) {

    currentPhotoMethod = method;

    const tabFile = document.getElementById("tabFile");

    const tabCamera = document.getElementById("tabCamera");

    const fileBox = document.getElementById("fileUploadBox");

    const cameraBox = document.getElementById("liveCameraBox");



    if (method === 'file') {

        tabFile.classList.add("active");

        tabCamera.classList.remove("active");

        fileBox.style.display = "block";

        cameraBox.style.display = "none";

        stopWebcamStream();

    } else {

        tabCamera.classList.add("active");

        tabFile.classList.remove("active");

        fileBox.style.display = "none";

        cameraBox.style.display = "flex";

        startWebcamStream();

    }

}



async function startWebcamStream() {

    capturedBlob = null;

    const video = document.getElementById("webcamVideo");

    const preview = document.getElementById("cameraPreview");

    const btnCapture = document.getElementById("btnCapture");

    const btnRetake = document.getElementById("btnRetake");



    video.style.display = "block";

    preview.style.display = "none";

    btnCapture.style.display = "inline-flex";

    btnRetake.style.display = "none";



    try {

        liveCameraStream = await navigator.mediaDevices.getUserMedia({

            video: { width: 640, height: 480, facingMode: "user" },

            audio: false

        });

        video.srcObject = liveCameraStream;

    } catch (err) {

        alert("Camera permission denied or camera not available: " + err.message);

        switchPhotoTab('file');

    }

}



function stopWebcamStream() {

    if (liveCameraStream) {

        liveCameraStream.getTracks().forEach(track => track.stop());

        liveCameraStream = null;

    }

}



function captureLiveSnapshot() {

    const video = document.getElementById("webcamVideo");

    const preview = document.getElementById("cameraPreview");

    const btnCapture = document.getElementById("btnCapture");

    const btnRetake = document.getElementById("btnRetake");



    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth || 640;

    canvas.height = video.videoHeight || 480;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);



    canvas.toBlob((blob) => {

        capturedBlob = blob;

        if (activePreviewUrl) URL.revokeObjectURL(activePreviewUrl);

        activePreviewUrl = URL.createObjectURL(blob);

        preview.src = activePreviewUrl;

        

        video.style.display = "none";

        preview.style.display = "block";

        btnCapture.style.display = "none";

        btnRetake.style.display = "inline-flex";

    }, "image/jpeg", 0.95);

}



function retakeLiveSnapshot() {

    capturedBlob = null;

    if (activePreviewUrl) {

        URL.revokeObjectURL(activePreviewUrl);

        activePreviewUrl = null;

    }

    const video = document.getElementById("webcamVideo");

    const preview = document.getElementById("cameraPreview");

    const btnCapture = document.getElementById("btnCapture");

    const btnRetake = document.getElementById("btnRetake");



    video.style.display = "block";

    preview.style.display = "none";

    btnCapture.style.display = "inline-flex";

    btnRetake.style.display = "none";

}



async function handleRegisterSubmit(event) {

    event.preventDefault();



    const empId = document.getElementById("regEmpId").value.trim();

    const name = document.getElementById("regName").value.trim();

    const fileInput = document.getElementById("regPhotoFile");



    let finalPhotoBlob = null;



    if (currentPhotoMethod === 'file') {

        if (!fileInput.files || fileInput.files.length === 0) {

            alert("Please select a photo file from your computer.");

            return;

        }

        finalPhotoBlob = fileInput.files[0];

    } else {

        if (!capturedBlob) {

            alert("Please click 'Take Snapshot' to capture your live photo first.");

            return;

        }

        finalPhotoBlob = capturedBlob;

    }



    const submitBtn = document.getElementById("btnRegisterSubmit");

    submitBtn.innerText = "Registering Face...";

    submitBtn.disabled = true;



    const formData = new FormData();

    formData.append("emp_id", empId);

    formData.append("name", name);

    formData.append("files", finalPhotoBlob, `${empId}_face.jpg`);



    try {

        const res = await fetch(`${BACKEND_BASE}/api/employees/register`, {

            method: "POST",

            body: formData

        });



        const data = await res.json();

        if (res.ok) {

            alert("Registered Successfully: " + data.message);

            closeRegisterModal();

            document.getElementById("registerForm").reset();

            fetchInitialLogs();

        } else {

            alert("Registration Failed: " + data.message);

        }

    } catch (err) {

        alert("Server Connection Error: " + err);

    } finally {

        submitBtn.innerText = "Submit & Register Employee";

        submitBtn.disabled = false;

    }

}



async function clearDashboard() {

    if (!confirm("Are you sure? A complete backup snapshot will be saved in backend/backups.")) return;

    try {

        const res = await fetch(`${BACKEND_BASE}/api/attendance/clear-dashboard`, { 

            method: "POST"

        });

        const data = await res.json();

        alert(data.message);

        globalRecords = [];

        renderFullTable(globalRecords);

        updateStats(globalRecords);

    } catch (err) {

        alert("Failed to clear: " + err);

    }

}



async function restoreBackup() {

    if (!confirm("Restore the latest attendance backup?")) return;

    try {

        const res = await fetch(`${BACKEND_BASE}/api/attendance/restore-latest-backup`, { 

            method: "POST"

        });

        const data = await res.json();

        alert(data.message);

        fetchInitialLogs();

    } catch (err) {

        alert("Failed to restore: " + err);

    }

}



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



        container.innerHTML = employees.map(emp => {

            let photo = DUMMY_AVATAR;

            if (emp.photo_url && emp.photo_url.trim() !== "") {

                photo = emp.photo_url.startsWith("http") ? emp.photo_url : `${BACKEND_BASE}${emp.photo_url.startsWith("/") ? "" : "/"}${emp.photo_url}`;

            }



            return `

                <div class="emp-item">

                    <div class="emp-info">

                        <img src="${photo}" class="avatar-img" onerror="this.onerror=null;this.src='${DUMMY_AVATAR}'">

                        <div>

                            <div style="font-weight: 600;">${escapeHtml(emp.name)}</div>

                            <div style="font-size: 12px; color: var(--text-muted);">${escapeHtml(emp.emp_id)}</div>

                        </div>

                    </div>

                    <button onclick="deleteEmployee('${escapeHtml(emp.emp_id)}')" style="background-color: rgba(239, 68, 68, 0.2); color: #f87171; padding: 6px 12px; font-size: 12px;">

                        Delete

                    </button>

                </div>

            `;

        }).join("");

    } catch (err) {

        container.innerHTML = `<p style="color: #f87171;">Failed to load employees: ${err}</p>`;

    }

}



async function deleteEmployee(empId) {

    if (!confirm(`Permanently delete employee ${empId}?`)) return;

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



function escapeHtml(str) {

    if (!str) return "";

    return String(str)

        .replace(/&/g, "&amp;")

        .replace(/</g, "&lt;")

        .replace(/>/g, "&gt;")

        .replace(/"/g, "&quot;")

        .replace(/'/g, "&#039;");

}



document.addEventListener("DOMContentLoaded", () => {

    loadClientConfig();

    fetchInitialLogs();

    initWebSocket();

});