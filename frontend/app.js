const tableBody = document.getElementById("logsTableBody");
const totalRecordsCard = document.getElementById("totalRecords");
const currentlyInOfficeCard = document.getElementById("currentlyInOffice");
const checkedOutCard = document.getElementById("checkedOut");
const liveDot = document.getElementById("liveDot");

const registerModal = document.getElementById("registerModal");
const empListModal = document.getElementById("empListModal");
const detailModal = document.getElementById("detailModal");
const liveCameraModal = document.getElementById("liveCameraModal");
const empListContainer = document.getElementById("empListContainer");

const photoDropdown = document.getElementById("photoDropdown");
const selectedPhotosContainer = document.getElementById("selectedPhotosContainer");
const webcamVideo = document.getElementById("webcamVideo");
const snapshotCanvas = document.getElementById("snapshotCanvas");
const snapBtn = document.getElementById("snapBtn");
const retakeBtn = document.getElementById("retakeBtn");
const usePhotoBtn = document.getElementById("usePhotoBtn");

let socket = null;
let reconnectTimer = null;
let allAttendanceData = [];
let masterEmployeeIds = new Set();
let activeFilter = "ALL";

let stagedRegistrationFiles = [];
let webcamStream = null;

function setFilter(filterType) {
    activeFilter = filterType;
    document.getElementById("filterAll").classList.toggle("active", filterType === "ALL");
    document.getElementById("filterEmp").classList.toggle("active", filterType === "EMPLOYEES");
    document.getElementById("filterUnknown").classList.toggle("active", filterType === "UNKNOWN");
    renderDashboard(allAttendanceData);
}

async function refreshMasterEmployeeList() {
    try {
        const res = await fetch("http://127.0.0.1:8000/api/employees/list");
        if (res.ok) {
            const list = await res.json();
            masterEmployeeIds = new Set(list.map(e => e.emp_id));
        }
    } catch (e) {
        console.error("Master employee sync error", e);
    }
}

function renderDashboard(logs) {
    if (!tableBody) return;
    tableBody.innerHTML = "";
    allAttendanceData = logs || [];

    let inOfficeCount = 0;
    let checkedOutCount = 0;

    // Strict Filter Logic: Isolates Genuine Employees from Unknown Visitors
    let filtered = allAttendanceData.filter(log => {
        const isUnknownVisitor = (log.user_id === "UNKNOWN" || log.status === "UNAUTHORIZED");
        const isRegisteredEmployee = !isUnknownVisitor && masterEmployeeIds.has(log.user_id);

        if (isRegisteredEmployee) {
            if (log.status === "IN OFFICE" || !log.logout_time || log.logout_time === "-") inOfficeCount++;
            else checkedOutCount++;
        }

        if (activeFilter === "EMPLOYEES") return isRegisteredEmployee;
        if (activeFilter === "UNKNOWN") return isUnknownVisitor;
        return true; // ALL
    });

    totalRecordsCard.innerText = allAttendanceData.length;
    currentlyInOfficeCard.innerText = inOfficeCount;
    checkedOutCard.innerText = checkedOutCount;

    if (!filtered || filtered.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: #94a3b8;">No records to display in this section.</td></tr>`;
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
            <td><span class="score-pill" id="score-pill-${log.user_id}">${log.similarity_score || "-"}</span></td>
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
        await refreshMasterEmployeeList();
        const res = await fetch("http://127.0.0.1:8000/attendance/logs");
        if (res.ok) {
            const logs = await res.json();
            renderDashboard(logs);
        }
    } catch (e) {
        console.error("Initial logs fetch failed", e);
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
            } else if (data.event === "LIVE_SCORE_UPDATE") {
                const el = document.getElementById(`score-pill-${data.user_id}`);
                if (el) el.innerText = data.similarity_score;
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

function openRegisterModal() {
    stagedRegistrationFiles = [];
    renderStagedPhotos();
    registerModal.style.display = "flex";
}

function closeRegisterModal() {
    registerModal.style.display = "none";
    if (photoDropdown) photoDropdown.classList.remove("show");
}

function togglePhotoMenu() {
    if (photoDropdown) photoDropdown.classList.toggle("show");
}

function triggerFileInput() {
    if (photoDropdown) photoDropdown.classList.remove("show");
    document.getElementById("regPhotos").click();
}

function handleSystemFilePicked(e) {
    const files = Array.from(e.target.files);
    files.forEach(f => stagedRegistrationFiles.push(f));
    renderStagedPhotos();
    e.target.value = "";
}

function removeStagedPhoto(idx) {
    stagedRegistrationFiles.splice(idx, 1);
    renderStagedPhotos();
}

function renderStagedPhotos() {
    if (!selectedPhotosContainer) return;
    selectedPhotosContainer.innerHTML = "";
    if (stagedRegistrationFiles.length === 0) {
        selectedPhotosContainer.innerHTML = `<span style="font-size: 11px; color: #64748b; margin: auto;">No photos added yet (Mandatory)</span>`;
        return;
    }
    stagedRegistrationFiles.forEach((file, idx) => {
        const url = URL.createObjectURL(file);
        const div = document.createElement("div");
        div.className = "thumb-preview-wrap";
        div.innerHTML = `
            <img src="${url}">
            <div class="del-btn" onclick="removeStagedPhoto(${idx})">✕</div>
        `;
        selectedPhotosContainer.appendChild(div);
    });
}

async function openLiveCameraModal() {
    if (photoDropdown) photoDropdown.classList.remove("show");
    liveCameraModal.style.display = "flex";
    snapBtn.style.display = "inline-block";
    retakeBtn.style.display = "none";
    usePhotoBtn.style.display = "none";
    webcamVideo.style.display = "block";
    snapshotCanvas.style.display = "none";

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
            audio: false
        });
        webcamVideo.srcObject = webcamStream;
    } catch (err) {
        alert("Camera Error: Please allow camera permissions. (" + err.message + ")");
        closeLiveCameraModal();
    }
}

function captureLiveSnapshot() {
    if (!webcamStream) return;
    snapshotCanvas.width = webcamVideo.videoWidth || 640;
    snapshotCanvas.height = webcamVideo.videoHeight || 480;
    const ctx = snapshotCanvas.getContext("2d");
    ctx.drawImage(webcamVideo, 0, 0, snapshotCanvas.width, snapshotCanvas.height);

    webcamVideo.style.display = "none";
    snapshotCanvas.style.display = "block";
    snapBtn.style.display = "none";
    retakeBtn.style.display = "inline-block";
    usePhotoBtn.style.display = "inline-block";
}

function retakeLiveSnapshot() {
    webcamVideo.style.display = "block";
    snapshotCanvas.style.display = "none";
    snapBtn.style.display = "inline-block";
    retakeBtn.style.display = "none";
    usePhotoBtn.style.display = "none";
}

function useCapturedPhoto() {
    snapshotCanvas.toBlob((blob) => {
        if (blob) {
            const fileName = `webcam_${Date.now()}.jpg`;
            const file = new File([blob], fileName, { type: "image/jpeg" });
            stagedRegistrationFiles.push(file);
            renderStagedPhotos();
            closeLiveCameraModal();
        }
    }, "image/jpeg", 0.98);
}

function closeLiveCameraModal() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    liveCameraModal.style.display = "none";
}

async function handleRegister(e) {
    e.preventDefault();
    const empId = document.getElementById("regEmpId").value.trim();
    const name = document.getElementById("regName").value.trim();
    const saveBtn = document.getElementById("saveEmpBtn");

    if (stagedRegistrationFiles.length === 0) {
        alert("⚠️ Photo is strictly required! Please add at least 1 photo.");
        return;
    }

    const formData = new FormData();
    formData.append("emp_id", empId);
    formData.append("name", name);
    stagedRegistrationFiles.forEach(f => {
        formData.append("files", f);
    });

    saveBtn.disabled = true;
    saveBtn.innerText = "Registering...";

    try {
        const res = await fetch("http://127.0.0.1:8000/api/employees/register", {
            method: "POST",
            body: formData
        });
        const d = await res.json();
        if (res.ok && d.status === "success") {
            alert(d.message);
            closeRegisterModal();
            document.getElementById("registerForm").reset();
            stagedRegistrationFiles = [];
            renderStagedPhotos();
            await refreshMasterEmployeeList();
            await openEmployeeListModal();
            renderDashboard(allAttendanceData);
        } else {
            alert("Registration Failed: " + (d.message || "Unknown error"));
        }
    } catch (err) {
        alert("Network Error: " + err.message);
    } finally {
        saveBtn.disabled = false;
        saveBtn.innerText = "Save Employee";
    }
}

async function deleteEmployee(empId) {
    if (!confirm(`Are you sure you want to delete Employee ${empId}?\nThis will remove their facial embeddings permanently.`)) {
        return;
    }

    try {
        const res = await fetch(`http://127.0.0.1:8000/api/employees/${encodeURIComponent(empId)}`, {
            method: "DELETE"
        });
        const d = await res.json();
        if (res.ok && d.status === "success") {
            alert(d.message);
            await refreshMasterEmployeeList();
            await openEmployeeListModal();
            renderDashboard(allAttendanceData);
        } else {
            alert("Failed to delete: " + (d.message || "Unknown error"));
        }
    } catch (err) {
        alert("Delete request failed: " + err.message);
    }
}

async function openEmployeeListModal() {
    empListModal.style.display = "flex";
    try {
        const res = await fetch("http://127.0.0.1:8000/api/employees/list");
        const list = await res.json();
        if (!list || list.length === 0) {
            empListContainer.innerHTML = "<p style='color: #94a3b8; font-size: 13px; text-align: center; padding: 12px;'>No verified employees registered yet.</p>";
            return;
        }
        empListContainer.innerHTML = list.map(emp => `
            <div class="emp-list-item">
                <div style="display: flex; align-items: center;">
                    <img src="${emp.photo_url}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'40\\' height=\\'40\\' viewBox=\\'0 0 24 24\\' fill=\\'%23666\\'><circle cx=\\'12\\' cy=\\'12\\' r=\\'10\\'/></svg>'">
                    <div>
                        <strong style="color: #f1f5f9; display: block;">${emp.name}</strong>
                        <span style="color: #38bdf8; font-size: 11px; font-family: monospace;">${emp.emp_id} (${emp.vector_count || 1} sample)</span>
                    </div>
                </div>
                <button class="btn btn-delete" onclick="deleteEmployee('${emp.emp_id}')" title="Delete Employee">
                    🗑️ Delete
                </button>
            </div>
        `).join("");
    } catch (e) {
        empListContainer.innerHTML = "<p style='color: #ef4444; font-size: 13px;'>Failed to load employee list.</p>";
    }
}

function closeEmployeeListModal() { empListModal.style.display = "none"; }

window.addEventListener("DOMContentLoaded", () => {
    initWebSocket();
});