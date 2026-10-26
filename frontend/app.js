const BACKEND_HOST = window.location.hostname || "127.0.0.1";
const BACKEND_BASE = `http://${BACKEND_HOST}:8000`;
const WS_URL = `ws://${BACKEND_HOST}:8000/ws/attendance`;

let globalRecords = [];
let socket = null;
let currentPhotoMethod = "file";
let liveCameraStream = null;
let capturedBlob = null;
let activePreviewUrl = null;
let isWsConnecting = false;

let dashboardRefreshIntervalMs = 15000;
let autoRefreshTimer = null;
let lastRenderedLogsSignature = null;

const DUMMY_AVATAR =
    "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='44' height='44' viewBox='0 0 24 24' fill='%2364748b'><path d='M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z'/></svg>";

// --------------------------------------------------
// CLIENT CONFIGURATION AND AUTO REFRESH
// --------------------------------------------------

async function loadClientConfig() {
    try {
        const res = await fetch(`${BACKEND_BASE}/api/config/client`);

        if (res.ok) {
            const cfg = await res.json();

            if (cfg.refresh_interval_seconds) {
                dashboardRefreshIntervalMs =
                    Math.max(3, cfg.refresh_interval_seconds) * 1000;

                console.log(
                    `[CONFIG] Dashboard polling interval: ${
                        dashboardRefreshIntervalMs / 1000
                    } seconds`
                );
            }
        }
    } catch (error) {
        console.warn(
            "[CONFIG] Using default refresh interval:",
            error
        );
    }

    setupAutoRefreshTimer();
}

function setupAutoRefreshTimer() {
    if (autoRefreshTimer) {
        clearInterval(autoRefreshTimer);
    }

    autoRefreshTimer = setInterval(() => {
        // WebSocket handles live updates. Poll only as a fallback
        // while the WebSocket is disconnected.
        const wsConnected =
            socket && socket.readyState === WebSocket.OPEN;

        if (
            document.visibilityState === "visible" &&
            !wsConnected
        ) {
            fetchInitialLogs(true);
        }
    }, dashboardRefreshIntervalMs);
}

async function fetchInitialLogs(isSilent = false) {
    try {
        const res = await fetch(`${BACKEND_BASE}/attendance/logs`);

        if (!res.ok) {
            throw new Error(`HTTP ${res.status}`);
        }

        const logs = await res.json();

        globalRecords = Array.isArray(logs) ? logs : [];
        const logsSignature = JSON.stringify(globalRecords);

        if (logsSignature !== lastRenderedLogsSignature) {
            renderFullTable(globalRecords);
        }
        updateStats(globalRecords);
    } catch (error) {
        if (!isSilent) {
            console.error("[REST] Failed to load attendance logs:", error);
        }
    }
}

// --------------------------------------------------
// WEBSOCKET CONNECTION
// --------------------------------------------------

function initWebSocket() {
    if (
        socket &&
        (
            socket.readyState === WebSocket.OPEN ||
            socket.readyState === WebSocket.CONNECTING
        )
    ) {
        return;
    }

    if (isWsConnecting) {
        return;
    }

    isWsConnecting = true;

    try {
        socket = new WebSocket(WS_URL);

        socket.onopen = () => {
            isWsConnecting = false;

            console.log("[WS] Connected to attendance backend.");

            const dot = document.getElementById("statusDot");

            if (dot) {
                dot.classList.add("connected");
            }
        };

        socket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                if (data.event !== "ATTENDANCE_UPDATE") {
                    return;
                }

                if (
                    data.action === "initial_load" ||
                    data.action === "dashboard_cleared" ||
                    data.action === "backup_restored"
                ) {
                    globalRecords = Array.isArray(data.logs)
                        ? data.logs
                        : [];

                    renderFullTable(globalRecords);
                    updateStats(globalRecords);
                } else if (
                    data.action === "login_recorded" ||
                    data.action === "unknown_detected"
                ) {
                    globalRecords = Array.isArray(data.logs)
                        ? data.logs
                        : globalRecords;

                    if (data.record) {
                        insertSingleRowOnTop(data.record);
                    } else {
                        renderFullTable(globalRecords);
                    }

                    updateStats(globalRecords);
                } else if (data.action === "logout_recorded") {
                    globalRecords = Array.isArray(data.logs)
                        ? data.logs
                        : globalRecords;

                    if (data.record) {
                        updateExistingRowInPlace(data.record);
                    } else {
                        renderFullTable(globalRecords);
                    }

                    updateStats(globalRecords);
                }

                lastRenderedLogsSignature = JSON.stringify(globalRecords);
            } catch (error) {
                console.error("[WS] Message parsing failed:", error);
            }
        };

        socket.onclose = () => {
            isWsConnecting = false;

            const dot = document.getElementById("statusDot");

            if (dot) {
                dot.classList.remove("connected");
            }

            setTimeout(initWebSocket, 3000);
        };

        socket.onerror = (error) => {
            console.warn("[WS] Connection error:", error);

            if (socket) {
                socket.close();
            }
        };
    } catch (error) {
        isWsConnecting = false;

        console.error("[WS] Could not initialize WebSocket:", error);

        setTimeout(initWebSocket, 3000);
    }
}

// --------------------------------------------------
// ATTENDANCE TABLE
// --------------------------------------------------

function renderFullTable(records) {
    const tbody = document.getElementById("attendanceTableBody");

    if (!tbody) {
        return;
    }

    lastRenderedLogsSignature = JSON.stringify(
        Array.isArray(records) ? records : []
    );

    const badge = document.getElementById("filterCountBadge");

    if (!Array.isArray(records) || records.length === 0) {
        if (badge) {
            badge.innerText = "Showing 0 of 0 records";
        }
        tbody.innerHTML = `
            <tr>
                <td colspan="10"
                    style="text-align:center;color:var(--text-muted);padding:40px;">
                    No attendance records found.
                </td>
            </tr>
        `;
        return;
    }

    const filterElement = document.getElementById("recordFilter");

    const currentFilter = filterElement
        ? filterElement.value
        : "ALL";

    let filteredList = records;

    if (currentFilter === "IN_OFFICE") {
        filteredList = records.filter(
            record => record.status === "IN OFFICE"
        );
    } else if (currentFilter === "CHECKED_OUT") {
        filteredList = records.filter(
            record => record.status === "CHECKED OUT"
        );
    } else if (currentFilter === "UNAUTHORIZED") {
        filteredList = records.filter(
            record =>
                record.status === "UNAUTHORIZED" ||
                record.user_id === "UNKNOWN"
        );
    }

    if (badge) {
        badge.innerText = `Showing ${filteredList.length} of ${records.length} records`;
    }

    tbody.innerHTML = filteredList
        .map(record => generateRowHtml(record))
        .join("");
}

function generateRowHtml(record) {
    let photo = DUMMY_AVATAR;

    if (
        record.photo_url &&
        String(record.photo_url).trim() !== ""
    ) {
        photo = getPhotoUrl(record.photo_url);
    }

    const status = record.status || "IN OFFICE";

    let badgeClass = "badge-success";

    if (status === "CHECKED OUT") {
        badgeClass = "badge-checkedout";
    } else if (
        status === "UNAUTHORIZED" ||
        record.user_id === "UNKNOWN"
    ) {
        badgeClass = "badge-danger";
    }

    const score =
        record.similarity_score !== null &&
        record.similarity_score !== undefined &&
        record.similarity_score !== ""
            ? record.similarity_score
            : "-";

    const userId = String(record.user_id || "");
    const safeUserId = escapeHtml(userId);
    const safeName = escapeHtml(record.name || "-");
    const safeLogin = escapeHtml(record.login_time || "-");

    return `
        <tr id="row-${escapeHtml(record.id)}">
            <td>#${escapeHtml(record.id)}</td>

            <td>
                <div class="photo-wrapper" onclick="openPhotoLightbox('${photo}', '${safeUserId}', '${safeName}', '${safeLogin}')" title="Hover to zoom, Click for HD preview">
                    <img
                        src="${photo}"
                        class="avatar-img"
                        loading="lazy"
                        onerror="this.onerror=null;this.src='${DUMMY_AVATAR}'"
                        alt="Face"
                    >
                </div>
            </td>

            <td>
                <a
                    href="javascript:void(0)"
                    onclick="openEmployeeDetailModal('${safeUserId}')"
                    style="color:var(--accent-blue);font-weight:bold;text-decoration:none;"
                >
                    ${safeUserId}
                </a>
            </td>

            <td>${safeName}</td>
            <td class="col-login">${safeLogin}</td>
            <td class="col-logout">${escapeHtml(record.logout_time || "-")}</td>
            <td class="col-duration">${escapeHtml(record.duration || "-")}</td>

            <td class="col-status">
                <span class="badge ${badgeClass}">
                    ${escapeHtml(status)}
                </span>
            </td>

            <td>
                <span class="score-pill">${escapeHtml(score)}</span>
            </td>

            <td style="text-align:center;">
                <button
                    type="button"
                    class="btn"
                    style="background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.4);color:#60a5fa;padding:5px 12px;font-size:12px;border-radius:6px;"
                    onclick="viewRecordDetail('${escapeHtml(record.id)}')"
                    title="View Full Record Details"
                >
                    👁️ View
                </button>
            </td>
        </tr>
    `;
}

function insertSingleRowOnTop(record) {
    const tbody = document.getElementById("attendanceTableBody");

    if (!tbody || !record) {
        return;
    }

    const existing = document.getElementById(`row-${record.id}`);

    if (existing) {
        updateExistingRowInPlace(record);
        return;
    }

    if (
        tbody.children.length === 1 &&
        tbody.children[0].innerText.includes("No attendance records")
    ) {
        tbody.innerHTML = "";
    }

    const temporaryTable = document.createElement("table");
    temporaryTable.innerHTML = generateRowHtml(record);

    const newRow = temporaryTable.querySelector("tr");

    if (newRow) {
        tbody.insertBefore(newRow, tbody.firstChild);
    }
}

function updateExistingRowInPlace(record) {
    const row = document.getElementById(`row-${record.id}`);

    if (!row) {
        renderFullTable(globalRecords);
        return;
    }

    const logoutCell = row.querySelector(".col-logout");
    const durationCell = row.querySelector(".col-duration");
    const statusCell = row.querySelector(".col-status");

    if (logoutCell) {
        logoutCell.textContent = record.logout_time || "-";
    }

    if (durationCell) {
        durationCell.textContent = record.duration || "-";
    }

    if (statusCell) {
        const status = record.status || "CHECKED OUT";

        let badgeClass = "badge-success";

        if (status === "CHECKED OUT") {
            badgeClass = "badge-checkedout";
        } else if (
            status === "UNAUTHORIZED" ||
            record.user_id === "UNKNOWN"
        ) {
            badgeClass = "badge-danger";
        }

        statusCell.innerHTML = `
            <span class="badge ${badgeClass}">
                ${escapeHtml(status)}
            </span>
        `;
    }
}

// --------------------------------------------------
// DASHBOARD STATISTICS
// --------------------------------------------------

function updateStats(records) {
    const totalEl = document.getElementById("statTotal");
    const inOfficeEl = document.getElementById("statInOffice");
    const checkedOutEl = document.getElementById("statCheckedOut");
    const unauthorizedEl = document.getElementById("statUnauthorized");

    if (totalEl) {
        totalEl.textContent = records.length;
    }

    let inOffice = 0;
    let checkedOut = 0;
    let unauthorized = 0;

    records.forEach(record => {
        if (record.status === "IN OFFICE") {
            inOffice++;
        } else if (record.status === "CHECKED OUT") {
            checkedOut++;
        } else if (
            record.status === "UNAUTHORIZED" ||
            record.user_id === "UNKNOWN"
        ) {
            unauthorized++;
        }
    });

    if (inOfficeEl) {
        inOfficeEl.textContent = inOffice;
    }

    if (checkedOutEl) {
        checkedOutEl.textContent = checkedOut;
    }

    if (unauthorizedEl) {
        unauthorizedEl.textContent = unauthorized;
    }
}

// --------------------------------------------------
// EMPLOYEE PHOTO COMPARISON MODAL
// --------------------------------------------------

async function openEmployeeDetailModal(empId) {
    if (!empId || empId === "UNKNOWN") {
        alert("This record belongs to an unregistered Unknown Visitor.");
        return;
    }

    try {
        const safeEmpId = encodeURIComponent(empId);

        const res = await fetch(
            `${BACKEND_BASE}/api/employees/${safeEmpId}/photos`
        );

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw new Error(
                data.detail ||
                data.message ||
                `Could not load employee details (HTTP ${res.status}).`
            );
        }

        const modal = document.getElementById("photoComparisonModal");
        const title = document.getElementById("modalPhotoEmpId");
        const registeredImg = document.getElementById("imgRegisteredPhoto");
        const liveImg = document.getElementById("imgLivePhoto");

        if (!modal || !title || !registeredImg || !liveImg) {
            throw new Error(
                "Photo comparison modal HTML elements are missing."
            );
        }

        title.textContent =
            `${data.name || "Employee"} (${data.emp_id || empId})`;

        registeredImg.onerror = () => {
            registeredImg.onerror = null;
            registeredImg.removeAttribute("src");
            registeredImg.alt = "Registered photo unavailable";
        };

        liveImg.onerror = () => {
            liveImg.onerror = null;
            liveImg.removeAttribute("src");
            liveImg.alt = "Live photo unavailable";
        };

        registeredImg.alt = "Registered Photo";

        if (data.registered_photo) {
            registeredImg.src = getPhotoUrl(data.registered_photo);
        } else {
            registeredImg.removeAttribute("src");
            registeredImg.alt = "No registered photo available";
        }

        if (data.latest_live_photo) {
            liveImg.alt = "Live Photo (Latest Punch)";
            liveImg.src = getPhotoUrl(data.latest_live_photo);
        } else {
            liveImg.removeAttribute("src");
            liveImg.alt = "No live punch photo available";
        }

        modal.style.display = "flex";
    } catch (error) {
        console.error("[EMPLOYEE PHOTOS]", error);

        alert(
            `Failed to load photo comparison: ${
                error.message || error
            }`
        );
    }
}

function closePhotoComparisonModal() {
    const modal = document.getElementById("photoComparisonModal");

    if (modal) {
        modal.style.display = "none";
    }
}

function filterRecords() {
    renderFullTable(globalRecords);
}

// --------------------------------------------------
// PHOTO LIGHTBOX & RECORD DETAIL MODALS
// --------------------------------------------------

function openPhotoLightbox(photoUrl, userId, name, loginTime) {
    const modal = document.getElementById("photoLightboxModal");
    const img = document.getElementById("lightboxImg");
    const title = document.getElementById("lightboxTitle");
    const sub = document.getElementById("lightboxSubtitle");

    if (!modal || !img) return;

    img.src = photoUrl || DUMMY_AVATAR;
    if (title) title.innerText = name ? `${name} (${userId})` : userId;
    if (sub) sub.innerText = loginTime ? `Recorded at: ${loginTime} IST` : "Live Face Punch";

    modal.style.display = "flex";
}

function closePhotoLightbox() {
    const modal = document.getElementById("photoLightboxModal");
    if (modal) modal.style.display = "none";
}

function viewRecordDetail(recordId) {
    const record = globalRecords.find(r => String(r.id) === String(recordId));
    if (!record) return;

    const modal = document.getElementById("recordDetailModal");
    const body = document.getElementById("recordDetailContent") || document.getElementById("recordDetailBody");
    if (!modal || !body) return;

    const photo = record.photo_url ? getPhotoUrl(record.photo_url) : DUMMY_AVATAR;

    body.innerHTML = `
        <div style="display:flex;gap:18px;align-items:center;margin-bottom:16px;background:rgba(255,255,255,0.03);padding:14px;border-radius:10px;">
            <img src="${photo}" style="width:72px;height:72px;border-radius:10px;object-fit:cover;border:2px solid #3b82f6;" alt="Face">
            <div>
                <h3 style="color:#f8fafc;font-size:16px;margin-bottom:4px;">${escapeHtml(record.name || "-")}</h3>
                <p style="color:#60a5fa;font-size:13px;font-weight:600;margin:0;">User ID: ${escapeHtml(record.user_id || "-")}</p>
                <p style="color:#94a3b8;font-size:12px;margin:2px 0 0 0;">Log ID: #${escapeHtml(record.id)}</p>
            </div>
        </div>
        <table style="width:100%;font-size:13px;border-collapse:collapse;">
            <tr><td style="padding:6px 0;color:#94a3b8;">Status:</td><td style="padding:6px 0;font-weight:700;color:#f8fafc;">${escapeHtml(record.status || "-")}</td></tr>
            <tr><td style="padding:6px 0;color:#94a3b8;">Login Time:</td><td style="padding:6px 0;color:#34d399;font-weight:600;">${escapeHtml(record.login_time || "-")}</td></tr>
            <tr><td style="padding:6px 0;color:#94a3b8;">Logout Time:</td><td style="padding:6px 0;color:#a78bfa;font-weight:600;">${escapeHtml(record.logout_time || "-")}</td></tr>
            <tr><td style="padding:6px 0;color:#94a3b8;">Working Duration:</td><td style="padding:6px 0;color:#f8fafc;font-weight:600;">${escapeHtml(record.duration || "-")}</td></tr>
            <tr><td style="padding:6px 0;color:#94a3b8;">Camera Gate:</td><td style="padding:6px 0;color:#f8fafc;">${escapeHtml(record.camera_id || "-")}</td></tr>
            <tr><td style="padding:6px 0;color:#94a3b8;">Face Match Score:</td><td style="padding:6px 0;color:#fbbf24;font-weight:700;">${escapeHtml(record.similarity_score || "-")}</td></tr>
        </table>
    `;

    modal.style.display = "flex";
}

function closeRecordDetailModal() {
    const modal = document.getElementById("recordDetailModal");
    if (modal) modal.style.display = "none";
}

// --------------------------------------------------
// REGISTRATION AND EMPLOYEE MODALS
// --------------------------------------------------

function openRegisterModal() {
    const modal = document.getElementById("registerModal");

    if (modal) {
        modal.style.display = "flex";
    }

    switchPhotoTab("file");
}

function closeRegisterModal() {
    const modal = document.getElementById("registerModal");

    if (modal) {
        modal.style.display = "none";
    }

    stopWebcamStream();

    if (activePreviewUrl) {
        URL.revokeObjectURL(activePreviewUrl);
        activePreviewUrl = null;
    }

    capturedBlob = null;
}

function openEmployeeListModal() {
    const modal = document.getElementById("employeeModal");

    if (modal) {
        modal.style.display = "flex";
        loadEmployeeList();
    }
}

function closeModal(id) {
    const modal = document.getElementById(id);

    if (modal) {
        modal.style.display = "none";
    }
}

window.onclick = function (event) {
    if (!event.target || !event.target.classList.contains("modal")) {
        return;
    }

    if (event.target.id === "registerModal") {
        closeRegisterModal();
    } else if (event.target.id === "employeeModal") {
        closeModal("employeeModal");
    } else if (event.target.id === "photoComparisonModal") {
        closePhotoComparisonModal();
    }
};

// --------------------------------------------------
// PHOTO SOURCE TABS
// --------------------------------------------------

function switchPhotoTab(method) {
    currentPhotoMethod = method;

    const tabFile = document.getElementById("tabFile");
    const tabCamera = document.getElementById("tabCamera");
    const fileBox = document.getElementById("fileUploadBox");
    const cameraBox = document.getElementById("liveCameraBox");

    if (!tabFile || !tabCamera || !fileBox || !cameraBox) {
        console.error("[REGISTRATION] Photo tab HTML elements are missing.");
        return;
    }

    if (method === "file") {
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

// --------------------------------------------------
// REGISTRATION WEBCAM
// --------------------------------------------------

async function startWebcamStream() {
    capturedBlob = null;

    const video = document.getElementById("webcamVideo");
    const preview = document.getElementById("cameraPreview");
    const btnCapture = document.getElementById("btnCapture");
    const btnRetake = document.getElementById("btnRetake");

    if (!video || !preview || !btnCapture || !btnRetake) {
        console.error("[CAMERA] Webcam HTML elements are missing.");
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Your browser does not support webcam access.");
        switchPhotoTab("file");
        return;
    }

    video.style.display = "block";
    preview.style.display = "none";
    btnCapture.style.display = "inline-flex";
    btnRetake.style.display = "none";

    try {
        stopWebcamStream();

        liveCameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = liveCameraStream;
        await video.play();
    } catch (error) {
        alert(
            "Camera permission denied or camera unavailable: " +
            error.message
        );

        switchPhotoTab("file");
    }
}

function stopWebcamStream() {
    if (liveCameraStream) {
        liveCameraStream.getTracks().forEach(track => track.stop());
        liveCameraStream = null;
    }

    const video = document.getElementById("webcamVideo");

    if (video) {
        video.srcObject = null;
    }
}

function captureLiveSnapshot() {
    const video = document.getElementById("webcamVideo");
    const preview = document.getElementById("cameraPreview");
    const btnCapture = document.getElementById("btnCapture");
    const btnRetake = document.getElementById("btnRetake");

    if (!video || !preview || !btnCapture || !btnRetake) {
        alert("Webcam controls are missing from the page.");
        return;
    }

    if (!video.videoWidth || !video.videoHeight) {
        alert("Camera is not ready yet. Please wait and try again.");
        return;
    }

    const canvas = document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
        alert("Could not prepare the image canvas.");
        return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
        if (!blob) {
            alert("Could not capture the photo. Please try again.");
            return;
        }

        capturedBlob = blob;

        if (activePreviewUrl) {
            URL.revokeObjectURL(activePreviewUrl);
        }

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

    if (!video || !preview || !btnCapture || !btnRetake) {
        return;
    }

    video.style.display = "block";
    preview.style.display = "none";

    btnCapture.style.display = "inline-flex";
    btnRetake.style.display = "none";
}

// --------------------------------------------------
// EMPLOYEE REGISTRATION
// --------------------------------------------------

async function handleRegisterSubmit(event) {
    event.preventDefault();

    const empIdElement = document.getElementById("regEmpId");
    const nameElement = document.getElementById("regName");
    const fileInput = document.getElementById("regPhotoFile");
    const submitBtn = document.getElementById("btnRegisterSubmit");

    if (!empIdElement || !nameElement || !fileInput || !submitBtn) {
        alert("Registration form elements are missing.");
        return;
    }

    const empId = empIdElement.value.trim();
    const name = nameElement.value.trim();

    if (!empId || !name) {
        alert("Please enter both Employee ID and Name.");
        return;
    }

    let photoFiles = [];

    if (currentPhotoMethod === "file") {
        if (!fileInput.files || fileInput.files.length === 0) {
            alert("Please select at least one photo.");
            return;
        }

        photoFiles = Array.from(fileInput.files);
    } else {
        if (!capturedBlob) {
            alert("Please capture a photo before registering.");
            return;
        }

        photoFiles = [capturedBlob];
    }

    const formData = new FormData();

    formData.append("emp_id", empId);
    formData.append("name", name);

    photoFiles.forEach((photo, index) => {
        const filename =
            photo instanceof File && photo.name
                ? photo.name
                : `${empId}_face_${index + 1}.jpg`;

        formData.append("files", photo, filename);
    });

    submitBtn.textContent = "Registering Face...";
    submitBtn.disabled = true;

    try {
        const res = await fetch(
            `${BACKEND_BASE}/api/employees/register`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw new Error(
                data.detail ||
                data.message ||
                `Registration failed (HTTP ${res.status}).`
            );
        }

        alert(
            "Registered successfully: " +
            (data.message || "Employee registered.")
        );

        closeRegisterModal();

        const registerForm = document.getElementById("registerForm");

        if (registerForm) {
            registerForm.reset();
        }

        await fetchInitialLogs();
    } catch (error) {
        alert(`Registration failed: ${error.message || error}`);
        console.error("[REGISTRATION]", error);
    } finally {
        submitBtn.textContent = "Submit & Register Employee";
        submitBtn.disabled = false;
    }
}

// --------------------------------------------------
// CLEAR DASHBOARD AND RESTORE BACKUP
// --------------------------------------------------

async function clearDashboard() {
    const confirmed = confirm(
        "Are you sure? A complete backup will be saved before attendance records are cleared."
    );

    if (!confirmed) {
        return;
    }

    try {
        const res = await fetch(
            `${BACKEND_BASE}/api/attendance/clear-dashboard`,
            {
                method: "POST"
            }
        );

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw new Error(
                data.detail ||
                data.message ||
                `HTTP ${res.status}`
            );
        }

        globalRecords = [];

        renderFullTable(globalRecords);
        updateStats(globalRecords);

        alert(data.message || "Dashboard cleared successfully.");
    } catch (error) {
        alert(`Failed to clear dashboard: ${error.message || error}`);
    }
}

async function restoreBackup() {
    if (!confirm("Restore the latest attendance backup?")) {
        return;
    }

    try {
        const res = await fetch(
            `${BACKEND_BASE}/api/attendance/restore-latest-backup`,
            {
                method: "POST"
            }
        );

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw new Error(
                data.detail ||
                data.message ||
                `HTTP ${res.status}`
            );
        }

        await fetchInitialLogs();

        alert(data.message || "Latest backup restored successfully.");
    } catch (error) {
        alert(`Failed to restore backup: ${error.message || error}`);
    }
}

// --------------------------------------------------
// EMPLOYEE LIST
// --------------------------------------------------

async function loadEmployeeList() {
    const container = document.getElementById("employeeListContainer");

    if (!container) {
        console.error("[EMPLOYEES] Employee list container is missing.");
        return;
    }

    container.innerHTML = `
        <p style="color:var(--text-muted);">Loading employees...</p>
    `;

    try {
        const res = await fetch(`${BACKEND_BASE}/api/employees/list`);

        const employees = await res.json().catch(() => []);

        if (!res.ok) {
            throw new Error(`Employee list failed (HTTP ${res.status}).`);
        }

        if (!Array.isArray(employees) || employees.length === 0) {
            container.innerHTML = `
                <p style="color:var(--text-muted);padding:10px 0;">
                    No registered employees found.
                </p>
            `;
            return;
        }

        container.innerHTML = employees.map(emp => {
            const empId = String(emp.emp_id || "");
            const safeEmpId = escapeHtml(empId);
            const safeName = escapeHtml(emp.name || "");

            const uploadedPhoto = emp.photo_url
                ? getPhotoUrl(emp.photo_url)
                : DUMMY_AVATAR;

            return `
                <div
                    class="emp-item"
                    style="padding:10px 0;border-bottom:1px solid var(--border-color);display:flex;align-items:center;justify-content:space-between;"
                >
                    <div
                        class="emp-info"
                        style="display:flex;align-items:center;gap:12px;cursor:pointer;"
                        onclick="openEmployeeDetailModal('${safeEmpId}')"
                    >
                        <img
                            src="${uploadedPhoto}"
                            class="avatar-img"
                            title="Click to view registered and live photos"
                            onerror="this.onerror=null;this.src='${DUMMY_AVATAR}'"
                            alt="Employee photo"
                        >

                        <div>
                            <div style="font-weight:600;color:#f8fafc;">
                                ${safeName}
                            </div>

                            <div style="font-size:12px;color:var(--accent-blue);">
                                ${safeEmpId}
                                <span style="font-size:10px;color:var(--text-muted);">
                                    (Click to inspect)
                                </span>
                            </div>
                        </div>
                    </div>

                    <div style="display:flex;gap:8px;">
                        <button
                            type="button"
                            onclick="openEmployeeDetailModal('${safeEmpId}')"
                            style="background-color:#1e293b;color:#38bdf8;padding:6px 10px;font-size:11px;"
                        >
                            View Photos
                        </button>

                        <button
                            type="button"
                            onclick="deleteEmployee('${safeEmpId}')"
                            style="background-color:rgba(239,68,68,0.2);color:#f87171;padding:6px 10px;font-size:11px;"
                        >
                            Delete
                        </button>
                    </div>
                </div>
            `;
        }).join("");
    } catch (error) {
        console.error("[EMPLOYEES]", error);

        container.innerHTML = `
            <p style="color:#f87171;">
                Failed to load employees: ${escapeHtml(error.message || error)}
            </p>
        `;
    }
}

// --------------------------------------------------
// DELETE EMPLOYEE
// --------------------------------------------------

async function deleteEmployee(empId) {
    if (!confirm(`Permanently delete employee ${empId}?`)) {
        return;
    }

    try {
        const res = await fetch(
            `${BACKEND_BASE}/api/employees/${encodeURIComponent(empId)}`,
            {
                method: "DELETE"
            }
        );

        const data = await res.json().catch(() => ({}));

        if (!res.ok) {
            throw new Error(
                data.detail ||
                data.message ||
                `HTTP ${res.status}`
            );
        }

        alert(data.message || "Employee deleted.");

        await loadEmployeeList();
        await fetchInitialLogs();
    } catch (error) {
        alert(`Failed to delete employee: ${error.message || error}`);
    }
}

// --------------------------------------------------
// SHARED HELPERS
// --------------------------------------------------

function getPhotoUrl(value) {
    if (!value) {
        return DUMMY_AVATAR;
    }

    const path = String(value).trim();

    if (/^https?:\/\//i.test(path) || path.startsWith("data:")) {
        return path;
    }

    return `${BACKEND_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

function escapeHtml(value) {
    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

// --------------------------------------------------
// APPLICATION INITIALIZATION
// --------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    loadClientConfig();
    fetchInitialLogs();
    initWebSocket();
});
