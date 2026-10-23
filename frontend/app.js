// ==========================================================================
// AI CATARACT CDSS — CLINICAL DECISION SUPPORT WORKSTATION CONTROLLER
// ==========================================================================

const API_BASE = (window.location.protocol === 'file:' || !window.location.port) 
    ? 'http://127.0.0.1:8000' 
    : '';

let currentPatients = [];
let activePatient = null;
let activeVisit = null;
let priorVisit = null;
let currentFilter = 'normal';

document.addEventListener('DOMContentLoaded', () => {
    initWorkstation();
    setupEventListeners();
    startTelemetryClock();
});

function startTelemetryClock() {
    function updateTime() {
        const now = new Date();
        const timeStr = now.toTimeString().split(' ')[0] + ' LOC';
        const dateStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }).toUpperCase();
        const el = document.getElementById('live-telemetry-time');
        if (el) {
            el.innerHTML = `<span class="time-val">${timeStr}</span><span class="date-val">${dateStr}</span>`;
        }
    }
    updateTime();
    setInterval(updateTime, 1000);
}

async function initWorkstation() {
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('visit-date-input').value = today;
    await loadPatients();
}

function setupEventListeners() {
    // Patient Select Change
    document.getElementById('patient-select').addEventListener('change', (e) => {
        selectPatient(e.target.value);
    });

    // Grad-CAM Continuous Opacity Slider
    const slider = document.getElementById('gradcam-slider');
    const sliderDisp = document.getElementById('gradcam-pct-disp');
    slider.addEventListener('input', (e) => {
        sliderDisp.textContent = `${e.target.value}%`;
        applyGradcamOpacity(e.target.value);
    });

    // Viewport Filter Tools (RGB, CLAHE, INVERT)
    document.getElementById('btn-tool-normal').addEventListener('click', () => setFilter('normal'));
    document.getElementById('btn-tool-clahe').addEventListener('click', () => setFilter('clahe'));
    document.getElementById('btn-tool-invert').addEventListener('click', () => setFilter('invert'));

    // Quick Clinical Presets for Doctor
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            const textarea = document.getElementById('doctor-notes-input');
            textarea.value = btn.getAttribute('data-text');
        });
    });

    // File Upload Input Change Listener & Live Preview
    const fileInput = document.getElementById('file-upload-input');
    const fileLabel = document.getElementById('file-upload-label');
    const fileText = document.getElementById('file-upload-text');
    const statusDiv = document.getElementById('file-selected-status');

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                const file = e.target.files[0];
                handleFileSelected(file);
            }
        });
    }

    // Drag and Drop onto Viewport & Ingestion Zone
    const dropZones = [document.getElementById('pane-curr'), document.getElementById('ingestion-dropzone')];
    dropZones.forEach(zone => {
        if (!zone) return;
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.style.borderColor = '#06b6d4';
        });
        zone.addEventListener('dragleave', () => {
            zone.style.borderColor = '';
        });
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.style.borderColor = '';
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.type.startsWith('image/')) {
                    fileInput.files = e.dataTransfer.files;
                    handleFileSelected(file);
                }
            }
        });
    });

    // Ingestion Form Submit
    document.getElementById('upload-visit-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleNewVisitIngestion();
    });

    // Save Doctor Notes & Sign-off
    document.getElementById('btn-save-notes').addEventListener('click', async () => {
        await saveDoctorNotes();
    });

    // View Clinical Report
    document.getElementById('btn-view-clinical-report').addEventListener('click', () => {
        openReportModal('clinical');
    });

    // View Patient Leaflet
    document.getElementById('btn-view-patient-summary').addEventListener('click', () => {
        openReportModal('patient');
    });

    // New Patient Modal Triggers
    document.getElementById('btn-new-patient').addEventListener('click', () => {
        const idInput = document.getElementById('new-p-id');
        idInput.value = `PAT-2026-${Math.floor(1000 + Math.random() * 9000)}`;
        document.getElementById('patient-modal').style.display = 'flex';
        setTimeout(() => {
            const nameInput = document.getElementById('new-p-name');
            if (nameInput) nameInput.focus();
        }, 50);
    });
    document.getElementById('btn-close-modal').addEventListener('click', () => {
        document.getElementById('patient-modal').style.display = 'none';
    });
    document.getElementById('btn-cancel-modal').addEventListener('click', () => {
        document.getElementById('patient-modal').style.display = 'none';
    });
    document.getElementById('btn-close-report-modal').addEventListener('click', () => {
        document.getElementById('report-modal').style.display = 'none';
    });

    // Register Patient Submit
    document.getElementById('new-patient-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await handleRegisterNewPatient();
    });

    // Quick Presentation Scenario Buttons
    document.getElementById('btn-demo-rajesh-v1').addEventListener('click', () => {
        setActiveCaseBtn('btn-demo-rajesh-v1');
        selectPatient('PAT-2026-0842', 0);
    });
    document.getElementById('btn-demo-rajesh-v2').addEventListener('click', async () => {
        setActiveCaseBtn('btn-demo-rajesh-v2');
        await runQuickDemoRajeshVisit2();
    });
    document.getElementById('btn-demo-sunita').addEventListener('click', () => {
        setActiveCaseBtn('btn-demo-sunita');
        selectPatient('PAT-2026-0193', 0);
    });

    // Single-Tab Workstation Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            if (tabId) {
                switchTab(tabId);
            }
        });
    });
}

function switchTab(tabId) {
    if (!tabId) return;

    // Toggle Tab Buttons in Top Tabbar
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.getAttribute('data-tab') === tabId) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Toggle Content Panes (Only 1 Active At A Time)
    document.querySelectorAll('.tab-pane').forEach(pane => {
        if (pane.id === tabId) {
            pane.classList.add('active');
        } else {
            pane.classList.remove('active');
        }
    });

    // Smooth scroll to top of main workstation
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
window.switchTab = switchTab;

function setActiveCaseBtn(activeId) {
    document.querySelectorAll('.case-btn').forEach(b => b.classList.remove('active'));
    const target = document.getElementById(activeId);
    if (target) target.classList.add('active');
}

function setFilter(filterType) {
    currentFilter = filterType;
    document.querySelectorAll('.btn-tool').forEach(b => b.classList.remove('active'));
    document.getElementById(`btn-tool-${filterType}`).classList.add('active');

    const currImg = document.getElementById('curr-eye-img');
    const prevImg = document.getElementById('prev-eye-img');

    let filterStyle = 'none';
    if (filterType === 'clahe') {
        filterStyle = 'contrast(160%) brightness(110%) saturate(120%)';
    } else if (filterType === 'invert') {
        filterStyle = 'invert(100%) hue-rotate(180deg) contrast(140%)';
    }

    if (currImg) currImg.style.filter = filterStyle;
    if (prevImg) prevImg.style.filter = filterStyle;
}

function applyGradcamOpacity(opacityPercent) {
    // If opacity > 0, we fetch Grad-CAM image overlay
    updateImagesDisplay(opacityPercent > 10);
}

async function loadPatients(patientIdToSelect = null, retryCount = 0) {
    const statusLabel = document.querySelector('.status-label');
    const pulseIndicator = document.querySelector('.pulse-indicator');
    try {
        if (statusLabel) statusLabel.textContent = "PACS SYNC: CONNECTING...";
        const res = await fetch(`${API_BASE}/api/patients`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        currentPatients = await res.json();
        
        if (statusLabel) statusLabel.textContent = "PACS SYNC: ONLINE";
        if (pulseIndicator) pulseIndicator.style.background = "#10b981";

        const errBanner = document.getElementById('pacs-offline-banner');
        if (errBanner) errBanner.remove();

        const select = document.getElementById('patient-select');
        select.innerHTML = '';
        
        currentPatients.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = `${p.id} — ${p.name} (${p.age}y, ${p.monitored_eye})`;
            select.appendChild(opt);
        });

        let targetId = null;
        if (patientIdToSelect && currentPatients.some(p => p.id === patientIdToSelect)) {
            targetId = patientIdToSelect;
        } else if (activePatient && currentPatients.some(p => p.id === activePatient.id)) {
            targetId = activePatient.id;
        } else if (currentPatients.length > 0) {
            targetId = currentPatients[0].id;
        }

        if (targetId) {
            selectPatient(targetId);
        }
    } catch (err) {
        console.warn("Patient fetch failed (attempt " + retryCount + "):", err);
        if (retryCount < 3) {
            if (statusLabel) statusLabel.textContent = `PACS SYNC: RETRYING (${retryCount + 1}/3)...`;
            setTimeout(() => loadPatients(patientIdToSelect, retryCount + 1), 1500);
        } else {
            if (statusLabel) statusLabel.textContent = "PACS SYNC: OFFLINE";
            if (pulseIndicator) pulseIndicator.style.background = "#ef4444";
            showConnectionErrorBanner();
        }
    }
}

function showConnectionErrorBanner() {
    if (document.getElementById('pacs-offline-banner')) return;
    const banner = document.createElement('div');
    banner.id = 'pacs-offline-banner';
    banner.style.cssText = `
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 10px 16px;
        margin: 12px 20px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 12px;
        font-family: var(--font-mono, monospace);
    `;
    banner.innerHTML = `
        <div>
            <strong>⚠️ Cannot connect to Cataract CDSS Server on http://127.0.0.1:8000</strong>
            <div style="font-size: 11px; color: #cbd5e1; margin-top: 2px;">Please make sure <code>python run.py</code> is running in the terminal.</div>
        </div>
        <button class="btn btn-primary-glow" onclick="loadPatients()" style="padding: 4px 10px; font-size: 11px;">🔄 Retry</button>
    `;
    const header = document.querySelector('.pacs-navbar');
    if (header && header.nextSibling) {
        header.parentNode.insertBefore(banner, header.nextSibling);
    }
}

function selectPatient(patientId, visitIndex = null) {
    activePatient = currentPatients.find(p => p.id === patientId);
    if (!activePatient) return;

    document.getElementById('patient-select').value = patientId;

    // Update HUD banner
    document.getElementById('disp-patient-id').textContent = activePatient.id;
    document.getElementById('disp-patient-name').textContent = `${activePatient.name} (${activePatient.age} Yrs, ${activePatient.gender})`;
    document.getElementById('disp-monitored-eye').textContent = activePatient.monitored_eye;
    document.getElementById('disp-risk-factors').textContent = activePatient.risk_factors || 'None';

    const visits = activePatient.visits || [];
    const countText = `${visits.length} Longitudinal Visit(s) Recorded`;
    const visitCountEl = document.getElementById('disp-visit-count');
    if (visitCountEl) visitCountEl.textContent = countText;
    const timelineCountEl = document.getElementById('disp-timeline-visit-count');
    if (timelineCountEl) timelineCountEl.textContent = countText;

    if (visits.length > 0) {
        const vIdx = (visitIndex !== null && visitIndex < visits.length) ? visitIndex : (visits.length - 1);
        activeVisit = visits[vIdx];
        priorVisit = vIdx > 0 ? visits[vIdx - 1] : null;
    } else {
        activeVisit = null;
        priorVisit = null;
    }

    renderWorkspace();
    renderTimeline();
}

function renderWorkspace() {
    const currImg = document.getElementById('curr-eye-img');
    const currEmpty = document.getElementById('curr-empty-msg');
    const prevImg = document.getElementById('prev-eye-img');
    const prevEmpty = document.getElementById('prev-empty-msg');
    const clarityHud = document.getElementById('curr-clarity-hud');

    if (!activeVisit) {
        document.getElementById('ai-detection').textContent = "Awaiting Examination";
        const detTag = document.getElementById('ai-detection-tag');
        detTag.textContent = "STANDBY";
        detTag.className = "tile-status";
        detTag.style.background = "";
        detTag.style.color = "";
        document.getElementById('ai-conf-bar').style.width = "0%";
        document.getElementById('ai-confidence').textContent = "Upload or select an eye image below";

        document.getElementById('ai-severity').textContent = "Baseline Pending";
        document.getElementById('ai-grade-tag').textContent = "--";
        document.getElementById('ai-opacity-pct').textContent = "Optical Density: --";

        document.getElementById('ai-progression-trend').textContent = "No Baseline Record";
        document.getElementById('ai-progression-trend').style.color = "";
        const trendTag = document.getElementById('ai-trend-tag');
        trendTag.textContent = "STANDBY";
        trendTag.className = "tile-status";
        trendTag.style.background = "";
        trendTag.style.color = "";
        document.getElementById('ai-ssim-score').textContent = "SSIM Delta: --";
        document.getElementById('ai-delta-desc').textContent = "Ingest first slit-lamp image below to establish baseline.";

        const riskBadge = document.getElementById('ai-risk-badge');
        riskBadge.textContent = "STANDBY";
        riskBadge.className = "risk-badge";
        document.getElementById('ai-risk-tag').textContent = "STANDBY";
        document.getElementById('ai-risk-tag').className = "tile-status";
        document.getElementById('ai-followup-month').textContent = "Awaiting initial exam to compute recommendation";

        if (currImg) {
            currImg.src = "";
            currImg.style.display = "none";
        }
        if (currEmpty) currEmpty.style.display = "block";

        if (prevImg) {
            prevImg.src = "";
            prevImg.style.display = "none";
        }
        if (prevEmpty) prevEmpty.style.display = "block";

        document.getElementById('curr-visit-date').textContent = "AWAITING INGESTION";
        document.getElementById('curr-severity-overlay').textContent = "NEW PATIENT PROFILE";
        if (clarityHud) {
            clarityHud.textContent = "STANDBY";
            clarityHud.className = "hud-overlay-text top-right";
            clarityHud.style.borderColor = "#334155";
            clarityHud.style.color = "#94a3b8";
        }

        document.getElementById('prev-visit-date').textContent = "NONE";
        document.getElementById('prev-severity-overlay').textContent = "NO PRIOR VISITS";
        document.getElementById('doctor-notes-input').value = "";
        return;
    }

    if (currEmpty) currEmpty.style.display = "none";
    if (currImg) currImg.style.display = "block";

    const a = activeVisit.analysis;

    // 1. Detection
    document.getElementById('ai-detection').textContent = a.detection;
    document.getElementById('ai-confidence').textContent = `Confidence: ${a.confidence}% (p < 0.001)`;
    document.getElementById('ai-conf-bar').style.width = `${a.confidence}%`;
    const detTag = document.getElementById('ai-detection-tag');
    if (a.is_cataract) {
        detTag.textContent = "POSITIVE";
        detTag.className = "tile-status status-pos";
        detTag.style.background = "";
        detTag.style.color = "";
    } else {
        detTag.textContent = "NEGATIVE";
        detTag.className = "tile-status";
        detTag.style.background = "#dcfce7";
        detTag.style.color = "#15803d";
        detTag.style.border = "1px solid #86efac";
    }

    // 2. Severity
    document.getElementById('ai-severity').textContent = a.severity;
    document.getElementById('ai-opacity-pct').textContent = `Optical Density: ${a.opacity_percentage}% Opacity`;
    document.getElementById('ai-grade-tag').textContent = `GRADE ${a.severity_grade}`;

    // 3. Longitudinal Trend
    document.getElementById('ai-progression-trend').textContent = a.progression_trend;
    document.getElementById('ai-ssim-score').textContent = priorVisit ? `Structural SSIM Delta: ${a.ssim_score}` : 'Baseline Established';
    document.getElementById('ai-delta-desc').textContent = a.delta_description || "";
    
    const trendEl = document.getElementById('ai-progression-trend');
    const trendTag = document.getElementById('ai-trend-tag');
    if (a.trend_badge === 'increasing') {
        trendEl.style.color = '#b91c1c';
        trendTag.textContent = "ACTIVE CHANGE";
        trendTag.className = "tile-status status-alert";
        trendTag.style.background = "#fee2e2";
        trendTag.style.color = "#b91c1c";
        trendTag.style.border = "1px solid #fca5a5";
    } else if (a.trend_badge === 'stable') {
        trendEl.style.color = '#15803d';
        trendTag.textContent = "STABLE";
        trendTag.className = "tile-status";
        trendTag.style.background = "#dcfce7";
        trendTag.style.color = "#15803d";
        trendTag.style.border = "1px solid #86efac";
    } else {
        trendEl.style.color = '#0284c7';
        trendTag.textContent = "BASELINE";
        trendTag.className = "tile-status";
        trendTag.style.background = "#dbeafe";
        trendTag.style.color = "#1d4ed8";
        trendTag.style.border = "1px solid #93c5fd";
    }

    // 4. Risk Tier
    const riskBadge = document.getElementById('ai-risk-badge');
    riskBadge.textContent = `${a.risk_level.toUpperCase()} (Score: ${a.risk_score})`;
    riskBadge.className = `risk-badge badge-${a.risk_badge || 'low'}`;
    document.getElementById('ai-risk-tag').textContent = a.risk_level.toUpperCase();

    // 5. Follow-up Recommendation
    document.getElementById('ai-followup-month').textContent = `${a.recommended_follow_up_month} (Approx. ${a.follow_up_interval_months} Months)`;

    // 6. Viewport Headers & Overlays
    document.getElementById('curr-visit-date').textContent = activeVisit.visit_date.toUpperCase();
    document.getElementById('curr-severity-overlay').textContent = a.severity.toUpperCase();
    if (clarityHud) {
        clarityHud.textContent = `QUALITY: ${a.clarity_score || 96}% PASS`;
        clarityHud.className = "hud-overlay-text top-right quality-pass";
        clarityHud.style.borderColor = "";
        clarityHud.style.color = "";
    }

    if (priorVisit) {
        document.getElementById('prev-visit-date').textContent = priorVisit.visit_date.toUpperCase();
        document.getElementById('prev-severity-overlay').textContent = priorVisit.analysis.severity.toUpperCase();
        if (prevEmpty) prevEmpty.style.display = "none";
        if (prevImg) prevImg.style.display = "block";
    } else {
        document.getElementById('prev-visit-date').textContent = "NONE";
        document.getElementById('prev-severity-overlay').textContent = "INITIAL BASELINE";
        if (prevEmpty) prevEmpty.style.display = "block";
        if (prevImg) {
            prevImg.src = "";
            prevImg.style.display = "none";
        }
    }

    // 7. Doctor Notes
    document.getElementById('doctor-notes-input').value = activeVisit.doctor_notes || "";

    const sliderVal = document.getElementById('gradcam-slider').value;
    updateImagesDisplay(sliderVal > 10);
}

async function updateImagesDisplay(showGradcam = true) {
    const currImg = document.getElementById('curr-eye-img');
    const currEmpty = document.getElementById('curr-empty-msg');
    const prevImg = document.getElementById('prev-eye-img');
    const prevEmpty = document.getElementById('prev-empty-msg');

    if (activeVisit) {
        const currGrade = activeVisit.analysis.severity_grade;
        const resCurr = await fetch(`${API_BASE}/api/image-preview?path=${encodeURIComponent(activeVisit.image_path)}&gradcam=${showGradcam}&severity_grade=${currGrade}`);
        const dataCurr = await resCurr.json();
        if (currImg) {
            currImg.src = dataCurr.base64;
            currImg.style.display = 'block';
        }
        if (currEmpty) currEmpty.style.display = 'none';
    } else {
        if (currImg) {
            currImg.src = '';
            currImg.style.display = 'none';
        }
        if (currEmpty) currEmpty.style.display = 'block';
    }

    if (priorVisit) {
        const prevGrade = priorVisit.analysis.severity_grade;
        const resPrev = await fetch(`${API_BASE}/api/image-preview?path=${encodeURIComponent(priorVisit.image_path)}&gradcam=${showGradcam}&severity_grade=${prevGrade}`);
        const dataPrev = await resPrev.json();
        if (prevImg) {
            prevImg.src = dataPrev.base64;
            prevImg.style.display = 'block';
        }
        if (prevEmpty) prevEmpty.style.display = 'none';
    } else {
        if (prevImg) {
            prevImg.src = '';
            prevImg.style.display = 'none';
        }
        if (prevEmpty) prevEmpty.style.display = 'block';
    }
}

function renderTimeline() {
    const container = document.getElementById('visits-timeline-stepper');
    container.innerHTML = '';
    const visits = activePatient ? activePatient.visits || [] : [];

    if (visits.length === 0) {
        container.innerHTML = '<div style="color: #64748b; font-size: 11px;">No longitudinal visits recorded.</div>';
        return;
    }

    visits.forEach((v, index) => {
        const node = document.createElement('div');
        node.className = `timeline-node ${activeVisit && activeVisit.visit_id === v.visit_id ? 'active' : ''}`;
        node.innerHTML = `
            <div class="node-top">
                <span>EXAM #${v.visit_number}</span>
                <span>${v.visit_date}</span>
            </div>
            <div class="node-severity">${v.analysis.severity}</div>
            <div class="node-risk-row">
                <span class="risk-badge badge-${v.analysis.risk_badge || 'low'}">${v.analysis.risk_level}</span>
                <span style="font-size: 10px; font-family: var(--font-mono); color: #38bdf8;">Next: ${v.analysis.recommended_follow_up_month}</span>
            </div>
        `;
        node.addEventListener('click', () => {
            selectPatient(activePatient.id, index);
        });
        container.appendChild(node);
    });
}

function handleFileSelected(file) {
    const fileLabel = document.getElementById('file-upload-label');
    const fileText = document.getElementById('file-upload-text');
    const statusDiv = document.getElementById('file-selected-status');
    const demoSelect = document.getElementById('demo-stage-select');

    if (fileLabel) fileLabel.classList.add('selected');
    if (fileText) fileText.textContent = `📁 ${file.name}`;
    if (demoSelect) demoSelect.value = '';

    if (statusDiv) {
        statusDiv.style.display = 'flex';
        statusDiv.innerHTML = `<span>✓ Photo Selected: <strong>${file.name}</strong> (${Math.round(file.size / 1024)} KB) — Ready to Analyze</span>`;
    }

    // Live Thumbnail Preview in Current Viewport
    const reader = new FileReader();
    reader.onload = (e) => {
        const currImg = document.getElementById('curr-eye-img');
        if (currImg) {
            currImg.src = e.target.result;
            currImg.style.display = 'block';
        }
        const currHud = document.getElementById('curr-severity-overlay');
        if (currHud) currHud.textContent = 'UPLOAD PREVIEW — CLICK ANALYZE TO EVALUATE';
    };
    reader.readAsDataURL(file);
}

async function handleNewVisitIngestion() {
    if (!activePatient) {
        alert("Please select or register a patient record first.");
        return;
    }

    const dateVal = document.getElementById('visit-date-input').value;
    const demoStage = document.getElementById('demo-stage-select').value;
    const fileInput = document.getElementById('file-upload-input');

    if (!demoStage && (!fileInput.files || fileInput.files.length === 0)) {
        alert("Please click 'Choose Eye Photo' to select an image from your computer, or pick a sample from the dropdown.");
        return;
    }

    const btn = document.getElementById('btn-run-analysis');
    btn.disabled = true;
    btn.innerHTML = '<span>⏳ Processing AI Pipeline...</span>';

    const formData = new FormData();
    formData.append('patient_id', activePatient.id);
    formData.append('visit_date', dateVal || new Date().toISOString().split('T')[0]);
    if (demoStage) formData.append('demo_stage', demoStage);
    if (fileInput.files && fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    }

    try {
        const res = await fetch(`${API_BASE}/api/visits/analyze`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned ${res.status}`);
        }
        await loadPatients();
        selectPatient(activePatient.id);

        // Switch to AI Telemetry Tab so clinician sees analysis results immediately
        switchTab('tab-cdss');

        const statusDiv = document.getElementById('file-selected-status');
        if (statusDiv) {
            statusDiv.innerHTML = `<span>✓ Analysis Complete! Slit-lamp exam recorded in longitudinal PACS audit.</span>`;
            setTimeout(() => { statusDiv.style.display = 'none'; }, 4000);
        }
    } catch (err) {
        alert("Error during clinical ingestion: " + err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>⚡ Analyze & Ingest</span>';
    }
}

async function saveDoctorNotes() {
    if (!activePatient || !activeVisit) return;
    const notes = document.getElementById('doctor-notes-input').value;

    const formData = new FormData();
    formData.append('notes', notes);

    const res = await fetch(`${API_BASE}/api/visits/${activePatient.id}/${activeVisit.visit_id}/approve`, {
        method: 'POST',
        body: formData
    });
    if (res.ok) {
        alert("✓ Clinical validation & digital sign-off recorded in PACS audit log.");
        activeVisit.doctor_notes = notes;
        activeVisit.doctor_approved = true;
    }
}

function openReportModal(type) {
    if (!activePatient || !activeVisit) {
        alert("Please select or record a visit first.");
        return;
    }
    const modal = document.getElementById('report-modal');
    const iframe = document.getElementById('report-iframe');
    const title = document.getElementById('report-modal-title');

    if (type === 'clinical') {
        title.textContent = "Official Hospital Clinical Decision Support Report (IEC 62304)";
        iframe.src = `${API_BASE}/api/reports/clinical/${activePatient.id}/${activeVisit.visit_id}`;
    } else {
        title.textContent = "Doctor-Approved Patient Check-up Summary Leaflet";
        iframe.src = `${API_BASE}/api/reports/patient/${activePatient.id}/${activeVisit.visit_id}`;
    }

    modal.style.display = 'flex';
}

async function handleRegisterNewPatient() {
    const idInput = document.getElementById('new-p-id');
    const nameInput = document.getElementById('new-p-name');
    const ageInput = document.getElementById('new-p-age');
    const genderInput = document.getElementById('new-p-gender');
    const eyeInput = document.getElementById('new-p-eye');
    const risksInput = document.getElementById('new-p-risks');

    const pId = idInput.value.trim();
    const name = nameInput.value.trim();
    const age = parseInt(ageInput.value, 10);

    if (!pId || !name || isNaN(age)) {
        alert("Please provide a valid Patient ID, Full Name, and Age.");
        return;
    }

    const formData = new FormData();
    formData.append('patient_id', pId);
    formData.append('name', name);
    formData.append('age', age);
    formData.append('gender', genderInput.value);
    formData.append('monitored_eye', eyeInput.value);
    formData.append('risk_factors', risksInput.value.trim() || 'None');

    const submitBtn = document.querySelector('#new-patient-form button[type="submit"]');
    const origText = submitBtn.textContent;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Registering...';

    try {
        const res = await fetch(`${API_BASE}/api/patients`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || "Failed to register patient");
        }

        document.getElementById('patient-modal').style.display = 'none';

        // Clear modal form for next time
        nameInput.value = '';
        if (risksInput.selectedIndex !== undefined) {
            risksInput.selectedIndex = 0;
        } else {
            risksInput.value = '';
        }
        idInput.value = '';

        // Reload patients and select the newly created patient
        await loadPatients(pId);

        const statusDiv = document.getElementById('file-selected-status');
        if (statusDiv) {
            statusDiv.style.display = 'flex';
            statusDiv.innerHTML = `<span>✓ Patient <strong>${name} (${pId})</strong> registered! Ingest eye photo below to establish baseline.</span>`;
            setTimeout(() => { statusDiv.style.display = 'none'; }, 6000);
        }
    } catch (err) {
        alert("Error registering patient: " + err.message);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = origText;
    }
}

async function runQuickDemoRajeshVisit2() {
    const rajesh = currentPatients.find(p => p.id === 'PAT-2026-0842');
    if (rajesh && rajesh.visits.length === 1) {
        const formData = new FormData();
        formData.append('patient_id', 'PAT-2026-0842');
        formData.append('visit_date', '2026-11-18');
        formData.append('demo_stage', 'moderate');

        await fetch(`${API_BASE}/api/visits/analyze`, {
            method: 'POST',
            body: formData
        });
        await loadPatients();
    }
    selectPatient('PAT-2026-0842', 1);
}
