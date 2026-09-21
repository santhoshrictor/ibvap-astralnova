/**
 * IBVAP - Client-Side Surveillance Telemetry & AI Simulation Engine
 * 100% Standalone / Serverless for Vercel Static Hosting
 */

(function () {
  'use strict';

  // --- DOM References ---
  const statFps = document.getElementById('statFps');
  const statLatency = document.getElementById('statLatency');
  const statTargets = document.getElementById('statTargets');
  const statPersons = document.getElementById('statPersons');
  const statVehicles = document.getElementById('statVehicles');
  const statPlates = document.getElementById('statPlates');
  const statFaces = document.getElementById('statFaces');
  const statBreaches = document.getElementById('statBreaches');
  const statBreachSub = document.getElementById('statBreachSub');

  const cmdDefenseStatus = document.getElementById('cmdDefenseStatus');
  const cmdThreatLevel = document.getElementById('cmdThreatLevel');
  const cmdGpuName = document.getElementById('cmdGpuName');
  const defenseStatusHeader = document.getElementById('defenseStatusHeader');
  const clockDisplay = document.getElementById('clockDisplay');

  const anprRegistryBody = document.getElementById('anprRegistryBody');
  const anprCounterBadge = document.getElementById('anprCounterBadge');

  const cameraContainer = document.getElementById('cameraContainer');
  const btnLayout3Col = document.getElementById('btnLayout3Col');
  const btnLayoutFocus = document.getElementById('btnLayoutFocus');
  const auditTerminal = document.getElementById('auditTerminal');

  // --- State Variables ---
  let targetCount = 14;
  let personCount = 5;
  let vehicleCount = 3;
  let faceCount = 2;
  let breachCount = 0;
  let totalPlatesScanned = 16;
  let isBreachActive = false;
  let breachTimeout = null;

  const anprRegistryData = [];

  // Realistic License Plate Roster
  const plateDatabase = [
    { plate: 'TN-09-AB-1234', cam: 'CAM 2 // OUT POST', status: 'CLEARED' },
    { plate: 'DL-01-CX-4091', cam: 'CAM 2 // OUT POST', status: 'FLAGGED' },
    { plate: 'KA-05-MK-9821', cam: 'CAM 2 // OUT POST', status: 'CLEARED' },
    { plate: 'MH-12-PQ-5544', cam: 'CAM 2 // OUT POST', status: 'CLEARED' },
    { plate: 'HR-26-DK-7712', cam: 'CAM 1 // HAWKINS',  status: 'CLEARED' },
    { plate: 'GJ-01-EF-8822', cam: 'CAM 2 // OUT POST', status: 'CLEARED' },
    { plate: 'WB-02-AZ-3319', cam: 'CAM 2 // OUT POST', status: 'FLAGGED' },
    { plate: 'KL-07-CD-9011', cam: 'CAM 3 // ALPHA',    status: 'CLEARED' },
    { plate: 'UP-16-BR-9900', cam: 'CAM 2 // OUT POST', status: 'CLEARED' },
    { plate: 'TS-08-GH-6655', cam: 'CAM 2 // OUT POST', status: 'CLEARED' },
    { plate: 'RJ-14-XY-1008', cam: 'CAM 1 // HAWKINS',  status: 'FLAGGED' },
    { plate: 'CH-01-TA-4432', cam: 'CAM 2 // OUT POST', status: 'CLEARED' },
    { plate: 'AP-09-BC-7819', cam: 'CAM 2 // OUT POST', status: 'CLEARED' }
  ];

  let platePoolIndex = 0;

  // --- Real-Time UTC Clock ---
  function updateClock() {
    const now = new Date();
    const utcStr = now.toISOString().substring(11, 19);
    if (clockDisplay) {
      clockDisplay.textContent = `${utcStr} UTC`;
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // --- Audit Logger ---
  function logEvent(tag, msg, level = 'sys') {
    if (!auditTerminal) return;
    const now = new Date();
    const ts = now.toISOString().substring(11, 19);
    const div = document.createElement('div');
    div.className = 'log-line';
    div.innerHTML = `<span class="log-ts">[${ts}]</span><span class="log-tag ${level}">[${tag}]</span> ${msg}`;
    auditTerminal.prepend(div);

    // Keep buffer to max 60 items
    while (auditTerminal.children.length > 60) {
      auditTerminal.removeChild(auditTerminal.lastChild);
    }
  }

  // --- ANPR Registry Simulation ---
  function addAnprRecord(plateStr, camTag, statusStr) {
    totalPlatesScanned++;
    if (statPlates) statPlates.textContent = `PLATES: ${totalPlatesScanned} SCANNED`;

    const now = new Date();
    const ts = now.toISOString().substring(11, 19);

    const record = {
      timestamp: ts,
      plate: plateStr,
      camera: camTag,
      status: statusStr,
      isNew: true
    };

    anprRegistryData.unshift(record);
    if (anprRegistryData.length > 50) {
      anprRegistryData.pop();
    }

    renderAnprRegistry();

    const tagType = statusStr === 'FLAGGED' ? 'breach' : 'anpr';
    logEvent('ANPR', `Plate scan [${plateStr}] status: ${statusStr} @ ${camTag}`, tagType);

    if (statusStr === 'FLAGGED') {
      triggerPerimeterBreach(`ANPR Watchlist Match: [${plateStr}]`);
    }
  }

  function renderAnprRegistry() {
    if (!anprRegistryBody) return;

    if (anprRegistryData.length === 0) {
      anprRegistryBody.innerHTML = `
        <tr class="anpr-empty-row">
          <td colspan="3">AWAITING VEHICLE PASS-THROUGH...</td>
        </tr>`;
      if (anprCounterBadge) anprCounterBadge.textContent = '0 DETECTIONS';
      return;
    }

    if (anprCounterBadge) {
      anprCounterBadge.textContent = `${anprRegistryData.length} DETECTIONS`;
    }

    anprRegistryBody.innerHTML = anprRegistryData.map(entry => {
      const isFlagged = entry.status === 'FLAGGED';
      const statusClass = isFlagged ? 'flagged' : 'cleared';
      const animClass = entry.isNew ? 'new-entry' : '';
      entry.isNew = false;

      return `
        <tr class="${animClass}">
          <td class="anpr-ts">${entry.timestamp}</td>
          <td>
            <span class="anpr-plate-badge">
              ${entry.plate}
              <span class="anpr-cam-pill">${entry.camera}</span>
            </span>
          </td>
          <td style="text-align: right;">
            <span class="anpr-status ${statusClass}">${entry.status}</span>
          </td>
        </tr>`;
    }).join('');
  }

  // Initial Seed Plates
  function seedInitialPlates() {
    for (let i = 0; i < 4; i++) {
      const p = plateDatabase[i % plateDatabase.length];
      const now = new Date(Date.now() - (4 - i) * 15000);
      anprRegistryData.unshift({
        timestamp: now.toISOString().substring(11, 19),
        plate: p.plate,
        camera: p.cam,
        status: p.status,
        isNew: false
      });
    }
    platePoolIndex = 4;
    renderAnprRegistry();
  }

  // --- Threat Escalation & Breach Simulation ---
  function triggerPerimeterBreach(reason) {
    isBreachActive = true;
    breachCount++;
    if (statBreaches) statBreaches.textContent = breachCount;
    if (statBreachSub) statBreachSub.textContent = 'ALERT: ACTIVE PERIMETER BREACH';

    if (cmdDefenseStatus) {
      cmdDefenseStatus.className = 'cmd-val breach';
      cmdDefenseStatus.innerHTML = '<span class="dot-live" style="background:var(--rd);box-shadow:0 0 8px var(--rd);"></span> DEFCON 2 // BREACH DETECTED';
    }

    if (cmdThreatLevel) {
      cmdThreatLevel.innerHTML = '<span style="color: var(--rd); font-weight:800;">DEFCON 2 // CRITICAL ALERT</span>';
    }

    if (defenseStatusHeader) {
      defenseStatusHeader.textContent = 'DEFCON 2 // BREACH';
      defenseStatusHeader.style.color = 'var(--rd)';
    }

    logEvent('BREACH', `CRITICAL: ${reason}`, 'breach');

    clearTimeout(breachTimeout);
    breachTimeout = setTimeout(() => {
      resetPerimeterStatus();
    }, 6000);
  }

  function resetPerimeterStatus() {
    isBreachActive = false;
    if (statBreachSub) statBreachSub.textContent = 'VIRTUAL TRIPWIRE & CLIMB';

    if (cmdDefenseStatus) {
      cmdDefenseStatus.className = 'cmd-val armed';
      cmdDefenseStatus.innerHTML = '<span class="dot-live"></span> ARMED // ACTIVE WATCH';
    }

    if (cmdThreatLevel) {
      cmdThreatLevel.innerHTML = '<span style="color: var(--ac);">DEFCON 4 // GUARDED</span>';
    }

    if (defenseStatusHeader) {
      defenseStatusHeader.textContent = 'SYSTEM ARMED';
      defenseStatusHeader.style.color = '';
    }

    logEvent('SYS', 'Perimeter status normalized: DEFCON 4 Guarded', 'sys');
  }

  // --- Telemetry Fluctuation Engine ---
  function updateTelemetry() {
    // 1. Fluctuate FPS between 28.0 and 32.2 FPS
    const currentFps = (28.0 + Math.random() * 4.2).toFixed(1);
    const latency = (14.2 + Math.random() * 6.5).toFixed(1);

    if (statFps) statFps.textContent = currentFps;
    if (statLatency) statLatency.textContent = `LATENCY: ${latency}ms`;

    // 2. Fluctuate target and entity counts
    if (Math.random() < 0.35) {
      targetCount = Math.floor(10 + Math.random() * 8);
      personCount = Math.floor(3 + Math.random() * 5);
      vehicleCount = Math.floor(2 + Math.random() * 4);
      faceCount = Math.floor(1 + Math.random() * 3);

      if (statTargets) statTargets.textContent = targetCount;
      if (statPersons) statPersons.textContent = personCount;
      if (statVehicles) statVehicles.textContent = vehicleCount;
      if (statFaces) statFaces.textContent = faceCount;
    }
  }

  // Run telemetry updates every 650ms
  setInterval(updateTelemetry, 650);

  // --- Surveillance Audit Event Generation Stream ---
  const simulationEventLibrary = [
    { tag: 'AI-POSE', msg: 'YOLOv8 Pose: 17 skeletal joints tracking lock on Target #402', level: 'pose' },
    { tag: 'ANPR', msg: 'Vehicle entered scan zone: Outpost Checkpoint Lane 1', level: 'anpr' },
    { tag: 'FACE', msg: 'YuNet DNN Face Detector: Confidence 96.8% (Matched Authorized)', level: 'face' },
    { tag: 'SYS', msg: 'Optical flow matrix calibrated across all 3 checkpoints', level: 'sys' },
    { tag: 'AI-POSE', msg: 'Target centroid velocity normal: 1.4 m/s (Sector 1)', level: 'pose' },
    { tag: 'TRIPWIRE', msg: 'BLA Virtual Tripwire: Boundary proximity warning @ Hawkins Post', level: 'breach' },
    { tag: 'ANPR', msg: 'Plate OCR preprocessing: Contrast equalization complete', level: 'anpr' },
    { tag: 'FACE', msg: 'YuNet scan complete: 1 face vector verified in Camera 1', level: 'face' },
    { tag: 'SYS', msg: 'Telemetry packet broadcast verified to client HUD', level: 'sys' },
    { tag: 'TRIPWIRE', msg: 'BLA Tripwire Breach: Hawkins Post (Virtual boundary crossed)', level: 'breach' }
  ];

  let eventLibraryIndex = 0;
  function streamRandomAuditEvent() {
    const item = simulationEventLibrary[eventLibraryIndex % simulationEventLibrary.length];
    eventLibraryIndex++;

    if (item.tag === 'TRIPWIRE' && item.msg.includes('Breach')) {
      triggerPerimeterBreach('Virtual Tripwire Boundary Breach at Hawkins Post');
    } else {
      logEvent(item.tag, item.msg, item.level);
    }
  }

  // Stream audit events every 3.2 seconds
  setInterval(streamRandomAuditEvent, 3200);

  // Push new ANPR detections every 5.5 seconds
  setInterval(() => {
    const p = plateDatabase[platePoolIndex % plateDatabase.length];
    platePoolIndex++;
    addAnprRecord(p.plate, p.cam, p.status);
  }, 5500);

  // --- Layout Toggle Buttons ---
  if (btnLayout3Col && btnLayoutFocus && cameraContainer) {
    btnLayout3Col.addEventListener('click', () => {
      cameraContainer.className = 'camera-grid layout-3col';
      btnLayout3Col.classList.add('active');
      btnLayoutFocus.classList.remove('active');
      logEvent('UI', 'Switched layout to 3 Equal Columns', 'sys');
      resizeOverlays();
    });

    btnLayoutFocus.addEventListener('click', () => {
      cameraContainer.className = 'camera-grid layout-focus';
      btnLayoutFocus.classList.add('active');
      btnLayout3Col.classList.remove('active');
      logEvent('UI', 'Switched layout to Tactical Focus (1 Main + 2 Aux)', 'sys');
      resizeOverlays();
    });
  }

  // --- Video Source Switcher Controls ---
  window.changeVideoSource = function (videoElemId, selectElemId, camTag) {
    const video = document.getElementById(videoElemId);
    const select = document.getElementById(selectElemId);
    if (!video || !select) return;

    const newSource = select.value;
    video.src = newSource;
    video.load();
    video.play().catch(() => {});

    logEvent('STREAM', `${camTag} video source switched to: ${newSource}`, 'sys');
  };

  // --- Video Snapshot Tool ---
  window.snapshotVideo = function (videoId, camTag) {
    const video = document.getElementById(videoId);
    if (!video) return;

    try {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 854;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw tactical watermark
      ctx.fillStyle = 'rgba(0, 212, 255, 0.9)';
      ctx.font = 'bold 16px "JetBrains Mono", monospace';
      ctx.fillText(`IBVAP // ${camTag} [${new Date().toISOString()}]`, 20, 36);

      const link = document.createElement('a');
      link.download = `IBVAP_${camTag}_${Date.now()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();

      logEvent('SNAPSHOT', `Captured high-res snapshot from ${camTag}`, 'face');
    } catch (e) {
      logEvent('SNAPSHOT', `Snapshot notice: ${e.message}`, 'sys');
    }
  };

  // --- Fullscreen Toggle ---
  window.toggleFullscreen = function (viewportId) {
    const elem = document.getElementById(viewportId);
    if (!elem) return;

    if (!document.fullscreenElement) {
      elem.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  // --- Reload/Play Video ---
  window.reloadVideo = function (videoId, camTag) {
    const video = document.getElementById(videoId);
    if (!video) return;
    video.currentTime = 0;
    video.play().catch(() => {});
    logEvent('STREAM', `${camTag} feed restarted from beginning`, 'sys');
  };

  // --- Simulated AI Computer Vision Canvas HUD Overlays ---
  // Renders real-time tactical bounding boxes, skeleton pose joints, and tripwire vectors over the videos!
  const canvasOverlays = [
    { canvasId: 'overlayCam1', videoId: 'videoCam1', type: 'pose' },
    { canvasId: 'overlayCam2', videoId: 'videoCam2', type: 'anpr' },
    { canvasId: 'overlayCam3', videoId: 'videoCam3', type: 'patrol' }
  ];

  function resizeOverlays() {
    canvasOverlays.forEach(item => {
      const c = document.getElementById(item.canvasId);
      if (c && c.parentElement) {
        c.width = c.parentElement.clientWidth;
        c.height = c.parentElement.clientHeight;
      }
    });
  }
  window.addEventListener('resize', resizeOverlays);

  let animFrame = 0;
  function renderAiOverlays() {
    animFrame++;
    const t = animFrame * 0.03;

    // --- Overlay 1: Hawkins Post (Pose + Virtual Fence Tripwire) ---
    const c1 = document.getElementById('overlayCam1');
    if (c1) {
      const ctx = c1.getContext('2d');
      const w = c1.width;
      const h = c1.height;
      ctx.clearRect(0, 0, w, h);

      // Virtual Tripwire Line across perimeter
      const tripwireY = h * 0.72;
      ctx.beginPath();
      ctx.moveTo(w * 0.05, tripwireY);
      ctx.lineTo(w * 0.95, tripwireY);
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = isBreachActive ? 'rgba(255, 61, 61, 0.85)' : 'rgba(0, 212, 255, 0.65)';
      ctx.stroke();
      ctx.setLineDash([]);

      // Tripwire HUD label
      ctx.fillStyle = isBreachActive ? 'rgba(255, 61, 61, 0.9)' : 'rgba(0, 212, 255, 0.8)';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(isBreachActive ? '⚠ BLA TRIPWIRE: BREACH DETECTED' : '⚡ BLA TRIPWIRE: ARMED', w * 0.08, tripwireY - 6);

      // Moving Simulated Person Bounding Box
      const pX = w * (0.35 + Math.sin(t * 0.6) * 0.15);
      const pY = h * (0.32 + Math.cos(t * 0.4) * 0.05);
      const bW = w * 0.14;
      const bH = h * 0.48;

      ctx.strokeStyle = isBreachActive ? '#ff3d3d' : '#00e887';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(pX, pY, bW, bH);

      // Bounding box tag
      ctx.fillStyle = isBreachActive ? 'rgba(255, 61, 61, 0.8)' : 'rgba(0, 232, 135, 0.8)';
      ctx.fillRect(pX, pY - 18, 125, 18);
      ctx.fillStyle = '#000';
      ctx.font = 'bold 9px "JetBrains Mono", monospace';
      ctx.fillText(`PERSON #402 96.4%`, pX + 4, pY - 5);

      // Skeletal Keypoints Simulation
      const headX = pX + bW * 0.5;
      const headY = pY + bH * 0.15;
      const chestX = headX;
      const chestY = pY + bH * 0.38;

      ctx.fillStyle = '#00d4ff';
      [
        [headX, headY],
        [chestX, chestY],
        [pX + bW * 0.25, pY + bH * 0.35],
        [pX + bW * 0.75, pY + bH * 0.35],
        [pX + bW * 0.3, pY + bH * 0.85],
        [pX + bW * 0.7, pY + bH * 0.85]
      ].forEach(([kx, ky]) => {
        ctx.beginPath();
        ctx.arc(kx, ky, 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // --- Overlay 2: Outpost Checkpoint (Vehicle & ANPR OCR Bounding Box) ---
    const c2 = document.getElementById('overlayCam2');
    if (c2) {
      const ctx = c2.getContext('2d');
      const w = c2.width;
      const h = c2.height;
      ctx.clearRect(0, 0, w, h);

      // Vehicle Bounding Box
      const vX = w * (0.28 + Math.sin(t * 0.3) * 0.08);
      const vY = h * 0.28;
      const vW = w * 0.44;
      const vH = h * 0.55;

      ctx.strokeStyle = '#c084fc';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(vX, vY, vW, vH);

      ctx.fillStyle = 'rgba(192, 132, 252, 0.85)';
      ctx.fillRect(vX, vY - 18, 140, 18);
      ctx.fillStyle = '#000';
      ctx.font = 'bold 9px "JetBrains Mono", monospace';
      ctx.fillText('VEHICLE 98.2% (SUV)', vX + 4, vY - 5);

      // Plate Bounding Box
      const plX = vX + vW * 0.35;
      const plY = vY + vH * 0.72;
      const plW = vW * 0.32;
      const plH = vH * 0.14;

      ctx.strokeStyle = '#00d4ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(plX, plY, plW, plH);

      ctx.fillStyle = 'rgba(0, 212, 255, 0.9)';
      ctx.fillRect(plX, plY + plH, 150, 16);
      ctx.fillStyle = '#050a10';
      ctx.font = 'bold 9px "JetBrains Mono", monospace';
      ctx.fillText('ANPR: TN-09-AB-1234', plX + 4, plY + plH + 12);
    }

    // --- Overlay 3: Alpha Post (Aerial Surveillance Grid) ---
    const c3 = document.getElementById('overlayCam3');
    if (c3) {
      const ctx = c3.getContext('2d');
      const w = c3.width;
      const h = c3.height;
      ctx.clearRect(0, 0, w, h);

      // Crosshair Reticle tracking target
      const tX = w * (0.5 + Math.cos(t * 0.5) * 0.25);
      const tY = h * (0.5 + Math.sin(t * 0.5) * 0.2);

      ctx.strokeStyle = 'rgba(0, 212, 255, 0.75)';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(tX, tY, 22, 0, Math.PI * 2);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(tX - 30, tY); ctx.lineTo(tX + 30, tY);
      ctx.moveTo(tX, tY - 30); ctx.lineTo(tX + 30, tY);
      ctx.stroke();

      ctx.fillStyle = 'rgba(0, 212, 255, 0.9)';
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillText('AERIAL LOCK: SEC-3', tX + 26, tY - 10);
    }

    requestAnimationFrame(renderAiOverlays);
  }

  // --- Boot Sequence ---
  window.addEventListener('DOMContentLoaded', () => {
    resizeOverlays();
    seedInitialPlates();

    logEvent('SYS', 'IBVAP Serverless Telemetry Engine Initialized', 'sys');
    logEvent('ENGINES', 'Modular Pipelines: YOLOv8 Pose + Plate YOLO + EasyOCR Active', 'pose');
    logEvent('CAM', '3-Camera Mosaic Video Feeds Engaged', 'face');

    requestAnimationFrame(renderAiOverlays);
  });

})();
