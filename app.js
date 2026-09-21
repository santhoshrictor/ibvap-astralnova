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

    let newSource = select.value;
    if (!newSource.includes('?v=')) {
      newSource += (newSource.includes('?') ? '&' : '?') + 'v=2';
    }
    video.src = newSource;
    video.muted = true;
    video.defaultMuted = true;
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

  function renderAiOverlays() {
    // Clear canvas overlays so real baked-in GPU AI detections (Pose, ANPR, Skeletons)
    // from processed_*.mp4 show through clearly without duplicate simulated boxes
    canvasOverlays.forEach(item => {
      const c = document.getElementById(item.canvasId);
      if (c) {
        const ctx = c.getContext('2d');
        ctx.clearRect(0, 0, c.width, c.height);
      }
    });
  }

  // --- Boot Sequence ---
  window.addEventListener('DOMContentLoaded', () => {
    resizeOverlays();
    seedInitialPlates();

    // Start all 3 video feeds with robust autoplay handling
    ['videoCam1', 'videoCam2', 'videoCam3'].forEach(id => {
      const v = document.getElementById(id);
      if (v) {
        v.muted = true;
        v.defaultMuted = true;
        v.playsInline = true;
        const playPromise = v.play();
        if (playPromise !== undefined) {
          playPromise.catch(() => {
            // Autoplay blocked by browser policy: start upon first click anywhere
            const startOnInteraction = () => {
              v.play().catch(() => {});
              document.removeEventListener('click', startOnInteraction);
            };
            document.addEventListener('click', startOnInteraction, { once: true });
          });
        }
      }
    });

    logEvent('SYS', 'IBVAP Serverless Telemetry Engine Initialized', 'sys');
    logEvent('ENGINES', 'Modular Pipelines: YOLOv8 Pose + Plate YOLO + EasyOCR Active', 'pose');
    logEvent('CAM', '3-Camera Mosaic Video Feeds Engaged', 'face');

    requestAnimationFrame(renderAiOverlays);
  });

})();
