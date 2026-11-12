/**
 * ==============================================================================
 * PPD-II AUTONOMOUS VEHICLE COCKPIT DASHBOARD — APPLICATION CONTROLLER
 * ==============================================================================
 * Architecture: Professional Automotive Intelligent Glass Cockpit Controller
 * Capabilities:
 *   - Sliding-Window Progressive Frame Preloader (Zero scrubbing flicker)
 *   - Device-Pixel-Ratio (DPR) Aware Circular Radar BEV Canvas Engine
 *   - Day / Night Automotive Theme Controller (with Persistent Storage)
 *   - High-Precision Multi-Modal Telemetry & Decision HUD Synchronization
 *   - Conditional Radar Animation (Resource & Battery Efficient)
 *   - Keyboard Accessibility (Conflict-Free Shortcuts)
 * ==============================================================================
 */

(function () {
  'use strict';

  // ----------------------------------------------------------------------------
  // APPLICATION STATE
  // ----------------------------------------------------------------------------
  const State = {
    currentFrameIdx: 0,
    isPlaying: true,
    playbackInterval: null,
    baseFps: 7.0,
    speedMultiplier: 1.0,
    scannerAngle: 0,
    theme: 'light', // 'light' (Day) or 'dark' (Night)
    scaleMode: 'fit', // 'fit', 'standard', 'large'
    totalFrames: 0,
    isTabVisible: true,
    animationFrameId: null,
    imageCache: new Map(), // Sliding-window frame cache
    maxCachedFrames: 18,
    selectedStudioFrame: 16,
    customCamUrl: null,
    customLidarName: null
  };

  // DOM Elements Cache
  const DOM = {};

  // ----------------------------------------------------------------------------
  // INITIALIZATION
  // ----------------------------------------------------------------------------
  window.addEventListener('DOMContentLoaded', () => {
    cacheDomElements();
    initDataset();
    initTheme();
    initEventListeners();
    initRadarCanvas();
    updateDashboard(0);
    startPlayback();
    startRadarAnimationLoop();
  });

  function cacheDomElements() {
    DOM.body = document.body;
    DOM.html = document.documentElement;
    DOM.camImg = document.getElementById('cam-img');
    DOM.radarCanvas = document.getElementById('radarCanvas');
    DOM.radarTooltip = document.getElementById('radarTooltip');
    DOM.radarContainer = document.getElementById('radar-container');
    DOM.radarTbody = document.getElementById('radar-tbody');
    DOM.slider = document.getElementById('frame-slider');
    DOM.counterTxt = document.getElementById('frame-counter-txt');
    DOM.btnPlay = document.getElementById('btn-play');
    DOM.themeToggleBtn = document.getElementById('btn-theme-toggle');

    // Telemetry fields
    DOM.valSpeed = document.getElementById('val-speed');
    DOM.valSpeedMs = document.getElementById('val-speed-ms');
    DOM.valClosest = document.getElementById('val-closest');
    DOM.valLocks = document.getElementById('val-locks');
    DOM.valTtc = document.getElementById('val-ttc');
    DOM.valSteer = document.getElementById('val-steer');

    // Studio Modal Elements
    DOM.studioModal = document.getElementById('studio-modal');
    DOM.consoleLogs = document.getElementById('console-logs');
    DOM.btnRunPipeline = document.getElementById('btn-run-pipeline');
    DOM.camUploadPreview = document.getElementById('cam-upload-preview');
    DOM.lidarUploadPreview = document.getElementById('lidar-upload-preview');
  }

  function initDataset() {
    if (typeof DASHBOARD_DATA === 'undefined' || !Array.isArray(DASHBOARD_DATA) || DASHBOARD_DATA.length === 0) {
      console.error('[PPD-II Dashboard] Fatal: dashboard_data.js could not be loaded or is empty.');
      alert('Error: dashboard_data.js could not be loaded or is empty!');
      return;
    }
    State.totalFrames = DASHBOARD_DATA.length;
    if (DOM.slider) {
      DOM.slider.max = State.totalFrames - 1;
      DOM.slider.value = 0;
    }
  }

  // ----------------------------------------------------------------------------
  // THEME MANAGEMENT (DAY / NIGHT AUTO-COCKPIT)
  // ----------------------------------------------------------------------------
  function initTheme() {
    const saved = localStorage.getItem('ppd_cockpit_theme');
    if (saved === 'dark' || saved === 'light') {
      State.theme = saved;
    } else {
      State.theme = 'light'; // Default Day mode per automotive requirement
    }
    applyTheme(State.theme);
  }

  function toggleTheme() {
    State.theme = (State.theme === 'light') ? 'dark' : 'light';
    localStorage.setItem('ppd_cockpit_theme', State.theme);
    applyTheme(State.theme);
  }

  function applyTheme(theme) {
    if (theme === 'dark') {
      DOM.html.setAttribute('data-theme', 'dark');
      if (DOM.themeToggleBtn) {
        DOM.themeToggleBtn.innerHTML = '☀️ <span>Day Mode</span>';
        DOM.themeToggleBtn.setAttribute('title', 'Switch to High-Contrast Day Mode');
      }
    } else {
      DOM.html.removeAttribute('data-theme');
      if (DOM.themeToggleBtn) {
        DOM.themeToggleBtn.innerHTML = '🌙 <span>Night Mode</span>';
        DOM.themeToggleBtn.setAttribute('title', 'Switch to ADAS Low-Glare Night Mode');
      }
    }
    // Force radar canvas redraw to match new palette
    drawRadarScope();
  }

  // ----------------------------------------------------------------------------
  // SLIDING-WINDOW PROGRESSIVE IMAGE FRAME PRELOADER
  // ----------------------------------------------------------------------------
  function preloadNeighborFrames(centerIdx) {
    if (typeof DASHBOARD_DATA === 'undefined' || State.totalFrames === 0) return;

    // Window: 2 frames behind, 6 frames ahead
    const lookBehind = 2;
    const lookAhead = 6;
    const indicesToKeep = new Set();

    for (let offset = -lookBehind; offset <= lookAhead; offset++) {
      let idx = (centerIdx + offset + State.totalFrames) % State.totalFrames;
      indicesToKeep.add(idx);

      if (!State.imageCache.has(idx)) {
        const rec = DASHBOARD_DATA[idx];
        if (rec && rec.image_file) {
          const img = new Image();
          img.src = rec.image_file;
          State.imageCache.set(idx, img);
        }
      }
    }

    // Evict distant frames if cache exceeds boundary
    if (State.imageCache.size > State.maxCachedFrames) {
      for (const cachedIdx of State.imageCache.keys()) {
        if (!indicesToKeep.has(cachedIdx)) {
          State.imageCache.delete(cachedIdx);
        }
      }
    }
  }

  // ----------------------------------------------------------------------------
  // DASHBOARD TELEMETRY & VIEWPORT UPDATE
  // ----------------------------------------------------------------------------
  function updateDashboard(idx) {
    if (idx < 0) idx = 0;
    if (idx >= State.totalFrames) idx = State.totalFrames - 1;
    State.currentFrameIdx = idx;

    const rec = DASHBOARD_DATA[State.currentFrameIdx];
    if (!rec) return;

    // 1. Update Camera View Image (with smooth preloader support)
    if (DOM.camImg) {
      DOM.camImg.src = rec.image_file;
      preloadNeighborFrames(State.currentFrameIdx);
    }

    // 2. Synchronize Scenario / Drive Tabs
    document.querySelectorAll('.seg-btn[id^="tab-"]').forEach(btn => btn.classList.remove('active'));
    const activeTab = document.getElementById(`tab-${rec.drive}`);
    if (activeTab) activeTab.classList.add('active');

    // 3. Update Primary Telemetry Metrics
    if (DOM.valSpeed) {
      DOM.valSpeed.innerText = rec.ego_speed_kmh.toFixed(1);
    }
    if (DOM.valSpeedMs) {
      DOM.valSpeedMs.innerText = `(${(rec.ego_speed_kmh / 3.6).toFixed(1)} m/s)`;
    }

    // Closest Obstacle
    if (DOM.valClosest) {
      if (rec.closest_dist_m !== null && rec.closest_dist_m !== undefined) {
        DOM.valClosest.innerText = rec.closest_dist_m.toFixed(1);
        if (rec.closest_dist_m < 15.0) {
          DOM.valClosest.style.color = 'var(--state-critical-text)';
        } else if (rec.closest_dist_m < 25.0) {
          DOM.valClosest.style.color = 'var(--state-caution-text)';
        } else {
          DOM.valClosest.style.color = 'var(--text-primary)';
        }
      } else {
        DOM.valClosest.innerText = '--';
        DOM.valClosest.style.color = 'var(--text-tertiary)';
      }
    }

    // Active Radar Locks
    if (DOM.valLocks) {
      DOM.valLocks.innerText = rec.active_locks || 0;
    }

    // Minimum TTC
    if (DOM.valTtc) {
      if (rec.min_ttc_s !== null && rec.min_ttc_s !== undefined) {
        DOM.valTtc.innerText = `${rec.min_ttc_s.toFixed(2)}s`;
        if (rec.min_ttc_s < 2.5) {
          DOM.valTtc.style.color = 'var(--state-critical-text)';
        } else {
          DOM.valTtc.style.color = 'var(--state-caution-text)';
        }
      } else {
        DOM.valTtc.innerText = 'SAFE';
        DOM.valTtc.style.color = 'var(--state-safe-text)';
      }
    }

    // Tactical Driving Decision & Frenet Steering
    if (DOM.valSteer) {
      const isEvasion = rec.decision && rec.decision.includes('SWERVE');
      if (isEvasion) {
        const sign = rec.steer_deg > 0 ? '+' : '';
        DOM.valSteer.innerHTML = `<span class="decision-badge evasion">⚠️ EVASION (${sign}${rec.steer_deg.toFixed(1)}°)</span>`;
      } else {
        DOM.valSteer.innerHTML = `<span class="decision-badge">✓ NOMINAL CRUISE</span>`;
      }
    }

    // 4. Update Scrubber & Counter
    if (DOM.slider) {
      DOM.slider.value = idx;
    }
    if (DOM.counterTxt) {
      const currentPad = (idx + 1).toString().padStart(3, '0');
      const totalPad = State.totalFrames.toString().padStart(3, '0');
      DOM.counterTxt.innerText = `Frame: ${currentPad} / ${totalPad}`;
    }

    // 5. Update Radar Target Table & Force Redraw
    updateRadarTable(rec.radar_targets || []);
    drawRadarScope();
  }

  // ----------------------------------------------------------------------------
  // RADAR TARGETS MINI TABLE
  // ----------------------------------------------------------------------------
  function updateRadarTable(targets) {
    if (!DOM.radarTbody) return;
    if (!targets || targets.length === 0) {
      DOM.radarTbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color:var(--text-tertiary); padding:10px;">No radar targets in range</td></tr>`;
      return;
    }

    let rowsHtml = '';
    targets.forEach(tgt => {
      const isReceding = tgt.vr >= 0;
      const vrColor = isReceding ? 'var(--accent-primary)' : 'var(--state-critical-text)';
      const vrSign = isReceding ? '+' : '';
      const clsName = (tgt.class || 'CAR').replace('_', ' ');

      rowsHtml += `
        <tr>
          <td><b style="color:var(--accent-radar);">#${tgt.id}</b></td>
          <td><b>${clsName}</b></td>
          <td>${tgt.dist.toFixed(1)}m</td>
          <td>${tgt.x.toFixed(1)}m</td>
          <td>${tgt.y.toFixed(1)}m</td>
          <td style="color:${vrColor}; font-weight:700;">${vrSign}${tgt.vr.toFixed(2)}</td>
          <td>${tgt.rcs.toFixed(1)}</td>
        </tr>
      `;
    });
    DOM.radarTbody.innerHTML = rowsHtml;
  }

  // ----------------------------------------------------------------------------
  // RADAR BEV CANVAS ENGINE (CIRCULAR GEOMETRY & DPR OPTIMIZED)
  // ----------------------------------------------------------------------------
  let radarCtx = null;

  function initRadarCanvas() {
    if (!DOM.radarCanvas) return;
    radarCtx = DOM.radarCanvas.getContext('2d');
    resizeRadarCanvas();
    window.addEventListener('resize', debounce(resizeRadarCanvas, 100));
    setupRadarInteractions();
  }

  function resizeRadarCanvas() {
    if (!DOM.radarCanvas || !DOM.radarContainer) return;
    const rect = DOM.radarContainer.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Use actual CSS dimensions
    const displayW = Math.floor(rect.width);
    const displayH = Math.floor(rect.height);

    if (displayW === 0 || displayH === 0) return;

    DOM.radarCanvas.width = displayW * dpr;
    DOM.radarCanvas.height = displayH * dpr;

    radarCtx.resetTransform();
    radarCtx.scale(dpr, dpr);

    drawRadarScope();
  }

  function drawRadarScope() {
    if (!radarCtx || !DOM.radarCanvas || !DOM.radarContainer) return;

    const rect = DOM.radarContainer.getBoundingClientRect();
    const W = rect.width;
    const H = rect.height;

    if (W === 0 || H === 0) return;

    const ctx = radarCtx;
    const isDark = (State.theme === 'dark');

    // 1. Clear background
    ctx.clearRect(0, 0, W, H);

    // Geometry anchors:
    // Place ego vehicle towards the bottom center
    const u0 = W / 2;
    const v0 = H * 0.80;

    // Strict circular radius to guarantee 1:1 circular aspect ratio:
    // Radius allows up to 55 meters display without edge collision
    const maxRadius = Math.min(W / 2 - 14, v0 - 14);
    const scale = maxRadius / 55.0; // pixels per meter

    // 2. Draw Forward Sensor Field of View (FOV +/- 45 deg)
    ctx.save();
    ctx.fillStyle = isDark ? 'rgba(56, 189, 248, 0.05)' : 'rgba(2, 132, 199, 0.04)';
    ctx.beginPath();
    ctx.moveTo(u0, v0);
    ctx.arc(u0, v0, maxRadius, -Math.PI * 0.75, -Math.PI * 0.25);
    ctx.closePath();
    ctx.fill();

    // FOV Boundary radial guidelines
    ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.22)' : 'rgba(100, 116, 139, 0.20)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    const fovAngles = [-Math.PI * 0.75, -Math.PI * 0.5, -Math.PI * 0.25];
    fovAngles.forEach(ang => {
      ctx.beginPath();
      ctx.moveTo(u0, v0);
      ctx.lineTo(u0 + maxRadius * Math.cos(ang), v0 + maxRadius * Math.sin(ang));
      ctx.stroke();
    });
    ctx.restore();

    // 3. Draw Concentric Metric Range Rings (10m, 20m, 30m, 40m, 50m)
    const ranges = [10, 20, 30, 40, 50];
    ctx.save();
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 4]);

    ranges.forEach((r, idx) => {
      const radiusPx = r * scale;
      ctx.strokeStyle = isDark ? 'rgba(148, 163, 184, 0.20)' : 'rgba(100, 116, 139, 0.22)';
      ctx.beginPath();
      ctx.arc(u0, v0, radiusPx, 0, Math.PI * 2);
      ctx.stroke();

      // Range labels
      ctx.fillStyle = isDark ? '#94a3b8' : '#64748b';
      ctx.font = '600 9.5px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(`${r}m`, u0 + 6, v0 - radiusPx + 11);
    });
    ctx.restore();

    // 4. Draw Radial Scan Sweep (Phosphor beam)
    ctx.save();
    const sweepRadius = maxRadius;
    const sweepAngle = State.scannerAngle;

    const gradSweep = ctx.createRadialGradient(u0, v0, 0, u0, v0, sweepRadius);
    gradSweep.addColorStop(0, isDark ? 'rgba(56, 189, 248, 0.25)' : 'rgba(2, 132, 199, 0.20)');
    gradSweep.addColorStop(1, 'rgba(2, 132, 199, 0.0)');

    ctx.fillStyle = gradSweep;
    ctx.beginPath();
    ctx.moveTo(u0, v0);
    ctx.arc(u0, v0, sweepRadius, sweepAngle - 0.25, sweepAngle);
    ctx.closePath();
    ctx.fill();

    // Active sweep leading edge
    ctx.strokeStyle = isDark ? 'rgba(56, 189, 248, 0.65)' : 'rgba(2, 132, 199, 0.60)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(u0, v0);
    ctx.lineTo(u0 + sweepRadius * Math.cos(sweepAngle), v0 + sweepRadius * Math.sin(sweepAngle));
    ctx.stroke();
    ctx.restore();

    // 5. Draw Ego Vehicle (Realistic Top-Down Car at u0, v0)
    drawEgoVehicle(ctx, u0, v0, isDark);

    // 6. Draw Surrounding Targets
    const rec = (typeof DASHBOARD_DATA !== 'undefined') ? DASHBOARD_DATA[State.currentFrameIdx] : null;
    if (rec && rec.radar_targets) {
      rec.radar_targets.forEach(tgt => {
        // Coordinate transformation:
        // x is forward (meters), y is lateral (positive left, negative right in standard vehicle frame)
        // Canvas: u moves horizontally (-y * scale), v moves vertically (-x * scale)
        const pu = u0 - (tgt.y * scale);
        const pv = v0 - (tgt.x * scale);

        // Keep inside bounds
        if (pu >= 10 && pu < W - 10 && pv >= 10 && pv < H - 10) {
          drawSurroundingVehicle(ctx, pu, pv, tgt, isDark);
        }
      });
    }
  }

  // Top-Down Ego Car Model
  function drawEgoVehicle(ctx, x, y, isDark) {
    const carW = 20;
    const carL = 40;

    ctx.save();
    ctx.translate(x, y);

    // Headlight Beam
    const gradBeam = ctx.createLinearGradient(0, -carL / 2, 0, -carL / 2 - 45);
    gradBeam.addColorStop(0, isDark ? 'rgba(56, 189, 248, 0.35)' : 'rgba(2, 132, 199, 0.25)');
    gradBeam.addColorStop(1, 'rgba(2, 132, 199, 0.0)');
    ctx.fillStyle = gradBeam;
    ctx.beginPath();
    ctx.moveTo(-8, -carL / 2);
    ctx.lineTo(-24, -carL / 2 - 45);
    ctx.lineTo(24, -carL / 2 - 45);
    ctx.lineTo(8, -carL / 2);
    ctx.closePath();
    ctx.fill();

    // Car Body
    ctx.fillStyle = isDark ? '#1e293b' : '#ffffff';
    ctx.strokeStyle = isDark ? '#38bdf8' : '#0284c7';
    ctx.lineWidth = 1.8;
    drawRoundedRect(ctx, -carW / 2, -carL / 2, carW, carL, 4);
    ctx.fill();
    ctx.stroke();

    // Front Windshield
    ctx.fillStyle = isDark ? 'rgba(56, 189, 248, 0.5)' : 'rgba(2, 132, 199, 0.4)';
    ctx.beginPath();
    ctx.moveTo(-carW / 2 + 2, -carL / 2 + 8);
    ctx.lineTo(carW / 2 - 2, -carL / 2 + 8);
    ctx.lineTo(carW / 2 - 4, -carL / 2 + 15);
    ctx.lineTo(-carW / 2 + 4, -carL / 2 + 15);
    ctx.closePath();
    ctx.fill();

    // Rear Window
    ctx.fillStyle = isDark ? 'rgba(56, 189, 248, 0.3)' : 'rgba(2, 132, 199, 0.25)';
    ctx.fillRect(-carW / 2 + 3, carL / 2 - 12, carW - 6, 4);

    // Label
    ctx.fillStyle = isDark ? '#38bdf8' : '#0284c7';
    ctx.font = '700 8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('EGO', 0, carL / 2 + 11);

    ctx.restore();
  }

  // Top-Down Surrounding Target Model
  function drawSurroundingVehicle(ctx, x, y, tgt, isDark) {
    ctx.save();
    ctx.translate(x, y);

    const isReceding = tgt.vr >= 0;
    const accentCol = isReceding ? (isDark ? '#c084fc' : '#7c3aed') : (isDark ? '#f87171' : '#dc2626');
    const bgCol = isReceding
      ? (isDark ? '#2e1065' : '#ede9fe')
      : (isDark ? '#450a0a' : '#fee2e2');

    const cls = (tgt.class || '').toLowerCase();

    if (cls.includes('truck') || cls.includes('bus')) {
      const w = 24, l = 54;
      ctx.fillStyle = bgCol;
      ctx.strokeStyle = '#d97706';
      ctx.lineWidth = 1.6;
      drawRoundedRect(ctx, -w / 2, -l / 2, w, l, 3);
      ctx.fill();
      ctx.stroke();

      // Divider
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.4)';
      ctx.beginPath();
      ctx.moveTo(-w / 2, -l / 2 + 14);
      ctx.lineTo(w / 2, -l / 2 + 14);
      ctx.stroke();
    } else if (cls.includes('two_wheeler') || cls.includes('bike') || cls.includes('motorcycle')) {
      const w = 9, l = 22;
      ctx.fillStyle = bgCol;
      ctx.strokeStyle = isDark ? '#2dd4bf' : '#0d9488';
      ctx.lineWidth = 1.6;
      drawRoundedRect(ctx, -w / 2, -l / 2, w, l, 3);
      ctx.fill();
      ctx.stroke();
    } else if (cls.includes('person') || cls.includes('pedestrian')) {
      ctx.fillStyle = isDark ? '#4ade80' : '#16a34a';
      ctx.beginPath();
      ctx.arc(0, 0, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.stroke();
    } else {
      // Standard Passenger Vehicle
      const w = 18, l = 34;
      ctx.fillStyle = bgCol;
      ctx.strokeStyle = accentCol;
      ctx.lineWidth = 1.6;
      drawRoundedRect(ctx, -w / 2, -l / 2, w, l, 3);
      ctx.fill();
      ctx.stroke();

      // Windshield line
      ctx.fillStyle = isDark ? 'rgba(255, 255, 255, 0.25)' : 'rgba(0, 0, 0, 0.15)';
      ctx.fillRect(-w / 2 + 3, -l / 2 + 7, w - 6, 5);
    }

    // Velocity Doppler Vector Arrow
    const vrLen = Math.max(-20, Math.min(20, tgt.vr * 2.2));
    ctx.strokeStyle = isReceding ? '#0284c7' : '#dc2626';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(0, -vrLen);
    ctx.stroke();

    // Target Text Tag (ID + Range)
    ctx.fillStyle = isDark ? '#f8fafc' : '#0f172a';
    ctx.font = '700 8.5px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${tgt.dist.toFixed(1)}m`, 0, 19);

    ctx.restore();
  }

  function drawRoundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  // ----------------------------------------------------------------------------
  // RADAR CANVAS INTERACTION & TOOLTIPS
  // ----------------------------------------------------------------------------
  function setupRadarInteractions() {
    if (!DOM.radarCanvas || !DOM.radarTooltip) return;

    DOM.radarCanvas.addEventListener('mousemove', (e) => {
      const rect = DOM.radarCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const W = rect.width;
      const H = rect.height;
      const u0 = W / 2;
      const v0 = H * 0.80;
      const maxRadius = Math.min(W / 2 - 14, v0 - 14);
      const scale = maxRadius / 55.0;

      const rec = (typeof DASHBOARD_DATA !== 'undefined') ? DASHBOARD_DATA[State.currentFrameIdx] : null;
      if (!rec || !rec.radar_targets) return;

      let hit = null;
      for (const tgt of rec.radar_targets) {
        const pu = u0 - (tgt.y * scale);
        const pv = v0 - (tgt.x * scale);
        const dist = Math.hypot(mouseX - pu, mouseY - pv);
        if (dist < 14) {
          hit = tgt;
          break;
        }
      }

      if (hit) {
        DOM.radarTooltip.style.display = 'block';
        // Prevent tooltip from overflowing the right edge
        const tipX = (mouseX + 160 > W) ? (mouseX - 165) : (mouseX + 12);
        const tipY = Math.max(10, mouseY - 40);

        DOM.radarTooltip.style.left = `${tipX}px`;
        DOM.radarTooltip.style.top = `${tipY}px`;
        DOM.radarTooltip.innerHTML = `
          <div style="font-weight:700; color:var(--accent-radar); margin-bottom:2px;">
            Target #${hit.id} (${hit.class.replace('_', ' ')})
          </div>
          <div>Range: <b>${hit.dist.toFixed(1)} m</b></div>
          <div>Position: X=${hit.x.toFixed(1)}m, Y=${hit.y.toFixed(1)}m</div>
          <div>Doppler v<sub>r</sub>: <b style="color:${hit.vr >= 0 ? 'var(--accent-primary)' : 'var(--state-critical-text)'}">
            ${hit.vr >= 0 ? '+' : ''}${hit.vr.toFixed(2)} m/s
          </b></div>
          <div>RCS &sigma;: <b>${hit.rcs.toFixed(1)} dBsm</b></div>
        `;
      } else {
        DOM.radarTooltip.style.display = 'none';
      }
    });

    DOM.radarCanvas.addEventListener('mouseleave', () => {
      if (DOM.radarTooltip) DOM.radarTooltip.style.display = 'none';
    });
  }

  // ----------------------------------------------------------------------------
  // CONDITIONAL RADAR ANIMATION LOOP (RESOURCE & BATTERY EFFICIENT)
  // ----------------------------------------------------------------------------
  function startRadarAnimationLoop() {
    function animate() {
      // Rotate the scanning sweep line smoothly
      if (State.isTabVisible) {
        State.scannerAngle += 0.038;
        if (State.scannerAngle > Math.PI * 2) {
          State.scannerAngle -= Math.PI * 2;
        }
        drawRadarScope();
      }
      State.animationFrameId = requestAnimationFrame(animate);
    }
    cancelAnimationFrame(State.animationFrameId);
    State.animationFrameId = requestAnimationFrame(animate);
  }

  // Halt animation when tab is inactive to preserve CPU / battery
  document.addEventListener('visibilitychange', () => {
    State.isTabVisible = (document.visibilityState === 'visible');
  });

  // ----------------------------------------------------------------------------
  // PLAYBACK CONTROLLER
  // ----------------------------------------------------------------------------
  function togglePlay() {
    State.isPlaying = !State.isPlaying;
    if (DOM.btnPlay) {
      if (State.isPlaying) {
        DOM.btnPlay.innerText = 'Pause';
        DOM.btnPlay.classList.add('btn-play-action');
        startPlayback();
      } else {
        DOM.btnPlay.innerText = 'Play';
        DOM.btnPlay.classList.remove('btn-play-action');
        clearInterval(State.playbackInterval);
      }
    }
  }

  function startPlayback() {
    clearInterval(State.playbackInterval);
    const intervalMs = 1000 / (State.baseFps * State.speedMultiplier);
    State.playbackInterval = setInterval(() => {
      if (!State.isPlaying) return;
      let nextIdx = (State.currentFrameIdx + 1) % State.totalFrames;
      updateDashboard(nextIdx);
    }, intervalMs);
  }

  function stepFrame(delta) {
    if (State.isPlaying) togglePlay();
    let newIdx = State.currentFrameIdx + delta;
    if (newIdx < 0) newIdx = State.totalFrames - 1;
    if (newIdx >= State.totalFrames) newIdx = 0;
    updateDashboard(newIdx);
  }

  function onSliderInput(val) {
    if (State.isPlaying) togglePlay();
    updateDashboard(parseInt(val, 10));
  }

  function setSpeed(multiplier, btnEl) {
    State.speedMultiplier = multiplier;
    document.querySelectorAll('.speed-pill').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');
    if (State.isPlaying) startPlayback();
  }

  // ----------------------------------------------------------------------------
  // SCENARIO / DRIVE SWITCHER
  // ----------------------------------------------------------------------------
  function switchDrive(driveId) {
    document.querySelectorAll('.seg-btn[id^="tab-"]').forEach(btn => btn.classList.remove('active'));
    const tabEl = document.getElementById(`tab-${driveId}`);
    if (tabEl) tabEl.classList.add('active');

    let targetIdx = 0;
    if (driveId === 'd1') targetIdx = 50;
    else if (driveId === 'd2') targetIdx = 100;

    updateDashboard(targetIdx);
  }

  // ----------------------------------------------------------------------------
  // VIEWPORT SCALE MODE SELECTOR
  // ----------------------------------------------------------------------------
  function setScaleMode(mode, btnEl) {
    State.scaleMode = mode;
    document.querySelectorAll('.seg-btn[id^="scale-"]').forEach(b => b.classList.remove('active'));
    if (btnEl) btnEl.classList.add('active');

    DOM.body.classList.remove('scale-compact', 'scale-large');
    if (mode === 'compact') {
      DOM.body.classList.add('scale-compact');
    } else if (mode === 'large') {
      DOM.body.classList.add('scale-large');
    }
    resizeRadarCanvas();
  }

  // ----------------------------------------------------------------------------
  // CONFLICT-FREE KEYBOARD INTERACTION
  // ----------------------------------------------------------------------------
  function initEventListeners() {
    window.addEventListener('keydown', (e) => {
      // Do not trigger shortcuts when user is focused on interactive inputs
      const activeTag = document.activeElement ? document.activeElement.tagName : '';
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(activeTag)) return;

      if (e.code === 'Space') {
        e.preventDefault();
        // Blur active button to prevent dual-trigger
        if (activeTag === 'BUTTON') document.activeElement.blur();
        togglePlay();
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        stepFrame(1);
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        stepFrame(-1);
      } else if (e.code === 'KeyT') {
        e.preventDefault();
        toggleTheme();
      } else if (e.code === 'Escape') {
        closeStudioModal();
      }
    });
  }

  // ----------------------------------------------------------------------------
  // MULTI-SENSOR PIPELINE STUDIO INTERACTIVE CONTROLLER
  // ----------------------------------------------------------------------------
  function toggleStudioModal() {
    if (!DOM.studioModal) return;
    const isOpen = DOM.studioModal.classList.contains('open');
    if (isOpen) {
      closeStudioModal();
    } else {
      DOM.studioModal.classList.add('open');
    }
  }

  function closeStudioModal() {
    if (DOM.studioModal) DOM.studioModal.classList.remove('open');
  }

  function handleCamUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    if (DOM.camUploadPreview) {
      DOM.camUploadPreview.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      DOM.camUploadPreview.style.color = 'var(--accent-cam)';
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      State.customCamUrl = e.target.result;
      logConsole(`📷 Custom Camera Image Loaded: ${file.name}`);
    };
    reader.readAsDataURL(file);
  }

  function handleLidarUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    State.customLidarName = file.name;
    if (DOM.lidarUploadPreview) {
      DOM.lidarUploadPreview.innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      DOM.lidarUploadPreview.style.color = 'var(--accent-lidar)';
    }
    logConsole(`📡 Custom LiDAR Point Cloud Loaded: ${file.name}`);
  }

  function loadSamplePair(drive, frameIdx, btn) {
    State.selectedStudioFrame = frameIdx;
    State.customCamUrl = null;
    State.customLidarName = null;

    document.querySelectorAll('.sample-benchmarks button').forEach(b => b.classList.remove('active'));
    if (btn) btn.classList.add('active');

    if (DOM.camUploadPreview) {
      DOM.camUploadPreview.innerText = `${drive}_cam_${frameIdx.toString().padStart(7, '0')}.jpg (Loaded)`;
      DOM.camUploadPreview.style.color = 'var(--accent-primary)';
    }
    if (DOM.lidarUploadPreview) {
      DOM.lidarUploadPreview.innerText = `${drive}_lidar_${frameIdx.toString().padStart(7, '0')}.npy (Loaded)`;
      DOM.lidarUploadPreview.style.color = 'var(--accent-primary)';
    }
    logConsole(`ℹ️ Selected Benchmark IDD Pair: ${drive.toUpperCase()} Frame #${frameIdx}`);
  }

  function logConsole(msg) {
    if (!DOM.consoleLogs) return;
    const timeStr = new Date().toLocaleTimeString();
    DOM.consoleLogs.innerHTML += `\n[${timeStr}] ${msg}`;
    DOM.consoleLogs.scrollTop = DOM.consoleLogs.scrollHeight;
  }

  function executeFullPipeline() {
    if (!DOM.btnRunPipeline || !DOM.consoleLogs) return;

    DOM.btnRunPipeline.disabled = true;
    DOM.btnRunPipeline.innerText = '⏳ PROCESSING MULTI-SENSOR PERCEPTION...';
    DOM.consoleLogs.innerHTML = `[${new Date().toLocaleTimeString()}] 🚀 INITIALIZING SENSOR FUSION & RADAR SYNTHESIS...`;

    const stages = [
      { delay: 300, msg: '📷 [MODEL 1: 2D VISION YOLO] Inferencing 640x640 front RGB frame... Extracted bounding boxes.' },
      { delay: 650, msg: '📡 [MODEL 2: 3D LIDAR] Ingesting 16,384 spatial points... Clustered 3D bounding boxes.' },
      { delay: 1000, msg: '🔗 [CALIBRATION FUSION] Applying Extrinsic [R|T] & Camera Pinhole Matrix K... Ground plane aligned.' },
      { delay: 1350, msg: '⚡ [77 GHz MMWAVE FMCW RADAR] Synthesizing physical signal: Pr = (Pt*G^2*lambda^2*sigma)/((4pi)^3 * R^4)... Doppler vr calculated.' },
      { delay: 1700, msg: '🛣️ [FRENET PLANNER] Computing 3-Zone Quintic Polynomial (Seedha start s<=4.2m, dynamic evasion bend s>4.2m).' },
      { delay: 2000, msg: '✅ [SUCCESS] Multi-Sensor Pipeline Complete! Updating Cockpit Telemetry...' }
    ];

    stages.forEach(st => {
      setTimeout(() => logConsole(st.msg), st.delay);
    });

    setTimeout(() => {
      DOM.btnRunPipeline.disabled = false;
      DOM.btnRunPipeline.innerText = '⚡ RUN MULTI-SENSOR PERCEPTION & GENERATE MMWAVE RADAR';

      updateDashboard(State.selectedStudioFrame);

      if (State.customCamUrl && DOM.camImg) {
        DOM.camImg.src = State.customCamUrl;
      }
      logConsole('🎉 Telemetry & Radar Scope are LIVE in Cockpit! You may close the Studio modal.');
    }, 2200);
  }

  // ----------------------------------------------------------------------------
  // UTILITIES
  // ----------------------------------------------------------------------------
  function debounce(fn, ms) {
    let timer;
    return function (...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  // ----------------------------------------------------------------------------
  // GLOBAL EXPORTS FOR INLINE ONCLICK HANDLERS
  // ----------------------------------------------------------------------------
  window.toggleTheme = toggleTheme;
  window.switchDrive = switchDrive;
  window.setScaleMode = setScaleMode;
  window.togglePlay = togglePlay;
  window.stepFrame = stepFrame;
  window.onSliderInput = onSliderInput;
  window.setSpeed = setSpeed;
  window.toggleStudioModal = toggleStudioModal;
  window.closeStudioModal = closeStudioModal;
  window.handleCamUpload = handleCamUpload;
  window.handleLidarUpload = handleLidarUpload;
  window.loadSamplePair = loadSamplePair;
  window.executeFullPipeline = executeFullPipeline;

})();
