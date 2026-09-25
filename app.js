/* global F1, LapSim, Jev, TRACKS */
"use strict";

const $ = (id) => document.getElementById(id);
// Tolerant setters: a readout the page does not have must not take down the
// render loop with it.
const setText = (id, v) => { const e = $(id); if (e) e.textContent = v; };
const setStyle = (id, k, v) => { const e = $(id); if (e) e.style[k] = v; };
const fmtLap = (t) => (t == null ? "--:--.---"
  : `${Math.floor(t / 60)}:${(t % 60).toFixed(3).padStart(6, "0")}`);

const S = {
  key: "monza", track: null, ref: null, refDriver: null,
  car: null, policy: null, worker: null,
  training: false, gen: 0, history: [], rec: null,
  driverMode: "jev",        // jev | reference
  escalate: false, escThreshold: 0.35,
  follow: true,
  lapBest: null, lastLap: null, escalated: false, escCount: 0, ctrlCount: 0,
  trail: [], reliability: null, speed: 1, trainerBestLap: null, ckptAt: null,
  lastLiveGen: null, lastLiveAt: null, liveMisses: 0,
  crashes: [], attempts: 0, furthest: 0, crashInfo: null,
  policySource: "none", pretrained: null, ckptTag: null, lapDirty: false, ckptMismatch: null,
};

// ------------------------------------------------------------------ track art
function elevColor(t) {
  // Blue (low) -> teal -> amber (high). Distinct in both light and dark.
  const stops = [[0, 42, 78, 128], [0.5, 34, 132, 122], [1, 176, 124, 44]];
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ar, ag, ab] = stops[i], [b, br, bg, bb] = stops[i + 1];
    if (t <= b) {
      const f = (t - a) / (b - a);
      return `rgb(${ar + (br - ar) * f | 0},${ag + (bg - ag) * f | 0},${ab + (bb - ab) * f | 0})`;
    }
  }
  return "rgb(176,124,44)";
}

function buildTrackPath(tr) {
  const n = tr.n, L = { x: [], y: [] }, R = { x: [], y: [] };
  for (let i = 0; i < n; i++) {
    const nx = -tr.ty[i], ny = tr.tx[i];
    L.x.push(tr.x[i] + nx * tr.wl[i]); L.y.push(tr.y[i] + ny * tr.wl[i]);
    R.x.push(tr.x[i] - nx * tr.wr[i]); R.y.push(tr.y[i] - ny * tr.wr[i]);
  }
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < n; i++) { zmin = Math.min(zmin, tr.z[i]); zmax = Math.max(zmax, tr.z[i]); }
  return { L, R, zmin, zmax };
}

// ------------------------------------------------------------------- rendering
const view = { cx: 0, cy: 0, scale: 1 };

function fitView(tr, w, h) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (let i = 0; i < tr.n; i++) {
    x0 = Math.min(x0, tr.x[i]); x1 = Math.max(x1, tr.x[i]);
    y0 = Math.min(y0, tr.y[i]); y1 = Math.max(y1, tr.y[i]);
  }
  const pad = 70;
  return {
    cx: (x0 + x1) / 2, cy: (y0 + y1) / 2,
    scale: Math.min((w - pad) / (x1 - x0), (h - pad) / (y1 - y0)),
  };
}

function drawScene(ctx, w, h) {
  const tr = S.track, art = S.art;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.translate(w / 2, h / 2);
  ctx.scale(view.scale, -view.scale);
  ctx.translate(-view.cx, -view.cy);

  const lw = Math.max(0.6 / view.scale, 0.25);

  // Track surface, one quad per sample, tinted by elevation.
  const n = tr.n, step = view.scale > 2.2 ? 1 : 2;
  for (let i = 0; i < n; i += step) {
    const j = (i + step) % n;
    const t = (tr.z[i] - art.zmin) / Math.max(1e-6, art.zmax - art.zmin);
    ctx.fillStyle = elevColor(t);
    ctx.beginPath();
    ctx.moveTo(art.L.x[i], art.L.y[i]);
    ctx.lineTo(art.L.x[j], art.L.y[j]);
    ctx.lineTo(art.R.x[j], art.R.y[j]);
    ctx.lineTo(art.R.x[i], art.R.y[i]);
    ctx.closePath();
    ctx.fill();
  }

  // White lines
  ctx.lineWidth = lw * 1.6; ctx.strokeStyle = "rgba(255,255,255,0.85)";
  for (const side of [art.L, art.R]) {
    ctx.beginPath();
    ctx.moveTo(side.x[0], side.y[0]);
    for (let i = 1; i < n; i++) ctx.lineTo(side.x[i], side.y[i]);
    ctx.closePath(); ctx.stroke();
  }

  if ($("optLine").checked) {
    ctx.lineWidth = lw * 2; ctx.strokeStyle = "rgba(255,255,255,0.45)";
    ctx.setLineDash([lw * 12, lw * 10]);
    ctx.beginPath();
    ctx.moveTo(S.ref.line.x[0], S.ref.line.y[0]);
    for (let i = 1; i < n; i++) ctx.lineTo(S.ref.line.x[i], S.ref.line.y[i]);
    ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
  }

  // Start/finish
  const nx = -tr.ty[0], ny = tr.tx[0];
  ctx.lineWidth = lw * 4; ctx.strokeStyle = "#fff";
  ctx.beginPath();
  ctx.moveTo(tr.x[0] + nx * tr.wl[0], tr.y[0] + ny * tr.wl[0]);
  ctx.lineTo(tr.x[0] - nx * tr.wr[0], tr.y[0] - ny * tr.wr[0]);
  ctx.stroke();

  // Trail, coloured by the typed decision that was active
  if (S.trail.length > 1) {
    ctx.lineWidth = lw * 3.2; ctx.lineCap = "round";
    for (let i = 1; i < S.trail.length; i++) {
      const a = S.trail[i - 1], b = S.trail[i];
      ctx.strokeStyle = Jev.DECISIONS[b.dec].color;
      ctx.globalAlpha = 0.25 + 0.6 * (i / S.trail.length);
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  for (const c of S.crashes) {
    ctx.strokeStyle = "rgba(255,45,85,0.35)"; ctx.lineWidth = lw * 2.5;
    const r = 2.2;
    ctx.beginPath();
    ctx.moveTo(c.x - r, c.y - r); ctx.lineTo(c.x + r, c.y + r);
    ctx.moveTo(c.x - r, c.y + r); ctx.lineTo(c.x + r, c.y - r);
    ctx.stroke();
  }
  if (S.crashInfo) {
    ctx.strokeStyle = "#ff2d55"; ctx.lineWidth = lw * 4;
    const r = 4;
    ctx.beginPath();
    ctx.moveTo(S.crashInfo.x - r, S.crashInfo.y - r); ctx.lineTo(S.crashInfo.x + r, S.crashInfo.y + r);
    ctx.moveTo(S.crashInfo.x - r, S.crashInfo.y + r); ctx.lineTo(S.crashInfo.x + r, S.crashInfo.y - r);
    ctx.stroke();
  }

  drawCar(ctx, S.car, lw);
  ctx.restore();
}

function drawCar(ctx, car, lw) {
  const sp = car.spec;
  ctx.save();
  ctx.translate(car.X, car.Y);
  ctx.rotate(car.psi);
  const l = sp.length, w = sp.width;
  ctx.fillStyle = !car.alive ? "#ff2d55" : S.escalated ? "#ffb020" : "#1b3ecf";
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = lw;
  ctx.beginPath();
  ctx.moveTo(l * 0.55, 0);
  ctx.lineTo(l * 0.2, w * 0.28);
  ctx.lineTo(-l * 0.32, w * 0.3);
  ctx.lineTo(-l * 0.45, w * 0.5);
  ctx.lineTo(-l * 0.45, -w * 0.5);
  ctx.lineTo(-l * 0.32, -w * 0.3);
  ctx.lineTo(l * 0.2, -w * 0.28);
  ctx.closePath();
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#ffd400";
  ctx.fillRect(-l * 0.12, -w * 0.13, l * 0.2, w * 0.26);
  ctx.restore();
}

// ------------------------------------------------------------------- telemetry
function drawGauges() {
  const c = $("gauges"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  const car = S.car;

  // Friction circle
  const cx = h / 2 + 6, cy = h / 2, R = h / 2 - 14;
  ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1;
  for (const g of [1, 2, 3, 4, 5]) {
    ctx.beginPath(); ctx.arc(cx, cy, (R * g) / 5, 0, 7); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.28)";
  ctx.beginPath(); ctx.moveTo(cx - R, cy); ctx.lineTo(cx + R, cy);
  ctx.moveTo(cx, cy - R); ctx.lineTo(cx, cy + R); ctx.stroke();
  const gx = (car.ay / F1.G / 5) * R, gy = (-car.ax / F1.G / 5) * R;
  ctx.fillStyle = "#00e0a4";
  ctx.beginPath(); ctx.arc(cx + gx, cy + gy, 5, 0, 7); ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.5)"; ctx.font = "9px ui-monospace,monospace";
  ctx.fillText("5g", cx + R - 10, cy - 3);

  // Tyre usage bars
  const bx = h + 28, bw = w - bx - 12;
  const bars = [["FRONT", car.useF], ["REAR", car.useR]];
  bars.forEach(([label, u], i) => {
    const y = 16 + i * 30;
    ctx.fillStyle = "rgba(255,255,255,0.55)"; ctx.font = "10px ui-monospace,monospace";
    ctx.fillText(label, bx, y - 3);
    ctx.fillStyle = "rgba(255,255,255,0.1)";
    ctx.fillRect(bx, y, bw, 10);
    ctx.fillStyle = u > 0.97 ? "#ff2d55" : u > 0.85 ? "#ffa02e" : "#00e0a4";
    ctx.fillRect(bx, y, bw * Math.min(1, u), 10);
  });
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.beginPath(); ctx.moveTo(bx + bw, 10); ctx.lineTo(bx + bw, 60); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.fillText("grip limit", bx + bw - 52, 74);
}

function drawElevation() {
  const c = $("elev"), ctx = c.getContext("2d");
  const w = c.width, h = c.height, tr = S.track;
  ctx.clearRect(0, 0, w, h);
  const zmin = S.art.zmin, zmax = S.art.zmax, span = Math.max(1, zmax - zmin);
  ctx.beginPath();
  ctx.moveTo(0, h);
  for (let i = 0; i < tr.n; i++) {
    const x = (i / tr.n) * w, y = h - 8 - ((tr.z[i] - zmin) / span) * (h - 22);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(w, h); ctx.closePath();
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(176,124,44,0.65)");
  grad.addColorStop(1, "rgba(42,78,128,0.25)");
  ctx.fillStyle = grad; ctx.fill();

  const sNorm = ((S.car.proj.s % tr.length) + tr.length) % tr.length / tr.length;
  const px = sNorm * w;
  ctx.strokeStyle = "#00e0a4"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
  ctx.fillStyle = "rgba(255,255,255,0.6)"; ctx.font = "10px ui-monospace,monospace";
  ctx.fillText(`${zmax.toFixed(0)} m`, 4, 12);
  ctx.fillText(`${zmin.toFixed(0)} m`, 4, h - 3);
  const gpc = tr.grade[S.car.proj.i] * 100;
  ctx.fillStyle = "#e8eaf0";
  ctx.fillText(`${gpc >= 0 ? "+" : ""}${gpc.toFixed(1)}% grade`, w - 78, 12);
}

function drawLearning() {
  const c = $("curve"), ctx = c.getContext("2d");
  const w = c.width, h = c.height, hist = S.history;
  ctx.clearRect(0, 0, w, h);
  if (hist.length < 2) {
    ctx.fillStyle = "rgba(255,255,255,0.35)"; ctx.font = "11px ui-monospace,monospace";
    ctx.fillText("waiting for first generations...", 10, h / 2);
    return;
  }
  const hasMean = hist.some((r) => r.mean != null);
  let lo = Infinity, hi = -Infinity;
  for (const r of hist) {
    lo = Math.min(lo, r.best); hi = Math.max(hi, r.best);
    if (r.mean != null) { lo = Math.min(lo, r.mean); hi = Math.max(hi, r.mean); }
  }
  const pad = (hi - lo) * 0.08 + 1; lo -= pad; hi += pad;
  const X = (i) => (i / (hist.length - 1)) * (w - 40) + 34;
  const Y = (v) => h - 16 - ((v - lo) / (hi - lo)) * (h - 26);

  // Stage boundaries, where the curriculum changes and scores rebase.
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  for (let i = 1; i < hist.length; i++) {
    if (hist[i].duration !== hist[i - 1].duration) {
      ctx.beginPath(); ctx.moveTo(X(i), 6); ctx.lineTo(X(i), h - 14); ctx.stroke();
    }
  }
  const line = (get, color, width) => {
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
    hist.forEach((r, i) => (i ? ctx.lineTo(X(i), Y(get(r))) : ctx.moveTo(X(i), Y(get(r)))));
    ctx.stroke();
  };
  if (hasMean) line((r) => r.mean, "rgba(255,255,255,0.3)", 1);
  line((r) => r.best, "#00e0a4", 2);
  ctx.fillStyle = "rgba(255,255,255,0.45)"; ctx.font = "10px ui-monospace,monospace";
  ctx.fillText(hi.toFixed(0), 2, 12);
  ctx.fillText(lo.toFixed(0), 2, h - 16);
  ctx.fillText(`gen ${hist[hist.length - 1].gen}`, w - 62, h - 3);
  ctx.fillStyle = "#00e0a4"; ctx.fillText("policy", 34, 12);
  if (hasMean) { ctx.fillStyle = "rgba(255,255,255,0.45)"; ctx.fillText("population mean", 78, 12); }
}

// Jev panel: typed decision distribution + confidence + calibration.
function drawDecision() {
  const c = $("decisions"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  const p = S.policy, rowH = h / Jev.N_DEC;
  ctx.font = "10px ui-monospace,monospace";
  for (let i = 0; i < Jev.N_DEC; i++) {
    const d = Jev.DECISIONS[i], y = i * rowH;
    const chosen = i === p.decision;
    ctx.fillStyle = chosen ? "rgba(255,255,255,0.08)" : "transparent";
    ctx.fillRect(0, y, w, rowH - 1);
    ctx.fillStyle = chosen ? "#fff" : "rgba(255,255,255,0.5)";
    ctx.fillText(d.name, 6, y + rowH / 2 + 3);
    const bx = 104, bw = w - bx - 44;
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(bx, y + rowH / 2 - 4, bw, 8);
    ctx.fillStyle = d.color;
    ctx.globalAlpha = chosen ? 1 : 0.5;
    ctx.fillRect(bx, y + rowH / 2 - 4, bw * p.probs[i], 8);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.fillText((p.probs[i] * 100).toFixed(0).padStart(3) + "%", w - 36, y + rowH / 2 + 3);
  }
}

function drawReliability() {
  const c = $("reliability"), ctx = c.getContext("2d");
  const w = c.width, h = c.height, R = S.reliability;
  ctx.clearRect(0, 0, w, h);
  const pad = 22;
  ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1;
  ctx.strokeRect(pad, 6, w - pad - 8, h - pad);
  ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(255,255,255,0.3)";
  ctx.beginPath(); ctx.moveTo(pad, h - pad + 6); ctx.lineTo(w - 8, 6); ctx.stroke();
  ctx.setLineDash([]);
  if (R) {
    ctx.fillStyle = "#00e0a4";
    for (let b = 0; b < R.n.length; b++) {
      if (!R.n[b]) continue;
      const conf = (b + 0.5) / R.n.length, acc = R.hit[b] / R.n[b];
      const x = pad + conf * (w - pad - 8), y = h - pad + 6 - acc * (h - pad);
      const rad = 2 + 4 * Math.min(1, R.n[b] / 120);
      ctx.beginPath(); ctx.arc(x, y, rad, 0, 7); ctx.fill();
    }
  }
  ctx.fillStyle = "rgba(255,255,255,0.45)"; ctx.font = "9px ui-monospace,monospace";
  ctx.fillText("outcome", 2, 12);
  ctx.fillText("confidence", w - 62, h - 4);
}

// ------------------------------------------------------------------- live loop
const obsBuf = new Float64Array(F1.N_INPUTS);
const pend = [];
const REL_BINS = 10;

function resetRun() {
  if (S.respawnTimer) { clearTimeout(S.respawnTimer); S.respawnTimer = null; }
  // Flying start: cross the line at racing speed, the way a qualifying lap does.
  S.car.reset(0, S.ref.v[0] * 0.98, F1.PHYS_DT, {
    pose: { x: S.ref.line.x[0], y: S.ref.line.y[0], psi: S.ref.line.psi[0] },
    timed: true,
  });
  S.trail.length = 0; pend.length = 0;
  S.escCount = 0; S.ctrlCount = 0; S.lastLap = null; S.lapDirty = false;
  S.crashInfo = null; S.attempts++;
  S.reliability = { n: new Array(REL_BINS).fill(0), hit: new Array(REL_BINS).fill(0) };
  if (S.refDriver) S.refDriver.prevSteer = 0;
}

function controlStep() {
  const car = S.car, p = S.policy;
  car.observe(obsBuf);
  for (let i = 0; i < F1.N_INPUTS; i++) p.obs[i] = obsBuf[i];
  p.forward();
  let steer = p.steer, pedal = p.pedal;
  S.escalated = false;
  if (S.driverMode === "reference") {
    const f = S.refDriver.control(car);
    steer = f.steer; pedal = f.pedal;
  } else if (S.escalate && p.confidence < S.escThreshold) {
    const f = S.refDriver.control(car);
    steer = f.steer; pedal = f.pedal;
    S.escalated = true; S.escCount++;
  }
  S.ctrlCount++;
  car.setControls(steer, pedal);

  // Grade the confidence emitted ~0.75 s ago against what actually happened.
  pend.push({ conf: p.confidence, due: car.t + 0.75 });
  while (pend.length && pend[0].due <= car.t) {
    const e = pend.shift();
    const good = car.alive && car.useF < 1.02 && car.useR < 1.02 ? 1 : 0;
    const b = Math.min(REL_BINS - 1, Math.floor(e.conf * REL_BINS));
    S.reliability.n[b]++; S.reliability.hit[b] += good;
  }

  S.trail.push({ x: car.X, y: car.Y, dec: p.decision });
  if (S.trail.length > 420) S.trail.shift();
}

let lastFrame = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dtWall = Math.min(0.2, (now - lastFrame) / 1000); // tolerate a throttled pane
  lastFrame = now;

  const car = S.car;
  const steps = Math.round((dtWall * S.speed) / F1.PHYS_DT);
  for (let k = 0; k < steps; k++) {
    if (!car.alive) break;
    if (car.stepCount % F1.CTRL_EVERY === 0) controlStep();
    const lapsBefore = car.laps;
    car.step();
    if (car.laps > lapsBefore && car.lapTimes.length) {
      S.lastLap = car.lapTimes[car.lapTimes.length - 1];
      if (S.lastLap && !S.lapDirty && (S.lapBest == null || S.lastLap < S.lapBest)) {
        S.lapBest = S.lastLap;
      }
      S.lapDirty = false;
    }
  }
  if (!car.alive) {
    if (!S.crashInfo) {
      const frac = (((car.proj.s % S.track.length) + S.track.length) % S.track.length) / S.track.length;
      S.crashInfo = { x: car.X, y: car.Y, s: car.proj.s, frac, reason: car.reason, dist: car.dist };
      S.crashes.push({ x: car.X, y: car.Y, reason: car.reason });
      if (S.crashes.length > 12) S.crashes.shift();
      S.furthest = Math.max(S.furthest, car.dist);
    }
    const c = S.crashInfo;
    setText("status", `${c.reason.toUpperCase()} at ${c.s.toFixed(0)} m (${(c.frac * 100).toFixed(0)}% of lap)`);
    $("status").className = "pill bad";
    // Hold the wreck on screen long enough to read, instead of flickering
    // through respawns that tell you nothing.
    if (!S.respawnTimer) {
      S.respawnTimer = setTimeout(() => { S.respawnTimer = null; resetRun(); }, 1800);
    }
  } else {
    setText("status", S.escalated ? "ESCALATED TO FALLBACK" : "RUNNING");
    $("status").className = "pill " + (S.escalated ? "warn" : "ok");
  }

  // Camera
  const cv = $("track"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const tgt = S.follow
    ? { scale: 3.4, cx: car.X, cy: car.Y }
    : Object.assign(fitView(S.track, W, H), {});
  const k = S.snapView ? 1 : S.follow ? 0.12 : 0.08;
  S.snapView = false;
  view.scale += (tgt.scale - view.scale) * k;
  view.cx += (tgt.cx - view.cx) * k;
  view.cy += (tgt.cy - view.cy) * k;

  drawScene(ctx, W, H);
  drawGauges(); drawElevation(); drawDecision(); drawReliability(); drawLearning();
  updateReadouts();
}

function updateReadouts() {
  const car = S.car, p = S.policy;
  setText("speed", (car.speed * 3.6).toFixed(0));
  let gear = 1;
  for (let i = 0; i < car.spec.gears.length; i++) if (car.speed * 3.6 > car.spec.gears[i]) gear = i + 2;
  setText("gear", Math.min(8, gear));
  setStyle("thr", "width", (car.throttle * 100).toFixed(0) + "%");
  setStyle("brk", "width", (car.brake * 100).toFixed(0) + "%");
  setStyle("steerBar", "transform", `translateX(${(car.delta / car.spec.maxSteer) * 50}%)`);
  setText("decName", Jev.DECISIONS[p.decision].name);
  setStyle("decName", "color", Jev.DECISIONS[p.decision].color);
  setText("conf", (p.confidence * 100).toFixed(0) + "%");
  setStyle("confBar", "width", (p.confidence * 100).toFixed(0) + "%");
  setStyle("confBar", "background", p.confidence < S.escThreshold ? "#ffb020" : "#00e0a4");
  setText("escRate", S.ctrlCount ? ((100 * S.escCount) / S.ctrlCount).toFixed(1) + "%" : "0.0%");
  setText("lastLap", fmtLap(S.lastLap));
  setText("bestLap", fmtLap(S.lapBest));
  setText("delta", S.lapBest ? `${S.lapBest > S.ref.time ? "+" : ""}${(S.lapBest - S.ref.time).toFixed(3)}s` : "--");
  setText("gen", S.gen);
  setText("attempts", S.attempts);
  setText("furthest", `${(Math.max(S.furthest, car.dist) / 1000).toFixed(2)} km`);
  setText("progress", `${((Math.max(0, car.dist) / S.track.length) * 100).toFixed(0)}%`);
  if (S.rec) {
    setText("fit", S.rec.best.toFixed(0));
    setText("brier", S.rec.brier.toFixed(3));
    setText("stage", S.rec.duration != null ? `${S.rec.duration}s` : "—");
    setText("trainSrc", S.policySource === "live" ? "in-browser worker"
      : S.lastLiveAt != null && Date.now() - S.lastLiveAt < 45000 ? "server trainer (running)"
      : "server trainer (finished)");
    setText("trainerLap",
      S.policySource === "live" || !S.trainerBestLap ? "—" : fmtLap(S.trainerBestLap));
    // Checkpoints arrive every few seconds at most, so show the age rather
    // than leaving a static number that looks stuck.
    setText("ckptAge", S.ckptAt == null ? "—"
      : `${Math.round((performance.now() - S.ckptAt) / 1000)}s ago`);
  }
}

// ------------------------------------------------------------------------ init
function loadTrack(key) {
  S.key = key;
  S.track = new F1.Track(key, TRACKS[key]);
  S.ref = LapSim.reference(S.track, F1.RB21);
  S.refDriver = new Jev.ReferenceDriver(S.track, S.ref, 0.85);
  S.art = buildTrackPath(S.track);
  S.car = new F1.Car(S.track, F1.RB21);
  S.policy = new Jev.Policy(Jev.randomGenome(Jev.makeRng(7)));
  // Weights are shared across circuits, so switching track keeps the policy
  // and only resets the per-circuit readouts.
  S.ckptTag = null;
  loadCheckpoint();
  startCheckpointPolling();
  S.gen = 0; S.history = []; S.rec = null; S.lapBest = null;
  resetRun();

  const m = S.track.meta;
  $("trackName").textContent = S.track.name;
  $("trackInfo").textContent =
    `${(S.track.length / 1000).toFixed(3)} km · ${(S.art.zmax - S.art.zmin).toFixed(0)} m elevation range · ${F1.RB21.aero[m.aero].label}`;
  const hb = $("heldOut");
  if (hb) hb.style.display = m.heldOut ? "inline-block" : "none";
  $("qss").textContent = fmtLap(S.ref.time);
  $("pole").textContent = m.pole2025 ? fmtLap(m.pole2025) : "—";
  $("poleBy").textContent = m.poleBy;
  $("vmax").textContent = (S.ref.vmax * 3.6).toFixed(0) + " km/h";

  if (S.worker) S.worker.terminate();
  S.worker = new Worker("worker.js");
  S.worker.onmessage = (e) => {
    const m2 = e.data;
    if (m2.type === "gen") {
      S.gen = m2.rec.gen; S.rec = m2.rec;
      S.history.push(m2.rec);
      if (S.history.length > 900) S.history.shift();
      if (S.policySource === "live") S.policy.g = m2.theta; // drive with the latest weights
      updatePolicyBadge();
    }
  };
  S.worker.postMessage({ cmd: "init", track: key });
  S.workerSeeded = false;
  if (S.training) S.worker.postMessage({ cmd: "start" });
}

// Weights from the headless trainer. There is ONE set for every circuit --
// the policy is trained across several at once and is meant to transfer -- so
// these are not keyed by track. `unified.live.json` is rewritten every couple
// of generations while training runs; `unified.json` is the best finished
// checkpoint.
async function loadCheckpoint() {
  const got = [];
  for (const file of ["unified.live.json", "unified.json"]) {
    try {
      if (file.endsWith(".live.json") && S.liveMisses > 2) continue;
      const r = await fetch(`weights/${file}`, { cache: "no-store" });
      if (!r.ok) { if (file.endsWith(".live.json")) S.liveMisses++; continue; }
      if (file.endsWith(".live.json")) S.liveMisses = 0;
      const w = await r.json();
      w.live = file.endsWith(".live.json");
      if (w.params === Jev.N_PARAMS) got.push(w);
      else S.ckptMismatch = `${file}: ${w.params} params, build expects ${Jev.N_PARAMS}`;
    } catch (e) { /* try the next file */ }
  }
  if (!got.length) { if (S.policySource === "untrained") updatePolicyBadge(); return; }

  if (S.liveMisses > 2 && S.ckptTag) { clearInterval(S.pollTimer); S.pollTimer = null; }
  const live = got.find((w) => w.live);
  if (live && live.gen !== S.lastLiveGen) { S.lastLiveGen = live.gen; S.lastLiveAt = Date.now(); }
  const trainerActive = S.lastLiveAt != null && Date.now() - S.lastLiveAt < 45000;
  const best = trainerActive && live ? live : got.reduce((a, b) => (b.fitness > a.fitness ? b : a));

  const tag = `${best.gen}:${best.fitness.toFixed(0)}`;
  if (tag === S.ckptTag) return;
  S.pretrained = Float32Array.from(best.theta);
  S.ckptMeta = best;
  S.ckptTag = tag;
  S.ckptAt = performance.now();
  const mine = best.perTrack && best.perTrack.find((t) => t.key === S.key);
  S.trainerBestLap = mine ? mine.qualiLap : null;
  if (S.policySource !== "live") {
    S.gen = best.gen;
    S.rec = { gen: best.gen, best: best.fitness, mean: null,
              brier: best.brier, duration: best.duration };
    const last = S.history[S.history.length - 1];
    if (!last || last.gen !== best.gen) S.history.push(S.rec);
    if (S.history.length > 900) S.history.shift();
    usePretrained({ keepRunning: S.policySource === "pretrained" });
  }
}

// Poll for newer weights while the headless trainer is running.
function startCheckpointPolling() {
  clearInterval(S.pollTimer);
  S.pollTimer = setInterval(() => {
    if (S.policySource !== "live") loadCheckpoint();
  }, 3000);
}

function usePretrained(opts = {}) {
  if (!S.pretrained) return;
  S.policy.g = S.pretrained;
  S.policySource = "pretrained";
  // Swapping weights mid-lap invalidates the lap in progress rather than
  // banking a time that two different policies drove.
  if (opts.keepRunning && S.car.alive) S.lapDirty = true;
  else { S.lapBest = null; resetRun(); }
  updatePolicyBadge();
}

function updatePolicyBadge() {
  const b = $("policyBadge");
  if (S.driverMode === "reference") { b.textContent = "reference fallback"; b.className = "pill warn"; }
  else if (S.policySource === "live") { b.textContent = `live policy · gen ${S.gen}`; b.className = "pill ok"; }
  else if (S.policySource === "pretrained") {
    const m = S.ckptMeta, running = S.lastLiveAt != null && Date.now() - S.lastLiveAt < 45000;
    b.textContent = m ? `unified · gen ${m.gen}` : "trained";
    b.className = "pill ok";
  } else { b.textContent = "untrained"; b.className = "pill bad"; }
  $("btnUseBest").disabled = !S.pretrained || S.policySource === "pretrained";
}

function resize() {
  for (const id of ["track", "gauges", "elev", "curve", "decisions", "reliability"]) {
    const c = $(id), r = c.getBoundingClientRect();
    c.width = Math.max(1, Math.round(r.width));
    c.height = Math.max(1, Math.round(r.height));
  }
}

window.addEventListener("resize", resize);

document.addEventListener("DOMContentLoaded", () => {
  resize();
  loadTrack("monza");

  $("trackSel").onchange = (e) => loadTrack(e.target.value);
  $("btnTrain").onclick = () => {
    S.training = !S.training;
    // Hand the worker the trained weights to start from, not random ones.
    if (S.training && S.pretrained && !S.workerSeeded) {
      S.worker.postMessage({ cmd: "init", seed: Float32Array.from(S.pretrained) });
      S.workerSeeded = true;
    }
    S.worker.postMessage({ cmd: S.training ? "start" : "pause" });
    if (S.training) {
      S.policySource = "live"; S.lapBest = null;
      S.history = []; S.gen = 0; S.rec = null;   // separate run, separate curve
      resetRun();
    }
    $("btnTrain").textContent = S.training ? "Pause training" : "Start training";
    $("btnTrain").classList.toggle("active", S.training);
    updatePolicyBadge();
  };
  $("btnUseBest").onclick = usePretrained;
  $("btnReset").onclick = () => {
    S.worker.postMessage({ cmd: "reset" });
    S.workerSeeded = false;
    S.training = false;
    $("btnTrain").textContent = "Start training";
    $("btnTrain").classList.remove("active");
    S.gen = 0; S.history = []; S.rec = null; S.lapBest = null;
    if (S.pretrained) usePretrained(); else { S.policySource = "untrained"; resetRun(); }
    updatePolicyBadge();
  };
  $("btnRespawn").onclick = resetRun;
  $("optFollow").onchange = (e) => { S.follow = e.target.checked; S.snapView = true; };
  $("optEscalate").onchange = (e) => (S.escalate = e.target.checked);
  $("driverSel").onchange = (e) => {
    S.driverMode = e.target.value; S.lapBest = null; resetRun(); updatePolicyBadge();
  };
  $("speedSel").onchange = (e) => (S.speed = +e.target.value);

  requestAnimationFrame(frame);
});
