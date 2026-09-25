// Track geometry + Red Bull RB21 vehicle dynamics.
// Shared by the browser app and the headless trainer (tools/train.js).
(function (root) {
  "use strict";

  const G = 9.81;

  // Red Bull RB21 (2025). Dimensions/mass/PU from the 2025 FIA technical regs and
  // published team figures. Aero and tyre coefficients are not public, so they
  // were fitted (tools/cal2.js) until the quasi-steady-state limit lap in
  // lapsim.js lands ~1 % under real 2025 pole at Monza, Spa and COTA.
  const RB21 = {
    name: "Oracle Red Bull Racing RB21",
    mass: 815,            // kg: 800 kg regulation minimum (car + driver) + ~15 kg qualifying fuel
    wheelbase: 3.60,      // m (regulation maximum; RB21 runs at the limit)
    width: 2.00,          // m
    length: 5.63,         // m
    cgHeight: 0.28,       // m
    frontWeight: 0.455,   // static weight on front axle (regs require >= 44.5 %)
    Iz: 1050,             // kg m^2 yaw inertia
    power: 735e3,         // W: Honda RBPTH003 ~ 555 kW ICE + 120 kW MGU-K ... ~1000 hp combined
    driveEff: 0.93,       // gearbox/driveline efficiency
    gears: [80, 118, 152, 185, 220, 255, 292, 360], // km/h at shift (display only)
    maxSteer: 0.33,       // rad road-wheel lock
    // Aero packages (0.5*rho*C*A form, m^2). Teams bring different wings per circuit.
    aero: {
      low:    { ClA: 3.55, CdA: 1.08, balance: 0.42, label: "Low downforce (Monza spec)" },
      medium: { ClA: 4.30, CdA: 1.20, balance: 0.42, label: "Medium downforce (Spa spec)" },
      high:   { ClA: 5.05, CdA: 1.34, balance: 0.42, label: "High downforce (COTA spec)" },
    },
    // Pirelli P Zero slicks (305 mm front / 405 mm rear, 18")
    // 305 mm front / 405 mm rear, so the rear axle carries more lateral grip
    // per newton of load than the front.
    tyre: { mu0: 2.09, loadSens: 0.12, Fz0: 3500, peakSlip: 0.10, C: 1.35, rearGrip: 1.10 },
    brakeBiasOffset: 0.05, // forward of grip-proportional: keeps the rear stable
  };

  const TRACK_META = {
    monza: { aero: "low",    altitude: 162, pole2025: 78.792, poleBy: "M. Verstappen (RB21), 2025" },
    spa:   { aero: "medium", altitude: 400, pole2025: 100.562, poleBy: "L. Norris (MCL39), 2025" },
    cota:  { aero: "high",   altitude: 150, pole2025: 92.510,  poleBy: "M. Verstappen (RB21), 2025" },
    // Held out from training entirely, used only to test whether one policy
    // transfers to a circuit it has never seen. Real pole times are left null
    // rather than quoted from memory.
    zandvoort:   { aero: "high",   altitude: 5,   pole2025: null, poleBy: "" },
    catalunya:   { aero: "medium", altitude: 140, pole2025: null, poleBy: "" },
    hungaroring: { aero: "high",   altitude: 230, pole2025: null, poleBy: "" },
    suzuka:   { aero: "high",   altitude: 45, pole2025: null, poleBy: "held-out circuit", heldOut: true },
    shanghai: { aero: "medium", altitude: 5,  pole2025: null, poleBy: "held-out circuit", heldOut: true },
  };

  // Tyre peak friction coefficient under vertical load Fz (N): load-sensitive.
  function tyreMu(Fz, t) {
    return t.mu0 * Math.max(0.6, 1 - t.loadSens * (Fz / t.Fz0 - 1));
  }

  // ---------------------------------------------------------------- Track
  class Track {
    constructor(key, raw) {
      this.key = key;
      this.name = raw.name;
      this.country = raw.country;
      this.meta = TRACK_META[key];
      const n = (this.n = raw.x.length);
      this.x = Float64Array.from(raw.x);
      this.y = Float64Array.from(raw.y);
      this.z = Float64Array.from(raw.z);
      this.wl = Float64Array.from(raw.wl);
      this.wr = Float64Array.from(raw.wr);
      this.s = new Float64Array(n + 1);
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n;
        this.s[i + 1] = this.s[i] + Math.hypot(this.x[j] - this.x[i], this.y[j] - this.y[i]);
      }
      this.length = this.s[n];
      this.ds = this.length / n;

      this.tx = new Float64Array(n); this.ty = new Float64Array(n);
      this.heading = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const a = (i - 1 + n) % n, b = (i + 1) % n;
        const dx = this.x[b] - this.x[a], dy = this.y[b] - this.y[a];
        const l = Math.hypot(dx, dy);
        this.tx[i] = dx / l; this.ty[i] = dy / l;
        this.heading[i] = Math.atan2(dy, dx);
      }
      // Signed curvature (left positive), smoothed over ~15 m to suppress survey noise.
      const k = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const b = (i + 1) % n;
        k[i] = wrapAngle(this.heading[b] - this.heading[i]) / this.ds;
      }
      this.kappa = smoothLoop(k, 2);
      // Elevation derived quantities.
      const grade = new Float64Array(n), kv = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const a = (i - 1 + n) % n, b = (i + 1) % n;
        grade[i] = (this.z[b] - this.z[a]) / (2 * this.ds);
        kv[i] = (this.z[b] - 2 * this.z[i] + this.z[a]) / (this.ds * this.ds);
      }
      this.grade = grade;
      this.kv = smoothLoop(kv, 3); // vertical curvature: + compression, - crest
    }

    idx(s) { let i = Math.floor(s / this.ds) % this.n; return i < 0 ? i + this.n : i; }
    sample(arr, s) {
      s = ((s % this.length) + this.length) % this.length;
      const f = s / this.ds, i = Math.floor(f) % this.n, j = (i + 1) % this.n, t = f - Math.floor(f);
      return arr[i] * (1 - t) + arr[j] * t;
    }

    // Project world point onto the centreline near index hint. Returns {i, s, d}
    // where d is lateral offset (left positive).
    project(px, py, hint, out) {
      const n = this.n;
      let best = hint, bd = Infinity;
      for (let o = -12; o <= 24; o++) {
        const i = (hint + o + n) % n;
        const dx = px - this.x[i], dy = py - this.y[i];
        const d2 = dx * dx + dy * dy;
        if (d2 < bd) { bd = d2; best = i; }
      }
      const i = best;
      const dx = px - this.x[i], dy = py - this.y[i];
      let along = dx * this.tx[i] + dy * this.ty[i];
      out.i = i;
      out.s = this.s[i] + along;
      out.d = -dx * this.ty[i] + dy * this.tx[i];
      return out;
    }
  }

  function wrapAngle(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  }

  function smoothLoop(v, radius) {
    const n = v.length, out = new Float64Array(n);
    let w = 0;
    const ker = [];
    for (let o = -3 * radius; o <= 3 * radius; o++) { const k = Math.exp(-0.5 * (o / radius) ** 2); ker.push(k); w += k; }
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let o = -3 * radius, q = 0; o <= 3 * radius; o++, q++) acc += v[(i + o + n) % n] * ker[q];
      out[i] = acc / w;
    }
    return out;
  }

  // ---------------------------------------------------------------- Car
  const STALL_DISTANCE = 5;  // m of progress required ...
  const STALL_SECONDS = 3;   // ... within this long, or the run is over
  const STEER_RATE = 4.0;    // rad/s at the road wheel
  const PHYS_DT = 1 / 240;   // physics step
  const CTRL_EVERY = 4;      // controller runs at 60 Hz

  class Car {
    constructor(track, spec = RB21) {
      this.track = track;
      this.spec = spec;
      this.aero = spec.aero[track.meta.aero];
      this.proj = { i: 0, s: 0, d: 0 };
      this.reset();
    }

    // `opts.pose` places the car somewhere other than the centreline -- the
    // trainer spawns on the racing line, because the reference speed it
    // launches at belongs to the racing line's radius, not the centreline's.
    // `opts.timed` starts the lap clock immediately, for a flying start that
    // crosses the line already at speed the way a qualifying lap does.
    reset(s0 = 0, v0 = 0, dt = PHYS_DT, opts = {}) {
      const tr = this.track, i = tr.idx(s0), pose = opts.pose;
      this.dt = dt;
      if (pose) { this.X = pose.x; this.Y = pose.y; this.psi = pose.psi; }
      else { this.X = tr.x[i]; this.Y = tr.y[i]; this.psi = tr.heading[i]; }
      this.vx = v0; this.vy = 0; this.r = 0;
      this.delta = 0; this.steerCmd = 0; this.pedal = 0;
      this.throttle = 0; this.brake = 0;
      this.ax = 0; this.ay = 0;
      this.hint = i;
      this.dist = 0;             // cumulative progress along centreline (m)
      this.t = 0; this.alive = true; this.reason = "";
      this.laps = 0; this.lapStart = opts.timed ? 0 : null; this.lapTimes = [];
      this.useF = 0; this.useR = 0; this.tractionLimited = false;
      this.maxSpeed = 0;
      this.lastProgressT = 0; this.lastProgressDist = 0;
      tr.project(this.X, this.Y, this.hint, this.proj);
      this.hint = this.proj.i;
      this.sPrev = this.proj.s;
      this.d = this.proj.d;
      this.stepCount = 0;
    }

    get speed() { return Math.hypot(this.vx, this.vy); }

    setControls(steer, pedal) {
      this.steerCmd = Math.max(-1, Math.min(1, steer));
      this.pedal = Math.max(-1, Math.min(1, pedal));
    }

    // Steering authority shrinks with speed like a real driver's usable lock.
    steerLimit(v) { return Math.min(this.spec.maxSteer, 360 / (v * v + 1) + 0.03); }

    step() {
      const sp = this.spec, tr = this.track, a = this.aero, ty = sp.tyre;
      const dt = this.dt, m = sp.mass, L = sp.wheelbase;
      const lf = L * (1 - sp.frontWeight), lr = L * sp.frontWeight;

      const vx = Math.max(this.vx, 0);
      // Steering actuator lag (~50 ms) plus a slew-rate limit. The lag alone
      // permits ~13 rad/s at the road wheel, which let the policy bang the
      // wheel from lock to lock at controller rate -- it flipped sign on most
      // control steps and the car visibly weaved. A rack and a pair of arms
      // cannot do that; STEER_RATE caps it at lock-to-lock in ~0.17 s.
      const target = this.steerCmd * this.steerLimit(vx);
      const want = this.delta + (target - this.delta) * Math.min(1, dt / 0.05);
      const maxStep = STEER_RATE * dt;
      this.delta += Math.max(-maxStep, Math.min(maxStep, want - this.delta));
      const delta = this.delta;
      this.throttle = Math.max(0, this.pedal);
      this.brake = Math.max(0, -this.pedal);

      // Track state at car position
      const i = this.proj.i;
      const grade = tr.grade[i], kv = tr.kv[i];
      const cosTh = 1 / Math.sqrt(1 + grade * grade), sinTh = grade * cosTh;
      const alt = tr.meta.altitude + tr.z[i];
      const rho = 1.225 * Math.exp(-alt / 8500);
      const q = 0.5 * rho * vx * vx;
      const drag = q * a.CdA, down = q * a.ClA;
      const eps = wrapAngle(this.psi - tr.heading[i]);

      // Vertical load: gravity normal component + compression/crest + aero
      let Fz = m * (G * cosTh + vx * vx * kv) + down;
      Fz = Math.max(Fz, 0.08 * m * G);
      const dFz = (m * this.ax * sp.cgHeight) / L; // longitudinal load transfer
      let Fzf = (Fz - down) * sp.frontWeight + down * a.balance - dFz;
      let Fzr = Fz - Fzf;
      Fzf = Math.max(Fzf, 200); Fzr = Math.max(Fzr, 200);
      const muF = tyreMu(Fzf / 2, ty), muR = tyreMu(Fzr / 2, ty) * ty.rearGrip;
      const FmaxF = muF * Fzf, FmaxR = muR * Fzr;

      // Longitudinal forces
      const Pw = sp.power * sp.driveEff * this.throttle;
      let Fdrive = Pw / Math.max(vx, 4);
      // Brake-by-wire distributes retardation in proportion to the grip each
      // axle actually has, so both reach the friction limit together. A fixed
      // split would over-saturate the rear once load transfers forward, and an
      // axle with no lateral capacity left makes the car snap into a spin.
      const brakeTot = this.brake * (FmaxF + FmaxR);
      const biasF = Math.max(0.5, Math.min(0.75, FmaxF / (FmaxF + FmaxR) + sp.brakeBiasOffset));
      let Fxf = -biasF * brakeTot;
      let Fxr = Fdrive - (1 - biasF) * brakeTot - (this.throttle < 0.05 ? 0.012 * Fz : 0); // engine braking
      if (vx < 0.3 && this.brake > 0) { Fxf = 0; Fxr = Math.max(Fxr, 0); }
      // A tyre at full longitudinal slip still points part of its force vector
      // sideways, so cap Fx below the friction circle rather than on it --
      // otherwise a traction-limited axle has literally zero lateral stiffness
      // and any yaw disturbance diverges.
      const CLIP = 0.94;
      Fxf = Math.max(-CLIP * FmaxF, Math.min(CLIP * FmaxF, Fxf));
      Fxr = Math.max(-CLIP * FmaxR, Math.min(CLIP * FmaxR, Fxr));
      this.tractionLimited = Fdrive > CLIP * FmaxR;

      // Lateral: simplified Pacejka with friction-ellipse coupling
      const vxs = Math.max(vx, 3);
      const alphaF = delta - Math.atan2(this.vy + lf * this.r, vxs);
      const alphaR = -Math.atan2(this.vy - lr * this.r, vxs);
      const B = Math.tan(Math.PI / (2 * ty.C)) / ty.peakSlip;
      const DyF = Math.sqrt(Math.max(0, FmaxF * FmaxF - Fxf * Fxf));
      const DyR = Math.sqrt(Math.max(0, FmaxR * FmaxR - Fxr * Fxr));
      const Fyf = DyF * Math.sin(ty.C * Math.atan(B * alphaF));
      const Fyr = DyR * Math.sin(ty.C * Math.atan(B * alphaR));
      this.useF = Math.hypot(Fxf, Fyf) / FmaxF;
      this.useR = Math.hypot(Fxr, Fyr) / FmaxR;

      const cd = Math.cos(delta), sd = Math.sin(delta);
      const Fgrav = -m * G * sinTh * Math.cos(eps);
      const Fx = Fxr + Fxf * cd - Fyf * sd - drag + Fgrav;
      const Fy = Fyr + Fyf * cd + Fxf * sd;
      const Mz = lf * (Fyf * cd + Fxf * sd) - lr * Fyr;

      this.ax = Fx / m;
      this.ay = Fy / m;
      if (vx < 5) {
        // Low speed: kinematic bicycle (tyre model is stiff/singular here)
        this.vx = Math.max(0, vx + this.ax * dt);
        this.r = (this.vx * Math.tan(delta)) / L;
        this.vy *= 0.8;
      } else {
        this.vx = vx + (this.ax + this.vy * this.r) * dt;
        this.vy += (this.ay - vx * this.r) * dt;
        this.r += (Mz / sp.Iz) * dt;
      }
      this.psi += this.r * dt;
      const c = Math.cos(this.psi), s = Math.sin(this.psi);
      this.X += (this.vx * c - this.vy * s) * dt;
      this.Y += (this.vx * s + this.vy * c) * dt;
      this.t += dt;
      this.stepCount++;

      // Progress bookkeeping
      tr.project(this.X, this.Y, this.hint, this.proj);
      this.hint = this.proj.i;
      this.d = this.proj.d;
      let ds = this.proj.s - this.sPrev;
      if (ds < -tr.length / 2) ds += tr.length;
      else if (ds > tr.length / 2) ds -= tr.length;
      const before = this.dist;
      this.dist += ds;
      this.sPrev = this.proj.s;
      if (Math.floor(this.dist / tr.length) > Math.floor(before / tr.length) && this.dist > 0) {
        if (this.lapStart !== null) this.lapTimes.push(this.t - this.lapStart);
        this.lapStart = this.t;
        this.laps++;
      }
      const v = this.speed;
      if (v > this.maxSpeed) this.maxSpeed = v;

      // Track limits: all four wheels beyond the white line
      const half = sp.width / 2;
      if (this.d > tr.wl[i] + half || -this.d > tr.wr[i] + half) { this.alive = false; this.reason = "off track"; }
      else if (Math.abs(eps) > 1.4 && v > 5) { this.alive = false; this.reason = "spun"; }
      // A car that has stopped, crawled, or spun to a halt satisfies neither
      // test above, so without this it stays alive forever: frozen on screen,
      // and burning a whole training episode at no cost.
      else if (this.dist - this.lastProgressDist > STALL_DISTANCE) {
        this.lastProgressDist = this.dist; this.lastProgressT = this.t;
      } else if (this.t - this.lastProgressT > STALL_SECONDS) {
        this.alive = false; this.reason = "stalled";
      }
    }

    // Peak acceleration the tyres can deliver at this speed (m/s^2), treating
    // the four contact patches as equally loaded. Used only as an observation
    // feature, not in the force solve.
    gripAccel(v) {
      const sp = this.spec, tr = this.track, i = this.proj.i;
      const rho = 1.225 * Math.exp(-(tr.meta.altitude + tr.z[i]) / 8500);
      const Fz = sp.mass * G + 0.5 * rho * v * v * this.aero.ClA;
      return (tyreMu(Fz / 4, sp.tyre) * Fz) / sp.mass;
    }

    // Normalised observation vector for the driver network.
    observe(out) {
      const tr = this.track, i = this.proj.i, s = this.proj.s;
      const v = this.speed;
      const wl = tr.wl[i], wr = tr.wr[i];
      const eps = wrapAngle(this.psi - tr.heading[i]);
      let k = 0;
      out[k++] = v / 90;
      out[k++] = this.vy / 5;
      out[k++] = this.r;
      out[k++] = (this.d - (wl - wr) / 2) / ((wl + wr) / 2); // -1 right edge .. +1 left edge
      out[k++] = eps * 3;
      for (const la of LOOKAHEAD) {
        const kap = tr.sample(tr.kappa, s + la);
        out[k++] = Math.asinh(kap * 150) / 3;
      }
      // Braking demand: for several points ahead, how hard the car would have
      // to decelerate -- as a fraction of what the tyres can actually deliver --
      // to arrive at a speed that corner's radius can hold. The network can in
      // principle derive this from the raw curvature and speed it already sees,
      // but it involves a square root and a comparison per lookahead, which is
      // a lot to ask of a small MLP and is the one thing it must get right.
      const aMax = this.gripAccel(v);
      for (const la of BRAKE_LOOKAHEAD) {
        const kap = Math.abs(tr.sample(tr.kappa, s + la));
        const vCorner = Math.sqrt(aMax / Math.max(kap, 1e-5));
        const need = (v * v - vCorner * vCorner) / (2 * la);
        out[k++] = Math.max(-1, Math.min(2, need / aMax));
      }
      out[k++] = tr.sample(tr.grade, s + 40) * 6;
      out[k++] = Math.max(-2, Math.min(2, tr.sample(tr.kv, s + 25) * 300));
      out[k++] = this.delta / this.steerLimit(v);
      out[k++] = this.pedal;
      out[k++] = this.useF - 1;
      out[k++] = this.useR - 1;
      return out;
    }
  }
  const LOOKAHEAD = [6, 15, 28, 45, 68, 95, 130, 170, 220, 280];
  const BRAKE_LOOKAHEAD = [40, 80, 150, 240];
  const N_INPUTS = 5 + LOOKAHEAD.length + BRAKE_LOOKAHEAD.length + 6;

  const api = { G, RB21, TRACK_META, Track, Car, tyreMu, wrapAngle, smoothLoop, PHYS_DT, CTRL_EVERY, N_INPUTS, LOOKAHEAD, BRAKE_LOOKAHEAD, STEER_RATE };
  root.F1 = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
