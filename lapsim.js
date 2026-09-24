// Reference racing line + quasi-steady-state (QSS) lap simulation.
// Gives the theoretical limit for the RB21 model on each circuit, used to
// calibrate the physics against real 2025 pole times and as a target for the AI.
(function (root) {
  "use strict";
  const F1 = root.F1 || require("./physics.js");
  const { G, tyreMu } = F1;

  // Minimum-curvature line: projected gradient descent on the lateral offsets,
  // minimising sum |p[i-1] - 2p[i] + p[i+1]|^2 inside the track corridor.
  //
  // Two "improvements" were tried and measured worse, so they are deliberately
  // absent: solving the quadratic exactly with conjugate gradient plus an
  // active set (the first unconstrained solve pins far too many variables at
  // the bounds and never releases them -- objective 91 vs 76 here), and adding
  // an offset-smoothness term (lap times got monotonically worse as lambda
  // rose). A lower objective is also not a faster lap: 40k iterations reach a
  // lower objective but a 0.85 s slower QSS lap than the 6k used here.
  function racingLine(tr, iters = 6000, eta = 0.12, margin = 1.3) {
    const n = tr.n, nx = new Float64Array(n), ny = new Float64Array(n);
    for (let i = 0; i < n; i++) { nx[i] = -tr.ty[i]; ny[i] = tr.tx[i]; }
    const off = new Float64Array(n), px = new Float64Array(n), py = new Float64Array(n);
    const lo = new Float64Array(n), hi = new Float64Array(n);
    for (let i = 0; i < n; i++) { lo[i] = -tr.wr[i] + margin; hi[i] = tr.wl[i] - margin; }
    const cx = new Float64Array(n), cy = new Float64Array(n);
    for (let it = 0; it < iters; it++) {
      for (let i = 0; i < n; i++) { px[i] = tr.x[i] + nx[i] * off[i]; py[i] = tr.y[i] + ny[i] * off[i]; }
      for (let i = 0; i < n; i++) {
        const a = (i - 1 + n) % n, b = (i + 1) % n;
        cx[i] = px[a] - 2 * px[i] + px[b]; cy[i] = py[a] - 2 * py[i] + py[b];
      }
      for (let i = 0; i < n; i++) {
        const a = (i - 1 + n) % n, b = (i + 1) % n;
        const gx = cx[a] - 2 * cx[i] + cx[b], gy = cy[a] - 2 * cy[i] + cy[b];
        const o = off[i] - eta * (gx * nx[i] + gy * ny[i]);
        off[i] = o < lo[i] ? lo[i] : o > hi[i] ? hi[i] : o;
      }
    }
    for (let i = 0; i < n; i++) { px[i] = tr.x[i] + nx[i] * off[i]; py[i] = tr.y[i] + ny[i] * off[i]; }
    return { off, x: px, y: py };
  }

  function lineCurvature(x, y) {
    const n = x.length, k = new Float64Array(n), seg = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = (i - 1 + n) % n, b = (i + 1) % n;
      const ax = x[i] - x[a], ay = y[i] - y[a], bx = x[b] - x[i], by = y[b] - y[i];
      const la = Math.hypot(ax, ay), lb = Math.hypot(bx, by);
      k[i] = (2 * (ax * by - ay * bx)) / (la * lb * (la + lb));
      seg[i] = lb;
    }
    return { k: F1.smoothLoop(k, 1), seg };
  }

  // Forces for the point-mass model at speed v on sample i.
  function envelope(tr, spec, aero, i, v) {
    const m = spec.mass, grade = tr.grade[i];
    const cosTh = 1 / Math.sqrt(1 + grade * grade), sinTh = grade * cosTh;
    const rho = 1.225 * Math.exp(-(tr.meta.altitude + tr.z[i]) / 8500);
    const q = 0.5 * rho * v * v;
    const down = q * aero.ClA, drag = q * aero.CdA;
    const Fz = Math.max(m * (G * cosTh + v * v * tr.kv[i]) + down, 0.08 * m * G);
    const Fzf = (Fz - down) * spec.frontWeight + down * aero.balance, Fzr = Fz - Fzf;
    const gripF = tyreMu(Fzf / 2, spec.tyre) * Fzf, gripR = tyreMu(Fzr / 2, spec.tyre) * spec.tyre.rearGrip * Fzr;
    return { grip: gripF + gripR, gripR, drag, grav: -m * G * sinTh };
  }

  function qss(tr, spec, line) {
    const aero = spec.aero[tr.meta.aero], m = spec.mass, n = tr.n;
    const { k, seg } = lineCurvature(line.x, line.y);
    const vCorner = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let lo = 1, hi = 120;
      for (let it = 0; it < 40; it++) {
        const v = (lo + hi) / 2, e = envelope(tr, spec, aero, i, v);
        if (m * v * v * Math.abs(k[i]) <= e.grip) lo = v; else hi = v;
      }
      vCorner[i] = lo;
    }
    const lat = (i, v) => m * v * v * Math.abs(k[i]);
    const v = Float64Array.from(vCorner);
    // Backward pass (braking), wrapped twice for the closed loop.
    for (let pass = 0; pass < 2 * n; pass++) {
      const i = (n - 1 - (pass % n) + n) % n, j = (i + 1) % n;
      const e = envelope(tr, spec, aero, j, v[j]);
      const fx = Math.sqrt(Math.max(0, e.grip ** 2 - lat(j, v[j]) ** 2));
      const dec = (fx + e.drag - e.grav) / m;
      v[i] = Math.min(v[i], Math.sqrt(v[j] * v[j] + 2 * dec * seg[i]));
    }
    // Forward pass (traction/power limited, standing lap wraps to flying lap).
    for (let pass = 0; pass < 2 * n; pass++) {
      const i = pass % n, j = (i + 1) % n;
      const e = envelope(tr, spec, aero, i, v[i]);
      const fxMax = Math.sqrt(Math.max(0, e.gripR ** 2 - (lat(i, v[i]) * (1 - spec.frontWeight)) ** 2));
      const fx = Math.min(spec.power * spec.driveEff / Math.max(v[i], 1), fxMax);
      const acc = (fx - e.drag + e.grav) / m;
      v[j] = Math.min(v[j], Math.sqrt(Math.max(1, v[i] * v[i] + 2 * acc * seg[i])));
    }
    let t = 0, vmax = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      t += (2 * seg[i]) / (v[i] + v[j]);
      vmax = Math.max(vmax, v[i]);
    }
    return { time: t, v, vmax, kappa: k };
  }

  function reference(tr, spec) {
    const line = racingLine(tr);
    const n = line.x.length;
    line.psi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const a = (i - 1 + n) % n, b = (i + 1) % n;
      line.psi[i] = Math.atan2(line.y[b] - line.y[a], line.x[b] - line.x[a]);
    }
    const res = qss(tr, spec, line);
    return { line, ...res };
  }

  const api = { racingLine, qss, reference };
  root.LapSim = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
