// Jev-style driving policy.
//
// TypeSafe AI's Jev is a non-autoregressive model that emits a *typed,
// probabilistic decision* plus a *calibrated confidence*, so calling code can
// either act on it or escalate. It is a decision model, not a continuous
// controller, so this is that architecture applied to driving rather than the
// Jev weights themselves:
//
//   - a typed head chooses one of DECISIONS (HARD_BRAKE ... FULL_POWER)
//   - a continuous head trims steering, which genuinely needs to be continuous
//   - a confidence head predicts P(this decision turns out well)
//
// Training follows the RLCD idea — confidence is scored against what actually
// happened (did the car survive the next 0.75 s within the grip limit?), not
// against anyone's approval. Fitness = distance covered - Brier score of the
// confidence head, optimised by a genetic algorithm.
(function (root) {
  "use strict";
  const F1 = root.F1 || require("./physics.js");

  // Typed longitudinal decisions. Each maps to a pedal position in [-1, 1].
  const DECISIONS = [
    { name: "HARD_BRAKE",   pedal: -1.00, color: "#ff2d55" },
    { name: "BRAKE",        pedal: -0.70, color: "#ff6b3d" },
    { name: "TRAIL_BRAKE",  pedal: -0.35, color: "#ffa02e" },
    { name: "COAST",        pedal:  0.00, color: "#9aa4b2" },
    { name: "FEATHER",      pedal:  0.35, color: "#7fd4a0" },
    { name: "MODULATE",     pedal:  0.70, color: "#3ecf8e" },
    { name: "FULL_POWER",   pedal:  1.00, color: "#00e0a4" },
  ];
  const N_DEC = DECISIONS.length;

  const LAYERS = [F1.N_INPUTS, 40, 32, N_DEC + 2]; // + steer trim + confidence logit
  const N_PARAMS = (() => {
    let n = 0;
    for (let l = 0; l < LAYERS.length - 1; l++) n += LAYERS[l] * LAYERS[l + 1] + LAYERS[l + 1];
    return n;
  })();

  // Deterministic RNG so a seed reproduces a training run exactly.
  function makeRng(seed) {
    let s = seed >>> 0 || 1;
    return function () {
      s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0;
      return s / 4294967296;
    };
  }
  function gauss(rng) {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function randomGenome(rng) {
    const g = new Float32Array(N_PARAMS);
    let k = 0;
    for (let l = 0; l < LAYERS.length - 1; l++) {
      const fan = LAYERS[l], sd = Math.sqrt(1 / fan);
      for (let i = 0; i < fan * LAYERS[l + 1]; i++) g[k++] = gauss(rng) * sd;
      for (let i = 0; i < LAYERS[l + 1]; i++) g[k++] = 0;
    }
    return g;
  }

  // Forward pass. Scratch buffers are reused across calls (hot path).
  class Policy {
    constructor(genome) {
      this.g = genome;
      this.buf = LAYERS.map((n) => new Float64Array(n));
      this.probs = new Float64Array(N_DEC);
      this.obs = this.buf[0];
      this.decision = 3;
      this.confidence = 0.5;
      this.steer = 0;
      this.pedal = 0;
    }

    // `rng` samples the decision from the softmax instead of taking the argmax.
    // Used only while training: with argmax the fitness is piecewise-constant
    // in the weights, so a small perturbation usually changes no decision at
    // all and ES gets no gradient. That is how two circuits ended up never
    // once selecting FULL_POWER -- MODULATE was a local optimum nothing could
    // step out of. Evaluation stays greedy.
    forward(rng) {
      const g = this.g;
      let k = 0;
      for (let l = 0; l < LAYERS.length - 1; l++) {
        const a = this.buf[l], b = this.buf[l + 1], nIn = LAYERS[l], nOut = LAYERS[l + 1];
        const last = l === LAYERS.length - 2;
        for (let j = 0; j < nOut; j++) {
          let sum = 0;
          const base = k + j * nIn;
          for (let i = 0; i < nIn; i++) sum += a[i] * g[base + i];
          sum += g[k + nOut * nIn + j];
          b[j] = last ? sum : Math.tanh(sum);
        }
        k += nIn * nOut + nOut;
      }
      const out = this.buf[LAYERS.length - 1];
      // Typed decision: softmax over the logits, argmax is the emitted decision.
      let max = -Infinity;
      for (let i = 0; i < N_DEC; i++) if (out[i] > max) max = out[i];
      let sum = 0, best = 0;
      for (let i = 0; i < N_DEC; i++) { const e = Math.exp(out[i] - max); this.probs[i] = e; sum += e; }
      for (let i = 0; i < N_DEC; i++) { this.probs[i] /= sum; if (this.probs[i] > this.probs[best]) best = i; }
      if (rng) {
        let u = rng(), acc = 0;
        for (let i = 0; i < N_DEC; i++) { acc += this.probs[i]; if (u <= acc) { best = i; break; } }
      }
      this.decision = best;
      this.pedal = DECISIONS[best].pedal;
      this.steer = Math.tanh(out[N_DEC]);
      this.confidence = 1 / (1 + Math.exp(-out[N_DEC + 1]));
      return this;
    }
  }

  // ---------------------------------------------------------- safety fallback
  // Pure-pursuit on the reference line with a QSS speed profile. Used as the
  // escalation target when confidence is low, and as a benchmark driver.
  class ReferenceDriver {
    constructor(track, ref, margin = 0.97) {
      this.tr = track; this.ref = ref; this.margin = margin;
      const n = track.n, x = ref.line.x, y = ref.line.y;
      this.hd = new Float64Array(n); this.kap = new Float64Array(n); this.seg = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const a2 = (i - 1 + n) % n, b2 = (i + 1) % n;
        this.hd[i] = Math.atan2(y[b2] - y[a2], x[b2] - x[a2]);
        this.seg[i] = Math.hypot(x[b2] - x[i], y[b2] - y[i]);
      }
      const k = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const b2 = (i + 1) % n;
        k[i] = F1.wrapAngle(this.hd[b2] - this.hd[i]) / Math.max(this.seg[i], 0.1);
      }
      this.kap = F1.smoothLoop(k, 2);
      this.prevSteer = 0;
    }

    // Stanley steering on the reference line: curvature feed-forward plus
    // heading and cross-track feedback. Stable through chicanes, unlike pure
    // pursuit, whose look-ahead point flips sides in an S.
    control(car) {
      const tr = this.tr, ref = this.ref, n = tr.n, v = car.speed;
      let i = car.proj.i, bd = Infinity;
      for (let o = -6; o <= 6; o++) {
        const j = (i + o + n) % n;
        const d2 = (car.X - ref.line.x[j]) ** 2 + (car.Y - ref.line.y[j]) ** 2;
        if (d2 < bd) { bd = d2; i = j; }
      }
      const th = this.hd[i];
      const dx = car.X - ref.line.x[i], dy = car.Y - ref.line.y[i];
      const e = -dx * Math.sin(th) + dy * Math.cos(th);       // left of line positive
      const psiErr = F1.wrapAngle(car.psi - th);
      const ff = (i + Math.round((6 + 0.22 * v) / tr.ds)) % n;
      let delta = Math.atan(this.kap[ff] * car.spec.wheelbase)
                - 0.9 * psiErr
                - Math.atan((2.2 * e) / (v + 6));
      let steer = Math.max(-1, Math.min(1, delta / car.steerLimit(v)));
      const rate = 0.22;                                      // hands are not instant
      steer = Math.max(this.prevSteer - rate, Math.min(this.prevSteer + rate, steer));
      this.prevSteer = steer;

      // Slowest speed reachable inside the braking horizon.
      let target = Infinity;
      for (let a2 = 0; a2 < 120; a2++) {
        const k = (i + a2) % n;
        const vt = ref.v[k] * this.margin;
        target = Math.min(target, Math.sqrt(vt * vt + 2 * 32 * a2 * tr.ds));
      }
      const err = target - v;
      let pedal = Math.max(-1, Math.min(1, err > 0 ? err / 5 : err / 4));
      const cap = this.envelopeCaps(car, v);
      pedal = pedal > 0 ? Math.min(pedal, cap.drive) : Math.max(pedal, -cap.brake);
      return { steer, pedal };
    }

    // How much of the pedal range is still usable once the corner has claimed
    // its share of the friction circle. Trail-braking at 100 % while adding
    // steering leaves the rear axle with no lateral grip, which spins the car.
    envelopeCaps(car, v) {
      const sp = car.spec, a = car.aero, tr = this.tr, i = car.proj.i;
      const rho = 1.225 * Math.exp(-(tr.meta.altitude + tr.z[i]) / 8500);
      const q = 0.5 * rho * v * v, down = q * a.ClA;
      const Fz = sp.mass * F1.G + down;
      const Fzf = (Fz - down) * sp.frontWeight + down * a.balance, Fzr = Fz - Fzf;
      const FmaxF = F1.tyreMu(Fzf / 2, sp.tyre) * Fzf;
      const FmaxR = F1.tyreMu(Fzr / 2, sp.tyre) * sp.tyre.rearGrip * Fzr;
      const lat = Math.abs(sp.mass * car.ay);
      const frac = (Fmax) => Math.sqrt(Math.max(0, 1 - Math.min(1, (lat / Fmax) ** 2)));
      const brake = Math.max(0.05, 0.93 * frac(FmaxF + FmaxR));
      const availR = frac(FmaxR) * FmaxR * 0.90;
      const demand = (sp.power * sp.driveEff) / Math.max(v, 4);
      return { brake, drive: Math.max(0.04, Math.min(1, availR / demand)) };
    }
  }

  // ---------------------------------------------------------------- evaluation
  const CONF_HORIZON = 45;  // control steps (~0.75 s at 60 Hz) before scoring a decision
  // Pace (m/s) a crashed car is assumed to forfeit for each remaining second.
  // Well under racing pace, so surviving slowly is not rewarded over driving.
  const CRASH_FORFEIT_SPEED = 22;
  // Default penalty per unit of mean |change in steering command| between
  // control steps. A nudge, not a hammer: at 300 the optimiser simply stopped
  // steering and could not get round a corner at all.
  const CHATTER_WEIGHT = 60;

  // Run one episode. Returns fitness plus diagnostics. `recorder` (optional)
  // receives a frame each control step so the UI can replay a run.
  function evaluate(car, policy, opts) {
    const {
      s0 = 0, v0 = 0, duration = 25, dt = 1 / 240, ctrlEvery = 4,
      fallback = null, escalateBelow = 0, recorder = null, pose = null, timed = false,
      sampleSeed = null, chatterWeight = CHATTER_WEIGHT, paceRef = null,
    } = opts;
    // Common random numbers: every genome in a generation samples the same
    // draws, so fitness differences reflect the weights and not the dice.
    const sampler = sampleSeed == null ? null : makeRng(sampleSeed);

    car.reset(s0, v0, dt, { pose, timed });
    const obs = policy.obs;
    const pendConf = new Float64Array(CONF_HORIZON);
    const pendUsed = new Uint8Array(CONF_HORIZON);
    let head = 0, brierSum = 0, brierN = 0;
    let escalations = 0, ctrlSteps = 0, gripOver = 0;
    let steerWork = 0, prevSteer = null;
    const steps = Math.round(duration / dt);

    for (let n = 0; n < steps; n++) {
      if (n % ctrlEvery === 0) {
        car.observe(obs);
        policy.forward(sampler);
        let steer = policy.steer, pedal = policy.pedal, escalated = false;
        if (fallback && policy.confidence < escalateBelow) {
          const f = fallback.control(car);
          steer = f.steer; pedal = f.pedal; escalated = true; escalations++;
        }
        if (prevSteer !== null) steerWork += Math.abs(steer - prevSteer);
        prevSteer = steer;
        car.setControls(steer, pedal);

        // RLCD-style scoring: the confidence emitted CONF_HORIZON steps ago is
        // graded now against whether that decision actually turned out well.
        if (pendUsed[head]) {
          const good = car.alive && car.useF < 1.02 && car.useR < 1.02 ? 1 : 0;
          const d = pendConf[head] - good;
          brierSum += d * d; brierN++;
        }
        pendConf[head] = policy.confidence; pendUsed[head] = 1;
        head = (head + 1) % CONF_HORIZON;
        ctrlSteps++;

        if (recorder) recorder(car, policy, escalated);
      }
      car.step();
      if (car.useF > 1.02 || car.useR > 1.02) gripOver++;
      if (!car.alive) {
        // Everything still pending was, in hindsight, a bad decision.
        for (let i = 0; i < CONF_HORIZON; i++) {
          if (pendUsed[i]) { const d = pendConf[i] - 0; brierSum += d * d; brierN++; }
        }
        break;
      }
    }

    const brier = brierN ? brierSum / brierN : 0.25;
    const crashed = !car.alive;
    // Distance covered is the objective; lap time follows from it directly.
    //
    // A crash forfeits the distance the car could still have made in the time
    // left, rather than a flat penalty. With a flat penalty, braking costs
    // distance immediately while surviving only pays off later, so carrying
    // straight on into a braking zone scored about as well as getting through
    // it slowly -- every policy stalled at the same metre of the same corner
    // for hundreds of generations. Forfeiting the remaining time makes any
    // survivor beat any crasher, which is the gradient that was missing.
    const forfeit = crashed ? Math.max(0, duration - car.t) * CRASH_FORFEIT_SPEED : 0;
    // Control effort: a steady hand is worth a little, and it keeps the policy
    // from paying for nothing in commands the steering rack cannot follow.
    const chatter = ctrlSteps > 1 ? steerWork / (ctrlSteps - 1) : 0;
    let fitness;
    if (paceRef) {
      // Normalised: 1000 is limit pace on this circuit. Without this a long
      // fast circuit would dominate a short slow one purely through metres.
      fitness = 1000 * ((car.dist - forfeit) / paceRef - 0.2 * brier)
              - chatterWeight * chatter;
      if (car.dist < 5) fitness -= 500;
    } else {
      fitness = car.dist - forfeit - 200 * brier - chatterWeight * chatter;
      if (car.dist < 5) fitness -= 200; // punish genomes that just sit still
    }
    return {
      fitness, dist: car.dist, brier, crashed, reason: car.reason,
      time: car.t, laps: car.laps, lapTimes: car.lapTimes.slice(),
      maxSpeed: car.maxSpeed, escalations, ctrlSteps, gripOver, chatter,
    };
  }

  // ---------------------------------------------------------------- trainer
  // Evolution strategies (Salimans et al.): estimate a gradient of expected
  // fitness from antithetic parameter perturbations, rank-shape it so outliers
  // cannot dominate, and step with Adam. Far more effective than a genetic
  // algorithm at this parameter count, and it keeps one coherent policy (theta)
  // that visibly improves rather than a population that jumps around.
  class Trainer {
    // `circuits` is one {track, ref} or a list of them. Training on several at
    // once is the point: the observation vector is already track-agnostic
    // (curvature lookaheads, grade, braking demand), so one set of weights can
    // drive any circuit -- but only if it is scored on several, otherwise it
    // memorises one.
    constructor(circuits, spec, cfg = {}) {
      this.circuits = (Array.isArray(circuits) ? circuits : [circuits]).map((c) => ({
        track: c.track, ref: c.ref,
        car: new F1.Car(c.track, spec),
        // Metres the limit lap would cover per second on this circuit.
        pace: c.track.length / c.ref.time,
      }));
      this.track = this.circuits[0].track;
      this.ref = this.circuits[0].ref;
      this.cfg = Object.assign({
        pairs: 32, sigma: 0.09, lr: 0.04, seed: 12345,
        weightDecay: 2e-4, dt: 1 / 160, ctrlEvery: 3,
        starts: 3, startSpeed: 0.82, qualiSpeed: 0.98,
        stochastic: false, chatterWeight: 0,
      }, cfg);
      this.rng = makeRng(this.cfg.seed);
      this.theta = randomGenome(this.rng);
      for (let i = 0; i < N_PARAMS; i++) this.theta[i] *= 0.5;
      this.theta[N_PARAMS - 3] = 1.0; // bias toward FULL_POWER at init
      this.m = new Float32Array(N_PARAMS);
      this.v = new Float32Array(N_PARAMS);
      this.eps = [];
      for (let i = 0; i < this.cfg.pairs; i++) this.eps.push(new Float32Array(N_PARAMS));
      this.probe = new Float32Array(N_PARAMS);
      this.gen = 0;
      this.history = [];
      this.policy = new Policy(this.theta);
      this.best = this.theta;
      this.bestFitness = -Infinity;
      this.lastDuration = null;
      for (const c of this.circuits) c.starts = this.buildStarts(c, this.cfg.starts, this.cfg.startSpeed, this.cfg.qualiSpeed);
    }

    // The start points never change -- only how long each run lasts. Swapping
    // in fresh start points late in training collapsed a competent policy and
    // it never recovered: every probe then failed at the same place, and
    // rank-shaped ES has no gradient when all samples score the same floor.
    //
    // Episode length is relative to each circuit's own limit lap, so the final
    // stage always leaves room to finish a lap whether it takes 79 s or 99 s.
    schedule() {
      const g = this.gen;
      const frac = g < 100 ? 0.25 : g < 250 ? 0.5 : g < 450 ? 0.9 : 1.5;
      return { frac, sampleSeed: this.cfg.stochastic ? 7001 + g * 131 : null };
    }

    // Start 0 is the timed qualifying lap: it crosses the line at racing speed,
    // carrying the momentum a flying lap arrives with. The rest drop the car
    // around the lap so the whole circuit gets practised, on the racing line
    // because the reference speed they launch at belongs to its radius.
    buildStarts(c, n, speedFrac, qualiFrac) {
      const L = c.track.length, ref = c.ref;
      const poseAt = (i) => ({ x: ref.line.x[i], y: ref.line.y[i], psi: ref.line.psi[i] });
      const starts = [{ s0: 0, v0: ref.v[0] * qualiFrac, pose: poseAt(0), timed: true }];
      for (let k = 1; k < n; k++) {
        const s0 = (L * k) / n, i = c.track.idx(s0);
        starts.push({ s0, v0: ref.v[i] * speedFrac, pose: poseAt(i), timed: false });
      }
      return starts;
    }

    scoreGenome(genome, sc) {
      this.policy.g = genome;
      let total = 0, count = 0;
      const perTrack = [];
      for (const c of this.circuits) {
        const duration = Math.round(c.ref.time * sc.frac);
        const paceRef = c.pace * duration;
        let sum = 0, quali = null;
        for (const st of c.starts) {
          const r = evaluate(c.car, this.policy, {
            s0: st.s0, v0: st.v0, pose: st.pose, timed: st.timed, duration,
            dt: this.cfg.dt, ctrlEvery: this.cfg.ctrlEvery,
            sampleSeed: sc.sampleSeed, chatterWeight: this.cfg.chatterWeight,
            paceRef,
          });
          sum += r.fitness; total += r.fitness; count++;
          if (st.timed) quali = r;
        }
        const laps = quali ? quali.lapTimes.filter(Boolean) : [];
        perTrack.push({
          key: c.track.key, fitness: sum / c.starts.length, duration,
          qualiLap: laps.length ? Math.min(...laps) : null,
          qualiDist: quali ? quali.dist : 0,
          brier: quali ? quali.brier : 0,
        });
      }
      return { fitness: total / count, perTrack };
    }

    step() {
      const cfg = this.cfg, sc = this.schedule(), P = cfg.pairs;
      const fp = new Float64Array(P), fm = new Float64Array(P);
      for (let i = 0; i < P; i++) {
        const e = this.eps[i];
        for (let j = 0; j < N_PARAMS; j++) e[j] = gauss(this.rng);
        for (let j = 0; j < N_PARAMS; j++) this.probe[j] = this.theta[j] + cfg.sigma * e[j];
        fp[i] = this.scoreGenome(this.probe, sc).fitness;
        for (let j = 0; j < N_PARAMS; j++) this.probe[j] = this.theta[j] - cfg.sigma * e[j];
        fm[i] = this.scoreGenome(this.probe, sc).fitness;
      }

      // Centred-rank fitness shaping: only the ordering of returns matters, so
      // one lucky run cannot dominate the step.
      const all = new Array(2 * P);
      for (let i = 0; i < P; i++) { all[i] = { f: fp[i], i, sign: 1 }; all[P + i] = { f: fm[i], i, sign: -1 }; }
      all.sort((a, b) => a.f - b.f);
      const u = new Float64Array(2 * P);
      for (let r = 0; r < 2 * P; r++) u[r] = r / (2 * P - 1) - 0.5;
      const adv = new Float64Array(P);
      for (let r = 0; r < 2 * P; r++) adv[all[r].i] += all[r].sign * u[r];

      // Adam step on the estimated gradient.
      const scale = 1 / (P * cfg.sigma);
      const b1 = 0.9, b2 = 0.999, t = this.gen + 1;
      const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
      for (let j = 0; j < N_PARAMS; j++) {
        let g = 0;
        for (let i = 0; i < P; i++) g += adv[i] * this.eps[i][j];
        g = g * scale - cfg.weightDecay * this.theta[j];
        this.m[j] = b1 * this.m[j] + (1 - b1) * g;
        this.v[j] = b2 * this.v[j] + (1 - b2) * g * g;
        this.theta[j] += (cfg.lr * (this.m[j] / c1)) / (Math.sqrt(this.v[j] / c2) + 1e-8);
      }
      this.gen++;

      // Score theta itself so the reported curve tracks the deployed policy.
      const cur = this.scoreGenome(this.theta, sc);
      const dur = cur.perTrack[0].duration;
      if (dur !== this.lastDuration) { this.bestFitness = -Infinity; this.lastDuration = dur; }
      this.bestFitness = Math.max(this.bestFitness, cur.fitness);
      const rec = {
        gen: this.gen, best: cur.fitness,
        mean: (fp.reduce((a, b) => a + b, 0) + fm.reduce((a, b) => a + b, 0)) / (2 * P),
        sigma: cfg.sigma, duration: dur,
        perTrack: cur.perTrack,
        brier: cur.perTrack.reduce((a, t) => a + t.brier, 0) / cur.perTrack.length,
        qualiLap: cur.perTrack[0].qualiLap,
      };
      this.history.push(rec);
      return rec;
    }
  }

  const api = { DECISIONS, N_DEC, LAYERS, N_PARAMS, Policy, ReferenceDriver, Trainer,
                evaluate, randomGenome, makeRng, CONF_HORIZON };
  root.Jev = api;
  if (typeof module !== "undefined") module.exports = api;
})(typeof window !== "undefined" ? window : globalThis);
