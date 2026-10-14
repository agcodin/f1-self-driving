// Export the simulator's driving decisions as an Endor dataset.
//
// Endor (endor.thetokencompany.com) fine-tunes decision models, and its
// primitives line up with this driver almost exactly: a Choice is the typed
// pedal decision, a Noul is the confidence head, and soft labels take the
// policy's own softmax. So the simulator can act as a decision-data generator
// -- it produces rows whose outcomes are known rather than guessed at, which
// is the part that is usually hard to come by.
//
// The policy's action is distilled (soft label = its softmax), so the trained
// model's ceiling is this policy. Confidence is NOT distilled: it is labelled
// with what actually happened, graded the same RLCD way training grades it --
// did the car stay on track and inside the grip limit over the next 0.75 s.
//
// Clean laps alone make the confidence question untrainable: the shipped
// policy almost never fails, so every row is labelled true and a model that
// always answers yes scores perfectly. So alongside the clean lap each circuit
// also contributes PERTURBED runs -- spawned off the racing line, too fast, and
// with occasional random inputs -- which genuinely go wrong and supply the
// negatives. The action label on a perturbed row is still the policy's own
// softmax for the state it is actually in.
//
// State deliberately carries NO circuit name or position. The policy itself
// has no track identifier, and leaving one in would let a fine-tune memorise
// layouts and make the held-out circuits meaningless.
//
//   node tools/export_endor.js [outDir] [--stride N]
//
// Writes endor_train.jsonl (trained circuits) and endor_heldout.jsonl
// (Suzuka and Shanghai, never trained on).
const fs = require("fs"), path = require("path");
require("../tracks.js");
const F1 = require("../physics.js"), LS = require("../lapsim.js"), J = require("../driver.js");

const outDir = process.argv[2] && !process.argv[2].startsWith("--")
  ? process.argv[2] : path.join(__dirname, "..", "data", "endor");
const strideArg = process.argv.indexOf("--stride");
const STRIDE = strideArg > -1 ? +process.argv[strideArg + 1] : 3;

// Endor caps a dataset at 50,000 rows.
const MAX_ROWS = 50000;
// Match the confidence horizon the trainer grades against.
const OUTCOME_STEPS = 45;
// Perturbed runs per circuit, the source of the confidence question's negatives.
const PERTURBED_RUNS = 26;

const DECISION_CRITERIA = {
  HARD_BRAKE:  "Maximum braking. The corner ahead needs every bit of retardation available.",
  BRAKE:       "Heavy braking, short of the limit.",
  TRAIL_BRAKE: "Light braking carried into the corner while steering is applied.",
  COAST:       "Neither throttle nor brake. Slowing on drag and engine braking alone.",
  FEATHER:     "A small amount of throttle, trimming speed without accelerating hard.",
  MODULATE:    "Partial throttle. Accelerating, but holding some in reserve for grip.",
  FULL_POWER:  "Full throttle. The road ahead can take everything the car has.",
};

const round = (x, n = 3) => +x.toFixed(n);

// Human-meaningful state. The network sees normalised floats, but a decision
// model reads named quantities, so these are emitted in real units.
function describeState(car, tr) {
  const v = car.speed, i = car.proj.i, s = car.proj.s;
  const wl = tr.wl[i], wr = tr.wr[i];
  const aMax = car.gripAccel(v);
  const curvature = {}, braking = {};
  for (const la of F1.LOOKAHEAD) {
    const kap = tr.sample(tr.kappa, s + la);
    // Signed corner radius reads better than curvature; cap the straights.
    curvature[`${la}m`] = Math.abs(kap) < 1 / 2000 ? "straight" : round(1 / kap, 1);
  }
  for (const la of F1.BRAKE_LOOKAHEAD) {
    const kap = Math.abs(tr.sample(tr.kappa, s + la));
    const vCorner = Math.sqrt(aMax / Math.max(kap, 1e-5));
    const need = (v * v - vCorner * vCorner) / (2 * la);
    braking[`${la}m`] = round(Math.max(-1, Math.min(2, need / aMax)), 3);
  }
  let gear = 1;
  for (let g = 0; g < car.spec.gears.length; g++) if (v * 3.6 > car.spec.gears[g]) gear = g + 2;
  return {
    speed_kmh: round(v * 3.6, 1),
    gear: Math.min(8, gear),
    lateral_offset_m: round(car.d, 2),
    track_edge_left_m: round(wl - car.d, 2),
    track_edge_right_m: round(wr + car.d, 2),
    heading_error_deg: round((F1.wrapAngle(car.psi - tr.heading[i]) * 180) / Math.PI, 2),
    yaw_rate_deg_s: round((car.r * 180) / Math.PI, 2),
    slip_speed_ms: round(car.vy, 2),
    corner_radius_ahead_m: curvature,
    // Fraction of available deceleration needed to make the corner at each
    // distance. Below 0 means no braking required.
    braking_demand_ahead: braking,
    gradient_pct: round(tr.sample(tr.grade, s + 40) * 100, 1),
    front_grip_used: round(car.useF, 3),
    rear_grip_used: round(car.useR, 3),
    steering_frac_of_lock: round(car.delta / car.spec.maxSteer, 3),
    previous_pedal: round(car.pedal, 2),
  };
}

const QUESTIONS = {
  pedal: {
    type: "choice",
    instructions:
      "An F1 car is on a qualifying lap. Given the car's state and the road ahead, " +
      "which longitudinal input gets round the lap fastest without leaving the track " +
      "or exceeding the tyres' grip?",
    criteria: DECISION_CRITERIA,
  },
  confident: {
    type: "noul",
    instructions:
      "Will this input turn out well? That is, will the car still be on track and " +
      "inside the limit of its tyres three quarters of a second from now?",
    criteria: {
      true: "The car stays on track and within the grip limit.",
      false: "The car leaves the track, spins, or exceeds the grip limit.",
    },
  },
};

function lapRows(key, theta, opts = {}) {
  const { seed = null, s0 = 0, speedScale = 0.98, offset = 0, noise = 0, maxTime = null } = opts;
  const tr = new F1.Track(key, globalThis.TRACKS[key]);
  const ref = LS.reference(tr, F1.RB21);
  const car = new F1.Car(tr, F1.RB21);
  const p = new J.Policy(theta);
  const obs = new Float64Array(F1.N_INPUTS);
  const rng = seed == null ? null : J.makeRng(seed);

  const i0 = tr.idx(s0);
  const nx = -tr.ty[i0], ny = tr.tx[i0];
  car.reset(s0, ref.v[i0] * speedScale, F1.PHYS_DT, {
    pose: {
      x: ref.line.x[i0] + nx * offset,
      y: ref.line.y[i0] + ny * offset,
      psi: ref.line.psi[i0],
    },
    timed: s0 === 0,
  });
  const limit = maxTime != null ? maxTime : ref.time * 3;

  // Each step is held until its outcome is known, OUTCOME_STEPS later.
  const pending = [];
  const rows = [];
  let step = 0;
  const grade = (good) => { for (const q of pending.splice(0)) { q.row.labels.confident = good; rows.push(q.row); } };

  while (car.alive && car.laps < 1 && car.t < limit) {
    if (car.stepCount % F1.CTRL_EVERY === 0) {
      car.observe(obs);
      for (let i = 0; i < F1.N_INPUTS; i++) p.obs[i] = obs[i];
      p.forward();
      let steer = p.steer, pedal = p.pedal;
      // Occasional wrong input, so some runs genuinely fail.
      if (rng && rng() < noise) {
        p.decision = Math.floor(rng() * J.N_DEC);
        pedal = J.DECISIONS[p.decision].pedal;
        steer = Math.max(-1, Math.min(1, steer + (rng() * 2 - 1) * 0.6));
      }
      car.setControls(steer, pedal);

      if (step % STRIDE === 0) {
        const probs = {};
        for (let i = 0; i < J.N_DEC; i++) probs[J.DECISIONS[i].name] = round(p.probs[i], 4);
        // Soft labels must sum to 1 within tolerance; fix up rounding drift.
        const sum = Object.values(probs).reduce((a, b) => a + b, 0);
        probs[J.DECISIONS[p.decision].name] = round(probs[J.DECISIONS[p.decision].name] + (1 - sum), 4);
        pending.push({
          due: step + OUTCOME_STEPS,
          row: {
            id: `${key}-${opts.tag || "clean"}-${String(step).padStart(5, "0")}`,
            state: describeState(car, tr),
            questions: QUESTIONS,
            labels: { pedal: { probs } },
          },
        });
      }
      // Grade anything whose horizon has elapsed.
      const ok = car.alive && car.useF < 1.02 && car.useR < 1.02;
      for (let i = pending.length - 1; i >= 0; i--) {
        if (pending[i].due <= step) {
          pending[i].row.labels.confident = ok;
          rows.push(pending[i].row);
          pending.splice(i, 1);
        }
      }
      step++;
    }
    car.step();
  }
  // The run ended; everything still pending was, in hindsight, a bad decision.
  if (!car.alive) grade(false);
  else grade(true);
  return { rows, lap: car.lapTimes.filter(Boolean)[0] || null, dist: car.dist };
}

function main() {
  const w = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "weights", "unified.json")));
  if (w.params !== J.N_PARAMS) throw new Error(`weights have ${w.params} params, build expects ${J.N_PARAMS}`);
  const theta = Float32Array.from(w.theta);
  fs.mkdirSync(outDir, { recursive: true });

  const split = { train: [], heldout: [] };
  for (const key of Object.keys(globalThis.TRACKS)) {
    const held = !!F1.TRACK_META[key].heldOut;
    const tr = new F1.Track(key, globalThis.TRACKS[key]);
    const bucket = split[held ? "heldout" : "train"];

    const clean = lapRows(key, theta, { tag: "clean" });
    bucket.push(...clean.rows);

    // Perturbed runs: off the line, over the reference speed, with occasional
    // random inputs. These are where the failures come from.
    let perturbed = 0, failed = 0;
    for (let k = 0; k < PERTURBED_RUNS; k++) {
      const seed = 9001 + k * 977 + key.length * 13;
      const rng = J.makeRng(seed);
      const r = lapRows(key, theta, {
        tag: `p${k}`, seed,
        s0: rng() * tr.length,
        // Deliberately hard: a failure only labels the ~0.75 s before it, so
        // many short bad runs produce far more negatives than a few long ones.
        speedScale: 0.95 + rng() * 0.45,          // up to 40 % over the limit lap
        offset: (rng() * 2 - 1) * 5.0,            // metres off the racing line
        noise: 0.02 + rng() * 0.10,               // 2-12 % of steps get a wrong input
        maxTime: 12,
      });
      bucket.push(...r.rows);
      perturbed += r.rows.length;
      if (r.rows.some((x) => x.labels.confident === false)) failed++;
    }
    console.log(`${key.padEnd(12)}${held ? "[held out] " : "           "}` +
      `${String(clean.rows.length).padStart(5)} clean + ${String(perturbed).padStart(5)} perturbed ` +
      `(${failed}/${PERTURBED_RUNS} runs failed)  ` +
      (clean.lap ? `lap ${clean.lap.toFixed(3)}s` : `no lap, ${clean.dist.toFixed(0)}m`));
  }

  for (const [name, rows] of Object.entries(split)) {
    // Endor weights rows in training. The decision classes are wildly
    // imbalanced by nature -- a lap is mostly full throttle -- so weight each
    // row by the inverse frequency of its own majority class, capped so a rare
    // class cannot dominate.
    const counts = {};
    const major = (r) => Object.entries(r.labels.pedal.probs).sort((a, b) => b[1] - a[1])[0][0];
    for (const r of rows) counts[major(r)] = (counts[major(r)] || 0) + 1;
    const mean = rows.length / Object.keys(counts).length;
    // The confidence question is rarer still -- most of a lap genuinely is
    // fine 0.75 s later -- so negatives get their own inverse-frequency boost,
    // and a row takes whichever of the two weights is larger.
    const negs = rows.filter((r) => r.labels.confident === false).length;
    const negWeight = negs ? Math.min(12, (rows.length - negs) / negs) : 1;
    for (const r of rows) {
      const byClass = Math.min(8, Math.max(0.25, mean / counts[major(r)]));
      r.weight = round(r.labels.confident === false ? Math.max(byClass, negWeight) : byClass, 3);
    }

    if (rows.length > MAX_ROWS) {
      console.log(`${name}: ${rows.length} rows exceeds Endor's ${MAX_ROWS} cap, trimming`);
      rows.length = MAX_ROWS;
    }
    const file = path.join(outDir, `endor_${name}.jsonl`);
    fs.writeFileSync(file, rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
    const neg = rows.filter((r) => r.labels.confident === false).length;
    console.log(`wrote ${file}  ${rows.length} rows, ` +
      `confident=false on ${(100 * neg / rows.length).toFixed(1)}%`);
    console.log("  class balance: " + Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([k, c]) => `${k} ${(100 * c / rows.length).toFixed(1)}%`).join(", "));
  }
}

main();
