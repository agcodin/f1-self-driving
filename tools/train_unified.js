// Train ONE policy across several circuits at once, so the same weights drive
// any of them. Circuits flagged heldOut in TRACK_META are never trained on and
// are only used to check transfer.
const fs = require("fs"), path = require("path");
require("../tracks.js");
const F1 = require("../physics.js"), LS = require("../lapsim.js"), J = require("../driver.js");

const gens = +(process.argv[2] || 900);
const trainKeys = Object.keys(globalThis.TRACKS).filter((k) => !F1.TRACK_META[k].heldOut);
const heldKeys = Object.keys(globalThis.TRACKS).filter((k) => F1.TRACK_META[k].heldOut);

const build = (k) => { const t = new F1.Track(k, globalThis.TRACKS[k]); return { track: t, ref: LS.reference(t, F1.RB21) }; };
const circuits = trainKeys.map(build);
const held = heldKeys.map(build);

console.log("training on:", trainKeys.join(", "));
console.log("held out   :", heldKeys.join(", "));
console.log("params", J.N_PARAMS);

const T = new J.Trainer(circuits, F1.RB21, {});
const out = path.join(__dirname, "..", "weights", "unified.json");
const live = path.join(__dirname, "..", "weights", "unified.live.json");
fs.mkdirSync(path.dirname(out), { recursive: true });

const fmt = (t) => (t ? `${Math.floor(t / 60)}:${(t % 60).toFixed(3).padStart(6, "0")}` : "dnf");

// One flying qualifying lap at browser timing. Used for reporting and for the
// held-out transfer check.
function qualify(theta, c) {
  const car = new F1.Car(c.track, F1.RB21), p = new J.Policy(theta);
  const r = J.evaluate(car, p, {
    s0: 0, v0: c.ref.v[0] * 0.98, timed: true,
    pose: { x: c.ref.line.x[0], y: c.ref.line.y[0], psi: c.ref.line.psi[0] },
    duration: c.ref.time * 3, dt: F1.PHYS_DT, ctrlEvery: F1.CTRL_EVERY,
  });
  const laps = r.lapTimes.filter(Boolean);
  return { lap: laps.length ? Math.min(...laps) : null, dist: r.dist, reason: r.reason, limit: c.ref.time };
}

let bestSeen = -Infinity;
const t0 = Date.now();
for (let g = 0; g < gens; g++) {
  const r = T.step();
  if (g % 2 === 0) {
    fs.writeFileSync(live, JSON.stringify({
      gen: r.gen, fitness: r.best, brier: r.brier, duration: r.duration,
      perTrack: r.perTrack, params: J.N_PARAMS,
      theta: Array.from(T.theta, (v) => +v.toFixed(5)),
    }));
  }
  // Checkpoint on the final stage, where episodes are long enough for a lap.
  if (r.duration >= T.circuits[0].ref.time * 1.4 && r.best > bestSeen) {
    bestSeen = r.best;
    fs.writeFileSync(out, JSON.stringify({
      gen: r.gen, fitness: r.best, brier: r.brier, duration: r.duration,
      perTrack: r.perTrack, params: J.N_PARAMS,
      theta: Array.from(T.theta, (v) => +v.toFixed(5)),
    }));
  }
  if (g % 25 === 0 || g === gens - 1) {
    const per = r.perTrack.map((t) => `${t.key}=${t.fitness.toFixed(0)}`).join(" ");
    console.log(`gen ${String(r.gen).padStart(4)} fit ${r.best.toFixed(0).padStart(5)} | ${per} | dur ${r.duration}s [${((Date.now() - t0) / 1000).toFixed(0)}s]`);
  }
  if (g % 150 === 0 && g > 0) {
    const line = [...circuits, ...held].map((c) => {
      const q = qualify(T.theta, c);
      const tag = F1.TRACK_META[c.track.key].heldOut ? "*" : "";
      return `${c.track.key}${tag} ${q.lap ? fmt(q.lap) : "dnf@" + q.dist.toFixed(0) + "m"}`;
    }).join("  ");
    console.log(`  qualifying (* = never trained on): ${line}`);
  }
}
console.log("saved", out, "bestFitness", bestSeen.toFixed(0));
