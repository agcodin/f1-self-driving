// Watch the live checkpoint and keep whichever one generalises best.
//
// Training fitness only measures the circuits being trained on, and transfer
// turned out to be volatile -- a checkpoint that gets round an unseen circuit
// at one generation can lose it a few generations later. Selecting the shipped
// weights on held-out performance is the only way to avoid shipping a
// checkpoint that happens to suit the training set.
const fs = require("fs"), path = require("path");
require("../tracks.js");
const F1 = require("../physics.js"), LS = require("../lapsim.js"), J = require("../driver.js");

const W = path.join(__dirname, "..", "weights");
const live = path.join(W, "unified.live.json");
// NOT unified.json: the trainer writes that on its own schedule, and its final
// fitness-based save clobbered this one mid-run.
const out = path.join(W, "unified.heldout.json");

const build = (k) => { const t = new F1.Track(k, globalThis.TRACKS[k]); return { track: t, ref: LS.reference(t, F1.RB21) }; };
const keys = Object.keys(globalThis.TRACKS);
const held = keys.filter((k) => F1.TRACK_META[k].heldOut).map(build);
const trained = keys.filter((k) => !F1.TRACK_META[k].heldOut).map(build);

const fmt = (t) => (t ? `${Math.floor(t / 60)}:${(t % 60).toFixed(3).padStart(6, "0")}` : null);

function qualify(theta, c) {
  const car = new F1.Car(c.track, F1.RB21), p = new J.Policy(theta);
  const r = J.evaluate(car, p, {
    s0: 0, v0: c.ref.v[0] * 0.98, timed: true,
    pose: { x: c.ref.line.x[0], y: c.ref.line.y[0], psi: c.ref.line.psi[0] },
    duration: c.ref.time * 3, dt: F1.PHYS_DT, ctrlEvery: F1.CTRL_EVERY,
  });
  const laps = r.lapTimes.filter(Boolean);
  const lap = laps.length ? Math.min(...laps) : null;
  // A completed lap scores its pace against the limit; a DNF scores only how
  // far it got, halved, so finishing always beats not finishing.
  return { lap, dist: r.dist, score: lap ? c.ref.time / lap : 0.5 * (r.dist / c.track.length) };
}

let bestScore = -Infinity, bestGen = null;
try { bestScore = JSON.parse(fs.readFileSync(out)).heldOutScore ?? -Infinity; } catch (e) {}

setInterval(() => {
  let w;
  try { w = JSON.parse(fs.readFileSync(live)); } catch (e) { return; }
  if (w.gen === bestGen || w.params !== J.N_PARAMS) return;
  bestGen = w.gen;
  const theta = Float32Array.from(w.theta);
  const h = held.map((c) => ({ key: c.track.key, ...qualify(theta, c) }));
  const score = h.reduce((a, x) => a + x.score, 0) / h.length;
  const line = h.map((x) => `${x.key} ${x.lap ? fmt(x.lap) : "dnf@" + x.dist.toFixed(0) + "m"}`).join("  ");
  if (score > bestScore) {
    bestScore = score;
    const t = trained.map((c) => ({ key: c.track.key, ...qualify(theta, c) }));
    fs.writeFileSync(out, JSON.stringify({
      ...w, heldOutScore: score,
      heldOut: h.map(({ key, lap, dist }) => ({ key, lap, dist })),
      trained: t.map(({ key, lap, dist }) => ({ key, lap, dist })),
    }));
    console.log(`gen ${w.gen} NEW BEST heldOut ${score.toFixed(3)} | ${line}`);
  } else {
    console.log(`gen ${w.gen} heldOut ${score.toFixed(3)} (best ${bestScore.toFixed(3)}) | ${line}`);
  }
}, 45000);
