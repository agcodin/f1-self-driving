// Training worker. Runs ES generations off the main thread and ships the
// current policy weights back so the UI can drive a car with them live.
//
// Trains ONE policy across every non-held-out circuit at once, matching
// tools/train_unified.js, so in-browser training produces the same kind of
// transferable weights rather than a single-circuit specialist.
/* global importScripts, TRACKS, F1, LapSim, Jev */
importScripts("tracks.js", "physics.js", "lapsim.js", "driver.js");

let trainer = null;
let running = false;
let seed = null;   // shipped weights, used to warm start

function boot() {
  const keys = Object.keys(TRACKS).filter((k) => !F1.TRACK_META[k].heldOut);
  const circuits = keys.map((k) => {
    const t = new F1.Track(k, TRACKS[k]);
    return { track: t, ref: LapSim.reference(t, F1.RB21) };
  });
  trainer = new Jev.Trainer(circuits, F1.RB21, {});
  // Warm start from the shipped weights. Training several circuits at once
  // from random weights stalls -- early gains are circuit-specific and
  // averaging cancels them, so every probe scores alike and rank-shaped ES has
  // no gradient. Starting from a policy that can already drive turns this into
  // refinement rather than a run that goes nowhere.
  if (seed && seed.length === Jev.N_PARAMS) trainer.theta.set(seed);
  postMessage({ type: "ready", tracks: keys, warmStarted: !!seed });
}

function loop() {
  if (!running) return;
  const t0 = Date.now();
  let rec = null;
  while (Date.now() - t0 < 180) rec = trainer.step();
  const theta = new Float32Array(trainer.theta);
  postMessage({ type: "gen", rec, theta }, [theta.buffer]);
  setTimeout(loop, 0);
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "init") { seed = m.seed ? new Float32Array(m.seed) : null; boot(); }
  else if (m.cmd === "start") { if (!running) { running = true; loop(); } }
  else if (m.cmd === "pause") running = false;
  else if (m.cmd === "reset") { running = false; boot(); }
};
