// Training worker. Runs ES generations off the main thread and ships the
// current policy weights back so the UI can drive a car with them live.
/* global importScripts, TRACKS, F1, LapSim, Jev */
importScripts("tracks.js", "physics.js", "lapsim.js", "driver.js");

let trainer = null;
let running = false;
let trackKey = null;

function boot(key) {
  trackKey = key;
  const track = new F1.Track(key, TRACKS[key]);
  const ref = LapSim.reference(track, F1.RB21);
  trainer = new Jev.Trainer(track, F1.RB21, ref, {});
  postMessage({
    type: "ready",
    track: key,
    refTime: ref.time,
    refLine: { x: Array.from(ref.line.x), y: Array.from(ref.line.y) },
    refV: Array.from(ref.v),
  });
}

function loop() {
  if (!running) return;
  const t0 = Date.now();
  let rec = null;
  // Batch generations so we are not posting messages more often than the UI
  // can paint them.
  while (Date.now() - t0 < 180) rec = trainer.step();
  const theta = new Float32Array(trainer.theta);
  postMessage({ type: "gen", rec, theta }, [theta.buffer]);
  setTimeout(loop, 0);
}

onmessage = (e) => {
  const m = e.data;
  if (m.cmd === "init") boot(m.track);
  else if (m.cmd === "start") { if (!running) { running = true; loop(); } }
  else if (m.cmd === "pause") running = false;
  else if (m.cmd === "reset") { running = false; boot(trackKey); }
};
