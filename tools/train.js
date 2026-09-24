const fs=require("fs"),path=require("path");
require("../tracks.js"); const F1=require("../physics.js"); const LS=require("../lapsim.js"); const J=require("../driver.js");
const key=process.argv[2]||"monza", gens=+(process.argv[3]||900);
const tr=new F1.Track(key,globalThis.TRACKS[key]); const ref=LS.reference(tr,F1.RB21);
const T=new J.Trainer(tr,F1.RB21,ref,{});
console.log(key,"QSS",ref.time.toFixed(2),"params",J.N_PARAMS);
const out=path.join(__dirname,"..","weights",key+".json");
const live=path.join(__dirname,"..","weights",key+".live.json");
fs.mkdirSync(path.dirname(out),{recursive:true});
let bestSeen=-Infinity,bestLap=null;
const t0=Date.now();
for(let g=0;g<gens;g++){
  const r=T.step();
  // Checkpoint only on the final curriculum stage, where the episode is long
  // enough to reflect a full lap rather than a short drill. Episode length is
  // lap-relative, so this threshold has to be too -- a fixed 120 s silently
  // excluded Monza, whose final stage is 118 s.
  if(r.duration>=ref.time*1.4&&r.best>bestSeen){
    bestSeen=r.best;
    if(r.qualiLap) bestLap=r.qualiLap;
    // Same field set as the live checkpoint, so consumers do not have to
    // special-case which file a checkpoint came from.
    fs.writeFileSync(out,JSON.stringify({track:key,gen:r.gen,fitness:r.best,
      brier:r.brier,duration:r.duration,dist:r.dist,bestLap,params:J.N_PARAMS,
      theta:Array.from(T.theta,v=>+v.toFixed(5))}));
  }
  // Stream the current policy for the browser display to pick up. Every other
  // generation: in the final stage a generation takes ~5 s, and writing every
  // eighth left the display looking frozen for 40 s at a time.
  if(g%2===0){
    fs.writeFileSync(live,JSON.stringify({track:key,gen:r.gen,fitness:r.best,
      brier:r.brier,duration:r.duration,dist:r.dist,params:J.N_PARAMS,
      theta:Array.from(T.theta,v=>+v.toFixed(5))}));
  }
  if(g%10===0||g===gens-1) console.log(`gen ${String(r.gen).padStart(4)} best ${r.best.toFixed(0).padStart(6)} mean ${r.mean.toFixed(0).padStart(6)} dist ${r.dist.toFixed(0).padStart(5)} brier ${r.brier.toFixed(3)} quali ${r.qualiLap?r.qualiLap.toFixed(2)+"s":r.qualiDist.toFixed(0)+"m"} dur ${r.duration} [${((Date.now()-t0)/1000).toFixed(0)}s]`);
}
console.log("saved",out,"bestFitness",bestSeen.toFixed(0),"bestLap",bestLap);
