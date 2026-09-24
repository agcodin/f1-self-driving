require("../tracks.js"); const F1=require("../physics.js"); const LS=require("../lapsim.js");
const trs=["monza","spa","cota"].map(k=>new F1.Track(k,globalThis.TRACKS[k]));
function err(mu){ F1.RB21.tyre.mu0=mu; let e=0; const r=[];
  for(const tr of trs){ const q=LS.reference(tr,F1.RB21); const tgt=tr.meta.pole2025*0.99; e+=(q.time-tgt)**2; r.push([tr.key,q.time,tgt,q.vmax*3.6]); }
  return [e,r]; }
let lo=1.7,hi=2.3;
for(let i=0;i<30;i++){ const m=(lo+hi)/2; const a=err(m-0.005)[0], b=err(m+0.005)[0]; if(a<b) hi=m; else lo=m; }
const mu=(lo+hi)/2; const [e,r]=err(mu);
console.log("mu0 =",mu.toFixed(4));
for(const [k,t,tgt,v] of r) console.log(k.padEnd(6),"QSS",t.toFixed(3),"target",tgt.toFixed(3),"delta",(t-tgt).toFixed(3),"vmax",v.toFixed(0),"km/h");
