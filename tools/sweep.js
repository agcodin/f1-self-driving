require("../tracks.js"); const F1=require("../physics.js"); const LS=require("../lapsim.js"); const J=require("../driver.js");
for(const key of ["monza","spa","cota"]){
 const tr=new F1.Track(key,globalThis.TRACKS[key]); const ref=LS.reference(tr,F1.RB21);
 const out=[];
 for(const m of [0.98,0.94,0.90,0.85,0.80]){
  const car=new F1.Car(tr,F1.RB21); const rd=new J.ReferenceDriver(tr,ref,m);
  car.reset(0,0,1/240);
  for(let n=0;n<240*400 && car.alive && car.laps<2;n++){ if(n%4===0){const c=rd.control(car);car.setControls(c.steer,c.pedal);} car.step(); }
  out.push(`m=${m} ${car.alive?"OK":"DEAD@"+car.proj.s.toFixed(0)+" "+car.reason} laps=${car.laps} ${car.lapTimes.map(t=>t.toFixed(2)).join("/")}`);
 }
 console.log(key,"QSS",ref.time.toFixed(2),"| "+out.join(" | "));
}
