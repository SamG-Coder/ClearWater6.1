import {chromium,devices} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']}),errors=[];
const url=process.env.WATER_URL||'http://127.0.0.1:5191/';
const point=(lat,lon)=>{lat*=Math.PI/180;lon*=Math.PI/180;return [Math.sin(lon)*Math.cos(lat),Math.sin(lat),Math.cos(lon)*Math.cos(lat),0];};
const select=(page,id,value)=>page.locator('#'+id).evaluate((e,v)=>{e.value=String(v);e.dispatchEvent(new Event('change'));},value);
const ready=page=>page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});
await mkdir('captures',{recursive:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(url+'?t=4');await ready(page);assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 const initial=await page.evaluate(()=>waterLab.geologyState());assert.ok(initial.enabled&&initial.finite&&initial.plateCount===28);assert.ok(initial.depth<2,'Shallows must spawn at the generated coast');
 const plates=await page.evaluate(()=>waterLab.plateData());assert.equal(plates.length,224);
 let minSeparation=10;for(let i=0;i<28;i++){assert.ok(Math.abs(Math.hypot(...plates.slice(i*8,i*8+3))-1)<1e-6);for(let j=0;j<i;j++)minSeparation=Math.min(minSeparation,Math.hypot(...plates.slice(i*8,i*8+3).map((v,k)=>v-plates[j*8+k])));}
 assert.ok(minSeparation>.15,'Sites must avoid clumping');
 // Equal-area spiral is an independent sampling oracle, not the generation grid.
 const points=Array.from({length:4096},(_,i)=>{const y=1-2*(i+.5)/4096,a=i*Math.PI*(3-Math.sqrt(5)),r=Math.sqrt(1-y*y);return [r*Math.sin(a),y,r*Math.cos(a),0];});
 const fields=await page.evaluate(p=>waterLab.geologyAt(p),points);assert.ok(fields.every(Number.isFinite));
 let min=Infinity,max=-Infinity,land=0,shelf=0,plateMismatch=0,tectonicDifference=0,cacheSquared=0;
 for(let i=0;i<points.length;i++){
  const h=fields[i*12];min=Math.min(min,h);max=Math.max(max,h);land+=h>0?1:0;shelf+=h>-200?1:0;tectonicDifference=Math.max(tectonicDifference,Math.abs(h-fields[i*12+8]));cacheSquared+=(h-fields[i*12+4])**2;
  let closest=0,best=-2;for(let j=0;j<28;j++){const d=points[i].slice(0,3).reduce((v,n,k)=>v+n*plates[j*8+k],0);if(d>best){best=d;closest=j;}}if(closest!==fields[i*12+3])plateMismatch++;
 }
 const distribution={minElevation:min,maxElevation:max,prospectiveLandFraction:land/points.length,shelfFraction:shelf/points.length,minPlateSeparation:minSeparation,plateMismatch,tectonicDifference,cacheRmsMeters:Math.sqrt(cacheSquared/points.length)};
 console.log(JSON.stringify({distribution,initial}));assert.equal(plateMismatch,0);assert.ok(min<-5000&&max>100&&land/points.length>.05&&land/points.length<.65);assert.ok(tectonicDifference>500);
 const seam=[];for(const lat of [-89.999,-45,0,45,89.999]){const values=await page.evaluate(p=>waterLab.geologyAt(p),[point(lat,-179.99999),point(lat,179.99999)]);const delta=Math.abs(values[4]-values[16]);assert.ok(delta<1);seam.push(delta);}
 // Regenerating a seed reproduces exact GPU fields; a new seed changes them.
 await select(page,'worldSeed',73);await page.evaluate(()=>waterLab.seek(4));const changed=await page.evaluate(p=>waterLab.geologyAt(p),points.slice(0,32));assert.notDeepEqual(changed,fields.slice(0,384));
 await select(page,'worldSeed',61);await page.evaluate(()=>waterLab.seek(4));assert.deepEqual(await page.evaluate(p=>waterLab.geologyAt(p),points.slice(0,32)),fields.slice(0,384));
 const before=await page.evaluate(()=>waterLab.geologyState());await page.evaluate(()=>waterLab.seek(5));const after=await page.evaluate(()=>waterLab.geologyState());assert.equal(before.builds,after.builds,'Static geology must not rebuild each frame');
 await page.locator('#space').click();await page.evaluate(()=>waterLab.seek(4));await page.locator('#toggle').click();
 for(const [view,name] of [[3,'bathymetry'],[4,'plates'],[0,'ocean']]){await select(page,'view',view);await page.evaluate(()=>waterLab.seek(4));await page.screenshot({path:`captures/geology-${name}.png`});}
 const samples=points.map((p,i)=>({p,h:fields[i*12]})).sort((a,b)=>a.h-b.h);
 let phaseChecks=0;for(const sample of [samples[0],samples.at(-1)]){const before=await page.evaluate(()=>waterLab.waveModes());const [x,y,z]=sample.p;await page.evaluate(([lat,lon])=>waterLab.weatherLocation(lat,lon,5),[Math.asin(y)*180/Math.PI,Math.atan2(x,z)*180/Math.PI]);const g=await page.evaluate(()=>waterLab.geologyState());assert.ok(Math.abs(g.depth-Math.max(1.4,-g.elevation))<.002);const after=await page.evaluate(()=>waterLab.waveModes());for(let j=0;j<after.length;j++){assert.ok(Math.abs(after[j].phase-before[j].phase)<.0001,'Depth changes must preserve wave phase');const k=after[j].k,omega=Math.sqrt(9.81*k*Math.tanh(k*g.depth));assert.ok(Math.abs(omega-after[j].omega)<.0001);}phaseChecks++;}
 await select(page,'depthMode',0);await page.locator('#depth').evaluate(e=>{e.value='2.5';e.dispatchEvent(new Event('input'));});await page.evaluate(()=>waterLab.seek(4));assert.equal((await page.evaluate(()=>waterLab.geologyState())).depth,2.5);
 await select(page,'depthMode',1);await page.locator('#toggle').click();await page.locator('#space').click();await page.locator('#toggle').click();await page.evaluate(()=>waterLab.seek(4));
 const orbit=[];for(const size of [2560,3840]){await page.setViewportSize({width:size,height:size*9/16});await select(page,'quality',size);await page.evaluate(()=>waterLab.seek(4));const b=await page.evaluate(()=>waterLab.benchmark(40,true));assert.equal(b.width,size);orbit.push(b);}
 const phone=await browser.newPage({...devices['Pixel 7']});phone.on('pageerror',e=>errors.push(String(e)));await phone.goto(url+'?t=4');await ready(phone);const mobile=await phone.evaluate(()=>waterLab.geologyState());assert.equal(mobile.mapWidth,512);assert.ok(mobile.finite&&mobile.enabled);await phone.screenshot({path:'captures/geology-mobile.png'});await phone.close();
 errors.push(...await page.evaluate(()=>waterDiagnostics.errors));assert.deepEqual(errors,[]);
 const result={phaseChecks,initial,distribution,seam,deterministicSeed:true,seedVariation:true,staticCache:true,depthCoupled:true,manualDepthControl:true,orbit,mobile,errors,environment:'Desktop NVIDIA/Edge; phone viewport emulation, not physical Samsung/Safari'};
 await writeFile('captures/geology-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
