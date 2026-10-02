import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

const base=process.env.WATER_URL||'http://127.0.0.1:5191/';
const reference=process.env.WATER_REFERENCE_URL;
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const errors=[],report={};
const rms=v=>Math.sqrt(v.reduce((s,x)=>s+x*x,0)/v.length);
async function ready(page){await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:180000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);}
async function benchmarks(url,label){
 const page=await browser.newPage({viewport:{width:2560,height:1440}});page.on('pageerror',e=>errors.push(String(e)));
 try{
  await page.goto(url+'?mode=explorer&t=4&geology=study');await ready(page);
  const results=[];
  for(const [width,height] of [[2560,1440],[3840,2160]]){
   await page.setViewportSize({width,height});await page.locator('#quality').selectOption(String(width));await page.evaluate(()=>waterLab.seek(4));
   const state=await page.evaluate(()=>({width:waterDiagnostics.width,height:waterDiagnostics.height,samples:waterDiagnostics.pixelSamples,light:waterDiagnostics.lightMapSize}));
   assert.deepEqual(state,{width,height,samples:4,light:512});
   const timing=await page.evaluate(()=>waterLab.benchmark(40));assert.ok(timing.medianMs>0);
   results.push({...state,timing});await page.screenshot({path:`captures/appearance-${label}-${width}.png`});
  }
  return results;
 }finally{await page.close();}
}
try{
 await mkdir('captures',{recursive:true});
 if(reference)report.reference=await benchmarks(reference,'reference');
 const page=await browser.newPage({viewport:{width:1440,height:900}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(base+'?mode=explorer&t=4&geology=study');await ready(page);await page.locator('#quality').selectOption('1920');await page.evaluate(()=>waterLab.seek(4));
 const points=[],groups=[];
 for(const mode of [0,1,4,5,6])for(let i=0;i<80;i++){
  const x=Math.fround((i%10-5)*4.813),z=Math.fround((Math.floor(i/10)-4)*6.137),e=.002,index=points.length;
  points.push([x,z,mode,0],[Math.fround(x+e),z,mode,0],[Math.fround(x-e),z,mode,0],[x,Math.fround(z+e),mode,0],[x,Math.fround(z-e),mode,0]);groups.push({mode,index});
 }
 const values=await page.evaluate(p=>waterLab.patternSamples(p),points);assert.ok(values.every(Number.isFinite));
 const slopeErrors={};
 for(const {mode,index:i} of groups){
  const dx=(values[(i+1)*4]-values[(i+2)*4])/(points[i+1][0]-points[i+2][0]);
  const dz=(values[(i+3)*4]-values[(i+4)*4])/(points[i+3][1]-points[i+4][1]);
  slopeErrors[mode]=Math.max(slopeErrors[mode]||0,Math.abs(dx-values[i*4+1]),Math.abs(dz-values[i*4+2]));
 }
 for(const [mode,error] of Object.entries(slopeErrors))assert.ok(error<.015,`Field ${mode} normal disagrees with finite differences: ${error}`);
 const cachePoints=[];for(let i=0;i<96;i++){const x=(i%12-6)*4.71,z=(Math.floor(i/12)-4)*13.19;cachePoints.push([x,z,0,0],[x,z,1,0],[x,z,7,i%3===0?0:(i%3===1?27:410)]);}
 const cacheValues=await page.evaluate(p=>waterLab.patternSamples(p),cachePoints);let cacheError=0;
 for(let i=0;i<96;i++)cacheError=Math.max(cacheError,Math.abs(cacheValues[i*12]+cacheValues[i*12+4]-cacheValues[i*12+8]));
 assert.ok(cacheError<1e-7,'Cached iteration must match direct evaluation, even after crossing region boundaries');
 const bedCachePoints=[];for(let i=0;i<96;i++){const x=(i%12-6)*4.71,z=(Math.floor(i/12)-4)*13.19;bedCachePoints.push([x,z,5,0],[x,z,9,i%3===0?0:(i%3===1?2:19)]);}
 const bedCacheValues=await page.evaluate(p=>waterLab.patternSamples(p),bedCachePoints);let bedCacheError=0;
 for(let i=0;i<96;i++)bedCacheError=Math.max(bedCacheError,Math.abs(bedCacheValues[i*8]-bedCacheValues[i*8+4]));
 assert.ok(bedCacheError<1e-7,'Cached bed iteration must match the direct height across cell boundaries');
 const repetition=[];
 for(const cascade of [0,1]){
  const tile=cascade===0?6:96,p=[];
  for(let i=0;i<128;i++){
   const x=Math.fround((i%16+.37)*tile/16),z=Math.fround((Math.floor(i/16)+.61)*tile/8);
   p.push([x,z,cascade,0],[x+tile,z,cascade,0],[x,z,cascade+2,0],[x+tile,z,cascade+2,0]);
  }
  const v=await page.evaluate(p=>waterLab.patternSamples(p),p),changed=[],control=[],signal=[];
  for(let i=0;i<128;i++){changed.push(v[i*16]-v[i*16+4]);control.push(v[i*16+8]-v[i*16+12]);signal.push(v[i*16]);}
  assert.ok(rms(control)<.00002,'Unwarped negative control must repeat');
  assert.ok(rms(changed)>rms(signal)*.1,`Cascade ${cascade} must differ across its old tile period`);
  repetition.push({cascade,tile,changedRms:rms(changed),signalRms:rms(signal),periodicControlRms:rms(control)});
 }
 // Exact representable coordinates avoid confusing float32 input rounding
 // with a chart seam. Sample both sides of each wrapped coordinate.
 const wrapPoints=[];
 for(const c of [0,1])for(const [x,z] of [[.125,9.25],[-.125,19.5],[17.5,-.125]])wrapPoints.push([x,z,c,0],[x+6144,z,c,0],[x,z+6144,c,0]);
 const wrap=await page.evaluate(p=>waterLab.patternSamples(p),wrapPoints);let wrapError=0;
 for(let i=0;i<wrapPoints.length;i+=3)for(let k=0;k<3;k++)for(let j=1;j<3;j++)wrapError=Math.max(wrapError,Math.abs(wrap[i*4+k]-wrap[(i+j)*4+k]));
 assert.ok(wrapError<.003,`Navigation chart wrap error: ${wrapError}`);
 const bed=await page.evaluate(()=>waterLab.bedTest()),sediment=await page.evaluate(()=>waterLab.sandTest()),interpolation=await page.evaluate(()=>waterLab.interpolationTest());
 assert.ok(bed.finite&&bed.sandSlopeError<.01&&bed.stoneSlopeError<.02);
 assert.ok(sediment.shallow.finite&&sediment.shallow.rms>.0001);for(const key of ['flat','deep','paused'])assert.equal(sediment[key].max,0);
 assert.ok(interpolation.finite&&interpolation.cubic.height<interpolation.bilinear.height*.3);
 await page.locator('#quality').selectOption('2560');await page.setViewportSize({width:2560,height:1440});await page.evaluate(()=>waterLab.lookDown());await page.evaluate(()=>waterLab.seek(4));await page.locator('#toggle').click();
 await page.screenshot({path:'captures/appearance-sand-1440p.png'});
 report.fields={slopeErrors,cacheError,bedCacheError,repetition,wrapError,bed,sediment,interpolation};await page.close();
 report.current=await benchmarks(base,'current');
 const phone=await browser.newPage({...devices['Pixel 7']});phone.on('pageerror',e=>errors.push(String(e)));
 await phone.goto(base+'?mode=explorer&t=4&geology=study');await ready(phone);await phone.evaluate(()=>waterLab.lookDown());
 assert.equal(await phone.evaluate(()=>waterDiagnostics.pixelSamples),1);assert.equal(await phone.evaluate(()=>waterDiagnostics.lightMapSize),256);
 report.mobile=await phone.evaluate(()=>waterLab.benchmark(40));await phone.screenshot({path:'captures/appearance-mobile.png'});await phone.close();
 assert.deepEqual(errors,[]);report.errors=errors;report.environment='Edge WebGPU on desktop hardware; mobile viewport emulation is not a physical-phone benchmark';
 await writeFile('captures/appearance-validation.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}
