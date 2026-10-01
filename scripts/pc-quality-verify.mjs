import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
await mkdir('captures',{recursive:true});
const completionSamples=page=>page.evaluate(()=>new Promise((resolve,reject)=>{
   const times=[],fps=[];let lastFrame=waterDiagnostics.frames;
   const timer=setTimeout(()=>reject(Error('Frame completion sampling timed out')),10000);
   function poll(){
    if(waterDiagnostics.errors.length){clearTimeout(timer);reject(Error(waterDiagnostics.errors.join('; ')));return;}
    if(waterDiagnostics.frames!==lastFrame){lastFrame=waterDiagnostics.frames;times.push(waterDiagnostics.frameMs);fps.push(waterDiagnostics.fps);}
    if(times.length===40){clearTimeout(timer);times.sort((a,b)=>a-b);fps.sort((a,b)=>a-b);resolve({frames:40,medianMs:times[20],p95Ms:times[38],medianFps:fps[20],scope:'CPU encoding through GPU completion, including canvas copy; excludes physical display latency'});return;}
    requestAnimationFrame(poll);
   }requestAnimationFrame(poll);
  }));
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto((process.env.WATER_URL||'http://127.0.0.1:5191/')+'?t=4&geology=study');
 await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length);
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 assert.equal(await page.evaluate(()=>waterDiagnostics.pixelSamples),4);
 const bed=await page.evaluate(()=>waterLab.bedTest());assert.ok(bed.finite&&bed.stonePoints>0&&bed.maxStoneHeight>.006);assert.ok(bed.sandSlopeError<.01);assert.ok(bed.stoneSlopeError<.02);
 const sandTransport=await page.evaluate(()=>waterLab.sandTest());
 assert.ok(Object.values(sandTransport).every(s=>s.finite));
 assert.ok(sandTransport.shallow.rms>.0001,'Actual FFT waves must reshape shallow sand');
 assert.equal(sandTransport.flat.max,0,'Flat water must not create sediment motion');
 assert.equal(sandTransport.deep.max,0,'Deep water must leave sand unchanged');
 assert.equal(sandTransport.paused.max,0,'Zero timestep must preserve sand');
 const reconstruction=await page.evaluate(()=>waterLab.interpolationTest());
 assert.ok(reconstruction.finite);assert.ok(reconstruction.cubic.height<reconstruction.bilinear.height*.3);assert.ok(reconstruction.cubic.slope<reconstruction.bilinear.slope*.6);
 const flat=await page.evaluate(()=>waterLab.flatCausticsTest());assert.ok(flat.maxDeviationFromUniform<.001);
 // Freeze identical water/camera states before and after live sediment evolution.
 await page.locator('#depth').evaluate(e=>{e.value='.5';e.dispatchEvent(new Event('input'));});
 await page.evaluate(()=>waterLab.lookDown());await page.evaluate(()=>waterLab.seek(4));
 const beforeSand=await page.evaluate(()=>waterLab.screenshot());
 await page.screenshot({path:'captures/sand-fft-before.png'});
 await page.evaluate(()=>waterLab.resume());await page.waitForTimeout(4000);await page.evaluate(()=>waterLab.seek(4));
 const afterSand=await page.evaluate(()=>waterLab.screenshot());let sedimentChannels=0;
 for(let i=0;i<beforeSand.rgba.length;i++)if(beforeSand.rgba[i]!==afterSand.rgba[i])sedimentChannels++;
 assert.ok(sedimentChannels>100,'Sand must visibly evolve with water frozen at the identical instant');
 await page.screenshot({path:'captures/sand-fft-after.png'});
 await page.locator('#depth').evaluate(e=>{e.value='1.4';e.dispatchEvent(new Event('input'));});
 await page.evaluate(()=>waterLab.lookAt(0,-.32));await page.evaluate(()=>waterLab.seek(4));
 const frames=[],timings=[];
 for(const samples of [1,4]){
  await page.evaluate(count=>waterLab.setPixelSamples(count),samples);
  const state=await page.evaluate(()=>waterLab.inspect());assert.ok(state.finite);
  const pixels=await page.evaluate(()=>waterLab.screenshot());frames.push({samples,state,pixels});
  timings.push({pixelSamples:samples,...await page.evaluate(()=>waterLab.benchmark(120))});
  await page.screenshot({path:`captures/pc-quality-${samples}-samples.png`});
 }
 assert.deepEqual(frames[0].state.camera,frames[1].state.camera);
 assert.equal(frames[0].state.heightRms,frames[1].state.heightRms);
 assert.deepEqual(frames[0].state.causticMean,frames[1].state.causticMean);
 let changed=0,total=0;
 const a=frames[0].pixels,b=frames[1].pixels;assert.equal(a.width,b.width);assert.equal(a.height,b.height);
 for(let i=0;i<a.rgba.length;i++){const d=Math.abs(a.rgba[i]-b.rgba[i]);if(d)changed++;total+=d;}
 assert.ok(changed>0,'Spatial samples must affect the image');
 assert.ok(total/a.rgba.length<8,'Supersampling must retain the overall scene appearance');
 assert.ok(timings.every(t=>t.medianMs>0&&Number.isFinite(t.medianMs)));
 const resolutions=[];let stress;
 for(const [quality,width,height] of [['2560',2560,1440],['3840',3840,2160]]){
  await page.setViewportSize({width,height});
  await page.locator('#quality').evaluate((e,value)=>{e.value=value;e.dispatchEvent(new Event('change'));},quality);
  await page.evaluate(()=>waterLab.seek(4));
  const layout=await page.evaluate(()=>({width:waterDiagnostics.width,height:waterDiagnostics.height,samples:waterDiagnostics.pixelSamples,canvasWidth:document.querySelector('canvas').width,canvasHeight:document.querySelector('canvas').height}));
  assert.deepEqual(layout,{width,height,samples:4,canvasWidth:width,canvasHeight:height});
  assert.equal(await page.evaluate(()=>waterDiagnostics.lightMapSize),512);assert.equal(await page.evaluate(()=>waterDiagnostics.photonRays),1024);
  const gpu=await page.evaluate(()=>waterLab.benchmark(120));
  const completion=await completionSamples(page);
  assert.ok(gpu.medianMs>0&&Number.isFinite(gpu.medianMs));
  const state=await page.evaluate(()=>waterLab.inspect());assert.ok(state.finite);
  await page.screenshot({path:`captures/pc-quality-${width}x${height}.png`});
  resolutions.push({width,height,pixelSamples:4,gpu,completion});
  if(width===3840){
   await page.screenshot({path:'captures/pc-quality-grid-closeup.png',clip:{x:1400,y:1250,width:900,height:700}});
   await page.screenshot({path:'captures/pc-bed-closeup.png',clip:{x:1000,y:1550,width:1200,height:570}});
   for(const [id,value] of [['depth',.5],['wind',14],['energy',2.5]])await page.locator('#'+id).evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input'));},value);
   await page.evaluate(()=>{waterLab.resume();});await page.evaluate(()=>waterLab.lookDown());
   await page.mouse.move(1700,1300);await page.mouse.down();
   for(let i=0;i<12;i++){await page.mouse.move(1700+i*35,1300+i*20);await page.waitForTimeout(20);}
   await page.mouse.up();await page.evaluate(()=>waterLab.seek(7));
   const state=await page.evaluate(()=>waterLab.inspect());assert.ok(state.finite&&state.forceEnergy>0);assert.equal(await page.evaluate(()=>waterDiagnostics.activeCascades),3);
   const timing=await page.evaluate(()=>waterLab.benchmark(120)),completion=await completionSamples(page);stress={width,height,depth:.5,wind:14,energy:2.5,downwardCamera:true,pressureActive:true,timing,completion};
   await page.screenshot({path:'captures/pc-quality-4k-stress.png'});
  }
 }
 await page.evaluate(()=>waterLab.setPixelSamples(0));
 await page.locator('#quality').evaluate(e=>{e.value='768';e.dispatchEvent(new Event('change'));});
 await page.evaluate(()=>waterLab.seek(4));
 assert.equal(await page.evaluate(()=>waterDiagnostics.pixelSamples),1);
 await page.locator('#quality').evaluate(e=>{e.value='mobile';e.dispatchEvent(new Event('change'));});
 await page.evaluate(()=>waterLab.seek(4));await page.evaluate(()=>waterLab.setPixelSamples(4));
 assert.equal(await page.evaluate(()=>waterDiagnostics.pixelSamples),1,'Mobile must keep one sample even with a PC override');
 assert.deepEqual(errors,[]);assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 const result={sandTransport,sedimentChannels,environment:'Desktop NVIDIA/Edge; not a benchmark for every PC or phone',bed,reconstruction,flat,width:a.width,height:a.height,linearRadianceAveraging:true,meanChannelDifference:total/a.rgba.length,changedChannels:changed,simulationUnchanged:true,mobileSingleSample:true,timings,resolutions,stress,errors};
 await writeFile('captures/pc-quality-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
