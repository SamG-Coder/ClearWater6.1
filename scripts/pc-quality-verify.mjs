import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';

const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
await mkdir('captures',{recursive:true});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto((process.env.WATER_URL||'http://127.0.0.1:5191/')+'?t=4');
 await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length);
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 assert.equal(await page.evaluate(()=>waterDiagnostics.pixelSamples),4);
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
 const resolutions=[];
 for(const [quality,width,height] of [['2560',2560,1440],['3840',3840,2160]]){
  await page.setViewportSize({width,height});
  await page.locator('#quality').evaluate((e,value)=>{e.value=value;e.dispatchEvent(new Event('change'));},quality);
  await page.evaluate(()=>waterLab.seek(4));
  const layout=await page.evaluate(()=>({width:waterDiagnostics.width,height:waterDiagnostics.height,samples:waterDiagnostics.pixelSamples,canvasWidth:document.querySelector('canvas').width,canvasHeight:document.querySelector('canvas').height}));
  assert.deepEqual(layout,{width,height,samples:4,canvasWidth:width,canvasHeight:height});
  const gpu=await page.evaluate(()=>waterLab.benchmark(120));
  const completion=await page.evaluate(()=>new Promise((resolve,reject)=>{
   const times=[],fps=[];let lastFrame=waterDiagnostics.frames;
   const timer=setTimeout(()=>reject(Error('Frame completion sampling timed out')),10000);
   function poll(){
    if(waterDiagnostics.errors.length){clearTimeout(timer);reject(Error(waterDiagnostics.errors.join('; ')));return;}
    if(waterDiagnostics.frames!==lastFrame){lastFrame=waterDiagnostics.frames;times.push(waterDiagnostics.frameMs);fps.push(waterDiagnostics.fps);}
    if(times.length===40){clearTimeout(timer);times.sort((a,b)=>a-b);fps.sort((a,b)=>a-b);resolve({frames:40,medianMs:times[20],p95Ms:times[38],medianFps:fps[20],scope:'CPU encoding through GPU completion, including canvas copy; excludes physical display latency'});return;}
    requestAnimationFrame(poll);
   }requestAnimationFrame(poll);
  }));
  assert.ok(gpu.medianMs>0&&Number.isFinite(gpu.medianMs));
  const state=await page.evaluate(()=>waterLab.inspect());assert.ok(state.finite);
  await page.screenshot({path:`captures/pc-quality-${width}x${height}.png`});
  resolutions.push({width,height,pixelSamples:4,gpu,completion});
 }
 await page.evaluate(()=>waterLab.setPixelSamples(0));
 await page.locator('#quality').evaluate(e=>{e.value='768';e.dispatchEvent(new Event('change'));});
 await page.evaluate(()=>waterLab.seek(4));
 assert.equal(await page.evaluate(()=>waterDiagnostics.pixelSamples),1);
 await page.locator('#quality').evaluate(e=>{e.value='mobile';e.dispatchEvent(new Event('change'));});
 await page.evaluate(()=>waterLab.seek(4));await page.evaluate(()=>waterLab.setPixelSamples(4));
 assert.equal(await page.evaluate(()=>waterDiagnostics.pixelSamples),1,'Mobile must keep one sample even with a PC override');
 assert.deepEqual(errors,[]);assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 const result={environment:'Desktop NVIDIA/Edge; not a benchmark for every PC or phone',width:a.width,height:a.height,linearRadianceAveraging:true,meanChannelDifference:total/a.rgba.length,changedChannels:changed,simulationUnchanged:true,mobileSingleSample:true,timings,resolutions,errors};
 await writeFile('captures/pc-quality-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
