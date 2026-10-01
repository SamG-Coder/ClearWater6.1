import {chromium,devices} from 'playwright';
import {execFileSync} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {compile,serializableArtifact} from '../vendor/webcuda/compiler/compiler.js';
// Reference only our own last published source, never other water examples.
const referenceCommit=process.env.WATER_REFERENCE_COMMIT||'ffeb5a0',oldSource=execFileSync('git',['show',`${referenceCommit}:src/water.cu`],{encoding:'utf8'}),oldHost=execFileSync('git',['show',`${referenceCommit}:app.js`],{encoding:'utf8'}).replace('window.waterLab={','window.waterLab={'+"\n async lookAt(yaw,pitch){return exclusive(async()=>{runtime.device.queue.writeBuffer(camera.gpuBuffer,16,new Float32Array([yaw,pitch,0,0]));compute(0);await runtime.idle();});},");
const oldHtml=execFileSync('git',['show',`${referenceCommit}:index.html`],{encoding:'utf8'});
const oldBuild=execFileSync('git',['show',`${referenceCommit}:scripts/build.mjs`],{encoding:'utf8'});
const artifacts=new Map();for(const m of oldSource.matchAll(/__global__ void (\w+)/g)){const name=m[1];artifacts.set(name,JSON.stringify(serializableArtifact(compile(oldSource,{entry:name,workgroupSize:['render','render_pc'].includes(name)&&(oldBuild.includes("name==='render'?[32,2,1]")||oldBuild.includes("['render','render_pc'].includes(name)?[32,2,1]"))?[32,2,1]:name==='fft_local'?[64,1,1]:['camera_step','brush_pick'].includes(name)?[1,1,1]:[8,8,1]}))));}
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']}),base=process.env.WATER_URL||'http://127.0.0.1:5191/';
const cases=[['shallows',4,1.4,5,false],['late',12,1.4,5,false],['deep',7,8,5,false],['highWindShallow',7,.5,14,false],['lowWindDeep',7,12,2,false],['down',4,1.4,5,true],['sun',4,1.4,5,'sun']],reference=[];let before,after;
await mkdir('captures',{recursive:true});
try{
 for(const old of [true,false]){
  const page=await browser.newPage(process.env.WATER_DESKTOP==='1'?{viewport:{width:1440,height:900}}:{...devices['Pixel 7']});
  if(old){await page.route(url=>url.href===base+'?t=4',r=>r.fulfill({contentType:'text/html',body:oldHtml}));await page.route('**/app.js*',r=>r.fulfill({contentType:'text/javascript',body:oldHost}));await page.route('**/kernels/*.json*',r=>{const name=new URL(r.request().url()).pathname.split('/').at(-1).replace('.json','');return r.fulfill({contentType:'application/json',body:artifacts.get(name)});});}
  await page.goto(base+'?t=4');await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);if(!old&&process.env.WATER_DESKTOP==='1')await page.evaluate(()=>waterLab.setPixelSamples(1));
  for(const [name,time,depth,wind,down] of cases){
   for(const [id,value] of [['depth',depth],['wind',wind]])await page.locator('#'+id).evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input'));},value);
   if(down===true)await page.evaluate(()=>waterLab.lookDown());if(down==='sun')await page.evaluate(()=>waterLab.lookAt(-.5880026,.716));await page.evaluate(t=>waterLab.seek(t),time);
   const pix=await page.evaluate(()=>waterLab.screenshot());
   if(old)reference.push({name,width:pix.width,height:pix.height,rgba:pix.rgba});
   else{const ref=reference.find(r=>r.name===name);assert.equal(pix.width,ref.width);assert.equal(pix.height,ref.height);let max=0,total=0,changed=0;for(let i=0;i<pix.rgba.length;i++){const d=Math.abs(pix.rgba[i]-ref.rgba[i]);max=Math.max(max,d);total+=d;if(d)changed++;}Object.assign(ref,{maxChannelDifference:max,meanChannelDifference:total/pix.rgba.length,changedChannels:changed});assert.ok(max===0,JSON.stringify({name,max,total,changed}));}
   if(name==='shallows'){const b=await page.evaluate(()=>waterLab.benchmark(80));if(old)before=b;else after=b;}
   if(!old&&name==='shallows'){
    // A drag in the sky enables the pressure path without injecting force.
    // Its zero field must produce the same pixels as the skipped-field path.
    await page.mouse.move(20,20);await page.mouse.down();await page.mouse.move(40,20);
    await page.waitForFunction(()=>waterDiagnostics.activeCascades===3);await page.mouse.up();
    const state=await page.evaluate(()=>waterLab.inspect());assert.equal(state.forceEnergy,0);
    const active=await page.evaluate(()=>waterLab.screenshot());assert.ok(active.rgba.every((v,i)=>v===pix.rgba[i]));
    reference.push({name:'activatedZeroPressure',width:active.width,height:active.height,maxChannelDifference:0,meanChannelDifference:0,changedChannels:0});
   }

  }
  if(!old){const shared=await page.evaluate(()=>waterLab.fftTest()),staged=await page.evaluate(()=>waterLab.fftTest('staged'));assert.ok(shared.maxError<1e-5&&staged.maxError<1e-5);reference.push({shared,staged});}
  await page.close();
 }
 // The displayed FPS must measure wall time, even when simulation dt is capped.
 const slow=await browser.newPage({...devices['Pixel 7']});await slow.addInitScript(()=>{const raf=window.requestAnimationFrame.bind(window);window.requestAnimationFrame=callback=>raf(()=>setTimeout(()=>callback(performance.now()),160));});
 await slow.goto(base+'?t=4');await slow.waitForFunction(()=>waterDiagnostics.frames>=3,null,{timeout:120000});const fps=await slow.evaluate(()=>waterDiagnostics.fps);assert.ok(fps>0&&fps<10);await slow.close();
 const result={referenceCommit,profile:process.env.WATER_DESKTOP==='1'?'desktop RGB':'mobile monochrome',environment:'Desktop Chromium/Edge with NVIDIA GPU; viewport emulation is not physical phone hardware',before,after,speedup:before.medianMs&&after.medianMs?before.medianMs/after.medianMs:null,frames:reference.map(({rgba,...rest})=>rest),slowFrameFps:fps};await writeFile('captures/optimization-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
