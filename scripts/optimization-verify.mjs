import {chromium,devices} from 'playwright';
import {execFileSync} from 'node:child_process';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {compile,serializableArtifact} from '../vendor/webcuda/compiler/compiler.js';
// Reference only our own last published source, never other water examples.
const referenceCommit=process.env.WATER_REFERENCE_COMMIT||'dbfa572',oldSource=execFileSync('git',['show',`${referenceCommit}:src/water.cu`],{encoding:'utf8'}),oldHost=execFileSync('git',['show',`${referenceCommit}:app.js`],{encoding:'utf8'});
const oldHtml=execFileSync('git',['show',`${referenceCommit}:index.html`],{encoding:'utf8'});
const artifacts=new Map();for(const m of oldSource.matchAll(/__global__ void (\w+)/g)){const name=m[1];artifacts.set(name,JSON.stringify(serializableArtifact(compile(oldSource,{entry:name,workgroupSize:name==='fft_local'?[64,1,1]:['camera_step','brush_pick'].includes(name)?[1,1,1]:[8,8,1]}))));}
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']}),base=process.env.WATER_URL||'http://127.0.0.1:5191/';
const cases=[['shallows',4,1.4,5,false],['late',12,1.4,5,false],['deep',7,8,5,false],['highWindShallow',7,.5,14,false],['lowWindDeep',7,12,2,false],['down',4,1.4,5,true]],reference=[];let before,after;
await mkdir('captures',{recursive:true});
try{
 for(const old of [true,false]){
  const page=await browser.newPage({...devices['Pixel 7']});
  if(old){await page.route(url=>url.href===base+'?t=4',r=>r.fulfill({contentType:'text/html',body:oldHtml}));await page.route('**/app.js*',r=>r.fulfill({contentType:'text/javascript',body:oldHost}));await page.route('**/kernels/*.json*',r=>{const name=new URL(r.request().url()).pathname.split('/').at(-1).replace('.json','');return r.fulfill({contentType:'application/json',body:artifacts.get(name)});});}
  await page.goto(base+'?t=4');await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
  for(const [name,time,depth,wind,down] of cases){
   for(const [id,value] of [['depth',depth],['wind',wind]])await page.locator('#'+id).evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('input'));},value);
   if(down)await page.evaluate(()=>waterLab.lookDown());await page.evaluate(t=>waterLab.seek(t),time);
   const pix=await page.evaluate(()=>waterLab.screenshot());
   if(old)reference.push({name,width:pix.width,height:pix.height,rgba:pix.rgba});
   else{const ref=reference.find(r=>r.name===name);assert.equal(pix.width,ref.width);assert.equal(pix.height,ref.height);let max=0,total=0,changed=0;for(let i=0;i<pix.rgba.length;i++){const d=Math.abs(pix.rgba[i]-ref.rgba[i]);max=Math.max(max,d);total+=d;if(d)changed++;}Object.assign(ref,{maxChannelDifference:max,meanChannelDifference:total/pix.rgba.length,changedChannels:changed});assert.ok(max<=1&&total/pix.rgba.length<.0001,JSON.stringify({name,max,total,changed}));}
   if(name==='shallows'){const b=await page.evaluate(()=>waterLab.benchmark(80));if(old)before=b;else after=b;}
  }
  if(!old){const shared=await page.evaluate(()=>waterLab.fftTest()),staged=await page.evaluate(()=>waterLab.fftTest('staged'));assert.ok(shared.maxError<1e-5&&staged.maxError<1e-5);reference.push({shared,staged});}
  await page.close();
 }
 // The displayed FPS must measure wall time, even when simulation dt is capped.
 const slow=await browser.newPage({...devices['Pixel 7']});await slow.addInitScript(()=>{const raf=window.requestAnimationFrame.bind(window);window.requestAnimationFrame=callback=>raf(()=>setTimeout(()=>callback(performance.now()),160));});
 await slow.goto(base+'?t=4');await slow.waitForFunction(()=>waterDiagnostics.frames>=3,null,{timeout:120000});const fps=await slow.evaluate(()=>waterDiagnostics.fps);assert.ok(fps>0&&fps<10);await slow.close();
 const result={referenceCommit,environment:'Desktop Chromium/Edge with NVIDIA GPU and mobile viewport; not Samsung A34 hardware',before,after,speedup:before.medianMs&&after.medianMs?before.medianMs/after.medianMs:null,frames:reference.map(({rgba,...rest})=>rest),slowFrameFps:fps};await writeFile('captures/optimization-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
