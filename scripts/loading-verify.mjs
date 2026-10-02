import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const base=process.env.WATER_URL||'http://127.0.0.1:5191/';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const report={layouts:[],checks:[]},errors=[];
const shot=(p,name)=>p.screenshot({path:`captures/loading/${name}.png`});
const loading=p=>p.evaluate(()=>waterDiagnostics.loading);
const select=(p,value)=>p.locator('#quality').selectOption(value);
async function heldDownload(context){
 let release;const hold=new Promise(resolve=>release=resolve);
 await context.route('**/kernels/render_pc.json?*',async route=>{await hold;await route.continue().catch(()=>{});});
 return release;
}
async function downloaded(p){await p.waitForFunction(()=>waterDiagnostics.loading?.phase==='download'&&waterDiagnostics.loading.completed===waterDiagnostics.loading.total-1,null,{timeout:60000});}
async function fits(p,selectors){for(const selector of selectors){const box=await p.locator(selector).boundingBox(),size=p.viewportSize();assert.ok(box&&box.x>=-1&&box.x+box.width<=size.width+1,`${selector} fits horizontally`);assert.ok(box.y>=0&&box.y+box.height<=size.height+1,`${selector} fits vertically`);}}
try{
 await mkdir('captures/loading',{recursive:true});
 // Real downloads are held before compilation: responsive UI checks need no GPU scene.
 for(const name of ['Pixel 7','iPhone 13']){
  const context=await browser.newContext({...devices[name],deviceScaleFactor:1});const release=await heldDownload(context),p=await context.newPage();
  await p.goto(base);await downloaded(p);await shot(p,`${name}-preflight`);
  await fits(p,['#loadTitle','#loadProgress','#bootSettings']);assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await p.locator('#bootSettings').click();await shot(p,`${name}-settings`);await fits(p,['#flightLoading','#settingsBack']);await p.locator('#settingsBack').click();
  await p.setViewportSize({width:844,height:390});await shot(p,`${name}-landscape`);await fits(p,['#loadTitle','#loadProgress','#bootSettings']);
  await p.emulateMedia({reducedMotion:'reduce'});assert.equal(await p.locator('.load-activity').evaluate(el=>getComputedStyle(el).animationName),'none');
  assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);report.layouts.push({name,portrait:true,landscape:true,reducedMotion:true,scope:'Emulated layout with shader download held; not a physical phone GPU benchmark'});
  await context.close();release();console.log('LAYOUT',name);
 }
 // A real HTTP failure must remain visible; late successful downloads cannot mark ready.
 {
  const context=await browser.newContext({viewport:{width:1440,height:900}});
  await context.route('**/kernels/render_pc.json?*',route=>route.fulfill({status:503,body:'Unavailable'}));
  const p=await context.newPage();await p.goto(base);await p.waitForFunction(()=>waterDiagnostics.errors.length>0);
  await p.waitForTimeout(350);assert.equal(await p.locator('#loadRetry').isVisible(),true);assert.equal((await loading(p)).phase,'download');assert.equal(await p.evaluate(()=>waterDiagnostics.ready),false);assert.ok(await p.locator('#loadDescription').innerText().then(s=>s.includes('503')));
  await shot(p,'download-error');report.checks.push('HTTP failure stays visible with retry; no false readiness');await context.close();
 }
 {
  const context=await browser.newContext();await context.addInitScript(()=>Object.defineProperty(navigator,'gpu',{value:undefined}));const p=await context.newPage();await p.goto(base);await p.waitForFunction(()=>waterDiagnostics.errors.length>0);assert.equal((await loading(p)).phase,'device');assert.ok(await p.locator('#loadRetry').isVisible());report.checks.push('Unsupported graphics shows device-stage failure');await context.close();
 }
 const context=await browser.newContext({viewport:{width:2560,height:1440},deviceScaleFactor:1});const release=await heldDownload(context);
 await context.addInitScript(()=>{
  localStorage.setItem('clearwater6.1.settings.v1',JSON.stringify({quality:'768'}));
  window.preflightProbe={hold:'camera_step',waiting:null,release:null,frameHold:true,frameWaiting:false,idleCalls:0};
  const compile=GPUDevice.prototype.createComputePipelineAsync;
  GPUDevice.prototype.createComputePipelineAsync=async function(descriptor){
   const pipeline=await compile.call(this,descriptor);
   if(descriptor.label===preflightProbe.hold){preflightProbe.waiting=descriptor.label;await new Promise(resolve=>preflightProbe.release=()=>{preflightProbe.waiting=null;resolve();});}
   return pipeline;
  };
  const idle=GPUQueue.prototype.onSubmittedWorkDone;
  GPUQueue.prototype.onSubmittedWorkDone=async function(){await idle.call(this);if(++preflightProbe.idleCalls===2&&preflightProbe.frameHold){preflightProbe.frameWaiting=true;await new Promise(resolve=>preflightProbe.releaseFrame=resolve);}};
 });
 const p=await context.newPage();p.on('pageerror',e=>errors.push(String(e)));await p.goto(base);await downloaded(p);const before=await loading(p);await p.waitForTimeout(1100);assert.deepEqual(await loading(p),before,'Elapsed time must not advance actual download progress');
 await shot(p,'download-1440p');await p.setViewportSize({width:3840,height:2160});await shot(p,'download-4k');await p.setViewportSize({width:2560,height:1440});report.checks.push('Download count freezes while a real response is held');release();
 await p.waitForFunction(()=>preflightProbe.waiting==='camera_step',null,{timeout:240000});const compiling=await loading(p);assert.equal(compiling.phase,'shaders');assert.equal(compiling.completed,0);await p.waitForTimeout(1100);assert.deepEqual(await loading(p),compiling);assert.ok(await p.locator('#newGame').isDisabled());
 await shot(p,'shaders-1440p');await p.locator('#bootSettings').click();await shot(p,'startup-settings');await p.locator('#settingsBack').click();await p.evaluate(()=>preflightProbe.release());console.log('REAL GPU COMPILATION STARTED');
 await p.waitForFunction(()=>preflightProbe.frameWaiting||waterDiagnostics.errors.length,null,{timeout:240000});assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);assert.equal((await loading(p)).phase,'frame');assert.equal(await p.evaluate(()=>waterDiagnostics.ready),false);assert.ok(await p.locator('#newGame').isDisabled());await shot(p,'first-view');await p.evaluate(()=>preflightProbe.releaseFrame());
 await p.waitForFunction(()=>waterDiagnostics.ready&&!waterDiagnostics.loading.active);assert.equal(await p.locator('#flightLoading').isHidden(),true);assert.equal((await p.evaluate(()=>waterLab.sessionState())).shell.page,'main');await shot(p,'ready-1440p');report.checks.push('Shader count waits for GPU pipeline promises; launch waits for the first completed frame');console.log('STARTUP READY');
 // Compile a previously unused PC profile while paused; also change the request mid-compile.
 await p.locator('#mainSettings').click();await p.evaluate(()=>preflightProbe.hold='render_pc');await select(p,'2560');await p.waitForFunction(()=>waterDiagnostics.loading.active);
 const paused=await p.evaluate(()=>waterLab.sessionState());await shot(p,'quality-preparing');
 await p.waitForFunction(()=>preflightProbe.waiting==='render_pc'||waterDiagnostics.errors.length,null,{timeout:240000});assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);await select(p,'3840');assert.equal((await loading(p)).profile,'4K');await p.setViewportSize({width:3840,height:2160});await shot(p,'quality-4k');
 assert.equal((await p.evaluate(()=>waterLab.sessionState())).clock,paused.clock);assert.equal(await p.locator('#newGame').isDisabled(),true);await p.keyboard.press('Escape');assert.equal((await p.evaluate(()=>waterLab.sessionState())).shell.page,'main');assert.equal(await p.locator('#flightLoading').isVisible(),true);await p.locator('#mainSettings').click();
 await p.evaluate(()=>preflightProbe.release());await p.waitForFunction(()=>!waterDiagnostics.loading.active&&waterDiagnostics.width===3840&&waterDiagnostics.height===2160,null,{timeout:240000});
 assert.equal((await p.evaluate(()=>waterLab.sessionState())).clock,paused.clock);await p.locator('#settingsBack').click();await shot(p,'ready-4k');report.checks.push('Profile changes remain paused, track the latest selection and unlock after a real 4K frame');
 const readyState=await p.evaluate(()=>waterLab.sessionState());await p.waitForTimeout(1100);assert.equal((await p.evaluate(()=>waterLab.sessionState())).frames,readyState.frames);assert.equal(await p.locator('#flightLoading').isHidden(),true);report.checks.push('Loader stops and menu GPU dispatch remains idle after readiness');
 // Launch and Escape still work after graphics preparation.
 await p.locator('#mainSettings').click();await select(p,'768');await p.waitForFunction(()=>waterDiagnostics.width===768);assert.equal((await loading(p)).active,false);await p.locator('#settingsBack').click();await p.locator('#newGame').click();await p.waitForFunction(()=>!waterLab.sessionState().shell.pending);await p.locator('#pilotName').fill('Test Pilot');await p.locator('#newShipName').fill('Test Craft');await p.locator('#launchFlight').click();await p.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');await p.keyboard.press('Escape');await p.waitForFunction(()=>waterLab.sessionState().shell.page==='pause');report.checks.push('Previously prepared profiles skip compilation; New Game and Escape still work');
 report.gpu=await p.evaluate(()=>({adapter:waterDiagnostics.adapter,kernels:waterDiagnostics.compiledKernels,errors:waterDiagnostics.errors}));assert.deepEqual(report.gpu.errors,[]);assert.deepEqual(errors,[]);await context.close();
 await writeFile('captures/loading/validation.json',JSON.stringify(report,null,2));console.log('LOADING PASSED',JSON.stringify(report));
}finally{await browser.close();}
