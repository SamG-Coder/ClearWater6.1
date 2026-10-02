import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {validSave,SAVE_KEY} from '../game-shell.js';
import {validIdentity,DEFAULT_COLOURS} from '../ship-profile.js';
const base=process.env.WATER_URL||'http://127.0.0.1:5191/';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const errors=[],report={};
const state=p=>p.evaluate(()=>waterLab.sessionState());
const ready=async p=>{p.on('pageerror',e=>errors.push(String(e)));await p.goto(base);await p.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:240000});assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);};
const settled=p=>p.waitForFunction(()=>!waterLab.sessionState().shell.pending&&!waterLab.sessionState().shell.saveBusy);
const screenshot=(p,name)=>p.screenshot({path:`captures/registration/${name}.png`});
async function colour(p,key,value){const frames=(await state(p)).frames;await p.locator('#'+key+'Colour').fill(value);await p.waitForFunction(n=>waterDiagnostics.frames>n,frames);}
async function setup(p,pilot='Sam',ship='Horizon Runner'){await p.locator('#newGame').click();await settled(p);await p.locator('#pilotName').fill(pilot);await p.locator('#newShipName').fill(ship);}
const linear=hex=>[1,3,5].map(i=>{const v=parseInt(hex.slice(i,i+2),16)/255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4;});
try{
 await mkdir('captures/registration',{recursive:true});
 const context=await browser.newContext({viewport:{width:2560,height:1440},deviceScaleFactor:1}),p=await context.newPage();await ready(p);console.log('REGISTRATION GPU READY');
 await p.locator('#newGame').click();await settled(p);assert.equal((await state(p)).shell.page,'setup');assert.ok(await p.locator('#launchFlight').isDisabled());await screenshot(p,'empty-1440p');
 await p.locator('#pilotName').fill('   ');await p.locator('#newShipName').fill('Test');assert.ok(await p.locator('#launchFlight').isDisabled());
 await p.locator('#pilotName').fill('Sam');await p.locator('#newShipName').fill('Horizon Runner');assert.ok(await p.locator('#launchFlight').isEnabled());
 const compiled=await p.evaluate(()=>waterDiagnostics.compiledKernels),time=(await state(p)).clock;
 for(const [key,value] of Object.entries({primary:'#3158c8',secondary:'#e8bb54',booster:'#ff3628'}))await colour(p,key,value);
 const palette=(await p.evaluate(()=>waterLab.shipState())).data.slice(44,56);
 for(const [index,value] of Object.values({primary:'#3158c8',secondary:'#e8bb54',booster:'#ff3628'}).entries())for(let channel=0;channel<3;channel++)assert.ok(Math.abs(palette[index*4+channel]-linear(value)[channel])<.00002);
 assert.equal(await p.evaluate(()=>waterDiagnostics.compiledKernels),compiled,'Colours must not compile new shaders');assert.equal((await state(p)).clock,time);await screenshot(p,'colours-1440p');
 const before=(await p.evaluate(()=>waterLab.shipState())).data.slice(8,12);await p.mouse.move(1840,610);await p.mouse.down();await p.mouse.move(2020,650,{steps:6});await p.mouse.up();await p.waitForTimeout(100);const after=(await p.evaluate(()=>waterLab.shipState())).data.slice(8,12);assert.notEqual(after[0],before[0]);report.preview={liveColours:true,linearPalette:palette,noShaderRecompile:true,orbit:true,pausedClock:true};
 await p.setViewportSize({width:3840,height:2160});await p.locator('#quality').evaluate(el=>{el.value='3840';el.dispatchEvent(new Event('change',{bubbles:true}));});await p.waitForFunction(()=>waterDiagnostics.width===3840);await screenshot(p,'colours-4k');
 await p.locator('#launchFlight').click();await p.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');await p.keyboard.press('Escape');await settled(p);
 const saved=JSON.parse(await p.evaluate(key=>localStorage.getItem(key),SAVE_KEY));assert.ok(validSave(saved));assert.deepEqual(saved.identity,{pilotName:'Sam',shipName:'Horizon Runner',colours:{primary:'#3158c8',secondary:'#e8bb54',booster:'#ff3628'}});assert.equal(await p.locator('#shipNameHUD').textContent(),'Horizon Runner');
 const old={...saved};delete old.identity;assert.ok(validSave(old));assert.equal(validSave({...saved,identity:{...saved.identity,colours:{...DEFAULT_COLOURS,booster:'red'}}}),false);assert.equal(validIdentity({...saved.identity,pilotName:'x'.repeat(33)}),false);
 await p.locator('#returnMain').click();await settled(p);await setup(p,'Cancelled pilot','Cancelled craft');await colour(p,'booster','#00ff00');await p.locator('#launchFlight').click();assert.equal((await state(p)).shell.page,'confirm');await p.locator('#cancelNewGame').click();await p.locator('#cancelSetup').click();assert.deepEqual(JSON.parse(await p.evaluate(key=>localStorage.getItem(key),SAVE_KEY)).identity,saved.identity);report.cancelPreservesSave=true;
 await p.locator('#continueGame').click();await p.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');await p.keyboard.press('Escape');await settled(p);await p.locator('#pauseSettings').click();await p.locator('#settingsShipTab').click();await settled(p);await screenshot(p,'ship-settings');
 await colour(p,'booster','#14ff5b');await p.waitForTimeout(450);await settled(p);assert.equal(JSON.parse(await p.evaluate(key=>localStorage.getItem(key),SAVE_KEY)).identity.colours.booster,'#14ff5b');await p.locator('#settingsBack').click();await p.locator('#returnMain').click();await settled(p);
 // Reload exercises persistence through actual initialization and CUDA palette upload.
 await p.reload();await p.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:240000});assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);await p.locator('#continueGame').click();await p.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');await p.keyboard.press('Escape');await settled(p);assert.equal((await state(p)).shell.identity.colours.booster,'#14ff5b');assert.equal(await p.locator('#shipNameHUD').textContent(),'Horizon Runner');report.save={identityRoundTrip:true,legacyAccepted:true,malformedRejected:true};console.log('NAMES AND SAVE PASSED');
 // Independent colour oracle: subtract black emitters at identical points so
 // foam, fixed navigation lamps and landing lights cancel out of the result.
 await p.locator('#quality').evaluate(el=>{el.value='768';el.dispatchEvent(new Event('change',{bubbles:true}));});await p.evaluate(async()=>{document.getElementById('depthMode').value='0';await waterLab.weatherLocation(0,0,6);await waterLab.shipAltitude(6);await waterLab.weatherSeek(7200);await waterLab.shipView(.38,.78,28);});
 await p.locator('#pauseSettings').click();await p.locator('#settingsShipTab').click();await settled(p);
 const wash=await p.evaluate(()=>waterLab.shipWashState()),points=wash.engines.map(e=>[e[0],0,e[1],0]);points.push([-4.65,.22,4.8,1],[4.65,.22,4.8,1]);
 const samples={};for(const [label,hex] of [['black','#000000'],['red','#ff0000'],['green','#00ff00'],['blue','#0000ff']]){await colour(p,'booster',hex);samples[label]=await p.evaluate(points=>waterLab.shipEffectSamples(points),points);assert.ok(samples[label].every(Number.isFinite));}
 for(let source=0;source<4;source++)for(const [channel,label] of ['red','green','blue'].entries()){const delta=samples[label].slice(source*4,source*4+3).map((v,i)=>v-samples.black[source*4+i]);assert.ok(delta[channel]>.00001,`${label} must light source ${source}`);for(let c=0;c<3;c++)if(c!==channel)assert.ok(Math.abs(delta[c])<.00001,'Unselected light channels must stay unchanged');}
 await colour(p,'booster','#ff793a');await p.locator('#settingsBack').click();await p.locator('#quality').evaluate(el=>{el.value='2560';el.dispatchEvent(new Event('change',{bubbles:true}));});await p.setViewportSize({width:2560,height:1440});await p.evaluate(()=>waterLab.shipView(.38,.78,28));await screenshot(p,'orange-night-water');report.lighting={points,samples,independentChannelOracle:true};
 assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);await context.close();console.log('BOOSTER LIGHTING PASSED');
 // Phone UI and a real mobile-profile CUDA preview in emulated Chromium.
 const mobile=await browser.newContext({...devices['Pixel 7'],deviceScaleFactor:1}),phone=await mobile.newPage();await ready(phone);await setup(phone,'Sam','Aurora');await colour(phone,'booster','#b167ff');await screenshot(phone,'phone-setup');assert.equal(await phone.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);await phone.locator('#launchFlight').scrollIntoViewIfNeeded();await screenshot(phone,'phone-controls');await phone.locator('#launchFlight').click();await phone.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');await phone.locator('#toggle').click();await settled(phone);assert.equal((await state(phone)).shell.identity.shipName,'Aurora');assert.deepEqual(await phone.evaluate(()=>waterDiagnostics.errors),[]);report.mobile={layoutAndLaunch:true,scope:'Pixel 7 browser emulation; physical Android/iPhone GPU performance not measured'};await mobile.close();
 assert.deepEqual(errors,[]);report.errors=errors;await writeFile('captures/registration/validation.json',JSON.stringify(report,null,2));console.log('SHIP PROFILE PASSED',JSON.stringify(report));
}finally{await browser.close();}
