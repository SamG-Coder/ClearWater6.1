import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {validSave,SAVE_KEY} from '../game-shell.js';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const base=process.env.WATER_URL||'http://127.0.0.1:5191/',errors=[],report=process.argv.includes('--touch-only')?JSON.parse(await readFile('captures/menu/desktop-validation.json','utf8')):{};
const state=p=>p.evaluate(()=>waterLab.sessionState());
const select=(p,id,value)=>p.locator('#'+id).evaluate((el,value)=>{el.value=String(value);el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));},value);
const idleSave=p=>p.waitForFunction(()=>!waterLab.sessionState().shell.saveBusy&&!waterLab.sessionState().shell.pending);
const ready=async p=>{p.on('pageerror',e=>errors.push(String(e)));await p.goto(base);await p.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:240000});assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);};
const screen=(p,name)=>p.screenshot({path:`captures/menu/${name}.png`});
try{
 await mkdir('captures/menu',{recursive:true});
 if(!process.argv.includes('--touch-only')){
 const page=await browser.newPage({viewport:{width:2560,height:1440},deviceScaleFactor:1});await ready(page);
 console.log('MENU READY');assert.equal((await state(page)).shell.page,'main');assert.equal((await state(page)).playing,false);assert.ok(await page.locator('#continueGame').isDisabled());
 const menuBefore=await state(page);await page.waitForTimeout(650);assert.deepEqual(await state(page),menuBefore,'Main menu must stop rendering and simulation');
 await screen(page,'main-1440p');await page.locator('#mainSettings').click();await select(page,'quality','2560');await page.waitForFunction(()=>waterDiagnostics.width===2560);await screen(page,'graphics-1440p');
 await page.locator('#graphicsTab').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('#controlsTab').getAttribute('aria-selected'),'true');await select(page,'sensitivity','1.35');await page.locator('#invertY').check();await page.locator('#invertY').uncheck();await screen(page,'controls-1440p');await page.keyboard.press('Escape');assert.equal((await state(page)).shell.page,'main');
 await page.locator('#mainSettings').click();await select(page,'quality','3840');await page.setViewportSize({width:3840,height:2160});await page.waitForFunction(()=>waterDiagnostics.width===3840&&waterDiagnostics.height===2160);await page.locator('#settingsBack').click();await screen(page,'main-4k');
 report.resolutions=await page.evaluate(()=>({width:waterDiagnostics.width,height:waterDiagnostics.height,samples:waterDiagnostics.pixelSamples}));
 await page.setViewportSize({width:1440,height:900});await page.locator('#mainSettings').click();await select(page,'quality','768');await page.waitForFunction(()=>waterDiagnostics.width===768);await page.locator('#settingsBack').click();
 await page.locator('#newGame').click();await page.waitForFunction(()=>waterLab.sessionState().shell.page==='playing',null,{timeout:90000});assert.equal((await state(page)).flightProfile,1);assert.ok(await page.locator('#shipSpace').isHidden());
 await page.locator('#shipFly').click();await page.waitForFunction(()=>['locked','drag'].includes(waterDiagnostics.pointerLock));report.pointerLock=await page.evaluate(()=>waterDiagnostics.pointerLock);
 await page.keyboard.press('Escape');await page.waitForFunction(()=>waterLab.sessionState().shell.page==='pause');assert.equal(await page.evaluate(()=>document.pointerLockElement),null);await idleSave(page);await page.locator('#resumeGame').click();
 await page.mouse.move(900,450);await page.mouse.wheel(0,-10000);await page.waitForTimeout(80);assert.equal((await state(page)).throttle,1);
 await page.keyboard.down('KeyW');await page.waitForTimeout(800);await page.keyboard.press('Escape');await page.keyboard.up('KeyW');await idleSave(page);
 assert.equal((await state(page)).shell.page,'pause');assert.equal((await state(page)).playing,false);
 await page.waitForTimeout(200);const frozen=await state(page),shipFrozen=await page.evaluate(()=>waterLab.shipState());await page.waitForTimeout(600);
 assert.deepEqual(await state(page),frozen,'Escape freezes clock, play time, frames and ship');assert.deepEqual((await page.evaluate(()=>waterLab.shipState())).data,shipFrozen.data);assert.ok(shipFrozen.speed>30&&shipFrozen.speed<181,'Keyboard flies the bounded normal craft');
 await screen(page,'pause');await page.locator('#pauseSettings').click();assert.ok(await page.locator('#settingsWorldTab').isHidden());await page.keyboard.press('KeyW');await page.keyboard.press('Escape');assert.equal((await state(page)).shell.page,'pause');
 report.pause={time:frozen.time,clock:frozen.clock,frames:frozen.frames,speed:shipFrozen.speed,noBackgroundFrames:true};
 // Sample the real CUDA integration, including diagonal thrust and re-entry.
 const samples=[];
 for(const [altitude,boost,cap] of [[22,false,180],[22,true,450],[90000,false,1800],[1000000,false,150000],[1000000,true,600000]]){
  await page.evaluate(alt=>waterLab.shipAltitude(alt),altitude);
  await page.evaluate(async boost=>{await waterLab.shipAdvance(150,1/60,{forward:1,boost,rise:1});},boost);
  const ship=await page.evaluate(()=>waterLab.shipState());assert.ok(ship.finite&&ship.speed<=cap*1.00001,JSON.stringify({altitude,boost,cap,speed:ship.speed}));assert.ok(ship.speed>cap*.1);samples.push({altitude,boost,cap,speed:ship.speed,finalAltitude:ship.position[1]});
 }
 await page.evaluate(async()=>{await waterLab.shipAltitude(100);await waterLab.shipAdvance(1,1/60,{});});const entry=await page.evaluate(()=>waterLab.shipState());assert.ok(entry.speed<=180.001,'Space velocity must not carry past the atmospheric cap');report.envelope={samples,reentrySpeed:entry.speed};console.log('ENVELOPE',JSON.stringify(report.envelope));
 // Brake to rest, save with UI, then verify Continue after a full reload.
 await page.evaluate(()=>waterLab.shipAdvance(150,1/60,{forward:-1}));await page.locator('#resumeGame').click();await page.keyboard.press('Escape');await idleSave(page);
 const saveText=await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY),saved=JSON.parse(saveText);assert.ok(validSave(saved));
 for(const damaged of [{...saved,version:2},{...saved,navigation:[0]},{...saved,speed:100000},{...saved,ship:saved.ship.map((x,i)=>i===1?-10:x)},{...saved,clock:NaN}])assert.equal(validSave(damaged),false);
 await page.locator('#returnMain').click();await page.waitForFunction(()=>waterLab.sessionState().shell.page==='main');await page.reload();await page.waitForFunction(()=>waterDiagnostics.ready,null,{timeout:240000});assert.ok(await page.locator('#continueGame').isEnabled());assert.equal(await page.locator('#quality').inputValue(),'768');assert.equal(await page.locator('#sensitivity').inputValue(),'1.35');
 const stored=JSON.parse(await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY));await page.locator('#continueGame').click();await page.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');await page.keyboard.press('Escape');await idleSave(page);
 const restored=await page.evaluate(()=>waterLab.shipState()),planet=await page.evaluate(()=>waterLab.planetState()),now=await state(page);
 assert.ok(Math.hypot(...restored.position.map((x,i)=>x-stored.ship[i]))<.03,'Resting craft position must round-trip');
 const maxNavError=Math.max(...planet.navigation.map((x,i)=>Math.abs(x-(stored.navigation[i*2]+stored.navigation[i*2+1]))));assert.ok(maxNavError<1e-7,'High/low navigation must retain its precision');assert.ok(now.clock>=stored.clock&&now.clock-stored.clock<20);assert.ok(Math.abs(now.playTime-stored.playTime)<.5);
 console.log('CONTINUE RESTORED');report.save={bytes:saveText.length,maxNavError,position:restored.position,clockError:now.clock-stored.clock,preferencesRestored:true,corruptSaveRejected:true};
 // A denied write must preserve the previous save and show the failure.
 const safeSave=await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY);
 await page.locator('#resumeGame').click();await page.evaluate(key=>{const original=Storage.prototype.setItem;window.restoreStorage=()=>Storage.prototype.setItem=original;Storage.prototype.setItem=function(k,v){if(k===key)throw new DOMException('Full','QuotaExceededError');return original.call(this,k,v);};},SAVE_KEY);
 await page.keyboard.press('Escape');await idleSave(page);assert.ok((await state(page)).shell.saveError);assert.ok(await page.locator('#menuStatus.warning').isVisible());assert.equal(await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY),safeSave);await page.evaluate(()=>window.restoreStorage());report.storageFailureVisible=true;
 await page.locator('#returnMain').click();await page.waitForFunction(()=>waterLab.sessionState().shell.page==='main');const normalSave=await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY);
 await page.locator('#newGame').click();assert.equal((await state(page)).shell.page,'confirm');await page.locator('#cancelNewGame').click();assert.equal(await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY),normalSave);
 await page.locator('#freeRoam').click();await page.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');assert.equal((await state(page)).flightProfile,0);assert.ok(await page.locator('#shipMoon').isVisible());await page.keyboard.press('Escape');await page.locator('#pauseSettings').click();await page.locator('#settingsWorldTab').click();assert.ok(await page.locator('#worldControls').isVisible());await select(page,'worldSeed',72);await page.keyboard.press('Escape');await page.locator('#returnMain').click();await page.waitForFunction(()=>waterLab.sessionState().shell.page==='main');assert.equal(await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY),normalSave,'Free Roam must never replace normal progress');report.freeRoamIsolated=true;
 await page.locator('#newGame').click();await page.locator('#confirmNewGame').click();await page.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');await page.keyboard.press('Escape');await idleSave(page);assert.ok((await state(page)).playTime<1);assert.equal(JSON.parse(await page.evaluate(key=>localStorage.getItem(key),SAVE_KEY)).seed,61);
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);await page.close();await writeFile('captures/menu/desktop-validation.json',JSON.stringify(report,null,2));console.log('DESKTOP PASSED');
 }
 for(const name of ['Pixel 7','iPhone 13']){
  const context=await browser.newContext({...devices[name],deviceScaleFactor:1});const p=await context.newPage();await ready(p);assert.equal(await p.locator('#quality').inputValue(),'mobile');await screen(p,`${name}-main`);assert.ok((await p.locator('#fullscreen').boundingBox()).height<=50,'Fullscreen must stay a compact button');
  assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await p.locator('#mainSettings').click();await screen(p,`${name}-settings`);await p.locator('#settingsBack').click();await p.locator('#newGame').click();await p.waitForFunction(()=>waterLab.sessionState().shell.page==='playing');
  await p.locator('#toggle').click();await idleSave(p);assert.equal((await state(p)).shell.page,'pause');const before=await state(p);await p.waitForTimeout(300);assert.deepEqual(await state(p),before);await screen(p,`${name}-pause`);
  await p.setViewportSize({width:844,height:390});await p.locator('#pauseSettings').click();await screen(p,`${name}-landscape-settings`);assert.ok((await p.locator('#fullscreen').boundingBox()).height<=50);assert.equal(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
  await p.locator('#settingsBack').click();await p.locator('#resumeGame').click();await p.waitForTimeout(200);assert.equal((await state(p)).playing,true);
  assert.deepEqual(await p.evaluate(()=>waterDiagnostics.errors),[]);await context.close();console.log('TOUCH PASSED',name);
 }
 assert.deepEqual(errors,[]);report.errors=errors;await writeFile('captures/menu/validation.json',JSON.stringify(report,null,2));console.log('Game menu validation passed',JSON.stringify(report));
}finally{await browser.close();}
