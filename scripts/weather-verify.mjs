import {chromium,devices} from 'playwright';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await mkdir('captures',{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const url=process.env.WATER_URL||'http://127.0.0.1:5191/',errors=[];
const ready=page=>page.waitForFunction(()=>window.waterDiagnostics?.ready||window.waterDiagnostics?.errors.length,null,{timeout:120000});
const point=(lat,lon,time=50400)=>{lat*=Math.PI/180;lon*=Math.PI/180;return [Math.sin(lon)*Math.cos(lat),Math.sin(lat),Math.cos(lon)*Math.cos(lat),time];};
const change=(a,b)=>a.reduce((sum,v,i)=>sum+Math.abs(v-b[i]),0)/a.length;
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(url+'?t=4&geology=study');await ready(page);assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 const initial=await page.evaluate(()=>waterLab.weatherState());assert.ok(initial.finite&&initial.enabled);assert.ok(Math.abs(initial.localHour-14)<.001);
 await page.locator('#toggle').click();await page.screenshot({path:'captures/weather-surface.png'});
 // Independently check the solar period and tilt at each season.
 const seasons=[];
 for(const season of [81,172,265,355]){
  await page.evaluate(s=>waterLab.weatherSeek(50400,s),season);
  const out=await page.evaluate(points=>waterLab.weatherAt(points),[point(0,0,0),point(0,0,43200),point(0,0,86400)]);
  const midnight=out.slice(8,11),noon=out.slice(24,27),next=out.slice(40,43);
  assert.ok(Math.hypot(...midnight.map((v,i)=>v-next[i]))<.008,'Sun must return after one mean solar day');
  assert.ok(Math.abs(midnight[2]+noon[2])<.008&&noon[2]>.9&&midnight[2]<-.9);
  const declination=Math.asin(noon[1])*180/Math.PI;assert.ok(Math.abs(declination)<=23.45);
  if(season===172)assert.ok(declination>23);if(season===355)assert.ok(declination<-23);
  seasons.push({season,declination});
 }
 await page.evaluate(()=>waterLab.weatherSeek(50400));
 // Longitude wrap and both poles must remain finite and continuous.
 const seams=[];for(const lat of [-89.999,-45,0,45,89.999]){
  const p=[point(lat,-179.99999),point(lat,179.99999)];
  const out=await page.evaluate(points=>waterLab.weatherAt(points),p);assert.ok(out.every(Number.isFinite));
  const delta=Math.max(...out.slice(0,8).map((v,i)=>Math.abs(v-out[16+i])));assert.ok(delta<.001,{lat,delta});seams.push(delta);
 }
 // Statistical wind belts are checked over all longitudes; storm eddies may
 // locally reverse them. Velocity must also remain tangent to the sphere.
 const circulation=[];for(const latitude of [-75,-45,-15,15,45,75]){
  const points=Array.from({length:48},(_,i)=>point(latitude,i*7.5-180));
  const out=await page.evaluate(p=>waterLab.weatherAt(p),points);let zonal=0,maxRadial=0;
  points.forEach((p,i)=>{const v=out.slice(i*16+4,i*16+7),lon=(i*7.5-180)*Math.PI/180;zonal+=(v[0]*Math.cos(lon)-v[2]*Math.sin(lon))/48;maxRadial=Math.max(maxRadial,Math.abs(v.reduce((sum,a,j)=>sum+a*p[j],0)));});
  assert.ok(maxRadial<.0001);assert.ok(Math.abs(latitude)===45?zonal>4:zonal<0,{latitude,zonal});circulation.push({latitude,zonal,maxRadial});
 }
 const evolving=await page.evaluate(p=>waterLab.weatherAt(p),[point(35,50,50400),point(35,50,50400+86400*3)]);
 assert.ok(change(evolving.slice(0,8),evolving.slice(16,24))>.02,'Weather must evolve in Earth-fixed coordinates');
 // Real controls visit the strongest daylight rain system in the shared map.
 await page.locator('#toggle').click();await page.locator('#weatherPanel summary').click();await page.locator('#findRain').click();await page.waitForTimeout(100);
 const rain=await page.evaluate(()=>waterLab.weatherState());assert.ok(rain.rain>.15&&rain.cloud>.6,JSON.stringify(rain));
 await page.locator('#toggle').click();await page.screenshot({path:'captures/weather-rain-arrival.png'});
 const before=await page.evaluate(()=>waterLab.inspect());await page.evaluate(()=>waterLab.weatherAdvance(1200,120));
 const after=await page.evaluate(()=>waterLab.inspect()),memory=await page.evaluate(()=>waterLab.weatherState());
 assert.ok(after.finite&&memory.finite&&after.imaginaryResidual<.00001);
 assert.ok(Math.abs(memory.energy[0]-1)>.05&&Math.abs(memory.energy[1]-1)>.05,'Wind must change retained FFT energy');
 assert.ok(Math.abs(after.heightRms-before.heightRms)>.001,'The resolved water must respond to weather');
 const studyEnergy=memory.energy;
 await page.locator('#weatherMode').evaluate(e=>{e.value='0';e.dispatchEvent(new Event('change'));});await page.evaluate(()=>waterLab.seek(4));
 const off=await page.evaluate(()=>waterLab.weatherState());assert.deepEqual(off.energy,[1,1],'Study mode is the wind-coupling negative control');
 await page.locator('#weatherMode').evaluate(e=>{e.value='1';e.dispatchEvent(new Event('change'));});await page.evaluate(()=>waterLab.weatherAdvance(600,60));
 await page.screenshot({path:'captures/weather-rain.png'});
 const paused=await page.evaluate(()=>waterLab.weatherState()),imageA=await page.evaluate(()=>waterLab.screenshot());await page.waitForTimeout(250);
 const pausedAgain=await page.evaluate(()=>waterLab.weatherState()),imageB=await page.evaluate(()=>waterLab.screenshot());
 assert.equal(paused.clock,pausedAgain.clock);assert.deepEqual(imageA.rgba,imageB.rgba,'Paused water, rain and sky must be stable');
 const rainyGpu=await page.evaluate(()=>waterLab.benchmark(50,true));await page.evaluate(()=>waterLab.pause());
 await page.setViewportSize({width:3840,height:2160});await page.locator('#quality').evaluate(e=>{e.value='3840';e.dispatchEvent(new Event('change'));});await page.evaluate(()=>waterLab.seek(4));
 const rainy4k=await page.evaluate(()=>waterLab.benchmark(50,true));await page.evaluate(()=>waterLab.pause());await page.evaluate(()=>waterLab.planetState());await page.screenshot({path:'captures/weather-rain-4k.png'});
 await page.locator('#toggle').click();await page.locator('#space').click();await page.waitForTimeout(100);await page.locator('#toggle').click();
 const orbit=[];for(const [width,height] of [[2560,1440],[3840,2160]]){
  await page.setViewportSize({width,height});await page.locator('#quality').evaluate((e,w)=>{e.value=String(w);e.dispatchEvent(new Event('change'));},width);
  await page.evaluate(()=>waterLab.seek(4));const gpu=await page.evaluate(()=>waterLab.benchmark(50,true));assert.equal(gpu.width,width);assert.equal(gpu.height,height);orbit.push(gpu);
  await page.evaluate(()=>waterLab.pause());await page.evaluate(()=>waterLab.planetState());await page.screenshot({path:`captures/weather-globe-${width}.png`});
 }
 await page.setViewportSize({width:1440,height:900});await page.locator('#quality').evaluate(e=>{e.value='1152';e.dispatchEvent(new Event('change'));});
 for(const altitude of [30000,2000,300]){await page.evaluate(h=>waterLab.setAltitude(h),altitude);await page.evaluate(()=>waterLab.lookAt(0,-.16));await page.screenshot({path:`captures/weather-flight-${altitude}.png`});}
 await page.locator('#space').evaluate(e=>e.click());await page.waitForTimeout(60);
 for(const [name,clock] of [['noon',43200],['sunset',64800],['night',0]]){
  await page.evaluate(t=>waterLab.weatherSeek(t),clock);await page.screenshot({path:`captures/weather-globe-${name}.png`});
 }
 await page.locator('#toggle').click();await page.locator('#shallows').click();await page.waitForTimeout(100);await page.evaluate(()=>waterLab.weatherSeek(50400));
 await page.locator('#fly').click();await page.waitForFunction(()=>document.pointerLockElement===document.querySelector('canvas')&&waterDiagnostics.pointerLock==='locked');
 assert.match(await page.locator('#fly').textContent(),/Flying/);const flight=await page.evaluate(()=>waterLab.planetState());await page.mouse.wheel(0,-400);await page.waitForTimeout(80);const faster=await page.evaluate(()=>waterLab.planetState());assert.ok(faster.speed>flight.speed*2);await page.keyboard.press('Escape');
 await page.close();
 const phone=await browser.newPage({...devices['Pixel 7']});phone.on('pageerror',e=>errors.push(String(e)));await phone.goto(url+'?t=4&geology=study');await ready(phone);
 const phoneWeather=await phone.evaluate(()=>waterLab.weatherState());assert.ok(phoneWeather.finite&&phoneWeather.enabled);
 const mobileGpu=await phone.evaluate(()=>waterLab.benchmark(50,true));await phone.evaluate(()=>waterLab.pause());await phone.screenshot({path:'captures/weather-phone.png'});await phone.close();
 // A denied browser lock must not cause an unhandled error or stop the render.
 const denied=await browser.newPage({viewport:{width:1440,height:900}});denied.on('pageerror',e=>errors.push(String(e)));
 await denied.addInitScript(()=>{HTMLCanvasElement.prototype.requestPointerLock=()=>Promise.reject(new DOMException('Test: lock denied','NotAllowedError'));});
 await denied.goto(url+'?t=4&geology=study');await ready(denied);await denied.locator('#fly').click();await denied.waitForFunction(()=>waterDiagnostics.pointerLock==='drag');assert.match(await denied.locator('#flightStatus').textContent(),/blocked/);
 const pose=await denied.evaluate(()=>waterLab.inspect());await denied.mouse.move(850,400);await denied.mouse.down();await denied.mouse.move(990,460,{steps:8});await denied.mouse.up();await denied.waitForTimeout(100);const looked=await denied.evaluate(()=>waterLab.inspect());assert.notEqual(looked.camera[4],pose.camera[4]);assert.equal(looked.forceEnergy,0);await denied.close();
 const shader=JSON.parse(await readFile('kernels/render_pc.json','utf8')).wgsl;assert.ok(!shader.includes('cw_d_add'));assert.deepEqual(errors,[]);
 const result={environment:'Desktop NVIDIA/Edge with real 1440p/4K framebuffers; Pixel 7 is touch/viewport emulation, not physical phone testing',initial,seasons,seams,circulation,rain,windEnergy:studyEnergy,paused:true,rainyGpu,rainy4k,orbit,mobileGpu,deniedMouseLockFallback:true,errors};await writeFile('captures/weather-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
