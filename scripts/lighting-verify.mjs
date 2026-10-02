import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';

const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const base=process.env.WATER_URL||'http://127.0.0.1:5191/',R=6371000,errors=[],results=[];
const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
const distance=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
const hourDifference=(a,b)=>Math.abs(((a-b+36)%24)-12);
// Independent solar position in Earth-fixed coordinates. Altitude and travel
// speed must not change this direction or the time over a given longitude.
function sunAt(clock,season){
 const days=clock/86400,declination=.3977885*Math.sin((season+days-81)*.01720242);
 const angle=(.5-(days-Math.floor(days)))*2*Math.PI,c=Math.sqrt(1-declination**2);
 return [Math.sin(angle)*c,declination,Math.cos(angle)*c];
}
async function state(page){
 const planet=await page.evaluate(()=>waterLab.planetState()),weather=await page.evaluate(()=>waterLab.weatherState());
 assert.ok(planet.finite&&weather.finite);
 const expected=sunAt(weather.clock,weather.season);
 assert.ok(distance(weather.sun,expected)<2e-6,'World Sun must follow the clock regardless of flight');
 // During live flight, separate diagnostics may straddle animation frames.
 // Compare sunlight and its basis from the same camera-buffer snapshot.
 const projected=[weather.frame.east,weather.frame.normal,weather.frame.back].map(axis=>dot(axis,expected));
 assert.ok(distance(weather.localSun,projected)<2e-6,'Local lighting must use the current transported globe frame');
 const longitude=Math.atan2(weather.frame.normal[0],weather.frame.normal[2]),hour=((weather.clock/3600+longitude*12/Math.PI)%24+24)%24;
 assert.ok(hourDifference(weather.localHour,hour)<1e-4,'Local hour must agree with the camera longitude');
 return {planet,weather};
}
function sameLocation(before,after,label){
 const moved=distance(before.planet.navigation.slice(0,3),after.planet.navigation.slice(0,3))*R;
 assert.ok(moved<.01,`${label} moved the observer ${moved} metres across the globe`);
 assert.ok(distance(before.planet.east,after.planet.east)<1e-7,`${label} rotated the transported frame`);
 assert.equal(before.weather.clock,after.weather.clock);
 assert.ok(hourDifference(before.weather.localHour,after.weather.localHour)<1e-4,`${label} changed local time`);
 assert.ok(distance(before.weather.localSun,after.weather.localSun)<1e-7,`${label} changed local sunlight`);
}
try{
 await mkdir('captures',{recursive:true});
 for(const mobile of [false,true]){
  const profile=mobile?'mobile':'pc',page=await browser.newPage(mobile?{...devices['Pixel 7']}:{viewport:{width:2560,height:1440}});
  page.on('pageerror',error=>errors.push(String(error)));
  await page.goto(new URL('?t=4&geology=study',base).href);
  await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});
  assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
  if(!mobile){await page.locator('#quality').selectOption('2560');await page.evaluate(()=>waterLab.seek(4));}
  await page.evaluate(()=>waterLab.weatherSeek(50400,172));
  let liveFlightSamples=0;
  if(!mobile){
   // Use real mouse lock, speed scrolling and WASD through normal compute
   // frames as well as the deterministic camera-only routes below.
   await page.evaluate(()=>waterLab.weatherLocation(0,0,1000));await page.evaluate(()=>waterLab.lookAt(0,0));
   await page.locator('#fly').click();await page.waitForFunction(()=>document.pointerLockElement===document.querySelector('#water'));
   for(let i=0;i<7;i++)await page.mouse.wheel(0,-1000);
   await page.keyboard.down('KeyD');let reachedNight=false;
   try{for(let i=0;i<30;i++){await page.waitForTimeout(25);const at=await state(page);liveFlightSamples++;assert.equal(at.weather.clock,50400);if(at.weather.localSun[1]<-.6){reachedNight=true;break;}}}
   finally{await page.keyboard.up('KeyD');await page.keyboard.press('Escape');}
   assert.ok(reachedNight,'Normal high-speed WASD flight must cross into night');
  }
  await page.evaluate(()=>waterLab.weatherLocation(0,0,5));await page.evaluate(()=>waterLab.lookAt(0,0));
  const day=await state(page);assert.ok(day.weather.localSun[1]>.7);
  // Cross to the opposite hemisphere quickly, retaining the paused clock.
  await page.evaluate(()=>waterLab.flightTest(256,Math.PI*(6371000+5)/5000000,0,1,0,5000000));
  const night=await state(page);assert.ok(night.weather.localSun[1]<-.7);assert.equal(night.weather.clock,day.weather.clock);
  // Exercise the actual Space control. The old reset path silently teleported
  // to longitude zero, changing 02:00 to 14:00 after this flight.
  await page.locator('#space').evaluate(element=>element.click());await page.evaluate(()=>waterLab.seek(4));
  const orbit=await state(page);
  await writeFile(`captures/lighting-${profile}-space.json`,JSON.stringify({night,orbit},null,2));
  await page.screenshot({path:`captures/lighting-${profile}-night-orbit.png`});
  sameLocation(night,orbit,'Space button');assert.equal(orbit.planet.altitude,R*1.25);
  await page.evaluate(()=>waterLab.setAltitude(5));await page.evaluate(()=>waterLab.lookAt(0,0));
  const descended=await state(page);sameLocation(night,descended,'Descent');
  // Normal wheel zoom must retain the same point as well, including its tail.
  await page.locator('#water').dispatchEvent('wheel',{deltaY:1000});
  await page.waitForTimeout(1300);const zoomed=await state(page);
  sameLocation(night,zoomed,'Wheel zoom');assert.ok(zoomed.planet.altitude>300);
  await page.evaluate(()=>waterLab.setAltitude(5));await page.evaluate(()=>waterLab.lookAt(0,0));
  await page.evaluate(()=>waterLab.flightTest(256,Math.PI*(6371000+5)/5000000,0,1,0,5000000));
  const returned=await state(page);assert.ok(returned.weather.localSun[1]>.7);
  // Large per-step angular changes along an oblique route test parallel transport,
  // not just the equatorial special case used by the failing reproduction.
  await page.evaluate(()=>waterLab.weatherLocation(35,-80,5));await page.evaluate(()=>waterLab.lookAt(.9,0));
  const route=[];
  for(let i=0;i<8;i++){await page.evaluate(()=>waterLab.flightTest(1,.1,1,.4,0,30000000));const at=await state(page);route.push({normal:at.planet.normal,sun:at.weather.localSun,hour:at.weather.localHour});}
  const travelled=await state(page);await page.locator('#space').evaluate(element=>element.click());await page.evaluate(()=>waterLab.seek(4));sameLocation(travelled,await state(page),'Space after oblique flight');
  // Repeat with the actual continent profile and its ground-clearance path.
  await page.locator('#highlands').evaluate(element=>element.click());await page.evaluate(()=>waterLab.seek(4));
  const land=await state(page);await page.locator('#space').evaluate(element=>element.click());await page.evaluate(()=>waterLab.seek(4));const landOrbit=await state(page);sameLocation(land,landOrbit,'Space above continents');
  const dimensions=[];
  if(!mobile)for(const width of [2560,3840]){
   await page.setViewportSize({width,height:width*9/16});await page.locator('#quality').selectOption(String(width));await page.evaluate(()=>waterLab.seek(4));
   const gpu=await page.evaluate(()=>waterLab.benchmark(20));assert.equal(gpu.width,width);assert.equal(gpu.height,width*9/16);dimensions.push(gpu);
   await page.screenshot({path:`captures/lighting-orbit-${width}.png`});
  }
  assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
  results.push({profile,liveFlightSamples,day:day.weather.localHour,night:night.weather.localHour,orbit:orbit.weather.localHour,returned:returned.weather.localHour,route,continentLocationPreserved:true,dimensions});
  await page.close();
 }
 assert.deepEqual(errors,[]);
 const result={environment:'Edge/NVIDIA; mobile is viewport and touch emulation, not physical phone hardware',results,errors};
 await writeFile('captures/lighting-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
