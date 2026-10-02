import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const base=process.env.WATER_URL||'http://127.0.0.1:5191/',errors=[],result={};
const distance=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
async function ready(page){await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:180000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);}
try{
 await mkdir('captures',{recursive:true});
 assert.ok(!JSON.parse(await readFile('kernels/ship_render.json','utf8')).wgsl.includes('cw_f64'),'Ship pixel shader must stay float32');
 const page=await browser.newPage({viewport:{width:2560,height:1440}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(new URL('?mode=ship&t=4',base).href);await ready(page);console.log('Ship ready');
 result.geometry=await page.evaluate(()=>waterLab.shipGeometry());assert.ok(result.geometry.finite);assert.ok(result.geometry.triangles>60000);assert.equal(result.geometry.symmetryError,0,'Every paired component must mirror exactly');
 for(const fin of result.geometry.finBounds)assert.ok(fin[5]>2.5,'Both fins must point above the hull');
 result.rays=await page.evaluate(()=>waterLab.shipRayTest());assert.ok(result.rays.finite&&result.rays.hits>8&&result.rays.misses>0);assert.ok(result.rays.maxError<.002,'BVH must match an independent brute-force triangle oracle');console.log(JSON.stringify({geometry:result.geometry,rays:result.rays}));
 const surfaceCamera=await page.evaluate(()=>waterLab.shipState());await page.evaluate(()=>waterLab.shipAltitude(8000000));const orbitalCamera=await page.evaluate(()=>waterLab.shipState());assert.ok(distance(surfaceCamera.data.slice(28,31),orbitalCamera.data.slice(28,31))<.00001,'Orbital altitude must not quantize the local chase camera');await page.evaluate(()=>waterLab.shipAltitude(22));result.orbitCameraPrecision=true;
 await page.evaluate(()=>waterLab.shipView(0,.40,22));await page.screenshot({path:'captures/ship-symmetry-2560.png'});
 await page.evaluate(()=>waterLab.shipView(2.50,.38,16));await page.screenshot({path:'captures/ship-detail-2560.png'});
 await page.evaluate(()=>waterLab.shipView(.55,.68,23));await page.screenshot({path:'captures/ship-top-2560.png'});
 await page.locator('#shipInspect').click();await page.evaluate(()=>waterLab.resume());
 const before=await page.evaluate(()=>waterLab.shipState()),navBefore=await page.evaluate(()=>waterLab.planetState());
 await page.keyboard.down('KeyW');await page.waitForTimeout(900);await page.keyboard.down('KeyD');await page.waitForTimeout(500);await page.keyboard.up('KeyD');await page.keyboard.up('KeyW');await page.evaluate(()=>waterLab.pause());
 const moving=await page.evaluate(()=>waterLab.shipState()),navAfter=await page.evaluate(()=>waterLab.planetState());
 assert.ok(moving.finite&&moving.speed>2);assert.ok(moving.angles[0]>.1&&moving.angles[2]<-.02);assert.ok(distance(navBefore.navigation.slice(0,3),navAfter.navigation.slice(0,3))*6371000>2);
 assert.ok(moving.clearance>=5.9,'Terrain clearance must hold');
 await page.mouse.move(1100,700);await page.mouse.wheel(0,-500);await page.waitForTimeout(100);const wheel=await page.evaluate(()=>waterLab.shipState());assert.ok(wheel.data[15]>before.data[15]*2,'Wheel up increases the thrust limit');
 await page.locator('#shipInspect').click();const beforeOrbit=await page.evaluate(()=>waterLab.shipState());await page.mouse.move(1200,650);await page.mouse.down({button:'right'});await page.mouse.move(1500,800,{steps:12});await page.mouse.up({button:'right'});const orbit=await page.evaluate(()=>waterLab.shipState());assert.equal(orbit.angles[0],beforeOrbit.angles[0]);assert.ok(Math.abs(orbit.data[8]-beforeOrbit.data[8])>.2);
 await page.locator('#shipFly').click();await page.waitForTimeout(100);result.pointerLock=await page.evaluate(()=>waterDiagnostics.pointerLock);assert.ok(['locked','drag'].includes(result.pointerLock));await page.keyboard.press('Escape');
 const ground=await page.evaluate(()=>waterLab.planetState()),weatherBefore=await page.evaluate(()=>waterLab.weatherState());await page.locator('#shipSpace').click();await page.waitForTimeout(200);const space=await page.evaluate(()=>waterLab.shipState()),spacePlanet=await page.evaluate(()=>waterLab.planetState()),weatherAfter=await page.evaluate(()=>waterLab.weatherState());
 assert.ok(space.position[1]>4000000);assert.ok(distance(ground.navigation.slice(0,3),spacePlanet.navigation.slice(0,3))<1e-6);assert.equal(weatherBefore.clock,weatherAfter.clock);assert.ok(distance(weatherBefore.sun,weatherAfter.sun)<1e-6);await page.screenshot({path:'captures/ship-orbit-2560.png'});
 await page.locator('#shipSurface').click();await page.waitForTimeout(200);const returned=await page.evaluate(()=>waterLab.shipState());assert.ok(returned.position[1]<2000&&returned.clearance>=5.9);result.flight={before:before.position,moving:moving.position,speed:moving.speed,bank:moving.angles[2],wheelLimit:wheel.data[15],inspectDoesNotSteer:true,spaceAltitude:space.position[1],returnedAltitude:returned.position[1],solarTimePreserved:true};console.log(JSON.stringify(result.flight));
 await page.evaluate(()=>waterLab.shipView(.34,.3,22));result.resolutions=[];
 for(const [width,height] of [[2560,1440],[3840,2160]]){
  await page.setViewportSize({width,height});await page.locator('#quality').evaluate((e,w)=>{e.value=String(w);e.dispatchEvent(new Event('change'));},width);await page.evaluate(()=>waterLab.seek(4));assert.equal(await page.evaluate(()=>waterDiagnostics.width),width);assert.equal(await page.evaluate(()=>waterDiagnostics.height),height);
  const timing=await page.evaluate(()=>waterLab.shipCost(32));result.resolutions.push({width,height,...timing});console.log(JSON.stringify({width,height,...timing}));await page.screenshot({path:`captures/ship-flight-${width}.png`});
 }
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);await page.close();
 const mobile=await browser.newPage({...devices['Pixel 7']});mobile.on('pageerror',e=>errors.push(String(e)));await mobile.goto(new URL('?mode=ship&t=4',base).href);await ready(mobile);assert.equal(await mobile.evaluate(()=>waterDiagnostics.mobile),true);assert.equal(await mobile.evaluate(()=>waterDiagnostics.pixelSamples),1);await mobile.screenshot({path:'captures/ship-mobile.png'});
 const initial=await mobile.evaluate(()=>waterLab.shipState());await mobile.evaluate(()=>waterLab.resume());const cdp=await mobile.context().newCDPSession(mobile),stick=await mobile.locator('#moveStick').boundingBox();
 await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:stick.x+stick.width/2,y:stick.y+stick.height*.2,id:1}]});await mobile.waitForTimeout(450);await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await mobile.evaluate(()=>waterLab.pause());const moved=await mobile.evaluate(()=>waterLab.shipState());assert.ok(moved.speed>initial.speed+1);
 await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:290,y:360,id:2}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:345,y:340,id:2}]});await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await mobile.evaluate(()=>waterLab.seek(4));const looked=await mobile.evaluate(()=>waterLab.shipState());assert.ok(Math.abs(looked.angles[0]-moved.angles[0])>.05);assert.deepEqual((await mobile.evaluate(()=>waterLab.shipWashState())).pointer,[0,0,0,0],'Flight touch must not inject pointer water forces');
 result.mobile={emulation:'Pixel 7 touch layout on desktop NVIDIA; not physical phone timing',joystick:true,touchSteering:true,stateFinite:looked.finite,...await mobile.evaluate(()=>waterLab.benchmark(24))};console.log(JSON.stringify(result.mobile));assert.deepEqual(await mobile.evaluate(()=>waterDiagnostics.errors),[]);await mobile.close();
 result.errors=errors;assert.deepEqual(errors,[]);await writeFile('captures/ship-validation.json',JSON.stringify(result,null,2));console.log('Ship validation passed');
}finally{await browser.close();}
