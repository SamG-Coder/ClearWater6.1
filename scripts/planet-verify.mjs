import {chromium,devices} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
await mkdir('captures',{recursive:true});const R=6371000,errors=[];
const base=process.env.WATER_URL||'http://127.0.0.1:5191/';
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(base+'?t=4');await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 const points=[];for(const h of [.45,2.6,100,10000,400000,R,500000000])for(const y of [-1,-.85,-.5,-.1,-.01,.1])points.push([Math.fround(h),Math.fround(Math.sqrt(1-y*y)),Math.fround(y),0]);
 const roots=await page.evaluate(p=>waterLab.geometryTest(p),points);let maxRelativeError=0;
 for(let i=0;i<points.length;i++){const [h,x,y,z]=points[i],dy=y/Math.hypot(x,y,z),b=(R+h)*dy,c=h*(2*R+h),disc=b*b-c,expected=dy<0&&disc>=0?c/(-b+Math.sqrt(disc)):-1,actual=roots[i*4];assert.equal(roots[i*4+3],R);if(expected<0)assert.equal(actual,-1);else{const error=Math.abs(actual-expected)/expected;maxRelativeError=Math.max(maxRelativeError,error);assert.ok(error<.00005,JSON.stringify({h,dy,actual,expected}));}}
 await page.evaluate(()=>waterLab.lookAt(0,0));
 const start=await page.evaluate(()=>waterLab.planetState());assert.equal(start.radius,R);
 const surfaceRadius=R+start.altitude,seconds=2*Math.PI*surfaceRadius/5000000;
 await page.evaluate(s=>waterLab.flightTest(512,s,0,1,0,5000000),seconds);
 const circumnavigation=await page.evaluate(()=>waterLab.planetState());
 const returnError=Math.hypot(...circumnavigation.navigation.slice(0,3).map((v,i)=>v-start.navigation[i]))*R;
 assert.ok(returnError<20,JSON.stringify({returnError,circumnavigation}));assert.equal(circumnavigation.altitude,start.altitude);
 // Small steps at the opposite side remain metre-accurate after a long flight.
 await page.evaluate(s=>waterLab.flightTest(256,s,0,1,0,5000000),seconds/2);
 const far=await page.evaluate(()=>waterLab.planetState());
 await page.evaluate(()=>waterLab.flightTest(60,1,0,1,0,3));
 const fine=await page.evaluate(()=>waterLab.planetState());
 const fineDistance=Math.hypot(...fine.navigation.slice(0,3).map((v,i)=>v-far.navigation[i]))*surfaceRadius;
 assert.ok(Math.abs(fineDistance-3)<.002,JSON.stringify({fineDistance}));console.log(JSON.stringify({returnError,fineDistance}));
 await page.locator('#reset').click();await page.waitForTimeout(80);await page.evaluate(()=>waterLab.lookAt(0,0));
 await page.evaluate(s=>waterLab.flightTest(128,s,1,0,0,5000000),seconds/4);
 const pole=await page.evaluate(()=>waterLab.planetState());assert.ok(pole.finite&&pole.normal[1]>.999999);assert.ok(Math.abs(pole.east.reduce((a,v,i)=>a+v*pole.normal[i],0))<1e-5);
 await page.evaluate(s=>waterLab.flightTest(128,s,1,0,0,5000000),seconds/4);
 const beyondPole=await page.evaluate(()=>waterLab.planetState());assert.ok(beyondPole.finite&&beyondPole.normal[2]<-.999999);
 await page.locator('#reset').click();await page.waitForTimeout(80);await page.locator('#toggle').click();
 // Actual wheel input performs a continuous dolly from metres to orbital height.
 await page.mouse.move(900,600);await page.keyboard.down('Control');for(let i=0;i<6;i++){await page.mouse.wheel(0,800);await page.waitForTimeout(250);}await page.keyboard.up('Control');await page.waitForTimeout(1400);
 const zoomed=await page.evaluate(()=>waterLab.planetState());assert.ok(zoomed.altitude>1000000&&zoomed.speed>100000);
 await page.screenshot({path:'captures/planet-wheel-to-space.png'});
 await page.locator('#toggle').click();await page.locator('#fly').click();await page.waitForFunction(()=>document.pointerLockElement===document.querySelector('canvas'));
 const beforeSpeed=await page.evaluate(()=>waterLab.planetState());await page.mouse.wheel(0,-500);await page.waitForTimeout(100);const afterSpeed=await page.evaluate(()=>waterLab.planetState());assert.ok(afterSpeed.speed>beforeSpeed.speed*2);assert.ok(Math.abs(afterSpeed.altitude-beforeSpeed.altitude)<1000);
 await page.keyboard.press('Escape');await page.waitForFunction(()=>document.pointerLockElement===null,null,{timeout:5000});await page.locator('#space').click();await page.waitForTimeout(150);await page.locator('#toggle').click();
 const resolutions=[];
 for(const [quality,width,height] of [['2560',2560,1440],['3840',3840,2160]]){await page.setViewportSize({width,height});await page.locator('#quality').evaluate((e,v)=>{e.value=v;e.dispatchEvent(new Event('change'));},quality);await page.evaluate(()=>waterLab.seek(4));const gpu=await page.evaluate(()=>waterLab.benchmark(80));assert.equal(gpu.width,width);assert.equal(gpu.height,height);resolutions.push(gpu);await page.screenshot({path:`captures/planet-space-${width}.png`});}
 // Follow the same renderer back down through the limb, aircraft height and waves.
 await page.setViewportSize({width:1440,height:900});await page.locator('#quality').evaluate(e=>{e.value='1152';e.dispatchEvent(new Event('change'));});
 for(const altitude of [400000,10000,300,60,2.6]){await page.evaluate(h=>waterLab.setAltitude(h),altitude);await page.evaluate(()=>waterLab.lookAt(0,-.32));const s=await page.evaluate(()=>waterLab.planetState());assert.ok(s.finite);await page.screenshot({path:`captures/planet-altitude-${altitude}.png`});}
 await page.close();
 const phone=await browser.newPage({...devices['Pixel 7']});phone.on('pageerror',e=>errors.push(String(e)));await phone.goto(base+'?t=4');await phone.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});
 const cdp=await phone.context().newCDPSession(phone);await cdp.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x:60,y:400,id:1},{x:350,y:400,id:2}]});
 for(let i=0;i<10;i++){await cdp.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x:60+i*10,y:400,id:1},{x:350-i*10,y:400,id:2}]});await phone.waitForTimeout(30);}await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});await phone.waitForTimeout(1000);
 const pinch=await phone.evaluate(()=>waterLab.planetState());assert.ok(pinch.altitude>100);const state=await phone.evaluate(()=>waterLab.inspect());assert.equal(state.forceEnergy,0,'Two-finger zoom must not disturb the water');
 await phone.evaluate(()=>{document.querySelector('#space').click();});await phone.waitForTimeout(150);await phone.screenshot({path:'captures/planet-mobile-space.png'});assert.deepEqual(await phone.evaluate(()=>waterDiagnostics.errors),[]);await phone.close();
 // Expensive software double operations belong only to the one-invocation camera.
 const shader=JSON.parse(await readFile('kernels/render_pc.json','utf8')).wgsl;assert.ok(!shader.includes('cw_d_add'),'Pixel shaders must remain float32');assert.deepEqual(errors,[]);
 const result={earthRadiusMeters:R,maxRelativeGeometryError:maxRelativeError,circumnavigationReturnErrorMeters:returnError,farSideOneSecondTravelMeters:fineDistance,poleCrossing:true,zoomed,scrollSpeedIncreased:afterSpeed.speed/beforeSpeed.speed,pinchAltitude:pinch.altitude,resolutions,errors,environment:'NVIDIA/Edge; phone viewport/touch emulation, not physical Samsung or Safari'};await writeFile('captures/planet-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
