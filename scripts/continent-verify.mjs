import {chromium,devices} from 'playwright';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await mkdir('captures',{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']}),errors=[];
const url=process.env.WATER_URL||'http://127.0.0.1:5191/';
const ready=p=>p.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:240000});
const select=(page,id,value)=>page.locator('#'+id).evaluate((e,v)=>{e.value=String(v);e.dispatchEvent(new Event('change'));},value);
const screenOracle=async page=>{
 const size=await page.evaluate(()=>({width:waterDiagnostics.width,height:waterDiagnostics.height}));
 const points=Array.from({length:512},(_,i)=>[Math.floor(((i%32)+.37)/32*size.width),Math.floor((Math.floor(i/32)+.63)/16*size.height),i%4,0]);
 const camera=await page.evaluate(()=>waterLab.planetState()),horizon=(1+Math.tan(camera.camera[5])/.65)*size.height/2;
 if(horizon>=4&&horizon<size.height-4)for(let j=0;j<8;j++)for(let i=0;i<32;i++)points.push([Math.floor((i+.5)/32*size.width),Math.floor(horizon+j-3.5),i%4,0]);
 const result=await page.evaluate(p=>waterLab.terrainScreenTest(p),points);let mismatches=0,maxDistanceError=0,maxRelativeError=0,maxResidual=0,hits=0;const examples=[];
 for(let i=0;i<points.length;i++){const a=result[i*4],b=result[i*4+1];if((a>0)!==(b>0)){mismatches++;examples.push({point:points[i],a,b});}if(a>0&&b>0){if(Math.abs(a-b)/b>.002)examples.push({point:points[i],a,b,residual:result[i*4+2]});hits++;maxDistanceError=Math.max(maxDistanceError,Math.abs(a-b));maxRelativeError=Math.max(maxRelativeError,Math.abs(a-b)/b);maxResidual=Math.max(maxResidual,Math.abs(result[i*4+2]));}}
 const coarse=await page.evaluate(p=>waterLab.terrainScreenTest(p),points.map(p=>[...p.slice(0,3),24]));let coarseDifferences=0;
 for(let i=0;i<points.length;i++){const a=result[i*4+1],b=coarse[i*4+1];if((a>0)!==(b>0)||(a>0&&Math.abs(a-b)/a>.002))coarseDifferences++;}
 const report={rays:points.length,hits,coarseDifferences,mismatches,maxDistanceError,maxRelativeError,maxResidual,examples:examples.slice(0,8)};console.log(JSON.stringify({screenOracle:report}));
 assert.equal(mismatches,0,'Accelerated tracing must agree with the dense reference hit mask');assert.ok(maxRelativeError<.002,JSON.stringify(report));return report;
};
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}});page.on('pageerror',e=>errors.push(String(e)));const startup=Date.now();await page.goto(url+'?t=4');await ready(page);const startupSeconds=(Date.now()-startup)/1000;console.log(JSON.stringify({startupSeconds}));assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 await page.locator('#toggle').click();await page.screenshot({path:'captures/continents-coast.png'});const coastOracle=await screenOracle(page);const stageCosts=await page.evaluate(()=>waterLab.profileStages());console.log(JSON.stringify({stageCosts}));
 await page.setViewportSize({width:2560,height:1440});await select(page,'quality',2560);await page.evaluate(()=>waterLab.seek(4));const coast2k=await page.evaluate(()=>waterLab.benchmark(30,true));
 await page.setViewportSize({width:3840,height:2160});await select(page,'quality',3840);await page.evaluate(()=>waterLab.seek(4));const coast4k=await page.evaluate(()=>waterLab.benchmark(30,true));
 await page.setViewportSize({width:1440,height:900});await select(page,'quality',1152);await page.evaluate(()=>waterLab.seek(4));
 await page.locator('#toggle').click();await page.locator('#geologyPanel summary').click();await page.locator('#highlands').click();await page.evaluate(()=>waterLab.seek(4));
 const highland=await page.evaluate(()=>waterLab.geologyState()),camera=await page.evaluate(()=>waterLab.planetState());assert.ok(highland.elevation>300);assert.ok(camera.altitude>highland.elevation+899);
 await page.locator('#toggle').click();await page.screenshot({path:'captures/continents-highlands.png'});const highlandOracle=await screenOracle(page);assert.ok(highlandOracle.coarseDifferences>0,'A coarse negative control must expose missed foreground ridges');
 const locations=Array.from({length:64},(_,i)=>{const x=(i%8-3.5)*1100,z=(Math.floor(i/8)-3.5)*1100,n=camera.normal.map((v,k)=>v+(camera.east[k]*x+camera.back[k]*z)/6371000),length=Math.hypot(...n);return [...n.map(v=>v/length),0];});
 const cached=await page.evaluate(p=>waterLab.terrainAt(p),locations);await page.evaluate(()=>waterLab.terrainCache(false));const direct=await page.evaluate(p=>waterLab.terrainAt(p),locations);await page.evaluate(()=>waterLab.terrainCache(true));
 let cacheMax=0,cacheSquared=0;for(let i=0;i<locations.length;i++){const d=Math.abs(cached[i*4]-direct[i*4]);cacheMax=Math.max(cacheMax,d);cacheSquared+=d*d;}const cacheRms=Math.sqrt(cacheSquared/locations.length);assert.ok(cacheRms<2&&cacheMax<12,JSON.stringify({cacheRms,cacheMax}));
 const rays=[];for(let y=1;y<=5;y++)for(let x=-5;x<=5;x++){const length=Math.hypot(x*.15,-y*.16,-1);rays.push([x*.15/length,-y*.16/length,-1/length,0]);}rays.push([0,-1,0,0]);
 const hits=await page.evaluate(p=>waterLab.terrainAt(p,true),rays),worldPoints=[];let count=0,maxResidual=0;
 for(let i=0;i<rays.length;i++){const t=hits[i*4];if(t<=0)continue;count++;maxResidual=Math.max(maxResidual,Math.abs(hits[i*4+1]));const v=[rays[i][0]*t,6371000+camera.altitude+rays[i][1]*t,rays[i][2]*t],length=Math.hypot(...v),n=camera.normal.map((_,k)=>(camera.east[k]*v[0]+camera.normal[k]*v[1]+camera.back[k]*v[2])/length);worldPoints.push([...n,0]);}
 assert.ok(count>rays.length*.8,'Most downward highland rays must hit raised terrain');assert.ok(maxResidual<4,JSON.stringify({count,maxResidual}));
 const heights=await page.evaluate(p=>waterLab.terrainAt(p),worldPoints);let oracleError=0,j=0;
 for(let i=0;i<rays.length;i++){const t=hits[i*4];if(t<=0)continue;const r=rays[i],h=Math.hypot(r[0]*t,6371000+camera.altitude+r[1]*t,r[2]*t)-6371000;oracleError=Math.max(oracleError,Math.abs(h-heights[j*4]));j++;}
 assert.ok(oracleError<4,JSON.stringify({oracleError}));
 // A vertical ray has an independent exact distance from camera to height.
 assert.ok(Math.abs(hits.at(-4)-(camera.altitude-highland.elevation))<.1);
 await page.evaluate(()=>waterLab.setAltitude(1));const grounded=await page.evaluate(()=>waterLab.planetState()),ground=await page.evaluate(()=>waterLab.geologyState());assert.ok(grounded.altitude>=ground.elevation+1.99,'Fly camera must remain above land');
 await page.evaluate(()=>waterLab.lookAt(0,-.8));const centre={x:720,y:520};await page.mouse.move(centre.x,centre.y);await page.mouse.down();await page.mouse.move(centre.x+30,centre.y+20,{steps:5});await page.mouse.up();assert.equal((await page.evaluate(()=>waterLab.inspect())).forceEnergy,0,'Dry land must not inject water pressure');
 await page.evaluate(h=>waterLab.setAltitude(h+900),highland.elevation);await page.evaluate(()=>waterLab.lookAt(0,-.32));
 const terrainGpu=await page.evaluate(()=>waterLab.benchmark(30,true));
 await page.setViewportSize({width:2560,height:1440});await select(page,'quality',2560);await page.evaluate(()=>waterLab.seek(4));const highland2k=await page.evaluate(()=>waterLab.benchmark(30,true));
 await page.setViewportSize({width:3840,height:2160});await select(page,'quality',3840);await page.evaluate(()=>waterLab.seek(4));const highland4k=await page.evaluate(()=>waterLab.benchmark(30,true));
 // Rotation preserves the cache frame; travel rebuilds it in world space.
 await page.evaluate(()=>waterLab.pause());const beforeRotation=await page.evaluate(p=>waterLab.terrainAt(p),locations);
 await page.evaluate(()=>waterLab.lookAt(.45,-.36));assert.deepEqual(await page.evaluate(p=>waterLab.terrainAt(p),locations),beforeRotation);
 await page.evaluate(s=>waterLab.flightTest(1,1,0,1,0,s),5000/Math.max(1,camera.altitude*.06));
 const afterTravel=await page.evaluate(p=>waterLab.terrainAt(p),locations);let recenterMax=0;for(let i=0;i<locations.length;i++)recenterMax=Math.max(recenterMax,Math.abs(afterTravel[i*4]-beforeRotation[i*4]));assert.ok(recenterMax<2,JSON.stringify({recenterMax}));
 await page.locator('#toggle').click();await page.locator('#space').click();await page.locator('#toggle').click();await page.evaluate(()=>waterLab.weatherSeek(43200));
 const orbit=[];for(const size of [2560,3840]){await page.setViewportSize({width:size,height:size*9/16});await select(page,'quality',size);await page.evaluate(()=>waterLab.seek(4));const gpu=await page.evaluate(()=>waterLab.benchmark(40,true));assert.equal(gpu.width,size);orbit.push(gpu);await page.screenshot({path:`captures/continents-globe-${size}.png`});}
 await page.evaluate(()=>waterLab.weatherSeek(0));await page.screenshot({path:'captures/continents-night.png'});
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);await page.close();const phone=await browser.newPage({...devices['Pixel 7']});phone.on('pageerror',e=>errors.push(String(e)));await phone.goto(url+'?t=4');await ready(phone);await phone.locator('#toggle').click();await phone.locator('#geologyPanel summary').click();await phone.locator('#highlands').click();await phone.evaluate(()=>waterLab.seek(4));await phone.locator('#toggle').click();await phone.screenshot({path:'captures/continents-mobile.png'});const mobile=await phone.evaluate(()=>waterLab.benchmark(30,true));assert.ok((await phone.evaluate(()=>waterLab.geologyState())).elevation>300);assert.deepEqual(await phone.evaluate(()=>waterDiagnostics.errors),[]);await phone.close();
 assert.deepEqual(errors,[]);
 const shader=JSON.parse(await readFile('kernels/render_pc.json','utf8')).wgsl;assert.ok(!shader.includes('f64'),'Terrain rendering must remain float32');
 const result={startupSeconds,stageCosts,coastOracle,highlandOracle,coast2k,coast4k,highland2k,highland4k,cacheRms,cacheMax,recenterMax,rotationStable:true,highland,count,maxResidual,oracleError,cameraClearance:true,dryLandPressureRejected:true,terrainGpu,orbit,mobile,errors,environment:'Desktop NVIDIA/Edge; mobile viewport/input emulation, not a Samsung or Safari GPU benchmark'};await writeFile('captures/continents-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
