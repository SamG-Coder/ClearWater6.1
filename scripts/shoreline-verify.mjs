import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const base=process.env.WATER_URL||'http://127.0.0.1:5191/',baseline=process.env.SHORE_BASELINE==='1',tag=baseline?'before':'after',report={poses:[],timings:[]},errors=[];
const length=a=>Math.hypot(...a),sub=(a,b)=>a.map((x,i)=>x-b[i]);
try{
 await mkdir('captures',{recursive:true});
 const page=await browser.newPage({viewport:{width:1440,height:900}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(base+'?mode=ship&t=4');await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:180000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 await page.locator('#quality').evaluate(e=>{e.value='1152';e.dispatchEvent(new Event('change'));});await page.evaluate(()=>waterLab.seek(4));
 for(const [yaw,elevation] of [[0,.35],[.8,.35],[1.6,.35],[2.4,.35],[-2.8,.60],[-1.2,.15]]){
  await page.evaluate(({yaw,elevation})=>waterLab.shipView(yaw,elevation,26),{yaw,elevation});
  const s=await page.evaluate(async()=>({ship:await waterLab.shipState(),planet:await waterLab.planetState(),weather:await waterLab.weatherState(),width:waterDiagnostics.width,height:waterDiagnostics.height}));
  const a=s.ship.data,f=s.weather.frame,world=v=>f.east.map((x,i)=>x*v[0]+f.normal[i]*v[1]+f.back[i]*v[2]);
  const right=world(a.slice(16,19)),up=world(a.slice(20,23)),back=world(a.slice(24,27));
  const offset=right.map((x,i)=>x*a[28]+up[i]*a[29]+back[i]*a[30]);
  const expected=s.planet.navigation.slice(0,3).map((x,i)=>x*(6371000+s.ship.position[1])+offset[i]),actual=f.normal.map(x=>x*(6371000+s.planet.altitude));
  const anchorError=length(sub(expected,actual));
  const points=Array.from({length:256},(_,i)=>[Math.floor((i%32+.4)/32*s.width),Math.floor((Math.floor(i/32)+.6)/8*s.height),i%4,0]);
  const hits=await page.evaluate(p=>waterLab.terrainScreenTest(p),points);let mismatches=0,maxRelativeError=0,land=0;
  for(let i=0;i<points.length;i++){const x=hits[i*4],y=hits[i*4+1];if((x>0)!==(y>0))mismatches++;if(x>0&&y>0){land++;maxRelativeError=Math.max(maxRelativeError,Math.abs(x-y)/y);}}
  report.poses.push({yaw,elevation,anchorError,mismatches,maxRelativeError,land});console.log(report.poses.at(-1));
  if(!baseline){assert.ok(anchorError<1.2,'The terrain eye must follow the chase eye, not the ship centre');assert.equal(mismatches,0);assert.ok(maxRelativeError<.003);}
  await page.screenshot({path:`captures/shore-${tag}-${report.poses.length}.png`});
 }
 // Keep a matched wet-shore performance view before visiting the dry beach.
 await page.evaluate(()=>waterLab.shipView(.6,.45,26));report.wetTimings=[];
 for(const [width,height] of [[2560,1440],[3840,2160]]){await page.setViewportSize({width,height});await page.locator('#quality').evaluate((e,w)=>{e.value=String(w);e.dispatchEvent(new Event('change'));},width);await page.evaluate(()=>waterLab.seek(4));report.wetTimings.push(await page.evaluate(()=>waterLab.benchmark(30,true)));}
 await page.setViewportSize({width:1440,height:900});await page.locator('#quality').evaluate(e=>{e.value='1152';e.dispatchEvent(new Event('change'));});await page.evaluate(()=>waterLab.seek(4));
 // Move onto the generated beach so the rotation test covers foreground land,
 // not only distant land pixels at the wet spawn.
 const origin=await page.evaluate(()=>waterLab.planetState());
 const candidates=Array.from({length:625},(_,i)=>{const x=(i%25-12)*200,z=(Math.floor(i/25)-12)*200,n=origin.normal.map((v,k)=>v+(origin.east[k]*x+origin.back[k]*z)/6371000),len=Math.hypot(...n);return [...n.map(v=>v/len),0];});
 const candidateHeights=await page.evaluate(p=>waterLab.terrainAt(p),candidates);let best=-1,score=Infinity;
 for(let i=0;i<candidates.length;i++){const h=candidateHeights[i*4];if(h>4&&Math.abs(h-6)<score){score=Math.abs(h-6);best=i;}}
 assert.ok(best>=0,'The generated shore must provide a nearby dry beach');const point=candidates[best];
 await page.evaluate(p=>waterLab.weatherLocation(Math.asin(p[1])*180/Math.PI,Math.atan2(p[0],p[2])*180/Math.PI,24),point);
 await page.evaluate(()=>waterLab.shipAltitude(22));
 const patch=await page.evaluate(()=>waterLab.nearTerrainState()),frame=await page.evaluate(()=>waterLab.planetState());
 const fixedPoints=Array.from({length:81},(_,i)=>{const x=(i%9-4)*31,z=(Math.floor(i/9)-4)*31,n=frame.normal.map((v,k)=>v+(frame.east[k]*x+frame.back[k]*z)/6371000),len=Math.hypot(...n);return [...n.map(v=>v/len),0];});
 const fixedBefore=await page.evaluate(p=>waterLab.terrainAt(p),fixedPoints);report.dryPoses=[];
 for(const yaw of [0,.8,1.6,2.4,-2.8,-1.2]){
  await page.evaluate(y=>waterLab.shipView(y,.55,26),yaw);
  const values=await page.evaluate(p=>waterLab.terrainAt(p),fixedPoints),state=await page.evaluate(()=>waterLab.nearTerrainState());
  assert.deepEqual(values,fixedBefore,'Fixed terrain points must not change as the chase eye rotates');assert.equal(state.header[14],patch.header[14],'Rotation must not rebuild the terrain patch');
  const size=await page.evaluate(()=>({w:waterDiagnostics.width,h:waterDiagnostics.height})),points=Array.from({length:512},(_,i)=>[Math.floor((i%32+.4)/32*size.w),Math.floor((Math.floor(i/32)+.6)/16*size.h),i%4,0]);
  const hits=await page.evaluate(p=>waterLab.terrainScreenTest(p),points);let land=0,mismatches=0,maxRelativeError=0,maxError=0;
  for(let i=0;i<points.length;i++){const a=hits[i*4],b=hits[i*4+1];if((a>0)!==(b>0))mismatches++;if(a>0&&b>0){land++;maxRelativeError=Math.max(maxRelativeError,Math.abs(a-b)/b);maxError=Math.max(maxError,Math.abs(a-b));}}
  const result={yaw,land,mismatches,maxRelativeError,maxError};report.dryPoses.push(result);console.log('Dry beach',result);assert.ok(land>100);assert.equal(mismatches,0);assert.ok(maxRelativeError<.003||maxError<.15);
  await page.screenshot({path:`captures/shore-dry-${tag}-${report.dryPoses.length}.png`});
 }
 report.nearTerrain={...patch,rotationStable:true,points:fixedPoints.length};
 const gradients=await page.evaluate(p=>waterLab.terrainAt(p,2),fixedPoints);let heightError=0;
 for(let i=0;i<fixedPoints.length;i++)heightError=Math.max(heightError,Math.abs(gradients[i*4+3]-fixedBefore[i*4]));assert.ok(heightError<.001,'Shading and ray intersections must share one terrain height');report.nearTerrain.gradientHeightError=heightError;
 await page.evaluate(()=>waterLab.shipView(.6,.45,26));
 for(const [width,height] of [[2560,1440],[3840,2160]]){await page.setViewportSize({width,height});await page.locator('#quality').evaluate((e,w)=>{e.value=String(w);e.dispatchEvent(new Event('change'));},width);await page.evaluate(()=>waterLab.seek(4));assert.deepEqual(await page.evaluate(()=>[waterDiagnostics.width,waterDiagnostics.height,waterDiagnostics.pixelSamples]),[width,height,4]);report.timings.push(await page.evaluate(()=>waterLab.benchmark(30,true)));await page.screenshot({path:`captures/shore-${tag}-${width}.png`});}
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);assert.deepEqual(errors,[]);report.errors=errors;
 await writeFile(`captures/shoreline-${tag}.json`,JSON.stringify(report,null,2));console.log('Shoreline rotation audit complete');
}finally{await browser.close();}
