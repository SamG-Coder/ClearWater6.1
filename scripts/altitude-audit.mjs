import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
const tag=process.env.AUDIT_TAG||'after',folder=`captures/altitude-${tag}`;
const base=process.env.WATER_URL||'http://127.0.0.1:5191/';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const errors=[],results=[],flightTimings=[];
async function ready(page){await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:180000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);}
async function capture(page,name){
 const state=await page.evaluate(()=>waterLab.shipState()),weather=await page.evaluate(()=>waterLab.weatherState());assert.ok(state.finite&&weather.finite);
 await page.evaluate(()=>waterLab.planetState());
 await page.screenshot({path:`${folder}/${name}.png`});results.push({name,altitude:state.position[1],weather:{cloud:weather.cloud,clock:weather.clock},angles:state.angles});console.log(name,state.position[1]);
}
try{
 await mkdir(folder,{recursive:true});
 const page=await browser.newPage({viewport:{width:2560,height:1440}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(base+'?mode=ship&t=4');await ready(page);
 await page.evaluate(()=>waterLab.shipAltitude(4459700));await page.evaluate(()=>waterLab.shipView(0,1.05,22));await capture(page,'orbit');
 await page.evaluate(()=>waterLab.shipAltitude(300));await page.evaluate(()=>waterLab.shipView(.34,.65,22));await capture(page,'coastal-300');
 await page.evaluate(()=>waterLab.shipAltitude(22));await capture(page,'coastal-22');
 await page.goto(base+'?mode=ship&t=4&geology=study');await ready(page);
 await page.locator('#toggle').click();await page.locator('#weatherPanel summary').click();await page.locator('#findRain').click();await page.evaluate(()=>waterLab.pause());await page.locator('#toggle').click();
 await page.evaluate(()=>waterLab.shipAltitude(12000));await page.evaluate(()=>waterLab.shipView(.34,.60,22));await capture(page,'cloud-12000');
 // Continuous descent through the real flight kernel, not altitude teleports.
 let frames=0;
 for(const target of [8000,5000,2000,800,350,120,40,6]){
  while((await page.evaluate(()=>waterLab.shipState())).position[1]>target+.01){await page.evaluate(()=>waterLab.shipAdvance(4,.1,{rise:-1}));frames+=4;assert.ok(frames<800,'Descent must reach the surface');}
  await page.evaluate(()=>waterLab.shipView(.34,.60,22));await capture(page,`descent-${target}`);
 }
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 // Inspect broken clouds as well as the deliberately overcast rain-system
 // stress case. Select a real daylight weather cell, never override its cover.
 const candidates=[];for(let lat=-60;lat<=60;lat+=10)for(let lon=-180;lon<180;lon+=15){const a=lat*Math.PI/180,b=lon*Math.PI/180;candidates.push({lat,lon,p:[Math.sin(b)*Math.cos(a),Math.sin(a),Math.cos(b)*Math.cos(a),50400]});}
 await page.evaluate(()=>waterLab.weatherSeek(50400));
 const field=await page.evaluate(p=>waterLab.patternSamples(p),candidates.map(c=>[c.lat,c.lon,8,0]));let best=0,score=-1e9;
 for(let i=0;i<candidates.length;i++){const s=-Math.abs(field[i*4]-.25)*8-Math.abs(field[i*4+1]-.65)+field[i*4+2];if(s>score){score=s;best=i;}}
 await page.evaluate(c=>waterLab.weatherLocation(c.lat,c.lon,12000),candidates[best]);await page.evaluate(()=>waterLab.shipAltitude(12000));await page.evaluate(()=>waterLab.shipView(.34,.35,22));await capture(page,'broken-clouds-12000');
 const cloudPoints=candidates.map(c=>[c.lat,c.lon,8,0]),fromAbove=await page.evaluate(p=>waterLab.patternSamples(p),cloudPoints);
 await page.evaluate(()=>waterLab.shipAltitude(3500));await page.evaluate(()=>waterLab.shipView(.34,.15,22));await capture(page,'broken-clouds-3500');
 assert.deepEqual(await page.evaluate(p=>waterLab.patternSamples(p),cloudPoints),fromAbove,'The same cloud field must not morph with observer altitude');
 if(process.env.AUDIT_TIMINGS==='1'){
  for(const [width,height] of [[2560,1440],[3840,2160]]){
   await page.setViewportSize({width,height});await page.locator('#quality').evaluate((e,w)=>{e.value=String(w);e.dispatchEvent(new Event('change'));},width);await page.evaluate(()=>waterLab.seek(4));
   assert.deepEqual(await page.evaluate(()=>[waterDiagnostics.width,waterDiagnostics.height,waterDiagnostics.pixelSamples]),[width,height,4]);
   flightTimings.push(await page.evaluate(()=>waterLab.benchmark(30,true)));
   await page.screenshot({path:`${folder}/cloud-flight-${width}.png`});
  }
  await page.setViewportSize({width:2560,height:1440});
 }
 await page.goto(base+'?mode=explorer&t=4&geology=study');await ready(page);await page.evaluate(()=>waterLab.lookDown());
 await page.locator('#quality').selectOption('2560');await page.evaluate(()=>waterLab.seek(4));await page.locator('#toggle').click();
 assert.deepEqual(await page.evaluate(()=>[waterDiagnostics.width,waterDiagnostics.height]),[2560,1440]);
 await page.screenshot({path:`${folder}/sand.png`});
 const bed=await page.evaluate(()=>waterLab.bedTest());assert.ok(bed.finite&&bed.sandSlopeError<.01&&bed.stoneSlopeError<.02);
 assert.deepEqual(errors,[]);await writeFile(`${folder}/report.json`,JSON.stringify({frames,results,bed,flightTimings,errors},null,2));
}finally{await browser.close();}
