import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const base=process.env.WATER_URL||'http://127.0.0.1:5191/',errors=[],result={};
const delta=(a,b)=>a.map((v,i)=>v-b[i]);
const length=a=>Math.hypot(...a);
try{
 await mkdir('captures',{recursive:true});
 const page=await browser.newPage({viewport:{width:1280,height:800}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(new URL('?mode=ship&t=4&geology=study',base).href);await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:180000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 await page.locator('#quality').evaluate(e=>{e.value='768';e.dispatchEvent(new Event('change'));});await page.evaluate(()=>waterLab.seek(4));
 const step=(n,dt,input={})=>page.evaluate(async({n,dt,input})=>{await waterLab.shipAdvance(n,dt,input);return waterLab.shipState();},{n,dt,input});
 const reset=async()=>{await page.locator('#shipSurface').click();return step(1,0);};
 await step(60,1/60,{forward:1});const thrust=await page.evaluate(()=>waterLab.shipState());assert.ok(thrust.speed>25&&thrust.speed<41,JSON.stringify(thrust));
 const coasting=await step(20,1/60);assert.ok(coasting.speed>thrust.speed*.75,'Release thrust should coast');
 const braked=await step(30,1/60,{forward:-1});assert.ok(braked.speed<coasting.speed*.18,'S must brake decisively');
 const steeringStart=await reset(),first=await step(1,1/60,{turn:1}),turning=await step(29,1/60,{turn:1});
 assert.ok(first.angles[0]>0&&first.angles[0]<.01,'Steering must accelerate smoothly');assert.ok(turning.angles[0]>.3&&turning.angles[2]<-.3,'Turning must bank with the actual angular motion');
 const settled=await step(90,1/60);assert.ok(Math.abs(settled.angles[2])<.01,'Bank must smoothly return to level');
 result.response={thrustSpeed:thrust.speed,coastingSpeed:coasting.speed,brakedSpeed:braked.speed,firstTurn:first.angles,turning:turning.angles,settledBank:settled.angles[2]};console.log('Response',JSON.stringify(result.response));
 const trajectories=[];
 for(const hz of [30,120]){const start=await reset(),end=await step(hz*2,1/hz,{forward:1,turn:.4});trajectories.push({hz,displacement:delta(end.position,start.position),angles:end.angles,speed:end.speed});}
 const positionError=length(delta(trajectories[0].displacement,trajectories[1].displacement)),speedError=Math.abs(trajectories[0].speed-trajectories[1].speed),angleError=length(delta(trajectories[0].angles,trajectories[1].angles));
 result.frameRates={trajectories,positionError,speedError,angleError};console.log('Frame rates',JSON.stringify(result.frameRates));assert.ok(positionError<.9&&speedError<.3&&angleError<.03,'Flight response should stay consistent at 30 and 120 Hz');
 await reset();await page.evaluate(()=>waterLab.shipAltitude(6));await step(150,1/60);
 const hover=await page.evaluate(()=>waterLab.inspect()),sources=await page.evaluate(()=>waterLab.shipWashState());
 assert.ok(sources.engines.every(e=>e[3]>.2),'Both engines must press on nearby water');assert.ok(hover.forceEnergy>1e-6&&hover.disturbanceMax>.05&&hover.disturbanceMax<1,JSON.stringify(hover));assert.ok(hover.finite&&hover.imaginaryResidual<1e-5);
 const local=await page.evaluate(()=>waterLab.domainTest());assert.deepEqual(local.ghost,[0,0],'Engine wash must not repeat on neighboring FFT tiles');assert.ok(local.untaperedGhost>.01,'Negative control must expose the periodic copy');
 const beforeDomain=sources.domain;await step(45,1/60,{forward:.35});const movedSources=await page.evaluate(()=>waterLab.shipWashState());assert.ok(length(delta(movedSources.domain.slice(0,2),beforeDomain.slice(0,2)))>1);assert.ok((await page.evaluate(()=>waterLab.inspect())).forceEnergy>1e-6,'Moving the pressure domain must preserve the wake');
 await reset();await page.evaluate(()=>waterLab.shipAltitude(80));const beforeDecay=await page.evaluate(()=>waterLab.inspect());assert.ok((await page.evaluate(()=>waterLab.shipWashState())).engines.every(e=>e[3]===0));await step(240,1/60);const afterDecay=await page.evaluate(()=>waterLab.inspect());assert.ok(afterDecay.forceEnergy<beforeDecay.forceEnergy*.03,'Wash must decay after leaving the surface');
 result.wash={sources,hoverHeight: hover.disturbanceMax,hoverEnergy:hover.forceEnergy,imaginaryResidual:hover.imaginaryResidual,local,movingDomain:movedSources.domain,highAltitudeSourcesOff:true,energyBeforeDecay:beforeDecay.forceEnergy,energyAfterDecay:afterDecay.forceEnergy};console.log('Wash',JSON.stringify(result.wash));
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);assert.deepEqual(errors,[]);result.errors=errors;
 await writeFile('captures/flight-validation.json',JSON.stringify(result,null,2));console.log('Flight and engine wash validation passed');
}finally{await browser.close();}
