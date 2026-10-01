import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
await mkdir('captures',{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto('http://127.0.0.1:5191/?geology=study');await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 await page.locator('#toggle').click();await page.evaluate(()=>waterLab.lookDown());await page.evaluate(()=>waterLab.resume());
 await page.mouse.move(800,500);await page.mouse.move(1150,680,{steps:20});
 const hover=await page.evaluate(()=>waterLab.inspect());assert.equal(hover.forceEnergy,0,'Hover must not add energy');
 const cameraBefore=hover.camera;
 await page.mouse.move(730,570);await page.mouse.down();await page.waitForTimeout(100);
 const stationary=await page.evaluate(()=>waterLab.inspect());assert.equal(stationary.forceEnergy,0,'Stationary held click must not inject a moving force');
 for(let i=0;i<30;i++){await page.mouse.move(730+i*11,570+Math.sin(i*.18)*60);await page.waitForTimeout(16);}
 const dragging=await page.evaluate(()=>waterLab.inspect());assert.ok(dragging.forceEnergy>1e-7&&dragging.disturbanceMax>.02,JSON.stringify(dragging));assert.deepEqual(dragging.camera,cameraBefore,'Left drag must not turn the camera');
 await page.mouse.up();await page.waitForTimeout(300);const released=await page.evaluate(()=>waterLab.inspect());assert.ok(released.forceEnergy>1e-8,'Waves must persist after release');assert.ok(released.finite&&released.imaginaryResidual<1e-5);
 const locality=await page.evaluate(()=>waterLab.domainTest());
 assert.ok(locality.finite&&locality.local.every(x=>x>1e-5),JSON.stringify(locality));
 assert.deepEqual(locality.ghost,[0,0],'Touch must not repeat in neighboring FFT tiles');
 assert.ok(locality.untaperedGhost>1e-5,'The negative control must expose the old periodic wake');
 assert.ok(locality.seamSlopeError<.02,JSON.stringify(locality));
 assert.ok(locality.regionalDifference>1e-4,'Neighboring background regions must differ');
 await page.screenshot({path:'captures/drag-force.png'});
 // Move to another world region and draw again: the one pressure domain must
 // follow the brush while still suppressing the neighboring periodic copy.
 await page.keyboard.down('Shift');await page.keyboard.down('KeyD');await page.waitForTimeout(700);await page.keyboard.up('KeyD');await page.keyboard.up('Shift');
 await page.mouse.move(730,570);await page.mouse.down();
 for(let i=0;i<10;i++){await page.mouse.move(730+i*10,570);await page.waitForTimeout(16);}
 await page.mouse.up();const relocated=await page.evaluate(()=>waterLab.domainTest());
 assert.ok(Math.abs(relocated.domain[0]-locality.domain[0])>6,JSON.stringify(relocated));
 assert.ok(relocated.finite&&relocated.local[0]>1e-5);assert.deepEqual(relocated.ghost,[0,0]);
 await page.mouse.move(1100,600);await page.mouse.down({button:'right'});await page.mouse.move(1200,620,{steps:10});await page.mouse.up({button:'right'});await page.waitForTimeout(100);
 const rotated=await page.evaluate(()=>waterLab.inspect());assert.notEqual(rotated.camera[4],cameraBefore[4],'Right drag must still turn camera');
 assert.deepEqual(errors,[]);const benchmark=await page.evaluate(()=>waterLab.benchmark());
 const result={locality,relocated,hoverEnergy:hover.forceEnergy,stationaryEnergy:stationary.forceEnergy,draggingEnergy:dragging.forceEnergy,releasedEnergy:released.forceEnergy,draggingHeightMax:dragging.disturbanceMax,releasedHeightMax:released.disturbanceMax,imaginaryResidual:released.imaginaryResidual,cameraUnchangedDuringForce:true,rightDragRotatesCamera:true,benchmark,errors};
 await writeFile('captures/force-validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
