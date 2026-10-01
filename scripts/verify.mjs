import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
await mkdir('captures',{recursive:true});
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
try{
 const page=await browser.newPage({viewport:{width:1440,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));page.on('console',m=>{if(m.type()==='error')errors.push(m.text());});
 await page.goto('http://127.0.0.1:5191/?t=4&geology=study');
 await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:120000});
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 assert.equal(await page.evaluate(()=>waterDiagnostics.readbackBytes),0);
 await page.screenshot({path:'captures/shallows-ui.png'});await page.locator('#toggle').click();await page.screenshot({path:'captures/shallows.png'});await page.evaluate(()=>waterLab.lookDown());await page.screenshot({path:'captures/revised-down.png'});await page.locator('#toggle').click();await page.locator('#reset').click();await page.evaluate(()=>waterLab.seek(4));await page.locator('#toggle').click();
 const fft=await page.evaluate(()=>waterLab.fftTest());assert.ok(fft.maxError<1e-5);
 const flatCaustics=await page.evaluate(()=>waterLab.flatCausticsTest());assert.ok(flatCaustics.maxDeviationFromUniform<.001);
 const water=await page.evaluate(()=>waterLab.inspect());assert.ok(water.finite&&water.heightRms>.001&&water.imaginaryResidual<1e-5,JSON.stringify(water));
 assert.ok(water.causticMax-water.causticMin>.1,'Sunlight must focus');
 const pixelsBefore=await page.evaluate(()=>waterLab.screenshot());await page.evaluate(()=>waterLab.seek(7));const pixelsAfter=await page.evaluate(()=>waterLab.screenshot());let pixelChange=0;for(let i=0;i<pixelsBefore.rgba.length;i+=4)pixelChange+=Math.abs(pixelsBefore.rgba[i]-pixelsAfter.rgba[i]);pixelChange/=pixelsBefore.width*pixelsBefore.height;assert.ok(pixelChange>.5,'Waves and refracted sunlight must animate');
 const before=water.camera;await page.keyboard.down('KeyW');await page.waitForTimeout(300);await page.keyboard.up('KeyW');await page.keyboard.down('KeyE');await page.waitForTimeout(300);await page.keyboard.up('KeyE');
 const moved=(await page.evaluate(()=>waterLab.inspect())).camera;assert.ok(moved[2]<before[2]-.2);assert.ok(moved[1]>before[1]+.2);
 const balanced=await page.evaluate(()=>waterLab.benchmark());
 await page.locator('#toggle').click();await page.locator('#quality').selectOption('1920');await page.setViewportSize({width:1920,height:1080});await page.waitForFunction(()=>waterDiagnostics.width===1920&&waterDiagnostics.height===1080);
 const fullhd=await page.evaluate(()=>waterLab.benchmark());
 await page.locator('#ocean').click();await page.evaluate(()=>waterLab.seek(12));await page.locator('#toggle').click();await page.screenshot({path:'captures/open-water.png'});
 assert.deepEqual(errors,[]);const result={fft,flatCaustics,pixelChange,water,cameraMoved:moved,balanced,fullhd,errors};await writeFile('captures/validation.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
}finally{await browser.close();}
