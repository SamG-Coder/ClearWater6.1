import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
const browser=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']});
const base=process.env.WATER_URL||'http://127.0.0.1:5191/',errors=[],report={};
try{
 await mkdir('captures/horizon',{recursive:true});
 const page=await browser.newPage({viewport:{width:2560,height:1440}});page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(base+'?mode=ship&t=4&geology=study');await page.waitForFunction(()=>waterDiagnostics.ready||waterDiagnostics.errors.length,null,{timeout:240000});assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);
 await page.evaluate(()=>waterLab.weatherSeek(50400));
 const points=[];for(const altitude of [800,1200,1600,1601,2500,3500,7000,8200])for(let i=0;i<=128;i++)points.push([altitude,-.12+i*.002,10,0]);
 const values=await page.evaluate(p=>waterLab.patternSamples(p),points);
 assert.ok(values.every(Number.isFinite));for(let i=0;i<points.length;i++)assert.equal(values[i*4],1,'Inside the cloud bounds, haze must not jump at an imaginary middle-shell horizon');
 // Independent negative control: the old 1600 m shell abruptly changes from
 // unit visibility to almost zero on either side of its tangent at 3500 m.
 const R=6371000,altitude=3500,outer=R+altitude,inner=R+1600,tangent=-Math.sqrt(outer*outer-inner*inner)/outer;
 const oldVisibility=y=>{const b=outer*y,disc=b*b-(outer*outer-inner*inner);if(disc<0)return 1;return Math.exp(-Math.max(0,-b-Math.sqrt(disc))*.000055);};
 report.oldTangentJump=Math.abs(oldVisibility(tangent-.00001)-oldVisibility(tangent+.00001));assert.ok(report.oldTangentJump>.99);report.samples=points.length;
 const locations=[];for(let lat=-60;lat<=60;lat+=10)for(let lon=-180;lon<180;lon+=15)locations.push([lat,lon,8,0]);
 const field=await page.evaluate(p=>waterLab.patternSamples(p),locations);let best=0,score=-Infinity;
 for(let i=0;i<locations.length;i++){const s=-Math.abs(field[i*4]-.25)*8-Math.abs(field[i*4+1]-.65)+field[i*4+2];if(s>score){score=s;best=i;}}
 report.location=locations[best].slice(0,2);await page.evaluate(p=>waterLab.weatherLocation(p[0],p[1],3500),report.location);
 for(const altitude of [799,801,1599,1601,3500,8199,8201,12000]){await page.evaluate(h=>waterLab.shipAltitude(h),altitude);await page.evaluate(()=>waterLab.shipView(.34,.15,22));await page.screenshot({path:`captures/horizon/${altitude}.png`});}
 report.timings=[];
 for(const [width,height] of [[2560,1440],[3840,2160]]){await page.setViewportSize({width,height});await page.locator('#quality').evaluate((e,w)=>{e.value=String(w);e.dispatchEvent(new Event('change'));},width);await page.evaluate(()=>waterLab.shipAltitude(3500));await page.evaluate(()=>waterLab.seek(4));report.timings.push(await page.evaluate(()=>waterLab.benchmark(30,true)));await page.screenshot({path:`captures/horizon/final-${width}.png`});}
 assert.deepEqual(await page.evaluate(()=>waterDiagnostics.errors),[]);assert.deepEqual(errors,[]);report.errors=errors;
 await writeFile('captures/horizon-validation.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report,null,2));
}finally{await browser.close();}
