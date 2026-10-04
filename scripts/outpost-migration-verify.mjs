import {chromium} from 'playwright';import assert from 'node:assert/strict';import {writeFile} from 'node:fs/promises';
const b=await chromium.launch({channel:'msedge',headless:true,args:['--enable-unsafe-webgpu']}),p=await b.newPage();
try{
 await p.route('**/outpost-migration-fixture',route=>route.fulfill({contentType:'text/html',body:'<!doctype html><title>Outpost migration verification</title>'}));await p.goto('http://127.0.0.1:5191/outpost-migration-fixture');
 const results=[];for(const width of [512,1024]){const result=await p.evaluate(async(width)=>{
  const {GpuRuntime}=await import('/vendor/webcuda/runtime/runtime.js');const errors=[];const runtime=await GpuRuntime.create({onError:e=>errors.push(String(e))}),offset=64,camera=runtime.createBuffer((offset+width*width/2+32)*16),plates=runtime.createBuffer(28*32),nav=runtime.createBuffer(64),ship=runtime.createBuffer(1024),kernels={};
  for(const name of ['geology_seed','geology_map','outpost_init'])kernels[name]=await runtime.kernel(await fetch(`/kernels/${name}.json`).then(r=>r.json()));
  runtime.batch().dispatch(kernels.geology_seed.bind({plates,camera},{seedValue:61,mapWidth:width,offset}),[1,1,1]).dispatch(kernels.geology_map.bind({plates,camera},{mapWidth:width,offset,seedValue:61}),[width/8,width/16,1]).submit();await runtime.idle();
  const map=await runtime.read(camera);let index=0,min=Infinity;for(let y=30;y<width/2-30;y++)for(let x=0;x<width;x++){const h=map[(offset+y*width+x)*4];if(h<min){min=h;index=y*width+x;}}
  const lon=((index%width+.5)/width-.5)*Math.PI*2,lat=(.5-(Math.floor(index/width)+.5)/(width/2))*Math.PI,n=[Math.sin(lon)*Math.cos(lat),Math.sin(lat),Math.cos(lon)*Math.cos(lat)],east=[Math.cos(lon),0,-Math.sin(lon)];
  const navigation=new Float32Array([n[0],0,n[1],0,n[2],0,east[0],0,east[1],0,east[2],0,0,0,0,0]),saved=new Float32Array(256);saved.set([0,2000000,0,1]);saved.set([100,75,20,0],56);runtime.device.queue.writeBuffer(nav.gpuBuffer,0,navigation);runtime.device.queue.writeBuffer(ship.gpuBuffer,0,saved);
  runtime.batch().dispatch(kernels.outpost_init.bind({camera,navigationState:nav,ship},{startParked:0}),[1,1,1]).submit();await runtime.idle();const after=await runtime.read(ship),afterNav=await runtime.read(nav);const distance=Math.acos(Math.max(-1,Math.min(1,n[0]*after[112]+n[1]*after[113]+n[2]*after[114])))*6371000;
  runtime.device.queue.writeBuffer(ship.gpuBuffer,28*16+12,new Float32Array([-5]));runtime.batch().dispatch(kernels.outpost_init.bind({camera,navigationState:nav,ship},{startParked:0}),[1,1,1]).submit();await runtime.idle();const repaired=await runtime.read(ship);
  return {repairedInvalidAnchor:repaired[115]>3,mapWidth:width,originalOceanDepth:-min,outpostAltitude:after[115],distanceMetres:distance,mode:after[84],stamp:after[87],finite:after.every(Number.isFinite),navigationUnchanged:afterNav.every((v,i)=>v===navigation[i]),shipUnchanged:after.slice(0,84).every((v,i)=>v===saved[i]),errors};
 },width);
 assert.ok(result.originalOceanDepth>2000);assert.ok(result.outpostAltitude>3);assert.ok(result.distanceMetres>20000);assert.equal(result.mode,0);assert.equal(result.stamp,62);assert.ok(result.finite&&result.navigationUnchanged&&result.shipUnchanged&&result.repairedInvalidAnchor);assert.deepEqual(result.errors,[]);results.push(result);console.log(result);}
 await writeFile('captures/outpost/migration-validation.json',JSON.stringify(results,null,2));
}finally{await b.close();}
