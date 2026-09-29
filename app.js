import {GpuRuntime} from './vendor/webcuda/runtime/runtime.js';
const $=id=>document.getElementById(id),canvas=$('water'),keys=new Set();
const diagnostics=window.waterDiagnostics={ready:false,errors:[],frames:0,readbackBytes:0};
let runtime,context,kernels={},width=0,height=0,image,surface,light,camera,fft,photons,disturbance,brush;
let reset=1,playing=true,speed=3,time=0,last=0,lookX=0,lookY=0,drag=false,held=false,forceMoved=false,pointerX=0,pointerY=0,busy=false,failed=false;
const fixed=new URLSearchParams(location.search).get('t');if(fixed!==null){time=Number(fixed);playing=false;}
$('pause').textContent=playing?'Pause':'Resume';
function fail(e){failed=true;diagnostics.errors.push(String(e.message||e));$('error').hidden=false;$('error').textContent=diagnostics.errors.at(-1);$('loading').hidden=true;console.error(e);}
function labels(){for(const id of ['depth','energy','wind','exposure'])$(id+'Value').textContent=Number($(id).value).toFixed(2)+(id==='depth'?' m':id==='wind'?' m/s':'');}labels();
for(const id of ['depth','energy','wind','exposure'])$(id).oninput=labels;
$('toggle').onclick=()=>{document.body.classList.toggle('clean');$('toggle').textContent=document.body.classList.contains('clean')?'Show controls ↙':'Hide controls ↗';};
$('pause').onclick=()=>{playing=!playing;$('pause').textContent=playing?'Pause':'Resume';};$('reset').onclick=()=>reset=1;
$('fly').onclick=()=>canvas.requestPointerLock();
for(const id of ['shallows','ocean'])$(id).onclick=()=>{const ocean=id==='ocean';$('depth').value=ocean?8:1.4;$('energy').value=ocean?1.8:.8;reset=ocean?2:1;for(const p of ['shallows','ocean'])$(p).classList.toggle('active',p===id);labels();};
addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;keys.add(e.code);if(e.code==='KeyH')$('toggle').click();if(e.code.startsWith('Arrow')||e.code==='Space')e.preventDefault();});
addEventListener('keyup',e=>keys.delete(e.code));addEventListener('blur',()=>{keys.clear();drag=false;held=false;});
function pointer(e){const r=canvas.getBoundingClientRect(),x=2*(e.clientX-r.left)/r.width-1,y=1-2*(e.clientY-r.top)/r.height;if(x!==pointerX||y!==pointerY)forceMoved=true;pointerX=x;pointerY=y;}
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{pointer(e);forceMoved=false;held=e.button===0;drag=e.button===2;canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(document.pointerLockElement!==canvas)pointer(e);});
canvas.addEventListener('pointerup',()=>{drag=false;held=false;});canvas.addEventListener('pointercancel',()=>{drag=false;held=false;});canvas.addEventListener('lostpointercapture',()=>{drag=false;held=false;});
addEventListener('mousemove',e=>{if(drag||document.pointerLockElement===canvas){lookX+=e.movementX*.0025;lookY-=e.movementY*.0025;if(document.pointerLockElement===canvas){pointerX=0;pointerY=0;}}});
canvas.addEventListener('wheel',e=>{e.preventDefault();speed=Math.min(100,Math.max(.2,speed*Math.exp(-e.deltaY*.001)));},{passive:false});
function resize(){const w=Math.ceil(Math.min(Number($('quality').value),innerWidth*devicePixelRatio)/64)*64,h=Math.max(8,Math.round(w*innerHeight/innerWidth/8)*8);if(w===width&&h===height)return;if(image)runtime.destroyBuffer(image);width=w;height=h;canvas.width=w;canvas.height=h;image=runtime.createBuffer(w*h*4);context.configure({device:runtime.device,format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT,alphaMode:'opaque'});diagnostics.width=w;diagnostics.height=h;kernels.render.clear();}
function bind(name,buffers,scalars={}){return kernels[name].bind(buffers,scalars);}
function compute(dt,timestampWrites){
 const b=runtime.batch({timestampWrites});
 const axis=(a,z)=>(keys.has(a)?1:0)-(keys.has(z)?1:0);
 b.dispatch(bind('camera_step',{camera},{dt,forward:axis('KeyW','KeyS'),side:axis('KeyD','KeyA'),up:axis('KeyE','KeyQ'),lookX:lookX+axis('ArrowRight','ArrowLeft')*dt,lookY:lookY+axis('ArrowUp','ArrowDown')*dt,speed:speed*(keys.has('ShiftLeft')||keys.has('ShiftRight')?6:1),reset}),[1,1,1]);reset=0;lookX=lookY=0;
 b.dispatch(bind('brush_pick',{surface,camera,brush},{pointerX,pointerY,aspect:width/height,held:held?1:0,moving:forceMoved?1:0}),[1,1,1]);forceMoved=false;
 b.dispatch(bind('force_modes',{disturbance,brush},{dt:playing?dt:0,depth:Number($('depth').value),clear:0}),[16,16,1]);
 b.dispatch(bind('spectrum',{output:fft[0],disturbance},{time,wind:Number($('wind').value),depth:Number($('depth').value),energy:Number($('energy').value)}),[16,16,3]);
 let src=0;for(let axis=0;axis<2;axis++)for(let span=2;span<=128;span*=2){b.dispatch(bind('fft_stage',{input:fft[src],output:fft[1-src]},{span,axis}),[16,16,3]);src=1-src;}
 b.dispatch(bind('resolve',{input:fft[src],surface}),[16,16,3]);
 b.dispatch(bind('caustic_clear',{photons}),[32,32,1]);
 b.dispatch(bind('caustic_map',{surface,photons},{depth:Number($('depth').value)}),[64,64,1]);
 b.dispatch(bind('caustic_resolve',{photons,light}),[32,32,1]);
 b.dispatch(bind('render',{surface,light,camera,image},{width,height,depth:Number($('depth').value),exposure:Number($('exposure').value),view:Number($('view').value)}),[width/8,height/8,1]);b.submit();
 const enc=runtime.device.createCommandEncoder();enc.copyBufferToTexture({buffer:image.gpuBuffer,bytesPerRow:width*4},{texture:context.getCurrentTexture()},[width,height]);runtime.device.queue.submit([enc.finish()]);
}
async function frame(now){if(failed)return;try{const dt=last?Math.min(.05,(now-last)/1000):1/60;last=now;if(!busy&&!document.hidden){resize();if(playing)time+=dt;const start=performance.now();compute(dt);await runtime.idle();diagnostics.frameMs=performance.now()-start;diagnostics.frames++;diagnostics.ready=true;diagnostics.readbackBytes=runtime.stats.readbackBytes;$('loading').hidden=true;if(diagnostics.frames===1||diagnostics.frames%20===0)$('metrics').textContent=`${Math.round(1/dt)} FPS · ${width} × ${height} · ${speed.toFixed(1)} m/s`;}requestAnimationFrame(frame);}catch(e){fail(e);}}
async function exclusive(fn){busy=true;try{await runtime.idle();return await fn();}finally{busy=false;}}
window.waterLab={
 async lookDown(){return exclusive(async()=>{const update=new Float32Array([0,-.95,0,0]);runtime.device.queue.writeBuffer(camera.gpuBuffer,16,update);compute(0);await runtime.idle();});},
 pause(){playing=false;},resume(){playing=true;},
 async inspect(){return exclusive(async()=>{const a=await runtime.read(surface),c=await runtime.read(camera);let max=0,imag=0,sum=0,disturbanceMax=0;for(let i=0;i<a.length;i+=4){max=Math.max(max,Math.abs(a[i]));imag=Math.max(imag,Math.abs(a[i+3]));sum+=a[i]*a[i];if(i>=32768*4)disturbanceMax=Math.max(disturbanceMax,Math.abs(a[i]));}const l=await runtime.read(light),d=await runtime.read(disturbance);let forceEnergy=0;for(const v of d)forceEnergy+=v*v;return {forceEnergy,disturbanceMax,finite:a.every(Number.isFinite)&&l.every(Number.isFinite),heightMax:max,heightRms:Math.sqrt(sum/(a.length/4)),imaginaryResidual:imag,camera:Array.from(c),causticMin:Math.min(...l.filter((v,i)=>i%4===0)),causticMax:Math.max(...l.filter((v,i)=>i%4===0)),adapter:runtime.describe()};});},
 async seek(t){return exclusive(async()=>{playing=false;time=t;resize();compute(0);await runtime.idle();});},
 async screenshot(){return exclusive(async()=>{const p=await runtime.read(image,Uint32Array);return {width,height,rgba:Array.from(new Uint8Array(p.buffer))};});},
 async fftTest(){return exclusive(async()=>{
  const data=new Float32Array(49152*2),modes=[{x:3,z:7,c:0,re:.3,im:-.2},{x:51,z:89,c:0,re:.07,im:.11},{x:11,z:125,c:1,re:-.4,im:.15}];
  const reverse=x=>{let r=0;for(let i=0;i<7;i++){r=r*2+(x&1);x>>=1;}return r;};
  for(const m of modes){const idx=(m.c*16384+reverse(m.z)*128+reverse(m.x))*2;data[idx]=m.re;data[idx+1]=m.im;}
  const a=runtime.createBuffer(data),b=runtime.createBuffer(data.byteLength);const batch=runtime.batch();let src=0;const pair=[a,b];
  for(let axis=0;axis<2;axis++)for(let span=2;span<=128;span*=2){batch.dispatch(bind('fft_stage',{input:pair[src],output:pair[1-src]},{span,axis}),[16,16,3]);src=1-src;}batch.submit();
  const result=await runtime.read(pair[src]);let maxError=0;
  for(let c=0;c<3;c++)for(let z=0;z<128;z++)for(let x=0;x<128;x++){let re=0,im=0;for(const m of modes){if(m.c!==c)continue;const angle=2*Math.PI*(m.x*x+m.z*z)/128;re+=m.re*Math.cos(angle)-m.im*Math.sin(angle);im+=m.re*Math.sin(angle)+m.im*Math.cos(angle);}const idx=(c*16384+z*128+x)*2;maxError=Math.max(maxError,Math.abs(result[idx]-re),Math.abs(result[idx+1]-im));}
  runtime.destroyBuffer(a);runtime.destroyBuffer(b);kernels.fft_stage.clear();return {maxError,modes:3,axes:2,cascades:3,oracle:'Direct complex Fourier series evaluated independently in JavaScript for diagnostics only'};
 });},
 async flatCausticsTest(){return exclusive(async()=>{
  const flat=runtime.createBuffer(49152*16),out=runtime.createBuffer(65536*16);runtime.batch().dispatch(bind('caustic_clear',{photons}),[32,32,1]).dispatch(bind('caustic_map',{surface:flat,photons},{depth:1.4}),[64,64,1]).dispatch(bind('caustic_resolve',{photons,light:out}),[32,32,1]).submit();
  const a=await runtime.read(out);let error=0;for(let i=0;i<a.length;i++)if(i%4!==3)error=Math.max(error,Math.abs(a[i]-1));runtime.destroyBuffer(flat);runtime.destroyBuffer(out);kernels.caustic_map.clear();kernels.caustic_resolve.clear();return {maxDeviationFromUniform:error};
 });},
 async benchmark(samples=40){return exclusive(async()=>{if(!runtime.device.features.has('timestamp-query'))return {unsupported:true};playing=false;const qs=runtime.device.createQuerySet({type:'timestamp',count:2}),resolve=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const times=[];try{for(let i=0;i<samples+8;i++){compute(0,{querySet:qs,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1});const enc=runtime.device.createCommandEncoder();enc.resolveQuerySet(qs,0,2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,16);runtime.device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(read.getMappedRange());if(i>=8)times.push(Number(t[1]-t[0])/1e6);read.unmap();}times.sort((a,b)=>a-b);return {width,height,samples,medianMs:times[Math.floor(samples/2)],p95Ms:times[Math.floor(samples*.95)],scope:'GPU camera, spectrum, FFT, normals, caustics and render; excludes texture copy and browser presentation',adapter:runtime.describe()};}finally{qs.destroy();resolve.destroy();read.destroy();}});}
};
try{
 runtime=await GpuRuntime.create({onError:fail});context=canvas.getContext('webgpu');
 for(const name of ['camera_step','brush_pick','force_modes','spectrum','fft_stage','resolve','caustic_clear','caustic_map','caustic_resolve','render']){
  $('loadText').textContent=`Preparing ${name.replaceAll('_',' ')}…`;const artifact=await(await fetch(`./kernels/${name}.json`)).json();
  const kernel=await runtime.kernel(artifact),cache=new Map();kernels[name]={bind(buffers,scalars){const key=Object.values(buffers).map(b=>b.id).join(':');let v=cache.get(key);if(v)v.setScalars(scalars);else{v=kernel.bind(buffers,scalars);cache.set(key,v);}return v;},clear(){cache.clear();}};
 }
 fft=[runtime.createBuffer(49152*8),runtime.createBuffer(49152*8)];surface=runtime.createBuffer(49152*16);light=runtime.createBuffer(65536*16);photons=runtime.createBuffer(65536*16);camera=runtime.createBuffer(32);brush=runtime.createBuffer(32);disturbance=runtime.createBuffer(16384*16);diagnostics.adapter=runtime.describe();requestAnimationFrame(frame);
}catch(e){fail(e);}
