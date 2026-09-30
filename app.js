import {GpuRuntime} from './vendor/webcuda/runtime/runtime.js';
const $=id=>document.getElementById(id),canvas=$('water'),keys=new Set();
const touchDevice=matchMedia('(pointer:coarse)').matches||navigator.maxTouchPoints>0;
if(touchDevice){document.body.classList.add('touch','clean');$('quality').value='mobile';$('toggle').textContent='Settings';$('fly').textContent='Look around';}
let adaptiveScale=1,frameAverage=0,adaptCount=0,touchLook=false,lastWind=-1;
const mobileProfile=()=>$('quality').value==='mobile';
const diagnostics=window.waterDiagnostics={ready:false,errors:[],frames:0,readbackBytes:0};
let runtime,context,kernels={},width=0,height=0,image,surface,light,camera,fft,photons,disturbance,brush,seed,twiddles;
let reset=1,playing=true,speed=3,time=0,last=0,lookX=0,lookY=0,drag=false,held=false,forceMoved=false,pointerX=0,pointerY=0,busy=false,failed=false;
const fixed=new URLSearchParams(location.search).get('t');if(fixed!==null){time=Number(fixed);playing=false;}
$('pause').textContent=playing?'Pause':'Resume';
function fail(e){failed=true;diagnostics.errors.push(String(e.message||e));$('error').hidden=false;$('error').textContent=diagnostics.errors.at(-1);$('loading').hidden=true;console.error(e);}
function labels(){for(const id of ['depth','energy','wind','exposure'])$(id+'Value').textContent=Number($(id).value).toFixed(2)+(id==='depth'?' m':id==='wind'?' m/s':'');}labels();
for(const id of ['depth','energy','wind','exposure'])$(id).oninput=labels;
$('toggle').onclick=()=>{resetSticks();keys.clear();padPointers.clear();document.body.classList.toggle('clean');$('toggle').textContent=touchDevice?(document.body.classList.contains('clean')?'Settings':'Close settings'):(document.body.classList.contains('clean')?'Show controls ↙':'Hide controls ↗');};
$('pause').onclick=()=>{playing=!playing;$('pause').textContent=playing?'Pause':'Resume';};$('reset').onclick=()=>reset=1;
$('fly').onclick=()=>{if(touchDevice){touchLook=!touchLook;$('fly').textContent=touchLook?'Push water':'Look around';$('touchMode').textContent=touchLook?'Mode: Look':'Mode: Water';}else canvas.requestPointerLock?.();};
$('touchMode').onclick=()=>$('fly').click();
$('quality').onchange=()=>{adaptiveScale=1;frameAverage=0;adaptCount=0;};
for(const id of ['shallows','ocean'])$(id).onclick=()=>{const ocean=id==='ocean';$('depth').value=ocean?8:1.4;$('energy').value=ocean?1.8:.8;reset=ocean?2:1;for(const p of ['shallows','ocean'])$(p).classList.toggle('active',p===id);labels();};
addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;keys.add(e.code);if(e.code==='KeyH')$('toggle').click();if(e.code.startsWith('Arrow')||e.code==='Space')e.preventDefault();});
addEventListener('keyup',e=>keys.delete(e.code));addEventListener('blur',()=>{keys.clear();drag=false;held=false;});
function pointer(e){const r=canvas.getBoundingClientRect(),x=2*(e.clientX-r.left)/r.width-1,y=1-2*(e.clientY-r.top)/r.height;if(x!==pointerX||y!==pointerY)forceMoved=true;pointerX=x;pointerY=y;}
canvas.addEventListener('contextmenu',e=>e.preventDefault());
let touchPointer=null,touchLastX=0,touchLastY=0;
function endPointer(e){if(touchPointer===null||e.pointerId===touchPointer){drag=false;held=false;forceMoved=false;touchPointer=null;}}
canvas.addEventListener('pointerdown',e=>{
 if(e.pointerType==='touch'&&touchPointer!==null)return;
 pointer(e);forceMoved=false;
 if(e.pointerType==='touch'){touchPointer=e.pointerId;drag=touchLook;held=!touchLook;touchLastX=e.clientX;touchLastY=e.clientY;}
 else{held=e.button===0;drag=e.button===2;}
 canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove',e=>{
 if(e.pointerType==='touch'){
  if(e.pointerId!==touchPointer)return;
  if(drag){lookX+=(e.clientX-touchLastX)*.004;lookY-=(e.clientY-touchLastY)*.004;}
  touchLastX=e.clientX;touchLastY=e.clientY;
 }
 if(document.pointerLockElement!==canvas)pointer(e);
});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,endPointer);
addEventListener('mousemove',e=>{if((drag&&touchPointer===null)||document.pointerLockElement===canvas){lookX+=e.movementX*.0025;lookY-=e.movementY*.0025;if(document.pointerLockElement===canvas){pointerX=0;pointerY=0;}}});
const padPointers=new Map();
const sticks={move:{x:0,y:0,pointer:null},look:{x:0,y:0,pointer:null}};
function resetSticks(){for(const [name,stick] of Object.entries(sticks)){stick.x=stick.y=0;stick.pointer=null;const el=$(name+'Stick');el.classList.remove('active');el.querySelector('.stick-knob').style.transform='translate(0px,0px)';}}
for(const [name,stick] of Object.entries(sticks)){
 const el=$(name+'Stick'),knob=el.querySelector('.stick-knob');
 const update=e=>{const r=el.getBoundingClientRect(),radius=r.width*.32,dx=e.clientX-r.x-r.width/2,dy=e.clientY-r.y-r.height/2,length=Math.hypot(dx,dy),scale=length>radius?radius/length:1;
  const amount=Math.max(0,(Math.min(1,length/radius)-.12)/.88);stick.x=length?dx/length*amount:0;stick.y=length?dy/length*amount:0;
  knob.style.transform=`translate(${dx*scale}px,${dy*scale}px)`;
 };
 el.addEventListener('pointerdown',e=>{if(stick.pointer!==null)return;e.preventDefault();stick.pointer=e.pointerId;el.setPointerCapture(e.pointerId);el.classList.add('active');update(e);});
 el.addEventListener('pointermove',e=>{if(e.pointerId===stick.pointer)update(e);});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])el.addEventListener(event,e=>{if(e.pointerId!==stick.pointer)return;stick.x=stick.y=0;stick.pointer=null;el.classList.remove('active');knob.style.transform='translate(0px,0px)';});
}
for(const button of document.querySelectorAll('[data-move]')){
 button.addEventListener('pointerdown',e=>{e.preventDefault();padPointers.set(e.pointerId,button.dataset.move);keys.add(button.dataset.move);button.setPointerCapture(e.pointerId);button.classList.add('active');});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(event,e=>{const key=padPointers.get(e.pointerId);padPointers.delete(e.pointerId);if(key&&![...padPointers.values()].includes(key))keys.delete(key);button.classList.remove('active');});
}
addEventListener('blur',()=>{touchPointer=null;padPointers.clear();resetSticks();});
document.addEventListener('visibilitychange',()=>{last=0;held=drag=false;keys.clear();touchPointer=null;padPointers.clear();resetSticks();});
addEventListener('resize',resetSticks);
canvas.addEventListener('wheel',e=>{e.preventDefault();speed=Math.min(100,Math.max(.2,speed*Math.exp(-e.deltaY*.001)));},{passive:false});
function resize(){
 const mobile=mobileProfile(),aspect=innerWidth/innerHeight;
 const longest=mobile?960*adaptiveScale:Infinity;
 const desiredWidth=mobile?Math.min(innerWidth*devicePixelRatio,longest*Math.min(1,aspect)):Math.min(Number($('quality').value),innerWidth*devicePixelRatio);
 const w=Math.max(64,Math.floor(desiredWidth/64)*64),h=Math.max(8,Math.round(w/aspect/8)*8);
 diagnostics.mobile=mobile;diagnostics.targetFps=mobile?30:60;diagnostics.adaptiveScale=adaptiveScale;diagnostics.photonRays=mobile?256:512;
 if(w===width&&h===height)return;
 if(image)runtime.destroyBuffer(image);width=w;height=h;canvas.width=w;canvas.height=h;image=runtime.createBuffer(w*h*4);
 context.configure({device:runtime.device,format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT,alphaMode:'opaque'});
 diagnostics.width=w;diagnostics.height=h;kernels.render.clear();
}
function bind(name,buffers,scalars={}){return kernels[name].bind(buffers,scalars);}
function compute(dt,timestampWrites){
 const b=runtime.batch({timestampWrites});
 const axis=(a,z)=>(keys.has(a)?1:0)-(keys.has(z)?1:0);
 b.dispatch(bind('camera_step',{camera},{dt,forward:Math.max(-1,Math.min(1,axis('KeyW','KeyS')-sticks.move.y)),side:Math.max(-1,Math.min(1,axis('KeyD','KeyA')+sticks.move.x)),up:axis('KeyE','KeyQ'),lookX:lookX+(axis('ArrowRight','ArrowLeft')+sticks.look.x*1.8)*dt,lookY:lookY+(axis('ArrowUp','ArrowDown')-sticks.look.y*1.8)*dt,speed:speed*(keys.has('ShiftLeft')||keys.has('ShiftRight')?6:1),reset}),[1,1,1]);reset=0;lookX=lookY=0;
 b.dispatch(bind('brush_pick',{surface,camera,brush},{pointerX,pointerY,aspect:width/height,held:held?1:0,moving:forceMoved?1:0}),[1,1,1]);forceMoved=false;
 b.dispatch(bind('force_modes',{disturbance,brush},{dt:playing?dt:0,depth:Number($('depth').value),clear:0}),[16,16,1]);
 const wind=Number($('wind').value);if(wind!==lastWind){b.dispatch(bind('seed_modes',{seed,twiddles},{wind}),[16,16,3]);lastWind=wind;diagnostics.spectrumSeeds=(diagnostics.spectrumSeeds||0)+1;}
 b.dispatch(bind('spectrum',{output:fft[0],seed,disturbance},{time,depth:Number($('depth').value),energy:Number($('energy').value)}),[16,16,3]);
 for(let axis=0;axis<2;axis++)b.dispatch(bind('fft_local',{input:fft[axis],output:fft[1-axis],twiddles},{axis}),[128,1,3]);
 b.dispatch(bind('resolve',{input:fft[0],surface}),[16,16,3]);
 b.dispatch(bind('caustic_clear',{photons}),[32,32,1]);
 const rays=mobileProfile()?256:512,dispersion=mobileProfile()?0:1;
 b.dispatch(bind('caustic_map',{surface,photons},{depth:Number($('depth').value),rays,dispersion}),[rays/8,rays/8,1]);
 b.dispatch(bind('caustic_resolve',{photons,light},{normalization:4096*(rays/256)**2,dispersion}),[32,32,1]);
 b.dispatch(bind('render',{surface,light,camera,image},{width,height,depth:Number($('depth').value),exposure:Number($('exposure').value),view:Number($('view').value)}),[width/8,height/8,1]);b.submit();
 const enc=runtime.device.createCommandEncoder();enc.copyBufferToTexture({buffer:image.gpuBuffer,bytesPerRow:width*4},{texture:context.getCurrentTexture()},[width,height]);runtime.device.queue.submit([enc.finish()]);
}
async function frame(now){
 if(failed)return;
 try{
  const mobile=mobileProfile(),interval=mobile?1000/30:0;
  if(!busy&&!document.hidden&&(!last||now-last>=interval-.5)){
   const elapsed=last?(now-last)/1000:1/(mobile?30:60),dt=Math.min(.1,elapsed);last=now;resize();if(playing)time+=dt;
   const start=performance.now();compute(dt);await runtime.idle();
   diagnostics.frameMs=performance.now()-start;diagnostics.fps=1/elapsed;diagnostics.frames++;diagnostics.ready=true;diagnostics.readbackBytes=runtime.stats.readbackBytes;$('loading').hidden=true;
   frameAverage=frameAverage?frameAverage*.94+diagnostics.frameMs*.06:diagnostics.frameMs;
   if(mobile&&++adaptCount>=60){if(frameAverage>25&&adaptiveScale>.5)adaptiveScale=Math.max(.5,adaptiveScale-.1);else if(frameAverage<12&&adaptiveScale<1)adaptiveScale=Math.min(1,adaptiveScale+.05);adaptCount=0;}
   if(diagnostics.frames===1||diagnostics.frames%15===0)$('metrics').textContent=`${Math.round(diagnostics.fps)} FPS · ${width} × ${height} · ${speed.toFixed(1)} m/s`;
  }
  requestAnimationFrame(frame);
 }catch(e){fail(e);}
}
async function exclusive(fn){busy=true;try{await runtime.idle();return await fn();}finally{busy=false;}}
window.waterLab={
 async lookDown(){return exclusive(async()=>{const update=new Float32Array([0,-.95,0,0]);runtime.device.queue.writeBuffer(camera.gpuBuffer,16,update);compute(0);await runtime.idle();});},
 pause(){playing=false;},resume(){playing=true;},
 async inspect(){return exclusive(async()=>{const a=await runtime.read(surface),c=await runtime.read(camera);let max=0,imag=0,sum=0,disturbanceMax=0;for(let i=0;i<a.length;i+=4){max=Math.max(max,Math.abs(a[i]));imag=Math.max(imag,Math.abs(a[i+3]));sum+=a[i]*a[i];if(i>=32768*4)disturbanceMax=Math.max(disturbanceMax,Math.abs(a[i]));}const l=await runtime.read(light),d=await runtime.read(disturbance);let forceEnergy=0;for(const v of d)forceEnergy+=v*v;const causticMean=[0,0,0];for(let i=0;i<l.length;i+=4)for(let c=0;c<3;c++)causticMean[c]+=l[i+c]/65536;return {forceEnergy,disturbanceMax,causticMean,finite:a.every(Number.isFinite)&&l.every(Number.isFinite),heightMax:max,heightRms:Math.sqrt(sum/(a.length/4)),imaginaryResidual:imag,camera:Array.from(c.slice(0,8)),causticMin:Math.min(...l.filter((v,i)=>i%4===0)),causticMax:Math.max(...l.filter((v,i)=>i%4===0)),adapter:runtime.describe()};});},
 async seek(t){return exclusive(async()=>{playing=false;time=t;resize();compute(0);await runtime.idle();});},
 async screenshot(){return exclusive(async()=>{const p=await runtime.read(image,Uint32Array);return {width,height,rgba:Array.from(new Uint8Array(p.buffer))};});},
 async fftTest(strategy='shared'){return exclusive(async()=>{
  const data=new Float32Array(49152*2),modes=[{x:3,z:7,c:0,re:.3,im:-.2},{x:51,z:89,c:0,re:.07,im:.11},{x:11,z:125,c:1,re:-.4,im:.15}];
  const reverse=x=>{let r=0;for(let i=0;i<7;i++){r=r*2+(x&1);x>>=1;}return r;};
  for(const m of modes){const idx=(m.c*16384+reverse(m.z)*128+reverse(m.x))*2;data[idx]=m.re;data[idx+1]=m.im;}
  const a=runtime.createBuffer(data),b=runtime.createBuffer(data.byteLength);const batch=runtime.batch();let src=0;const pair=[a,b];
  if(strategy==='staged'){for(let axis=0;axis<2;axis++)for(let span=2;span<=128;span*=2){batch.dispatch(bind('fft_stage',{input:pair[src],output:pair[1-src]},{span,axis}),[16,16,3]);src=1-src;}}
  else{for(let axis=0;axis<2;axis++){batch.dispatch(bind('fft_local',{input:pair[src],output:pair[1-src],twiddles},{axis}),[128,1,3]);src=1-src;}}
  batch.submit();
  const result=await runtime.read(pair[src]);let maxError=0;
  for(let c=0;c<3;c++)for(let z=0;z<128;z++)for(let x=0;x<128;x++){let re=0,im=0;for(const m of modes){if(m.c!==c)continue;const angle=2*Math.PI*(m.x*x+m.z*z)/128;re+=m.re*Math.cos(angle)-m.im*Math.sin(angle);im+=m.re*Math.sin(angle)+m.im*Math.cos(angle);}const idx=(c*16384+z*128+x)*2;maxError=Math.max(maxError,Math.abs(result[idx]-re),Math.abs(result[idx+1]-im));}
  runtime.destroyBuffer(a);runtime.destroyBuffer(b);kernels.fft_stage.clear();kernels.fft_local.clear();return {maxError,strategy,modes:3,axes:2,cascades:3,oracle:'Direct complex Fourier series evaluated independently in JavaScript for diagnostics only'};
 });},
 async flatCausticsTest(mobile=false){return exclusive(async()=>{
  const rays=mobile?256:512,dispersion=mobile?0:1,normalization=mobile?4096:16384;
  const flat=runtime.createBuffer(49152*16),out=runtime.createBuffer(65536*16);runtime.batch().dispatch(bind('caustic_clear',{photons}),[32,32,1]).dispatch(bind('caustic_map',{surface:flat,photons},{depth:1.4,rays,dispersion}),[rays/8,rays/8,1]).dispatch(bind('caustic_resolve',{photons,light:out},{normalization,dispersion}),[32,32,1]).submit();
  const a=await runtime.read(out);let error=0;for(let i=0;i<a.length;i++)if(i%4!==3)error=Math.max(error,Math.abs(a[i]-1));runtime.destroyBuffer(flat);runtime.destroyBuffer(out);kernels.caustic_map.clear();kernels.caustic_resolve.clear();return {maxDeviationFromUniform:error};
 });},
 async benchmark(samples=40){return exclusive(async()=>{if(!runtime.device.features.has('timestamp-query'))return {unsupported:true};playing=false;const qs=runtime.device.createQuerySet({type:'timestamp',count:2}),resolve=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const times=[];try{for(let i=0;i<samples+8;i++){compute(0,{querySet:qs,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1});const enc=runtime.device.createCommandEncoder();enc.resolveQuerySet(qs,0,2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,16);runtime.device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(read.getMappedRange());if(i>=8)times.push(Number(t[1]-t[0])/1e6);read.unmap();}times.sort((a,b)=>a-b);return {width,height,samples,medianMs:times[Math.floor(samples/2)],p95Ms:times[Math.floor(samples*.95)],scope:'GPU camera, spectrum, FFT, normals, caustics and render; excludes texture copy and browser presentation',adapter:runtime.describe()};}finally{qs.destroy();resolve.destroy();read.destroy();}});}
};
try{
 if(!navigator.gpu)throw Error('WebGPU is unavailable in this browser. Use a supported Chrome device, or Safari 26 or newer on iPhone, and open the HTTPS site.');
 runtime=await GpuRuntime.create({onError:fail});context=canvas.getContext('webgpu');
 const names=['camera_step','brush_pick','force_modes','seed_modes','spectrum','fft_stage','fft_local','resolve','caustic_clear','caustic_map','caustic_resolve','render'];
 const artifacts=new Map(await Promise.all(names.map(async name=>{const response=await fetch(`./kernels/${name}.json`);if(!response.ok)throw Error(`Kernel ${name}: HTTP ${response.status}`);return [name,await response.json()];})));
 for(const name of ['camera_step','brush_pick','force_modes','seed_modes','spectrum','fft_stage','fft_local','resolve','caustic_clear','caustic_map','caustic_resolve','render']){
  $('loadText').textContent=`Preparing ${name.replaceAll('_',' ')}…`;const artifact=artifacts.get(name);
  const kernel=await runtime.kernel(artifact),cache=new Map();kernels[name]={bind(buffers,scalars){const key=Object.values(buffers).map(b=>b.id).join(':');let v=cache.get(key);if(v)v.setScalars(scalars);else{v=kernel.bind(buffers,scalars);cache.set(key,v);}return v;},clear(){cache.clear();}};
 }
 twiddles=runtime.createBuffer(64*8);seed=runtime.createBuffer(49152*16);fft=[runtime.createBuffer(49152*8),runtime.createBuffer(49152*8)];surface=runtime.createBuffer(49152*16);light=runtime.createBuffer(65536*16);photons=runtime.createBuffer(65536*16);camera=runtime.createBuffer(80);brush=runtime.createBuffer(32);disturbance=runtime.createBuffer(16384*16);diagnostics.adapter=runtime.describe();requestAnimationFrame(frame);
}catch(e){fail(e);}
