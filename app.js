import {GpuRuntime} from './vendor/webcuda/runtime/runtime.js';
const assetVersion=new URL(import.meta.url).searchParams.get('v')||'local';
const $=id=>document.getElementById(id),canvas=$('water'),keys=new Set();
const touchDevice=matchMedia('(pointer:coarse)').matches||navigator.maxTouchPoints>0;
if(touchDevice){document.body.classList.add('touch','clean');$('quality').value='mobile';$('toggle').textContent='Settings';$('fly').textContent='Push water';$('touchMode').textContent='Mode: Look';}
let adaptiveScale=1,frameAverage=0,adaptCount=0,touchLook=touchDevice,lastWind=-1,lastDepth=-1,disturbanceActive=false,pcSampleOverride=0;
const pcQuality=()=>$('quality').value!=='mobile'&&($('quality').value!=='768'||pcSampleOverride===4);
const mapSize=()=>pcQuality()?512:256;
const mobileProfile=()=>$('quality').value==='mobile';
const diagnostics=window.waterDiagnostics={ready:false,errors:[],frames:0,readbackBytes:0};
let sandState,coefficients=null,runtime,context,kernels={},width=0,height=0,image,surface,light,monoLight,camera,fft,photons,disturbance,brush,seed,twiddles,motion;
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
const fullscreenButton=$('fullscreen'),fullscreenRoot=document.documentElement;
const fullscreenElement=()=>document.fullscreenElement||document.webkitFullscreenElement;
function fullscreenLabel(){const active=!!fullscreenElement();fullscreenButton.setAttribute('aria-pressed',String(active));fullscreenButton.setAttribute('aria-label',active?'Exit fullscreen':'Enter fullscreen');fullscreenButton.title=active?'Exit fullscreen':'Enter fullscreen';fullscreenButton.querySelector('.fullscreen-label').textContent=active?' Exit fullscreen':' Fullscreen';}
if(!(fullscreenRoot.requestFullscreen||fullscreenRoot.webkitRequestFullscreen)){fullscreenButton.disabled=true;fullscreenButton.title='Fullscreen is unavailable in this browser';fullscreenButton.setAttribute('aria-label',fullscreenButton.title);}
fullscreenButton.onclick=async()=>{try{if(fullscreenElement()){await (document.exitFullscreen||document.webkitExitFullscreen).call(document);}else{await (fullscreenRoot.requestFullscreen||fullscreenRoot.webkitRequestFullscreen).call(fullscreenRoot);}fullscreenLabel();}catch{fullscreenButton.title='Fullscreen could not be opened. Tap to try again.';}};
for(const event of ['fullscreenchange','webkitfullscreenchange'])document.addEventListener(event,fullscreenLabel);

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
const sticks={move:{x:0,y:0,pointer:null}};
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
 const lightChannels=mobile?1:3,lightSize=mapSize(),texels=lightSize*lightSize;
 if(diagnostics.lightChannels!==lightChannels||diagnostics.lightMapSize!==lightSize){
  runtime.destroyBuffer(light);runtime.destroyBuffer(monoLight);runtime.destroyBuffer(photons);
  if(coefficients){for(const buffer of coefficients)runtime.destroyBuffer(buffer);coefficients=null;}
  if(pcQuality()){coefficients=[runtime.createBuffer(49152*16),runtime.createBuffer(49152*16)];if(!sandState)sandState=runtime.createBuffer(16384*16);}
  kernels.surface_coefficients.clear();
  light=runtime.createBuffer(mobile?16:texels*16);monoLight=runtime.createBuffer(mobile?texels*4:4);photons=runtime.createBuffer(mobile?texels*4:texels*16);
  diagnostics.lightMapSize=lightSize;diagnostics.smoothSurface=pcQuality();diagnostics.lightChannels=lightChannels;diagnostics.lightStorageBytes=mobile?texels*4+16:texels*16+4;diagnostics.photonStorageBytes=mobile?texels*4:texels*16;
  kernels.caustic_clear.clear();kernels.caustic_map.clear();kernels.caustic_resolve.clear();kernels.render.clear();kernels.render_pc.clear();kernels.render_pc_single.clear();
 }

 const longest=mobile?960*adaptiveScale:Infinity;
 const desiredWidth=mobile?Math.min(innerWidth*devicePixelRatio,longest*Math.min(1,aspect)):Math.min(Number($('quality').value),innerWidth*devicePixelRatio);
 const w=Math.max(64,Math.floor(desiredWidth/64)*64),h=Math.max(8,Math.round(w/aspect/8)*8);
 diagnostics.mobile=mobile;diagnostics.targetFps=mobile?30:60;diagnostics.adaptiveScale=adaptiveScale;diagnostics.photonRays=mobile?256:(pcQuality()?1024:512);
 if(w===width&&h===height)return;
 if(image)runtime.destroyBuffer(image);width=w;height=h;canvas.width=w;canvas.height=h;image=runtime.createBuffer(w*h*4);
 context.configure({device:runtime.device,format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT,alphaMode:'opaque'});
 diagnostics.width=w;diagnostics.height=h;kernels.render.clear();kernels.render_pc.clear();kernels.render_pc_single.clear();
}
function bind(name,buffers,scalars={}){return kernels[name].bind(buffers,scalars);}
function compute(dt,timestampWrites){
 const b=runtime.batch({timestampWrites});
 const depth=Number($('depth').value),wind=Number($('wind').value),energy=Number($('energy').value),exposure=Number($('exposure').value),view=Number($('view').value),rays=mobileProfile()?256:(pcQuality()?1024:512),dispersion=mobileProfile()?0:1,lightSize=mapSize();
 const samples=mobileProfile()?1:(pcSampleOverride||($('quality').value==='768'?1:4));diagnostics.pixelSamples=samples;
 if(held&&forceMoved)disturbanceActive=true;
 const cascades=disturbanceActive?3:2;diagnostics.activeCascades=cascades;
 const axis=(a,z)=>(keys.has(a)?1:0)-(keys.has(z)?1:0);
 b.dispatch(bind('camera_step',{camera},{dt,forward:Math.max(-1,Math.min(1,axis('KeyW','KeyS')-sticks.move.y)),side:Math.max(-1,Math.min(1,axis('KeyD','KeyA')+sticks.move.x)),up:axis('KeyE','KeyQ'),lookX:lookX+axis('ArrowRight','ArrowLeft')*dt,lookY:lookY+axis('ArrowUp','ArrowDown')*dt,speed:speed*(keys.has('ShiftLeft')||keys.has('ShiftRight')?6:1),reset,depth}),[1,1,1]);reset=0;lookX=lookY=0;
 b.dispatch(bind('brush_pick',{surface,camera,brush},{pointerX,pointerY,aspect:width/height,held:held?1:0,moving:forceMoved?1:0,pressureActive:disturbanceActive?1:0}),[1,1,1]);forceMoved=false;
 if(depth!==lastDepth){b.dispatch(bind('prepare_modes',{motion},{depth}),[16,16,3]);lastDepth=depth;diagnostics.dispersionSeeds=(diagnostics.dispersionSeeds||0)+1;}
 if(disturbanceActive)b.dispatch(bind('force_modes',{disturbance,brush,motion},{dt:playing?dt:0,clear:0}),[16,16,1]);
 if(wind!==lastWind){b.dispatch(bind('seed_modes',{seed,twiddles},{wind}),[16,16,3]);lastWind=wind;diagnostics.spectrumSeeds=(diagnostics.spectrumSeeds||0)+1;}
 b.dispatch(bind('spectrum',{output:fft[0],seed,disturbance,motion},{time,energy}),[16,16,cascades]);
 for(let axis=0;axis<2;axis++)b.dispatch(bind('fft_local',{input:fft[axis],output:fft[1-axis],twiddles},{axis}),[128,1,cascades]);
 b.dispatch(bind('resolve',{input:fft[0],surface}),[16,16,cascades]);
 if(pcQuality()&&depth<3&&playing&&dt>0)b.dispatch(bind('sand_transport',{surface,sandState},{dt,depth,pressureActive:disturbanceActive?1:0}),[16,16,1]);
 if(pcQuality())for(let axis=0;axis<2;axis++)b.dispatch(bind('surface_coefficients',{input:axis===0?surface:coefficients[0],output:coefficients[axis]},{axis}),[16,16,cascades]);
 b.dispatch(bind('caustic_clear',{photons},{dispersion,lightSize}),[lightSize/8,lightSize/8,1]);
 b.dispatch(bind('caustic_map',{surface:pcQuality()?coefficients[1]:surface,photons},{depth,rays,dispersion,lightSize}),[rays/8,rays/8,1]);
 b.dispatch(bind('caustic_resolve',{photons,light,monoLight},{normalization:4096*(rays/lightSize)**2,dispersion,lightSize}),[lightSize/8,lightSize/8,1]);
 b.dispatch(bind(pcQuality()?(samples===4?'render_pc':'render_pc_single'):'render',{...(pcQuality()?{sandState}:{}),surface,coefficients:pcQuality()?coefficients[1]:surface,light,monoLight,camera,image},{width,height,depth,exposure,view,pressureActive:disturbanceActive?1:0,dispersion,lightSize}),[width/32,height/2,1]);
 b.endPass();b.encoder.copyBufferToTexture({buffer:image.gpuBuffer,bytesPerRow:width*4},{texture:context.getCurrentTexture()},[width,height]);b.submit();
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
 async setPixelSamples(count){if(count!==0&&count!==1&&count!==4)throw Error('Pixel samples must be 0 (automatic), 1 or 4');return exclusive(async()=>{pcSampleOverride=count;resize();compute(0);await runtime.idle();});},
 async lookAt(yaw,pitch){return exclusive(async()=>{runtime.device.queue.writeBuffer(camera.gpuBuffer,16,new Float32Array([yaw,pitch,0,0]));compute(0);await runtime.idle();});},
 async lookDown(){return exclusive(async()=>{const update=new Float32Array([0,-.95,0,0]);runtime.device.queue.writeBuffer(camera.gpuBuffer,16,update);compute(0);await runtime.idle();});},
 pause(){playing=false;},resume(){playing=true;},
 async inspect(){return exclusive(async()=>{const a=await runtime.read(surface),c=await runtime.read(camera);let max=0,imag=0,sum=0,disturbanceMax=0;for(let i=0;i<a.length;i+=4){max=Math.max(max,Math.abs(a[i]));imag=Math.max(imag,Math.abs(a[i+3]));sum+=a[i]*a[i];if(i>=32768*4)disturbanceMax=Math.max(disturbanceMax,Math.abs(a[i]));}const rawLight=await runtime.read(mobileProfile()?monoLight:light),l=mobileProfile()?Float32Array.from({length:65536*4},(_,i)=>i%4===3?1:rawLight[Math.floor(i/4)]):rawLight,d=await runtime.read(disturbance);let forceEnergy=0;for(const v of d)forceEnergy+=v*v;const causticMean=[0,0,0];let causticMin=Infinity,causticMax=-Infinity;for(let i=0;i<l.length;i+=4){causticMin=Math.min(causticMin,l[i]);causticMax=Math.max(causticMax,l[i]);for(let c=0;c<3;c++)causticMean[c]+=l[i+c]/(l.length/4);}return {forceEnergy,disturbanceMax,causticMean,finite:a.every(Number.isFinite)&&l.every(Number.isFinite),heightMax:max,heightRms:Math.sqrt(sum/(a.length/4)),imaginaryResidual:imag,camera:Array.from(c.slice(0,8)),causticMin,causticMax,adapter:runtime.describe()};});},
 async seek(t){return exclusive(async()=>{playing=false;time=t;resize();compute(0);await runtime.idle();});},
 async screenshot(){return exclusive(async()=>{const p=await runtime.read(image,Uint32Array);return {width,height,rgba:Array.from(new Uint8Array(p.buffer))};});},
 async sandTest(){return exclusive(async()=>{
  const zero=runtime.createBuffer(49152*16),states=[0,1,2,3].map(()=>runtime.createBuffer(16384*16));
  try{
   for(let group=0;group<4;group++){const batch=runtime.batch();
   for(let step=0;step<30;step++)for(let j=0;j<4;j++)batch.dispatch(bind('sand_transport',{surface:j===0?zero:surface,sandState:states[j]},{dt:j===3?0:1/30,depth:j===2?8:.5,pressureActive:disturbanceActive?1:0}),[16,16,1]);
   batch.submit();}const results=[];
   for(const state of states){const v=await runtime.read(state);let max=0,rms=0;for(let i=0;i<v.length;i+=4){max=Math.max(max,Math.abs(v[i]));rms+=v[i]*v[i];}results.push({finite:Array.from(v).every(Number.isFinite),max,rms:Math.sqrt(rms/16384)});}
   return {flat:results[0],shallow:results[1],deep:results[2],paused:results[3]};
  }finally{runtime.destroyBuffer(zero);for(const state of states)runtime.destroyBuffer(state);kernels.sand_transport.clear();}
 });},
 async bedTest(){return exclusive(async()=>{
  const count=4096,out=runtime.createBuffer(count*3*16);
  try{runtime.batch().dispatch(bind('bed_quality_probe',{output:out},{count}),[count/64,1,1]).submit();const data=await runtime.read(out);let stonePoints=0,maxStoneHeight=0,sandSlopeError=0,stoneSlopeError=0;
   for(let i=0;i<count;i++){const a=i*12;if(data[a+4]>.0005)stonePoints++;maxStoneHeight=Math.max(maxStoneHeight,data[a+4]);sandSlopeError=Math.max(sandSlopeError,Math.abs(data[a+1]-data[a+8]),Math.abs(data[a+2]-data[a+9]));stoneSlopeError=Math.max(stoneSlopeError,Math.abs(data[a+5]-data[a+10]),Math.abs(data[a+6]-data[a+11]));}
   return {finite:data.every(Number.isFinite),points:count,stonePoints,maxStoneHeight,sandSlopeError,stoneSlopeError,oracle:'Centered finite differences of actual geometric relief; analytic normals evaluated independently'};
  }finally{runtime.destroyBuffer(out);kernels.bed_quality_probe.clear();}
 });},
 async interpolationTest(){return exclusive(async()=>{
  // Independent analytic Fourier surface; never used for live simulation.
  const modes=[{x:13,z:7,a:.08},{x:5,z:-9,a:.04}],field=new Float32Array(49152*4),points=new Float32Array(64*4);
  const oracle=(x,z)=>{let h=0,dx=0,dz=0;for(const m of modes){const kx=2*Math.PI*m.x/6,kz=2*Math.PI*m.z/6,p=kx*x+kz*z;h+=m.a*Math.cos(p);dx-=m.a*kx*Math.sin(p);dz-=m.a*kz*Math.sin(p);}return [h,dx,dz];};
  for(let z=0;z<128;z++)for(let x=0;x<128;x++){const wx=x*6/128,wz=z*6/128,d=6/128,p=oracle(wx,wz),i=(z*128+x)*4;p[1]=(oracle(wx+d,wz)[0]-oracle(wx-d,wz)[0])/(2*d);p[2]=(oracle(wx,wz+d)[0]-oracle(wx,wz-d)[0])/(2*d);field.set(p,i);}
  for(let i=0;i<64;i++){points[i*4]=Math.fround((i*1.713-.8)*6/128);points[i*4+1]=Math.fround((i*.937+127.4)*6/128);}
  const input=runtime.createBuffer(field),positions=runtime.createBuffer(points),output=runtime.createBuffer(64*2*16),temporary=runtime.createBuffer(field.byteLength),filtered=runtime.createBuffer(field.byteLength);
  try{runtime.batch().dispatch(bind('surface_coefficients',{input,output:temporary},{axis:0}),[16,16,1]).dispatch(bind('surface_coefficients',{input:temporary,output:filtered},{axis:1}),[16,16,1]).dispatch(bind('sample_quality_probe',{surface:input,coefficients:filtered,points:positions,output},{count:64}),[1,1,1]).submit();const values=await runtime.read(output),errors=[{height:0,slope:0},{height:0,slope:0}];
   for(let i=0;i<64;i++){const ref=oracle(points[i*4],points[i*4+1]);for(let q=0;q<2;q++){const offset=(i*2+q)*4;errors[q].height+=(values[offset]-ref[0])**2;errors[q].slope+=(values[offset+1]-ref[1])**2+(values[offset+2]-ref[2])**2;}}
   for(const e of errors){e.height=Math.sqrt(e.height/64);e.slope=Math.sqrt(e.slope/128);}
   return {bilinear:errors[0],cubic:errors[1],finite:values.every(Number.isFinite),points:64,oracle:'Independent analytic Fourier height and derivatives, including periodic boundaries'};
  }finally{runtime.destroyBuffer(input);runtime.destroyBuffer(positions);runtime.destroyBuffer(output);runtime.destroyBuffer(temporary);runtime.destroyBuffer(filtered);kernels.sample_quality_probe.clear();kernels.surface_coefficients.clear();}
 });},
 async fftTest(strategy='shared'){return exclusive(async()=>{
  const data=new Float32Array(49152*2),modes=[{x:3,z:7,c:0,re:.3,im:-.2},{x:51,z:89,c:0,re:.07,im:.11},{x:11,z:125,c:1,re:-.4,im:.15},{x:37,z:11,c:2,re:.03,im:-.02}];
  const reverse=x=>{let r=0;for(let i=0;i<7;i++){r=r*2+(x&1);x>>=1;}return r;};
  for(const m of modes){const idx=(m.c*16384+reverse(m.z)*128+reverse(m.x))*2;data[idx]=m.re;data[idx+1]=m.im;}
  const a=runtime.createBuffer(data),b=runtime.createBuffer(data.byteLength);const batch=runtime.batch();let src=0;const pair=[a,b];
  if(strategy==='staged'){for(let axis=0;axis<2;axis++)for(let span=2;span<=128;span*=2){batch.dispatch(bind('fft_stage',{input:pair[src],output:pair[1-src]},{span,axis}),[16,16,3]);src=1-src;}}
  else{for(let axis=0;axis<2;axis++){batch.dispatch(bind('fft_local',{input:pair[src],output:pair[1-src],twiddles},{axis}),[128,1,3]);src=1-src;}}
  batch.submit();
  const result=await runtime.read(pair[src]);let maxError=0;
  for(let c=0;c<3;c++)for(let z=0;z<128;z++)for(let x=0;x<128;x++){let re=0,im=0;for(const m of modes){if(m.c!==c)continue;const angle=2*Math.PI*(m.x*x+m.z*z)/128;re+=m.re*Math.cos(angle)-m.im*Math.sin(angle);im+=m.re*Math.sin(angle)+m.im*Math.cos(angle);}const idx=(c*16384+z*128+x)*2;maxError=Math.max(maxError,Math.abs(result[idx]-re),Math.abs(result[idx+1]-im));}
  runtime.destroyBuffer(a);runtime.destroyBuffer(b);kernels.fft_stage.clear();kernels.fft_local.clear();return {maxError,strategy,modes:modes.length,axes:2,cascades:3,oracle:'Direct complex Fourier series evaluated independently in JavaScript for diagnostics only'};
 });},
 async flatCausticsTest(mobile=false){return exclusive(async()=>{
  const lightSize=mobile?256:mapSize(),texels=lightSize*lightSize,rays=mobile?256:(lightSize===512?1024:512),dispersion=mobile?0:1,normalization=4096*(rays/lightSize)**2;
  const flat=runtime.createBuffer(49152*16),out=runtime.createBuffer(texels*16),outMono=runtime.createBuffer(texels*4),testPhotons=runtime.createBuffer(mobile?texels*4:texels*16);runtime.batch().dispatch(bind('caustic_clear',{photons:testPhotons},{dispersion,lightSize}),[lightSize/8,lightSize/8,1]).dispatch(bind('caustic_map',{surface:flat,photons:testPhotons},{depth:1.4,rays,dispersion,lightSize}),[rays/8,rays/8,1]).dispatch(bind('caustic_resolve',{photons:testPhotons,light:out,monoLight:outMono},{normalization,dispersion,lightSize}),[lightSize/8,lightSize/8,1]).submit();
  const a=await runtime.read(mobile?outMono:out);let error=0;for(let i=0;i<a.length;i++)if(mobile||i%4!==3)error=Math.max(error,Math.abs(a[i]-1));runtime.destroyBuffer(flat);runtime.destroyBuffer(out);runtime.destroyBuffer(outMono);runtime.destroyBuffer(testPhotons);kernels.caustic_clear.clear();kernels.caustic_map.clear();kernels.caustic_resolve.clear();return {maxDeviationFromUniform:error};
 });},
 async benchmark(samples=40){return exclusive(async()=>{if(!runtime.device.features.has('timestamp-query'))return {unsupported:true};playing=false;const qs=runtime.device.createQuerySet({type:'timestamp',count:2}),resolve=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const times=[];try{for(let i=0;i<samples+8;i++){compute(0,{querySet:qs,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1});const enc=runtime.device.createCommandEncoder();enc.resolveQuerySet(qs,0,2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,16);runtime.device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(read.getMappedRange());if(i>=8)times.push(Number(t[1]-t[0])/1e6);read.unmap();}times.sort((a,b)=>a-b);return {width,height,samples,medianMs:times[Math.floor(samples/2)],p95Ms:times[Math.floor(samples*.95)],scope:'GPU camera, spectrum, FFT, normals, caustics and render; excludes texture copy and browser presentation',adapter:runtime.describe()};}finally{qs.destroy();resolve.destroy();read.destroy();}});}
};
try{
 if(!navigator.gpu)throw Error('WebGPU is unavailable in this browser. Use a supported Chrome device, or Safari 26 or newer on iPhone, and open the HTTPS site.');
 runtime=await GpuRuntime.create({onError:fail});context=canvas.getContext('webgpu');
 const names=['camera_step','brush_pick','force_modes','seed_modes','prepare_modes','spectrum','fft_stage','fft_local','resolve','caustic_clear','caustic_map','caustic_resolve','render','render_pc','render_pc_single','sample_quality_probe','surface_coefficients','bed_quality_probe','sand_transport'];
 const artifacts=new Map(await Promise.all(names.map(async name=>{const response=await fetch(`./kernels/${name}.json?v=${encodeURIComponent(assetVersion)}`);if(!response.ok)throw Error(`Kernel ${name}: HTTP ${response.status}`);return [name,await response.json()];})));
 for(const name of ['camera_step','brush_pick','force_modes','seed_modes','prepare_modes','spectrum','fft_stage','fft_local','resolve','caustic_clear','caustic_map','caustic_resolve','render','render_pc','render_pc_single','sample_quality_probe','surface_coefficients','bed_quality_probe','sand_transport']){
  $('loadText').textContent=`Preparing ${name.replaceAll('_',' ')}…`;const artifact=artifacts.get(name);
  const kernel=await runtime.kernel(artifact),cache=new Map();kernels[name]={bind(buffers,scalars){const key=Object.values(buffers).map(b=>b.id).join(':');let v=cache.get(key);if(v)v.setScalars(scalars);else{v=kernel.bind(buffers,scalars);cache.set(key,v);}return v;},clear(){cache.clear();}};
 }
 motion=runtime.createBuffer(81920*4);twiddles=runtime.createBuffer(64*8);seed=runtime.createBuffer(49152*16);fft=[runtime.createBuffer(49152*8),runtime.createBuffer(49152*8)];surface=runtime.createBuffer(49152*16);light=runtime.createBuffer(16);monoLight=runtime.createBuffer(4);photons=runtime.createBuffer(4);camera=runtime.createBuffer(80);brush=runtime.createBuffer(32);disturbance=runtime.createBuffer(16384*16);diagnostics.adapter=runtime.describe();requestAnimationFrame(frame);
}catch(e){fail(e);}
