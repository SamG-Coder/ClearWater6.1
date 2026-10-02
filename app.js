import {GpuRuntime} from './vendor/webcuda/runtime/runtime.js';
import {createGameShell} from './game-shell.js';
const assetVersion=new URL(import.meta.url).searchParams.get('v')||'local';
const $=id=>document.getElementById(id),canvas=$('water'),keys=new Set();
const touchDevice=matchMedia('(pointer:coarse)').matches||navigator.maxTouchPoints>0;
if(touchDevice){document.body.classList.add('touch','clean');$('quality').value='mobile';$('toggle').textContent='Settings';$('fly').textContent='Push water';$('touchMode').textContent='Mode: Look';}
let adaptiveScale=1,frameAverage=0,adaptCount=0,touchLook=touchDevice,lastWind=-1,lastDepth=-1,disturbanceActive=false,pcSampleOverride=0,telemetryBusy=false;
const pcQuality=()=>$('quality').value!=='mobile'&&($('quality').value!=='768'||pcSampleOverride===4);
const mapSize=()=>pcQuality()?512:256;
const mobileProfile=()=>$('quality').value==='mobile';
const diagnostics=window.waterDiagnostics={ready:false,errors:[],frames:0,readbackBytes:0};
let visitRain=false,visitTerrain=false;
let geometryBytes=0,profileSkip=0,cloudScratch,cloudStorageBytes=0;
let plates,geologySeed=-1,geologySize=0;const geologyOffset=131104+(touchDevice?1048576:4194304),terrainWidth=touchDevice?512:1024,terrainOffset=geologyOffset+Math.ceil((touchDevice?512*256:1024*512)*4/3)+32;
const nearTerrainWidth=touchDevice?256:512,nearTerrainOffset=terrainOffset+Math.ceil(terrainWidth*terrainWidth*4/3)+32;
let seaMemory,weatherClock=14*3600,weatherSeason=172,weatherDirty=true,weatherRefresh=0,weatherMapAt=-Infinity,weatherSkyAt=-Infinity,lastWeatherMode=-1,weatherSize=0,zoomTail=false;
let kernelArtifacts;const kernelLoads=new Map();
let navigation,zoomDelta=0,sandState,coefficients=null,runtime,context,kernels={},width=0,height=0,image,surface,light,monoLight,camera,fft,photons,disturbance,brush,seed,twiddles,motion;
let flyFallback=false,lockPending=false;
let gameShell=null,menuRedraw=true,flightProfile=0,playTime=0;
let reset=1,playing=true,speed=3,time=0,last=0,lookX=0,lookY=0,drag=false,held=false,forceMoved=false,pointerX=0,pointerY=0,busy=false,failed=false;
const params=new URLSearchParams(location.search),shipMode=params.get('mode')==='ship'||(!params.has('t')&&params.get('mode')!=='explorer');
let shipMesh,shipBounds,shipData,shipInspect=false,shipAction=0;
if(shipMode){document.title='ClearWater6.1 · Spacecraft flight';document.body.classList.add('ship-mode','clean');speed=40;touchLook=true;$('toggle').textContent='Settings';if(!touchDevice)$('quality').value=innerWidth>=3840?'3840':innerWidth>=2560?'2560':'1920';}
diagnostics.shipMode=shipMode;
const fixed=params.get('t');if(fixed!==null){time=Number(fixed);playing=false;}
$('pause').textContent=playing?'Pause':'Resume';
if(new URLSearchParams(location.search).get('weather')==='study')$('weatherMode').value='0';
for(const id of ['weatherMode','weatherRate'])$(id).onchange=()=>{weatherDirty=true;};
$('dayTime').oninput=()=>{weatherClock=Math.floor(weatherClock/86400)*86400+Number($('dayTime').value)*3600;weatherDirty=true;weatherRefresh=1;};
$('season').onchange=()=>{weatherSeason=Number($('season').value);weatherDirty=true;weatherRefresh=1;};
if(new URLSearchParams(location.search).get('geology')==='study')$('depthMode').value='0';
$('depthMode').onchange=()=>{lastDepth=-1;weatherDirty=true;};
$('worldSeed').onchange=()=>{weatherDirty=true;};
function fail(e){failed=true;diagnostics.errors.push(String(e.message||e));$('error').hidden=false;$('error').textContent=diagnostics.errors.at(-1);$('loading').hidden=true;gameShell?.error(diagnostics.errors.at(-1));console.error(e);}
function labels(){for(const id of ['depth','energy','wind','exposure'])$(id+'Value').textContent=Number($(id).value).toFixed(2)+(id==='depth'?' m':id==='wind'?' m/s':'');}labels();
for(const id of ['depth','energy','wind','exposure'])$(id).oninput=()=>{if(id==='depth'){$('depthMode').value='0';lastDepth=-1;weatherDirty=true;}labels();};
$('toggle').onclick=()=>{resetSticks();keys.clear();padPointers.clear();document.body.classList.toggle('clean');$('toggle').textContent=touchDevice?(document.body.classList.contains('clean')?'Settings':'Close settings'):(document.body.classList.contains('clean')?'Show controls ↙':'Hide controls ↗');};
$('pause').onclick=()=>{playing=!playing;$('pause').textContent=playing?'Pause':'Resume';};$('reset').onclick=()=>{reset=1;if(shipMode)shipAction=1;};
$('findRain').onclick=()=>{visitRain=true;reset=0;weatherDirty=true;weatherRefresh=1;};
$('highlands').onclick=()=>{visitTerrain=true;reset=0;$('depthMode').value='1';weatherDirty=true;};
$('space').onclick=()=>{if(shipMode)shipAction=3;else reset=3;zoomDelta=0;};
$('shipMoon').onclick=()=>{shipAction=4;weatherDirty=true;canvas.focus({preventScroll:true});};
$('shipSpace').onclick=()=>{shipAction=3;weatherDirty=true;canvas.focus({preventScroll:true});};
$('shipSurface').onclick=()=>{reset=1;shipAction=1;weatherDirty=true;canvas.focus({preventScroll:true});};
$('shipInspect').onclick=()=>{shipInspect=!shipInspect;$('shipInspect').textContent=shipInspect?'Return to flight · V':'Inspect ship · V';$('shipModeLabel').textContent=shipInspect?'INSPECTING CRAFT':'THIRD-PERSON FLIGHT';weatherDirty=true;canvas.focus({preventScroll:true});};
$('shipFly').onclick=()=>$('fly').click();
function flyStatus(message){$('flightStatus').hidden=!message;$('flightStatus').textContent=message;}
function lockRefused(error){lockPending=false;flyFallback=true;diagnostics.pointerLock='drag';diagnostics.pointerLockReason=String(error?.message||'Browser refused mouse lock');$('fly').textContent='Retry mouse lock';$('fly').setAttribute('aria-pressed','true');flyStatus(shipMode?'Mouse lock was blocked. Drag to steer; W thrusts, S brakes, wheel changes speed.':'Mouse lock was blocked. Drag the water to look; WASD flies, wheel changes speed. Esc exits.');}
$('fly').onclick=async()=>{
 if(gameShell?.open)return;
 if(touchDevice){if(shipMode){$('shipInspect').click();return;}touchLook=!touchLook;$('fly').textContent=touchLook?'Push water':'Look around';$('touchMode').textContent=touchLook?'Mode: Look':'Mode: Water';return;}
 if(document.pointerLockElement===canvas){document.exitPointerLock();return;}
 if(lockPending)return;
 canvas.focus({preventScroll:true});
 if(!canvas.requestPointerLock){lockRefused(new Error('Mouse lock is unavailable in this browser'));return;}
 lockPending=true;
 try{await canvas.requestPointerLock();}catch(error){lockRefused(error);}
};
document.addEventListener('pointerlockerror',()=>lockRefused());
document.addEventListener('pointerlockchange',()=>{
 lockPending=false;const locked=document.pointerLockElement===canvas;flyFallback=false;drag=held=forceMoved=false;keys.clear();
 diagnostics.pointerLock=locked?'locked':'off';$('fly').textContent=locked?'Flying · Esc to exit':'Fly camera ↗';$('fly').setAttribute('aria-pressed',String(locked));
 $('shipFly').textContent=locked?'Release mouse · Esc':'Lock mouse ↗';flyStatus(locked?(shipMode?'Mouse locked · W thrust · S brake · Mouse steers · Wheel changes speed · Esc releases':'Mouse locked · WASD to fly · Wheel up / down changes speed · Esc releases'):'');
});
$('touchMode').onclick=()=>$('fly').click();
const fullscreenButton=$('fullscreen'),fullscreenRoot=document.documentElement;
const fullscreenElement=()=>document.fullscreenElement||document.webkitFullscreenElement;
function fullscreenLabel(){const active=!!fullscreenElement();fullscreenButton.setAttribute('aria-pressed',String(active));fullscreenButton.setAttribute('aria-label',active?'Exit fullscreen':'Enter fullscreen');fullscreenButton.title=active?'Exit fullscreen':'Enter fullscreen';fullscreenButton.querySelector('.fullscreen-label').textContent=active?' Exit fullscreen':' Fullscreen';}
if(!(fullscreenRoot.requestFullscreen||fullscreenRoot.webkitRequestFullscreen)){fullscreenButton.disabled=true;fullscreenButton.title='Fullscreen is unavailable in this browser';fullscreenButton.setAttribute('aria-label',fullscreenButton.title);}
fullscreenButton.onclick=async()=>{try{if(fullscreenElement()){await (document.exitFullscreen||document.webkitExitFullscreen).call(document);}else{await (fullscreenRoot.requestFullscreen||fullscreenRoot.webkitRequestFullscreen).call(fullscreenRoot);}fullscreenLabel();}catch{fullscreenButton.title='Fullscreen could not be opened. Tap to try again.';}};
for(const event of ['fullscreenchange','webkitfullscreenchange'])document.addEventListener(event,fullscreenLabel);

$('quality').onchange=()=>{adaptiveScale=1;frameAverage=0;adaptCount=0;};
for(const id of ['shallows','ocean'])$(id).onclick=()=>{const ocean=id==='ocean';$('depth').value=ocean?8:1.4;$('energy').value=ocean?1.8:.8;reset=ocean?2:1;for(const p of ['shallows','ocean'])$(p).classList.toggle('active',p===id);labels();};
addEventListener('keydown',e=>{if(gameShell?.open)return;if(shipMode&&e.code==='KeyV'&&!e.repeat&&!['INPUT','SELECT'].includes(document.activeElement.tagName))$('shipInspect').click();if(e.code==='Escape'){document.exitPointerLock?.();drag=false;held=false;forceMoved=false;flyFallback=false;lockPending=false;flyStatus('');if(!touchDevice){$('fly').textContent='Fly camera ↗';$('fly').setAttribute('aria-pressed','false');}}if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;keys.add(e.code);if(e.code==='KeyH')$('toggle').click();if(e.code.startsWith('Arrow')||e.code==='Space')e.preventDefault();});
addEventListener('keyup',e=>keys.delete(e.code));addEventListener('blur',()=>{keys.clear();drag=false;held=false;});
function pointer(e){const r=canvas.getBoundingClientRect(),x=2*(e.clientX-r.left)/r.width-1,y=1-2*(e.clientY-r.top)/r.height;if(x!==pointerX||y!==pointerY)forceMoved=true;pointerX=x;pointerY=y;}
canvas.addEventListener('contextmenu',e=>e.preventDefault());
const zoomTouches=new Map();let pinchGap=0;
let touchPointer=null,touchLastX=0,touchLastY=0;
function endPointer(e){zoomTouches.delete(e.pointerId);if(zoomTouches.size<2)pinchGap=0;if(touchPointer===null||e.pointerId===touchPointer){drag=false;held=false;forceMoved=false;touchPointer=null;}}
canvas.addEventListener('pointerdown',e=>{
 if(gameShell?.open)return;
 if(e.pointerType==='touch'){zoomTouches.set(e.pointerId,{x:e.clientX,y:e.clientY});if(zoomTouches.size===2){const [a,b]=[...zoomTouches.values()];pinchGap=Math.hypot(a.x-b.x,a.y-b.y);held=drag=forceMoved=false;touchPointer=null;canvas.setPointerCapture(e.pointerId);return;}}
 if(e.pointerType==='touch'&&touchPointer!==null)return;
 pointer(e);forceMoved=false;
 if(e.pointerType==='touch'){touchPointer=e.pointerId;drag=touchLook;held=!touchLook;touchLastX=e.clientX;touchLastY=e.clientY;}
 else{canvas.focus({preventScroll:true});if(shipMode){held=false;drag=true;}else{held=e.button===0&&!flyFallback;drag=e.button===2||(e.button===0&&flyFallback);}}
 canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener('pointermove',e=>{
 if(gameShell?.open)return;
 if(e.pointerType==='touch'&&zoomTouches.has(e.pointerId)){zoomTouches.set(e.pointerId,{x:e.clientX,y:e.clientY});if(zoomTouches.size===2){const [a,b]=[...zoomTouches.values()],gap=Math.hypot(a.x-b.x,a.y-b.y);if(pinchGap>0&&gap>0)zoomDelta=Math.max(-8,Math.min(8,zoomDelta+Math.log(pinchGap/gap)*3));pinchGap=gap;return;}}
 if(e.pointerType==='touch'){
  if(e.pointerId!==touchPointer)return;
  if(drag){lookX+=(e.clientX-touchLastX)*.004*(gameShell?.sensitivity??1);lookY-=(e.clientY-touchLastY)*.004*(gameShell?.sensitivity??1)*(gameShell?.invertY?-1:1);}
  touchLastX=e.clientX;touchLastY=e.clientY;
 }
 if(document.pointerLockElement!==canvas)pointer(e);
});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,endPointer);
addEventListener('mousemove',e=>{if(gameShell?.open)return;if((drag&&touchPointer===null)||document.pointerLockElement===canvas){lookX+=e.movementX*.0025*(gameShell?.sensitivity??1);lookY-=e.movementY*.0025*(gameShell?.sensitivity??1)*(gameShell?.invertY?-1:1);if(document.pointerLockElement===canvas){pointerX=0;pointerY=0;}}});
const padPointers=new Map();
const sticks={move:{x:0,y:0,pointer:null}};
function resetSticks(){for(const [name,stick] of Object.entries(sticks)){stick.x=stick.y=0;stick.pointer=null;const el=$(name+'Stick');el.classList.remove('active');el.querySelector('.stick-knob').style.transform='translate(0px,0px)';}}
for(const [name,stick] of Object.entries(sticks)){
 const el=$(name+'Stick'),knob=el.querySelector('.stick-knob');
 const update=e=>{const r=el.getBoundingClientRect(),radius=r.width*.32,dx=e.clientX-r.x-r.width/2,dy=e.clientY-r.y-r.height/2,length=Math.hypot(dx,dy),scale=length>radius?radius/length:1;
  const amount=Math.max(0,(Math.min(1,length/radius)-.12)/.88);stick.x=length?dx/length*amount:0;stick.y=length?dy/length*amount:0;
  knob.style.transform=`translate(${dx*scale}px,${dy*scale}px)`;
 };
 el.addEventListener('pointerdown',e=>{if(gameShell?.open)return;if(stick.pointer!==null)return;e.preventDefault();stick.pointer=e.pointerId;el.setPointerCapture(e.pointerId);el.classList.add('active');update(e);});
 el.addEventListener('pointermove',e=>{if(e.pointerId===stick.pointer)update(e);});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])el.addEventListener(event,e=>{if(e.pointerId!==stick.pointer)return;stick.x=stick.y=0;stick.pointer=null;el.classList.remove('active');knob.style.transform='translate(0px,0px)';});
}
for(const button of document.querySelectorAll('[data-move]')){
 button.addEventListener('pointerdown',e=>{if(gameShell?.open)return;e.preventDefault();padPointers.set(e.pointerId,button.dataset.move);keys.add(button.dataset.move);button.setPointerCapture(e.pointerId);button.classList.add('active');});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])button.addEventListener(event,e=>{const key=padPointers.get(e.pointerId);padPointers.delete(e.pointerId);if(key&&![...padPointers.values()].includes(key))keys.delete(key);button.classList.remove('active');});
}
addEventListener('blur',()=>{touchPointer=null;zoomTouches.clear();pinchGap=0;padPointers.clear();resetSticks();});
document.addEventListener('visibilitychange',()=>{last=0;held=drag=false;keys.clear();touchPointer=null;zoomTouches.clear();pinchGap=0;padPointers.clear();resetSticks();});
addEventListener('resize',resetSticks);
canvas.addEventListener('wheel',e=>{e.preventDefault();if(gameShell?.open)return;const delta=Math.max(-1000,Math.min(1000,e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?innerHeight:1)));if((shipMode||document.pointerLockElement===canvas||flyFallback)&&!e.ctrlKey){speed=flightProfile?Math.min(1,Math.max(.1,speed*Math.exp(-delta*.002))):Math.min(1000000,Math.max(.2,speed*Math.exp(-delta*.002)));gameShell?.updateThrottle(speed);menuRedraw=true;diagnostics.flightSpeed=undefined;if(shipMode)$('shipLimit').textContent=formatDistance(speed)+'/s';updateMetrics();}else{zoomDelta=Math.max(-8,Math.min(8,zoomDelta+delta*.003));}},{passive:false});
function resize(){
 const mobile=mobileProfile(),aspect=innerWidth/innerHeight;
 const lightChannels=mobile?1:3,lightSize=mapSize(),texels=lightSize*lightSize;
 if(diagnostics.lightChannels!==lightChannels||diagnostics.lightMapSize!==lightSize){
  runtime.destroyBuffer(light);runtime.destroyBuffer(monoLight);geometryBytes=0;runtime.destroyBuffer(photons);
  if(coefficients){for(const buffer of coefficients)runtime.destroyBuffer(buffer);coefficients=null;}
  if(pcQuality()){coefficients=[runtime.createBuffer(49152*16),runtime.createBuffer(49152*16)];if(!sandState)sandState=runtime.createBuffer(16384*16);}
  kernels.surface_coefficients.clear();
  light=runtime.createBuffer(mobile?16:texels*16);monoLight=runtime.createBuffer(mobile?texels*4:4);photons=runtime.createBuffer(mobile?texels*4:texels*16);
  diagnostics.lightMapSize=lightSize;diagnostics.smoothSurface=pcQuality();diagnostics.lightChannels=lightChannels;diagnostics.lightStorageBytes=mobile?texels*4+16:texels*16+4;diagnostics.photonStorageBytes=mobile?texels*4:texels*16;
  kernels.caustic_clear.clear();kernels.caustic_map.clear();kernels.caustic_resolve.clear();kernels.render.clear();kernels.render_pc.clear();kernels.render_pc_single.clear();
 }

 const longest=mobile?960*adaptiveScale:Infinity;
 const desiredWidth=mobile?Math.min(innerWidth*devicePixelRatio,longest*Math.min(1,aspect)):Math.min(Number($('quality').value),innerWidth*devicePixelRatio);
 const samples=mobile?1:(pcSampleOverride||($('quality').value==='768'?1:4));
 const pixelLimit=Math.floor((runtime.device.limits.maxStorageBufferBindingSize-(mobile?texels*4:0))/(samples*4+.125));
 const w=Math.max(64,Math.floor(Math.min(desiredWidth,Math.sqrt((pixelLimit-8192)*aspect))/64)*64),h=Math.max(8,Math.round(w/aspect/8)*8);
 const needed=(w*h*samples+2*Math.ceil(w/8)*Math.ceil(h/8)+(mobile?texels:0))*4;
 if(needed!==geometryBytes){runtime.destroyBuffer(monoLight);monoLight=runtime.createBuffer(needed);geometryBytes=needed;diagnostics.geometryStorageBytes=needed-(mobile?texels*4:0);kernels.caustic_resolve.clear();kernels.terrain_intersections.clear();kernels.terrain_tile_heights.clear();kernels.weather_cloud_view.clear();kernels.render.clear();kernels.render_pc.clear();kernels.render_pc_single.clear();}
 const cloudScale=Math.min(.5,(touchDevice||mobile?960:1920)/Math.max(w,h)),cloudWidth=Math.max(8,Math.ceil(w*cloudScale/8)*8),cloudHeight=Math.max(8,Math.ceil(h*cloudScale/8)*8),cloudBytes=cloudWidth*cloudHeight*16;
 if(cloudBytes!==cloudStorageBytes){if(cloudScratch)runtime.destroyBuffer(cloudScratch);cloudScratch=runtime.createBuffer(cloudBytes);cloudStorageBytes=cloudBytes;kernels.weather_cloud_filter.clear();}
 diagnostics.cloudWidth=cloudWidth;diagnostics.cloudHeight=cloudHeight;diagnostics.cloudFilterBytes=cloudBytes;diagnostics.nearTerrainSpacing=2048/nearTerrainWidth;diagnostics.nearTerrainBytes=nearTerrainWidth*nearTerrainWidth*32+128;
 diagnostics.mobile=mobile;diagnostics.targetFps=mobile?30:60;diagnostics.adaptiveScale=adaptiveScale;diagnostics.photonRays=mobile?256:(pcQuality()?1024:512);
 if(w===width&&h===height)return;weatherDirty=true;
 if(image)runtime.destroyBuffer(image);width=w;height=h;canvas.width=w;canvas.height=h;image=runtime.createBuffer(w*h*4);
 context.configure({device:runtime.device,format:'rgba8unorm',usage:GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT,alphaMode:'opaque'});
 diagnostics.width=w;diagnostics.height=h;kernels.ship_render.clear();kernels.render.clear();kernels.render_pc.clear();kernels.render_pc_single.clear();
 updateMetrics();
}
function qualityKernels(){return [...(mobileProfile()?['render']:pcQuality()?['render_pc','surface_coefficients','sand_transport']:['render']),...(shipMode?['ship_step','ship_wash_pick','ship_wash_modes','ship_render','ship_mesh','ship_bounds']:[])];}
async function ensureKernels(names){
 const missing=names.filter(name=>!kernels[name]?.loaded);if(!missing.length)return;
 await Promise.all(missing.map(name=>{
  if(kernelLoads.has(name))return kernelLoads.get(name);
  const promise=(async()=>{const kernel=await runtime.kernel(kernelArtifacts.get(name)),cache=new Map();kernels[name]={loaded:true,bind(buffers,scalars){const key=Object.values(buffers).map(b=>b.id).join(':');let v=cache.get(key);if(v)v.setScalars(scalars);else{v=kernel.bind(buffers,scalars);cache.set(key,v);}return v;},clear(){cache.clear();}};diagnostics.compiledKernels=(diagnostics.compiledKernels||0)+1;})();
  kernelLoads.set(name,promise);return promise;
 }));
}
function bind(name,buffers,scalars={}){return kernels[name].bind(buffers,scalars);}
function compute(dt,timestampWrites){
 const b=runtime.batch({timestampWrites});
 const geoWidth=touchDevice?512:1024,geoSeed=Math.max(1,Math.min(999999,Math.floor(Number($('worldSeed').value)||1))),geoEnabled=Number($('depthMode').value);
 if(geoSeed!==geologySeed||geoWidth!==geologySize){
  b.dispatch(bind('geology_seed',{plates,camera},{seedValue:geoSeed,mapWidth:geoWidth,offset:geologyOffset}),[1,1,1]);
  b.dispatch(bind('geology_map',{plates,camera},{mapWidth:geoWidth,offset:geologyOffset,seedValue:geoSeed}),[geoWidth/8,geoWidth/16,1]);
  for(let level=1;geoWidth/2**(level+1)>=1;level++)b.dispatch(bind('geology_mip',{camera},{width:geoWidth,level}),[Math.ceil(geoWidth/2**level/8),Math.ceil(geoWidth/2**(level+1)/8),1]);
  geologySeed=geoSeed;geologySize=geoWidth;weatherDirty=true;diagnostics.geologyBuilds=(diagnostics.geologyBuilds||0)+1;
 }
 diagnostics.geologyEnabled=geoEnabled!==0;diagnostics.geologyMapSize=geoWidth;diagnostics.worldSeed=geoSeed;
 const weatherEnabled=Number($('weatherMode').value),mapResolution=mobileProfile()?128:256,skyWidth=mobileProfile()?256:512,weatherDt=playing?dt*Number($('weatherRate').value):0;
 weatherClock+=weatherDt;
 const moving=(shipMode&&playing)||reset!==0||lookX!==0||lookY!==0||zoomDelta!==0||zoomTail||keys.size>0||sticks.move.x!==0||sticks.move.y!==0;
 if(zoomDelta!==0)zoomTail=true;
 if(lastWeatherMode!==weatherEnabled||weatherSize!==mapResolution){weatherDirty=true;weatherRefresh=1;}
 const updateMap=weatherDirty||weatherClock-weatherMapAt>=60||weatherClock<weatherMapAt;
 if(updateMap){b.dispatch(bind('weather_map',{camera},{clock:weatherClock,season:weatherSeason,mapSize:mapResolution}),[mapResolution/8,mapResolution/16,1]);weatherMapAt=weatherClock;diagnostics.weatherMapUpdates=(diagnostics.weatherMapUpdates||0)+1;}

 const depth=Number($('depth').value),wind=Number($('wind').value),energy=Number($('energy').value),exposure=Number($('exposure').value),view=Number($('view').value),rays=mobileProfile()?256:(pcQuality()?1024:512),dispersion=mobileProfile()?0:1,lightSize=mapSize();
 const samples=mobileProfile()?1:(pcSampleOverride||($('quality').value==='768'?1:4));diagnostics.pixelSamples=samples;
 if(shipMode||(held&&forceMoved))disturbanceActive=true;
 const cascades=disturbanceActive?3:2;diagnostics.activeCascades=cascades;
 const axis=(a,z)=>(keys.has(a)?1:0)-(keys.has(z)?1:0);
 if(visitRain){b.dispatch(bind('weather_visit',{camera,navigationState:navigation},{latitude:0,longitude:0,altitude:5,findRain:1,mapSize:mapResolution}),[1,1,1]);visitRain=false;weatherDirty=true;}
 if(visitTerrain){b.dispatch(bind('geology_visit',{camera,navigationState:navigation},{targetDepth:-2500,clock:weatherClock,season:weatherSeason}),[1,1,1]);visitTerrain=false;weatherDirty=true;}
 if(geoEnabled&&(reset===1||reset===2)){b.dispatch(bind('geology_visit',{camera,navigationState:navigation},{targetDepth:reset===1?1.4:4500,clock:weatherClock,season:weatherSeason}),[1,1,1]);reset=0;weatherDirty=true;}
 if(shipMode)b.dispatch(bind('ship_step',{camera,navigationState:navigation,ship:shipData},{deltaTime:playing?dt:0,forward:Math.max(-1,Math.min(1,axis('KeyW','KeyS')-sticks.move.y)),turn:Math.max(-1,Math.min(1,axis('KeyD','KeyA')+sticks.move.x)),rise:Math.max(-1,Math.min(1,axis('KeyE','KeyQ')+(keys.has('Space')?1:0))),lookX:lookX+axis('ArrowRight','ArrowLeft')*dt,lookY:lookY+axis('ArrowUp','ArrowDown')*dt,speed,boost:keys.has('ShiftLeft')||keys.has('ShiftRight')?1:0,action:shipAction,inspect:shipInspect?1:0,flightProfile,depth,aspect:width/height}),[1,1,1]);
 else b.dispatch(bind('camera_step',{camera,navigationState:navigation},{dt,forward:Math.max(-1,Math.min(1,axis('KeyW','KeyS')-sticks.move.y)),side:Math.max(-1,Math.min(1,axis('KeyD','KeyA')+sticks.move.x)),up:axis('KeyE','KeyQ'),lookX:lookX+axis('ArrowRight','ArrowLeft')*dt,lookY:lookY+axis('ArrowUp','ArrowDown')*dt,speed:speed*(keys.has('ShiftLeft')||keys.has('ShiftRight')?6:1),zoom:zoomDelta,reset,depth}),[1,1,1]);
 shipAction=0;zoomDelta=0;reset=0;lookX=lookY=0;
 b.dispatch(bind('terrain_cache_setup',{camera},{width:terrainWidth,offset:terrainOffset,enabled:geoEnabled,nearWidth:nearTerrainWidth,nearOffset:nearTerrainOffset}),[1,1,1]);
 b.dispatch(bind('terrain_cache',{camera},{width:terrainWidth}),[terrainWidth/8,terrainWidth/8,1]);
 for(let level=1;terrainWidth/2**level>=1;level++)b.dispatch(bind('terrain_cache_mip',{camera},{width:terrainWidth,level}),[Math.ceil(terrainWidth/2**level/8),Math.ceil(terrainWidth/2**level/8),1]);
 b.dispatch(bind('terrain_near_cache',{camera},{width:nearTerrainWidth}),[nearTerrainWidth/8,nearTerrainWidth/8,1]);
 b.dispatch(bind('terrain_near_coefficients',{camera},{width:nearTerrainWidth}),[nearTerrainWidth/8,nearTerrainWidth/8,1]);
 b.dispatch(bind('geology_update',{camera},{depth,enabled:geoEnabled}),[1,1,1]);
 b.dispatch(bind('terrain_camera_frame',{camera,navigationState:navigation}),[1,1,1]);
 b.dispatch(bind('weather_update',{camera},{clock:weatherClock,season:weatherSeason,time,dt:weatherDt,baseWind:wind,enabled:weatherEnabled,mapSize:mapResolution,skyWidth,refresh:weatherRefresh}),[1,1,1]);
 if(geoEnabled&&!(profileSkip&4))b.dispatch(bind('terrain_tile_heights',{camera,hits:monoLight},{width,height,samples,offset:mobileProfile()?lightSize*lightSize:0}),[Math.ceil(width/64),Math.ceil(height/64),1]);
 if(weatherEnabled&&(weatherDirty||moving||time-weatherSkyAt>=.1||time<weatherSkyAt)){
  b.dispatch(bind('weather_sky',{camera},{skyWidth}),[skyWidth/8,skyWidth/32,1]);
  const {cloudWidth,cloudHeight}=diagnostics;
  b.dispatch(bind('weather_cloud_view',{camera,hits:monoLight},{cloudWidth,cloudHeight,aspect:width/height}),[cloudWidth/8,cloudHeight/8,1]);diagnostics.cloudWidth=cloudWidth;diagnostics.cloudHeight=cloudHeight;weatherSkyAt=time;diagnostics.weatherSkyUpdates=(diagnostics.weatherSkyUpdates||0)+1;
  for(let axis=0;axis<2;axis++)b.dispatch(bind('weather_cloud_filter',{camera,scratch:cloudScratch},{width:cloudWidth,height:cloudHeight,axis}),[cloudWidth/8,cloudHeight/8,1]);
 }
 weatherDirty=false;weatherRefresh=0;lastWeatherMode=weatherEnabled;weatherSize=mapResolution;diagnostics.weatherClock=weatherClock;diagnostics.weatherEnabled=weatherEnabled===1;
 if(shipMode)b.dispatch(bind('ship_wash_pick',{camera,ship:shipData,brush}),[1,1,1]);
 else b.dispatch(bind('brush_pick',{surface,camera,brush},{pointerX,pointerY,aspect:width/height,held:held?1:0,moving:forceMoved?1:0,pressureActive:disturbanceActive?1:0}),[1,1,1]);forceMoved=false;
 if(geoEnabled||depth!==lastDepth){b.dispatch(bind('prepare_modes',{motion,camera},{depth,time}),[16,16,3]);lastDepth=depth;diagnostics.dispersionSeeds=(diagnostics.dispersionSeeds||0)+1;}
 if(shipMode)b.dispatch(bind('ship_wash_modes',{disturbance,brush,motion,ship:shipData},{dt:playing?dt:0}),[16,16,1]);
 else if(disturbanceActive)b.dispatch(bind('force_modes',{disturbance,brush,motion},{dt:playing?dt:0,clear:0}),[16,16,1]);
 if(wind!==lastWind){b.dispatch(bind('seed_modes',{seed,twiddles},{wind}),[16,16,3]);lastWind=wind;diagnostics.spectrumSeeds=(diagnostics.spectrumSeeds||0)+1;}
 b.dispatch(bind('spectrum',{output:fft[0],seed,disturbance,motion,camera,seaMemory},{time,energy}),[16,16,cascades]);
 for(let axis=0;axis<2;axis++)b.dispatch(bind('fft_local',{input:fft[axis],output:fft[1-axis],twiddles},{axis}),[128,1,cascades]);
 b.dispatch(bind('resolve',{input:fft[0],surface}),[16,16,cascades]);
 if(pcQuality()&&(geoEnabled||depth<3)&&playing&&dt>0)b.dispatch(bind('sand_transport',{brush,surface,camera,sandState},{dt,depth,pressureActive:disturbanceActive?1:0,useGlobeDepth:1}),[16,16,1]);
 if(pcQuality())for(let axis=0;axis<2;axis++)b.dispatch(bind('surface_coefficients',{input:axis===0?surface:coefficients[0],output:coefficients[axis]},{axis}),[16,16,cascades]);
 b.dispatch(bind('caustic_clear',{photons},{dispersion,lightSize}),[lightSize/8,lightSize/8,1]);
 b.dispatch(bind('caustic_map',{surface:pcQuality()?coefficients[1]:surface,photons,camera},{depth,rays,dispersion,lightSize}),[rays/8,rays/8,1]);
 b.dispatch(bind('caustic_resolve',{photons,light,monoLight},{normalization:4096*(rays/lightSize)**2,dispersion,lightSize}),[lightSize/8,lightSize/8,1]);
 if(geoEnabled&&!(profileSkip&1))b.dispatch(bind('terrain_intersections',{camera,hits:monoLight},{width,height,samples,offset:mobileProfile()?lightSize*lightSize:0,view}),[width/8,height/8,samples]);
 if(!(profileSkip&2))b.dispatch(bind(pcQuality()?(samples===4?'render_pc':'render_pc_single'):'render',{brush,...(pcQuality()?{sandState}:{}),surface,coefficients:pcQuality()?coefficients[1]:surface,light,monoLight,camera,image},{width,height,depth,exposure,view,pressureActive:disturbanceActive?1:0,dispersion,lightSize}),[width/32,height/2,1]);
 if(shipMode&&!(profileSkip&8))b.dispatch(bind('ship_render',{mesh:shipMesh,bounds:shipBounds,camera,ship:shipData,image},{width,height,samples,exposure}),[width/32,height/2,1]);
 b.endPass();b.encoder.copyBufferToTexture({buffer:image.gpuBuffer,bytesPerRow:width*4},{texture:context.getCurrentTexture()},[width,height]);b.submit();
}
function updateMetrics(){if(shipMode){if(gameShell&&!shipInspect)$('shipModeLabel').textContent=flightProfile?`NORMAL GAME · ${(diagnostics.shipAltitude??22)>=80000?'SPACE FLIGHT':'ATMOSPHERIC FLIGHT'}`:'FREE ROAM · UNRESTRICTED';$('shipSpeed').textContent=formatDistance(diagnostics.flightSpeed||0)+'/s';$('shipAltitude').textContent=formatDistance(diagnostics.shipAltitude??22);$('shipLimit').textContent=formatDistance(flightProfile?(diagnostics.thrustLimit||0):speed)+'/s';}if(diagnostics.ready)$('metrics').textContent=`${Math.round(diagnostics.fps)} FPS · ${width} × ${height} · ${formatDistance(diagnostics.altitude||2.6)} altitude · ${formatDistance(diagnostics.flightSpeed??speed)}/s`;}
function weatherLabels(){const w=diagnostics.localWeather;if(!w)return;const hour=Math.floor(w.hour),minute=Math.floor((w.hour-hour)*60);$('weatherStatus').textContent=`${String(hour).padStart(2,'0')}:${String(minute).padStart(2,'0')} local · ${w.wind.toFixed(1)} m/s · ${w.rain>.1?'Rain':w.cloud>.5?'Cloudy':'Fair'}`;if(document.activeElement!==$('dayTime'))$('dayTime').value=(weatherClock/3600)%24;}
function formatDistance(value){return value>=1000?(value/1000).toFixed(value>=100000?0:1)+' km':value.toFixed(1)+' m';}
async function frame(now){
 if(failed)return;
 try{
  const mobile=mobileProfile(),interval=mobile?1000/30:0;
  if(!busy&&!document.hidden&&(!gameShell?.open||menuRedraw)&&(!last||now-last>=interval-.5)){
   menuRedraw=false;
   while(qualityKernels().some(name=>!kernels[name]?.loaded)){if(diagnostics.ready){$('loading').hidden=false;$('loadText').textContent='Preparing this quality profile…';}await ensureKernels(qualityKernels());}
   const elapsed=last?(now-last)/1000:1/(mobile?30:60),dt=Math.min(.1,elapsed);last=now;resize();if(playing){time+=dt;if(gameShell&&!gameShell.open)playTime+=dt;}
   const start=performance.now();compute(dt);await runtime.idle();
   diagnostics.frameMs=performance.now()-start;diagnostics.fps=1/elapsed;diagnostics.frames++;const firstReady=!diagnostics.ready;diagnostics.ready=true;if(firstReady)gameShell?.ready();diagnostics.readbackBytes=runtime.stats.readbackBytes;$('loading').hidden=true;
   frameAverage=frameAverage?frameAverage*.94+diagnostics.frameMs*.06:diagnostics.frameMs;
   if(mobile&&++adaptCount>=60){if(frameAverage>25&&adaptiveScale>.5)adaptiveScale=Math.max(.5,adaptiveScale-.1);else if(frameAverage<12&&adaptiveScale<1)adaptiveScale=Math.min(1,adaptiveScale+.05);adaptCount=0;}
   if(((shipMode&&diagnostics.frames===1)||diagnostics.frames%15===0)&&!telemetryBusy){telemetryBusy=true;runtime.read(camera,Float32Array,256,128).then(v=>{diagnostics.altitude=v[0];diagnostics.flightSpeed=v[1];if(shipMode){diagnostics.shipAltitude=v[44];diagnostics.thrustLimit=v[3];}zoomTail=!shipMode&&Math.abs(v[3])>.00001;diagnostics.localWeather={wind:v[10],cloud:v[11],rain:v[12],sunlight:v[13],temperature:v[14],pressure:v[15],hour:v[39]};diagnostics.oceanDepth=v[52];$('geologyStatus').textContent=`${formatDistance(v[53]>0?v[53]:v[52])} ${v[53]>0?'elevation':'deep'} · seed ${geologySeed}`;weatherLabels();updateMetrics();}).catch(fail).finally(()=>telemetryBusy=false);}
   if(diagnostics.frames===1||diagnostics.frames%15===0)updateMetrics();
   if(gameShell)await gameShell.tick(now);
  }
  requestAnimationFrame(frame);
 }catch(e){fail(e);}
}
async function exclusive(fn){busy=true;try{await runtime.idle();await ensureKernels(qualityKernels());return await fn();}finally{busy=false;}}
window.waterLab={
 sessionState(){return {shell:gameShell?.state,playing,time,clock:weatherClock,playTime,flightProfile,throttle:speed,frames:diagnostics.frames};},
 async shipCost(samples=30){let full,water;try{profileSkip=0;full=await waterLab.benchmark(samples);profileSkip=8;water=await waterLab.benchmark(samples);}finally{profileSkip=0;}return {full,water,scope:'Sequential full-scene measurements with and without the ship render pass; not an isolated per-pass timing'};},
 async shipState(){return exclusive(async()=>{const data=await runtime.read(shipData);return {data:Array.from(data),position:Array.from(data.slice(0,3)),angles:Array.from(data.slice(4,7)),speed:data[39],clearance:data[14],finite:data.every(Number.isFinite),triangles:71680,components:126};});},
 async shipView(yaw=.34,elevation=.3,distance=22){return exclusive(async()=>{runtime.device.queue.writeBuffer(shipData.gpuBuffer,32,new Float32Array([yaw,elevation,distance,1]));shipInspect=true;weatherDirty=true;compute(0);await runtime.idle();const c=await runtime.read(camera,Float32Array,320);diagnostics.altitude=c[1];diagnostics.shipAltitude=c[76];diagnostics.flightSpeed=c[77];updateMetrics();});},
 async shipAltitude(metres){return exclusive(async()=>{runtime.device.queue.writeBuffer(shipData.gpuBuffer,4,new Float32Array([metres]));weatherDirty=true;compute(0);await runtime.idle();const state=await runtime.read(shipData);diagnostics.shipAltitude=state[1];diagnostics.flightSpeed=state[39];updateMetrics();});},
 async shipWashState(){return exclusive(async()=>{const data=await runtime.read(brush);return {pointer:Array.from(data.slice(0,4)),domain:Array.from(data.slice(8,12)),shift:Array.from(data.slice(12,16)),engines:[Array.from(data.slice(16,20)),Array.from(data.slice(20,24))]};});},
 async celestialState(){return exclusive(async()=>{const c=await runtime.read(camera,Float32Array,320);return {moon:[c[31],c[39],c[75]],sun:Array.from(c.slice(48,51)),eye:c.slice(24,27).map(x=>x*(6371000+c[1])),radius:1737400};});},
 async celestialSamples(points){return exclusive(async()=>{await ensureKernels(['celestial_probe']);const input=runtime.createBuffer(Float32Array.from(points.flat())),output=runtime.createBuffer(points.length*16);try{runtime.batch().dispatch(bind('celestial_probe',{camera,points:input,output},{count:points.length}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(output));}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);kernels.celestial_probe.clear();}});},
 async shipEffectSamples(points){return exclusive(async()=>{await ensureKernels(['ship_effect_probe']);const input=runtime.createBuffer(Float32Array.from(points.flat())),output=runtime.createBuffer(points.length*16);try{runtime.batch().dispatch(bind('ship_effect_probe',{camera,ship:shipData,brush,points:input,output},{count:points.length}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(output));}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);kernels.ship_effect_probe.clear();}});},
 // Deterministic diagnostic stepping uses the same input/dispatch path as play.
 async shipAdvance(frames=60,dt=1/60,input={}){return exclusive(async()=>{
  playing=true;shipInspect=false;keys.clear();sticks.move.x=input.turn||0;sticks.move.y=-(input.forward||0);
  if(input.rise)keys.add(input.rise>0?'KeyE':'KeyQ');if(input.boost)keys.add('ShiftLeft');
  try{for(let i=0;i<Math.min(600,frames);i++){lookX=(input.yawRate||0)*dt;lookY=(input.pitchRate||0)*dt;time+=dt;compute(dt);if(i%16===15)await runtime.idle();}await runtime.idle();}
  finally{playing=false;keys.clear();sticks.move.x=sticks.move.y=0;lookX=lookY=0;}
  const state=await runtime.read(shipData);diagnostics.shipAltitude=state[1];diagnostics.flightSpeed=state[39];updateMetrics();
 });},
 async shipRayTest(){return exclusive(async()=>{
  await ensureKernels(['ship_probe']);const mesh=await runtime.read(shipMesh),rays=[];
  for(let i=0;i<36;i++){const a=i*2.399963,origin=[Math.sin(a)*21,4+(i%3)*4,Math.cos(a)*21],target=[(i%5-2)*2,(i%4-1)*1.2,(i%7-3)*1.5],delta=target.map((v,j)=>v-origin[j]),length=Math.hypot(...delta);rays.push({origin,ray:delta.map(v=>v/length)});}
  const input=runtime.createBuffer(new Float32Array(rays.flatMap(({origin,ray})=>[...origin,0,...ray,0]))),out=runtime.createBuffer(rays.length*16);
  try{runtime.batch().dispatch(bind('ship_probe',{mesh:shipMesh,bounds:shipBounds,rays:input,output:out},{count:rays.length}),[1,1,1]).submit();const gpu=await runtime.read(out);let maxError=0,hits=0,misses=0;
   for(let i=0;i<rays.length;i++){const {origin:o,ray:d}=rays[i];let closest=100000;
    for(let t=0;t<71680;t++){const j=t*24,ex=mesh[j+4]-mesh[j],ey=mesh[j+5]-mesh[j+1],ez=mesh[j+6]-mesh[j+2],fx=mesh[j+8]-mesh[j],fy=mesh[j+9]-mesh[j+1],fz=mesh[j+10]-mesh[j+2],px=d[1]*fz-d[2]*fy,py=d[2]*fx-d[0]*fz,pz=d[0]*fy-d[1]*fx,det=ex*px+ey*py+ez*pz;if(Math.abs(det)<1e-8)continue;const dx=o[0]-mesh[j],dy=o[1]-mesh[j+1],dz=o[2]-mesh[j+2],u=(dx*px+dy*py+dz*pz)/det;if(u<0||u>1)continue;const qx=dy*ez-dz*ey,qy=dz*ex-dx*ez,qz=dx*ey-dy*ex,v=(d[0]*qx+d[1]*qy+d[2]*qz)/det;if(v<0||u+v>1)continue;const distance=(fx*qx+fy*qy+fz*qz)/det;if(distance>.001&&distance<closest)closest=distance;}
    if(closest<100000)hits++;else misses++;maxError=Math.max(maxError,Math.abs(closest-gpu[i*4]));
   }return {rays:rays.length,hits,misses,maxError,finite:gpu.every(Number.isFinite)};
  }finally{runtime.destroyBuffer(input);runtime.destroyBuffer(out);kernels.ship_probe.clear();}
 });},
 async shipGeometry(){return exclusive(async()=>{const vertices=await runtime.read(shipMesh),bounds=await runtime.read(shipBounds);let triangles=0;for(let t=0;t<71680;t++){const a=t*24,e=[vertices[a+4]-vertices[a],vertices[a+5]-vertices[a+1],vertices[a+6]-vertices[a+2]],f=[vertices[a+8]-vertices[a],vertices[a+9]-vertices[a+1],vertices[a+10]-vertices[a+2]];if(Math.hypot(e[1]*f[2]-e[2]*f[1],e[2]*f[0]-e[0]*f[2],e[0]*f[1]-e[1]*f[0])>1e-8)triangles++;}let symmetryError=0;for(let component=0;component<48;component++)for(let t=0;t<512;t++)for(let v=0;v<3;v++){const l=(6144+(component+32)*512+t)*24+v*4,r=(6144+(component+80)*512+t)*24+v*4;symmetryError=Math.max(symmetryError,Math.abs(vertices[l]+vertices[r]),Math.abs(vertices[l+1]-vertices[r+1]),Math.abs(vertices[l+2]-vertices[r+2]));}const finBounds=[43,91].map(i=>Array.from(bounds.slice((255+1024+i*85)*8,(255+1024+i*85)*8+8)));return {finite:vertices.every(Number.isFinite)&&bounds.every(Number.isFinite),triangles,symmetryError,finBounds,bounds:Array.from(bounds.slice(0,8))};});},
 async terrainScreenTest(points){return exclusive(async()=>{await ensureKernels(['terrain_screen_probe']);const input=runtime.createBuffer(new Float32Array(points.flat())),out=runtime.createBuffer(points.length*16);try{runtime.batch().dispatch(bind('terrain_screen_probe',{points:input,camera,hits:monoLight,output:out},{count:points.length}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(out));}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(out);kernels.terrain_screen_probe.clear();}});},
 async nearTerrainState(){return exclusive(async()=>({header:Array.from(await runtime.read(camera,Float32Array,64,nearTerrainOffset*16)),spacing:2048/nearTerrainWidth,bytes:diagnostics.nearTerrainBytes}));},
 async terrainBounds(enabled){return exclusive(async()=>{runtime.device.queue.writeBuffer(camera.gpuBuffer,508,new Float32Array([enabled?0:1]));weatherDirty=true;compute(0);await runtime.idle();});},
 async profileStages(){const out={};try{for(const [name,mask] of [['full',0],['withoutTerrainIntersections',1],['withoutShading',2],['withoutTerrainTiles',4]]){profileSkip=mask;out[name]=await waterLab.benchmark(20);}}finally{profileSkip=0;}return out;},
 async terrainCache(enabled){return exclusive(async()=>{runtime.device.queue.writeBuffer(camera.gpuBuffer,496,new Float32Array([enabled?0:1]));weatherDirty=true;compute(0);await runtime.idle();});},
 async terrainAt(points,rays=false){return exclusive(async()=>{await ensureKernels(["terrain_probe"]);const input=runtime.createBuffer(new Float32Array(points.flat())),out=runtime.createBuffer(points.length*16);try{runtime.batch().dispatch(bind('terrain_probe',{points:input,camera,output:out},{count:points.length,rays:rays===true?1:Number(rays)||0}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(out));}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(out);kernels.terrain_probe.clear();}});},
 async waveModes(){return exclusive(async()=>{const m=await runtime.read(motion);return [1,129,16385,16513].map(i=>{const j=i%16384,x=j%128,z=Math.floor(j/128),k=2*Math.PI*Math.hypot(x,z)/(i<16384?6:96);return {i,k,omega:m[i],phase:m[i]*time+m[81920+i]};});});},
 async geologyState(){return exclusive(async()=>{const c=await runtime.read(camera,Float32Array,64,336);return {depth:c[0],elevation:c[1],crust:c[2],enabled:!!c[3],offset:c[4],mapWidth:c[5],seed:c[6],plateCount:c[7],boundary:c[10],plate:c[11],finite:c.every(Number.isFinite),builds:diagnostics.geologyBuilds};});},
 async geologyAt(points){return exclusive(async()=>{await ensureKernels(["geology_probe"]);const input=runtime.createBuffer(new Float32Array(points.flat())),out=runtime.createBuffer(points.length*48);try{runtime.batch().dispatch(bind('geology_probe',{points:input,plates,camera,output:out},{count:points.length,seedValue:geologySeed}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(out));}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(out);kernels.geology_probe.clear();}});},
 async plateData(){return exclusive(async()=>Array.from(await runtime.read(plates)));},
 async weatherLocation(latitude,longitude,altitude=5){return exclusive(async()=>{runtime.batch().dispatch(bind('weather_visit',{camera,navigationState:navigation},{latitude,longitude,altitude,findRain:0,mapSize:weatherSize}),[1,1,1]).submit();weatherDirty=true;weatherRefresh=1;compute(0);await runtime.idle();});},
 async weatherState(){return exclusive(async()=>{const c=await runtime.read(camera,Float32Array,320),memory=await runtime.read(seaMemory);let fast=0,slow=0;for(let i=0;i<16384;i++){fast+=memory[i]/16384;slow+=memory[i+16384]/16384;}return {clock:weatherClock,season:weatherSeason,enabled:!!c[55],wind:Array.from(c.slice(40,43)),cloud:c[43],rain:c[44],sunlight:c[45],temperature:c[46],pressure:c[47],sun:Array.from(c.slice(48,51)),localSun:Array.from(c.slice(36,39)),frame:{east:Array.from(c.slice(20,23)),normal:Array.from(c.slice(24,27)),back:Array.from(c.slice(28,31))},latitude:c[70],localHour:c[71],energy:[fast,slow],finite:c.every(Number.isFinite)&&memory.every(Number.isFinite),mapUpdates:diagnostics.weatherMapUpdates,skyUpdates:diagnostics.weatherSkyUpdates};});},
 async weatherAt(points){return exclusive(async()=>{await ensureKernels(["weather_probe"]);const input=runtime.createBuffer(new Float32Array(points.flat())),out=runtime.createBuffer(points.length*64);try{runtime.batch().dispatch(bind('weather_probe',{points:input,output:out,camera},{count:points.length,season:weatherSeason}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(out));}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(out);kernels.weather_probe.clear();}});},
 async weatherSeek(seconds,season=172){return exclusive(async()=>{playing=false;weatherClock=seconds;weatherSeason=season;weatherDirty=true;weatherRefresh=1;compute(0);await runtime.idle();});},
 async weatherAdvance(seconds,steps=120){return exclusive(async()=>{playing=false;for(let i=0;i<steps;i++){weatherClock+=seconds/steps;time+=1/60;weatherDirty=true;compute(0);runtime.batch().dispatch(bind('weather_update',{camera},{clock:weatherClock,season:weatherSeason,time,dt:seconds/steps,baseWind:Number($('wind').value),enabled:Number($('weatherMode').value),mapSize:weatherSize,skyWidth:mobileProfile()?256:512,refresh:0}),[1,1,1]).dispatch(bind('spectrum',{output:fft[0],seed,disturbance,motion,camera,seaMemory},{time,energy:Number($('energy').value)}),[16,16,2]).submit();}weatherDirty=true;compute(0);await runtime.idle();});},

 async geometryTest(points){return exclusive(async()=>{await ensureKernels(["planet_probe"]);const input=runtime.createBuffer(new Float32Array(points.flat())),out=runtime.createBuffer(points.length*16);try{runtime.batch().dispatch(bind('planet_probe',{points:input,output:out},{count:points.length}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(out));}finally{runtime.destroyBuffer(input);runtime.destroyBuffer(out);kernels.planet_probe.clear();}});},
 async planetState(){return exclusive(async()=>{const c=await runtime.read(camera,Float32Array,320),raw=await runtime.read(navigation);diagnostics.altitude=c[1];diagnostics.flightSpeed=c[33];updateMetrics();return {altitude:c[0*4+1],radius:c[8*4+2],speed:c[8*4+1],normal:Array.from(c.slice(24,27)),east:Array.from(c.slice(20,23)),back:Array.from(c.slice(28,31)),camera:Array.from(c.slice(0,8)),navigation:Array.from({length:8},(_,i)=>raw[i*2]+raw[i*2+1]),finite:c.every(Number.isFinite)};});},
 async setAltitude(meters){return exclusive(async()=>{runtime.device.queue.writeBuffer(camera.gpuBuffer,4,new Float32Array([meters]));weatherDirty=true;compute(0);await runtime.idle();});},
 async flightTest(steps,seconds,forward,side,up,speed){return exclusive(async()=>{for(let i=0;i<steps;i++)runtime.batch().dispatch(bind('camera_step',{camera,navigationState:navigation},{dt:seconds/steps,forward,side,up,lookX:0,lookY:0,speed,zoom:0,reset:0,depth:Number($('depth').value)}),[1,1,1]).submit();weatherDirty=true;compute(0);await runtime.idle();});},
 async setPixelSamples(count){if(count!==0&&count!==1&&count!==4)throw Error('Pixel samples must be 0 (automatic), 1 or 4');return exclusive(async()=>{await ensureKernels(["render_pc_single", "render_pc", "surface_coefficients", "sand_transport"]);pcSampleOverride=count;resize();weatherDirty=true;compute(0);await runtime.idle();});},
 async lookAt(yaw,pitch){return exclusive(async()=>{runtime.device.queue.writeBuffer(camera.gpuBuffer,16,new Float32Array([yaw,pitch,0,0]));weatherDirty=true;compute(0);await runtime.idle();});},
 async lookDown(){return exclusive(async()=>{const update=new Float32Array([0,-.95,0,0]);runtime.device.queue.writeBuffer(camera.gpuBuffer,16,update);weatherDirty=true;compute(0);await runtime.idle();});},
 pause(){playing=false;},resume(){playing=true;},
 async inspect(){return exclusive(async()=>{const a=await runtime.read(surface),c=await runtime.read(camera,Float32Array,320);let max=0,imag=0,sum=0,disturbanceMax=0;for(let i=0;i<a.length;i+=4){max=Math.max(max,Math.abs(a[i]));imag=Math.max(imag,Math.abs(a[i+3]));sum+=a[i]*a[i];if(i>=32768*4)disturbanceMax=Math.max(disturbanceMax,Math.abs(a[i]));}const rawLight=await runtime.read(mobileProfile()?monoLight:light,Float32Array,mapSize()*mapSize()*(mobileProfile()?4:16)),l=mobileProfile()?Float32Array.from({length:65536*4},(_,i)=>i%4===3?1:rawLight[Math.floor(i/4)]):rawLight,d=await runtime.read(disturbance);let forceEnergy=0;for(const v of d)forceEnergy+=v*v;const causticMean=[0,0,0];let causticMin=Infinity,causticMax=-Infinity;for(let i=0;i<l.length;i+=4){causticMin=Math.min(causticMin,l[i]);causticMax=Math.max(causticMax,l[i]);for(let c=0;c<3;c++)causticMean[c]+=l[i+c]/(l.length/4);}return {forceEnergy,disturbanceMax,causticMean,finite:a.every(Number.isFinite)&&l.every(Number.isFinite),heightMax:max,heightRms:Math.sqrt(sum/(a.length/4)),imaginaryResidual:imag,camera:Array.from(c.slice(0,8)),causticMin,causticMax,adapter:runtime.describe()};});},
 async seek(t){return exclusive(async()=>{playing=false;time=t;resize();weatherDirty=true;compute(0);await runtime.idle();});},
 async screenshot(){return exclusive(async()=>{const p=await runtime.read(image,Uint32Array);return {width,height,rgba:Array.from(new Uint8Array(p.buffer))};});},
 async domainTest(){return exclusive(async()=>{await ensureKernels(["domain_probe"]);
  const domain=await runtime.read(brush),cx=domain[8],cz=domain[9],count=532,points=new Float32Array(count*4);
  for(let i=0;i<256;i++){const x=cx+(i%16-7.5)*.75,z=cz+(Math.floor(i/16)-7.5)*.75;points.set([x,z,0,0],i*4);points.set([x+24,z,0,0],(i+256)*4);}
  const centers=[[24,3],[3,24],[cx+7,cz],[cx+12,cz]];for(let j=0;j<4;j++)for(const [k,dx,dz] of [[0,0,0],[1,.001,0],[2,-.001,0],[3,0,.001],[4,0,-.001]])points.set([centers[j][0]+dx,centers[j][1]+dz,0,0],(512+j*5+k)*4);
  const input=runtime.createBuffer(points),output=runtime.createBuffer(count*4*16);
  try{runtime.batch().dispatch(bind('domain_probe',{surface,coefficients:pcQuality()?coefficients[1]:surface,brush,points:input,output},{count}),[9,1,1]).submit();const values=await runtime.read(output);const local=[0,0],ghost=[0,0];let untaperedGhost=0,regionalDifference=0;
   for(let i=0;i<256;i++){for(let j=0;j<2;j++){local[j]=Math.max(local[j],Math.abs(values[i*16+j*4]));ghost[j]=Math.max(ghost[j],Math.abs(values[(i+256)*16+j*4]));}untaperedGhost=Math.max(untaperedGhost,Math.abs(values[(i+256)*16+12]));regionalDifference=Math.max(regionalDifference,Math.abs(values[i*16+8]-values[(i+256)*16+8]));}
   let seamSlopeError=0;for(let j=0;j<4;j++)for(const offset of [4,8]){const i=512+j*5,sx=(values[(i+1)*16+offset]-values[(i+2)*16+offset])/.002,sz=(values[(i+3)*16+offset]-values[(i+4)*16+offset])/.002;seamSlopeError=Math.max(seamSlopeError,Math.abs(sx-values[i*16+offset+1]),Math.abs(sz-values[i*16+offset+2]));}
   return {seamSlopeError,finite:values.every(Number.isFinite),local,ghost,untaperedGhost,regionalDifference,domain:[cx,cz]};
  }finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);kernels.domain_probe.clear();}
 });},
 async sandTest(){return exclusive(async()=>{await ensureKernels(["sand_transport"]);
  const zero=runtime.createBuffer(49152*16),states=[0,1,2,3].map(()=>runtime.createBuffer(16384*16));
  try{
   for(let group=0;group<4;group++){const batch=runtime.batch();
   for(let step=0;step<30;step++)for(let j=0;j<4;j++)batch.dispatch(bind('sand_transport',{brush,surface:j===0?zero:surface,camera,sandState:states[j]},{dt:j===3?0:1/30,depth:j===2?8:.5,pressureActive:disturbanceActive?1:0,useGlobeDepth:0}),[16,16,1]);
   batch.submit();}const results=[];
   for(const state of states){const v=await runtime.read(state);let max=0,rms=0;for(let i=0;i<v.length;i+=4){max=Math.max(max,Math.abs(v[i]));rms+=v[i]*v[i];}results.push({finite:Array.from(v).every(Number.isFinite),max,rms:Math.sqrt(rms/16384)});}
   return {flat:results[0],shallow:results[1],deep:results[2],paused:results[3]};
  }finally{runtime.destroyBuffer(zero);for(const state of states)runtime.destroyBuffer(state);kernels.sand_transport.clear();}
 });},
 async patternSamples(points){return exclusive(async()=>{await ensureKernels(['appearance_probe']);
  const input=runtime.createBuffer(new Float32Array(points.flat())),output=runtime.createBuffer(points.length*16);
  try{runtime.batch().dispatch(bind('appearance_probe',{coefficients:coefficients[1],camera,points:input,output},{count:points.length}),[Math.ceil(points.length/64),1,1]).submit();return Array.from(await runtime.read(output));}
  finally{runtime.destroyBuffer(input);runtime.destroyBuffer(output);kernels.appearance_probe.clear();}
 });},
 async bedTest(){return exclusive(async()=>{await ensureKernels(["bed_quality_probe"]);
  const count=4096,out=runtime.createBuffer(count*3*16);
  try{runtime.batch().dispatch(bind('bed_quality_probe',{output:out},{count}),[count/64,1,1]).submit();const data=await runtime.read(out);let stonePoints=0,maxStoneHeight=0,sandSlopeError=0,stoneSlopeError=0;
   for(let i=0;i<count;i++){const a=i*12;if(data[a+4]>.0005)stonePoints++;maxStoneHeight=Math.max(maxStoneHeight,data[a+4]);sandSlopeError=Math.max(sandSlopeError,Math.abs(data[a+1]-data[a+8]),Math.abs(data[a+2]-data[a+9]));stoneSlopeError=Math.max(stoneSlopeError,Math.abs(data[a+5]-data[a+10]),Math.abs(data[a+6]-data[a+11]));}
   return {finite:data.every(Number.isFinite),points:count,stonePoints,maxStoneHeight,sandSlopeError,stoneSlopeError,oracle:'Centered finite differences of actual geometric relief; analytic normals evaluated independently'};
  }finally{runtime.destroyBuffer(out);kernels.bed_quality_probe.clear();}
 });},
 async interpolationTest(){return exclusive(async()=>{await ensureKernels(["sample_quality_probe", "surface_coefficients"]);
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
 async fftTest(strategy='shared'){return exclusive(async()=>{await ensureKernels(["fft_stage"]);
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
  const flat=runtime.createBuffer(49152*16),out=runtime.createBuffer(texels*16),outMono=runtime.createBuffer(texels*4),testPhotons=runtime.createBuffer(mobile?texels*4:texels*16);runtime.batch().dispatch(bind('caustic_clear',{photons:testPhotons},{dispersion,lightSize}),[lightSize/8,lightSize/8,1]).dispatch(bind('caustic_map',{surface:flat,photons:testPhotons,camera},{depth:1.4,rays,dispersion,lightSize}),[rays/8,rays/8,1]).dispatch(bind('caustic_resolve',{photons:testPhotons,light:out,monoLight:outMono},{normalization,dispersion,lightSize}),[lightSize/8,lightSize/8,1]).submit();
  const a=await runtime.read(mobile?outMono:out);let error=0;for(let i=0;i<a.length;i++)if(mobile||i%4!==3)error=Math.max(error,Math.abs(a[i]-1));runtime.destroyBuffer(flat);runtime.destroyBuffer(out);runtime.destroyBuffer(outMono);runtime.destroyBuffer(testPhotons);kernels.caustic_clear.clear();kernels.caustic_map.clear();kernels.caustic_resolve.clear();return {maxDeviationFromUniform:error};
 });},
 async benchmark(samples=40,animate=false){return exclusive(async()=>{if(!runtime.device.features.has('timestamp-query'))return {unsupported:true};playing=animate;const qs=runtime.device.createQuerySet({type:'timestamp',count:2}),resolve=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),read=runtime.device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const times=[];try{for(let i=0;i<samples+8;i++){if(animate)time+=1/60;compute(animate?1/60:0,{querySet:qs,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1});const enc=runtime.device.createCommandEncoder();enc.resolveQuerySet(qs,0,2,resolve,0);enc.copyBufferToBuffer(resolve,0,read,0,16);runtime.device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const t=new BigUint64Array(read.getMappedRange());if(i>=8)times.push(Number(t[1]-t[0])/1e6);read.unmap();}times.sort((a,b)=>a-b);return {width,height,samples,animated:animate,medianMs:times[Math.floor(samples/2)],p95Ms:times[Math.floor(samples*.95)],scope:'GPU camera, global weather, sky cache when due, spectrum, FFT, normals, caustics and render; excludes texture copy and browser presentation',adapter:runtime.describe()};}finally{qs.destroy();resolve.destroy();read.destroy();}});}
};
function clearGameInput(){
 keys.clear();lookX=lookY=zoomDelta=0;drag=held=forceMoved=flyFallback=lockPending=false;touchPointer=null;zoomTouches.clear();pinchGap=0;padPointers.clear();resetSticks();
 for(const el of document.querySelectorAll('[data-move]'))el.classList.remove('active');
 if(document.pointerLockElement===canvas)document.exitPointerLock();flyStatus('');
}
async function startSession(kind,saved){
 return exclusive(async()=>{
  playing=false;clearGameInput();flightProfile=kind==='normal'?1:0;speed=saved?.speed??(flightProfile ? .65 : 40);playTime=saved?.playTime??0;
  time=saved?.time??0;weatherClock=saved?.clock??14*3600;weatherSeason=saved?.season??172;
  for(const [id,value] of Object.entries({depthMode:1,worldSeed:saved?.seed??61,depth:1.4,energy:.8,wind:5,view:0,weatherMode:1,weatherRate:60,season:weatherSeason}))$(id).value=String(value);
  labels();shipInspect=false;$('shipInspect').textContent='Inspect ship · V';$('shipModeLabel').textContent=flightProfile?'NORMAL GAME · ATMOSPHERIC FLIGHT':'FREE ROAM · UNRESTRICTED';
  // Only small state/header buffers are reset. Terrain caches rebuild on demand.
  const enc=runtime.device.createCommandEncoder();
  for(const buffer of [navigation,shipData,brush,disturbance,seaMemory,motion,...(sandState?[sandState]:[])])enc.clearBuffer(buffer.gpuBuffer);
  enc.clearBuffer(camera.gpuBuffer,0,512);enc.clearBuffer(camera.gpuBuffer,nearTerrainOffset*16,128);runtime.device.queue.submit([enc.finish()]);
  geologySeed=-1;geologySize=0;lastDepth=lastWind=lastWeatherMode=-1;weatherMapAt=weatherSkyAt=-Infinity;weatherDirty=true;weatherRefresh=1;visitRain=visitTerrain=false;zoomTail=false;
  reset=saved?0:1;shipAction=saved?0:1;
  if(saved){runtime.device.queue.writeBuffer(navigation.gpuBuffer,0,new Float32Array(saved.navigation));runtime.device.queue.writeBuffer(shipData.gpuBuffer,0,new Float32Array(saved.ship));runtime.device.queue.writeBuffer(camera.gpuBuffer,0,new Float32Array(saved.camera));}
  resize();compute(0);await runtime.idle();menuRedraw=true;gameShell.updateThrottle(speed);
  const c=await runtime.read(camera,Float32Array,256,128);diagnostics.shipAltitude=c[44];diagnostics.flightSpeed=c[1];diagnostics.thrustLimit=c[3];updateMetrics();
 });
}
if(shipMode&&fixed===null){
 gameShell=createGameShell({touch:touchDevice,labels,notice:flyStatus,
  pause(){playing=false;clearGameInput();last=0;},
  resume(){clearGameInput();playing=true;last=0;canvas.focus({preventScroll:true});},
  redraw(){menuRedraw=true;weatherDirty=true;},
  throttle(value){speed=value;gameShell.updateThrottle(speed);},
  start:startSession,
  async snapshot(){return exclusive(async()=>{const [nav,ship,header]=await Promise.all([runtime.read(navigation),runtime.read(shipData),runtime.read(camera,Float32Array,320)]);return {navigation:Array.from(nav),ship:Array.from(ship),camera:Array.from(header),time,clock:weatherClock,season:weatherSeason,seed:geologySeed,speed,playTime};});}
 });
 labels();playing=false;shipAction=3;shipInspect=true;
 addEventListener('resize',()=>{menuRedraw=true;});
 $('worldControls').addEventListener('click',()=>{menuRedraw=true;});
}
try{
 if(!navigator.gpu)throw Error('WebGPU is unavailable in this browser. Use a supported Chrome device, or Safari 26 or newer on iPhone, and open the HTTPS site.');
 runtime=await GpuRuntime.create({onError:fail});context=canvas.getContext('webgpu');
 const names=['camera_step','brush_pick','force_modes','seed_modes','prepare_modes','spectrum','fft_stage','fft_local','resolve','caustic_clear','caustic_map','caustic_resolve','render','render_pc','render_pc_single','sample_quality_probe','appearance_probe','celestial_probe','surface_coefficients','bed_quality_probe','sand_transport','domain_probe','planet_probe','weather_map','weather_update','weather_probe','weather_sky','weather_visit','weather_cloud_view','weather_cloud_filter','geology_seed','geology_map','geology_update','geology_probe','geology_visit','terrain_probe','terrain_screen_probe','terrain_cache_setup','terrain_camera_frame','terrain_cache','terrain_cache_mip','terrain_near_cache','terrain_near_coefficients','terrain_intersections','geology_mip','terrain_tile_heights','ship_step','ship_wash_pick','ship_wash_modes','ship_mesh','ship_bounds','ship_render','ship_probe','ship_effect_probe'];
 kernelArtifacts=new Map(await Promise.all(names.map(async name=>{const response=await fetch(`./kernels/${name}.json?v=${encodeURIComponent(assetVersion)}`);if(!response.ok)throw Error(`Kernel ${name}: HTTP ${response.status}`);return [name,await response.json()];})));
 for(const name of names)kernels[name]={loaded:false,clear(){}};
 const optional=new Set(['ship_probe','ship_effect_probe','ship_step','ship_wash_pick','ship_wash_modes','ship_mesh','ship_bounds','ship_render','render','render_pc','render_pc_single','surface_coefficients','sand_transport','fft_stage','sample_quality_probe','appearance_probe','celestial_probe','bed_quality_probe','domain_probe','planet_probe','weather_probe','geology_probe','terrain_probe','terrain_screen_probe']);
 $('loadText').textContent=shipMode?'Preparing the spacecraft and planet…':'Preparing the globe and water…';
 await ensureKernels([...names.filter(name=>!optional.has(name)),...qualityKernels()]);
 motion=runtime.createBuffer(114688*4);twiddles=runtime.createBuffer(64*8);seed=runtime.createBuffer(49152*16);fft=[runtime.createBuffer(49152*8),runtime.createBuffer(49152*8)];surface=runtime.createBuffer(49152*16);light=runtime.createBuffer(16);monoLight=runtime.createBuffer(4);photons=runtime.createBuffer(4);navigation=runtime.createBuffer(8*8);camera=runtime.createBuffer((nearTerrainOffset+8+nearTerrainWidth*nearTerrainWidth*2)*16);plates=runtime.createBuffer(28*32);seaMemory=runtime.createBuffer(32768*4);brush=runtime.createBuffer(shipMode?176:48);disturbance=runtime.createBuffer(16384*16);diagnostics.adapter=runtime.describe();
 if(shipMode){shipMesh=runtime.createBuffer(71680*6*16);shipBounds=runtime.createBuffer((255+128*85+1024)*2*16);shipData=runtime.createBuffer(64*16);const b=runtime.batch();b.dispatch(bind('ship_mesh',{mesh:shipMesh,bounds:shipBounds}),[1120,1,1]);for(let level=0;level<=12;level++){const count=level===0?8960:level<5?128*(level===1?64:level===2?16:level===3?4:1):128>>(level-5);b.dispatch(bind('ship_bounds',{mesh:shipMesh,bounds:shipBounds},{level}),[Math.ceil(count/64),1,1]);}b.submit();diagnostics.shipTriangles=71680;}
 if(gameShell){resize();compute(0);await runtime.idle();runtime.device.queue.writeBuffer(shipData.gpuBuffer,32,new Float32Array([-.18,1.12,42,1]));weatherDirty=true;}
 requestAnimationFrame(frame);
}catch(e){fail(e);}
