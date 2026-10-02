// Browser presentation only. Counts advance when real resource/GPU work finishes.
const $=id=>document.getElementById(id);
const phases=['device','download','shaders','world','frame'];
const descriptions={
 device:['Connecting to your GPU','Checking graphics support and opening the flight display.'],
 download:['Downloading flight systems','Loading the shaders for the ocean, planet, atmosphere and spacecraft.'],
 shaders:['Preparing shaders','Your GPU is preparing the graphics pipelines. First-time preparation can take a few minutes.'],
 world:['Building your world','Preparing the spacecraft, ocean surface, terrain and atmosphere.'],
 frame:['Rendering the first view','Waiting for the GPU to finish the scene. Flight controls unlock when it is ready.']
};
function system(name){
 if(name.startsWith('combat_'))return 'Weapons & hostile craft';
 if(name.startsWith('ship_'))return 'Spacecraft';
 if(name.startsWith('render'))return 'Ocean lighting';
 if(name.startsWith('terrain_')||name.startsWith('geology_'))return 'Planet & terrain';
 if(name.startsWith('weather_'))return 'Atmosphere & clouds';
 if(name.startsWith('caustic_'))return 'Underwater light';
 return 'Waves & flight';
}
export function createFlightLoading(onChange){
 const root=$('flightLoading'),bar=$('loadProgress'),fill=$('loadProgressFill');
 let state={active:false,mode:'startup',phase:null,completed:0,total:null,error:null},run=0,phaseRun=0,timer=null,started=0;
 const finished=new Set();
 function elapsed(){const seconds=Math.floor((performance.now()-started)/1000);$('loadElapsed').textContent=`${Math.floor(seconds/60)}:${String(seconds%60).padStart(2,'0')} elapsed`;}
 function publish(){onChange({...state});}
 function paint(){
  const index=phases.indexOf(state.phase);
  root.dataset.phase=state.phase||'';root.dataset.mode=state.mode;root.classList.toggle('load-failed',!!state.error);
  for(const row of root.querySelectorAll('[data-load-phase]')){
   const id=row.dataset.loadPhase,active=id===state.phase,done=finished.has(id);
   row.hidden=state.mode==='profile'&&!['shaders','frame'].includes(id);
   row.classList.toggle('current',active);row.classList.toggle('complete',done);
   if(active)row.setAttribute('aria-current','step');else row.removeAttribute('aria-current');
   row.querySelector('.step-state').textContent=active?(state.error?'FAILED':'ACTIVE'):done?'READY':'WAITING';
  }
  const label=state.error?'Flight preparation interrupted':descriptions[state.phase]?.[0]||'Flight systems ready';
  $('loadTitle').textContent=label;
  $('loadDescription').textContent=state.error||descriptions[state.phase]?.[1]||'Your next horizon is ready.';
  $('loadPhase').textContent=state.mode==='profile'?`${state.profile} / GRAPHICS UPDATE`:`PRE-FLIGHT / ${String(index+1).padStart(2,'0')} OF 05`;
  $('loadCount').textContent=state.total===null?(state.error?'ATTENTION REQUIRED':'IN PROGRESS'):`${state.completed} / ${state.total} ${state.phase==='download'?'packages downloaded':'shaders ready'}`;
  bar.setAttribute('aria-label',label);bar.setAttribute('aria-valuetext',$('loadCount').textContent);
  if(state.total!==null){bar.setAttribute('aria-valuemin','0');bar.setAttribute('aria-valuemax',String(state.total));bar.setAttribute('aria-valuenow',String(state.completed));}
  else for(const key of ['aria-valuemin','aria-valuemax','aria-valuenow'])bar.removeAttribute(key);
  fill.style.width=state.total===null?'100%':`${state.total?100*state.completed/state.total:0}%`;
  bar.classList.toggle('indeterminate',state.total===null&&!state.error);
  $('loadRetry').hidden=!state.error;root.setAttribute('aria-busy',String(state.active&&!state.error));
  publish();
 }
 function stage(phase,total=null){
  if(!state.active||state.error)return null;
  if(state.phase&&state.phase!==phase)finished.add(state.phase);
  finished.delete(phase);phaseRun++;state={...state,phase,completed:0,total};$('loadPending').textContent='';paint();
  const stamp=run,step=phaseRun;
  return ()=>{if(stamp!==run||step!==phaseRun||!state.active||state.error)return;state.completed=Math.min(total,state.completed+1);paint();};
 }
 $('loadRetry').onclick=()=>location.reload();
 return {
  get active(){return state.active;},
  begin(mode='startup',profile=''){
   run++;phaseRun++;finished.clear();clearInterval(timer);started=performance.now();
   state={active:true,mode,profile,phase:null,completed:0,total:null,error:null};root.hidden=false;
   elapsed();timer=setInterval(elapsed,1000);
   stage(mode==='startup'?'device':'shaders');
  },
  stage,
  profile(name){if(state.active&&state.mode==='profile'&&!state.error){state.profile=name;paint();}},
  shaders(names){
   const done=stage('shaders',names.length),remaining=new Set(names),stamp=run,step=phaseRun;
   function pending(){const groups=[...new Set([...remaining].map(system))];$('loadPending').textContent=groups.length?`Preparing: ${groups.join(' · ')}`:'All requested shaders are ready.';}
   pending();
   return name=>{if(stamp!==run||step!==phaseRun||!state.active||state.error||!remaining.delete(name))return;done?.();pending();};
  },
  finish(){if(!state.active||state.error)return;clearInterval(timer);timer=null;state.active=false;state.phase=null;root.hidden=true;root.setAttribute('aria-busy','false');publish();},
  error(message){if(!started)started=performance.now();clearInterval(timer);timer=null;state={...state,active:true,error:message};root.hidden=false;elapsed();paint();}
 };
}
