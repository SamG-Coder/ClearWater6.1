// Browser UI and local persistence only. Flight and rendering stay in CUDA.
export const SAVE_KEY='clearwater6.1.save.v1';
const SETTINGS_KEY='clearwater6.1.settings.v1';
const $=id=>document.getElementById(id);
const finiteArray=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(Number.isFinite);
export function validSave(s){
 if(!s||s.version!==1||!finiteArray(s.navigation,16)||!finiteArray(s.ship,256)||!finiteArray(s.camera,80))return false;
 const n=[0,2,4].map(i=>s.navigation[i]+s.navigation[i+1]),e=[6,8,10].map(i=>s.navigation[i]+s.navigation[i+1]);
 return Math.abs(Math.hypot(...n)-1)<.001&&Math.abs(Math.hypot(...e)-1)<.001&&Math.abs(n.reduce((v,x,i)=>v+x*e[i],0))<.001&&
  s.ship[3]===1&&s.ship[1]>=0&&s.ship[1]<1e12&&s.ship.every(v=>Math.abs(v)<1e12)&&s.camera.every(v=>Math.abs(v)<1e12)&&
  s.navigation.every(v=>Math.abs(v)<1e12)&&Number.isFinite(s.time)&&s.time>=0&&s.time<1e10&&Number.isFinite(s.clock)&&s.clock>=0&&s.clock<1e12&&Number.isFinite(s.playTime)&&s.playTime>=0&&s.playTime<1e10&&
  Number.isInteger(s.seed)&&s.seed>=1&&s.seed<=999999&&s.season>=1&&s.season<=366&&s.speed>=.1&&s.speed<=1&&Number.isFinite(s.savedAt);
}
export function createGameShell(host){
 const root=$('gameMenu');let page='main',origin='main',ready=false,pending=false,kind=null,save=null,saveBusy=false,savePromise=null,lastSave=0,escapeAt=0,storageIssue='';
 const prefs={quality:$('quality').value,exposure:'1',sensitivity:'1',invertY:false,hud:true};
 try{const stored=JSON.parse(localStorage.getItem(SETTINGS_KEY));if(stored&&typeof stored==='object')for(const key of Object.keys(prefs))if(key in stored)prefs[key]=stored[key];}catch{}
 if(![...$('quality').options].some(o=>o.value===prefs.quality))prefs.quality=host.touch?'mobile':'1920';
 prefs.exposure=String(Math.max(.5,Math.min(1.7,Number(prefs.exposure)||1)));
 prefs.sensitivity=String(Math.max(.25,Math.min(2.5,Number(prefs.sensitivity)||1)));
 prefs.invertY=prefs.invertY===true;prefs.hud=prefs.hud!==false;
 for(const id of ['quality','exposure','sensitivity'])$(id).value=prefs[id];
 $('invertY').checked=prefs.invertY;$('showHUD').checked=prefs.hud;
 document.body.classList.add('game-shell','menu-open','menu-main');document.body.classList.toggle('minimal-hud',!prefs.hud);
 root.hidden=false;root.append($('fullscreen'));
 // Move the real controls, preserving one source of truth and diagnostic IDs.
 for(const id of ['quality','exposure'])$('graphicsControls').append($(id).closest('label'));
 for(const id of ['geologyPanel','weatherPanel'])$('worldControls').append($(id));
 for(const id of ['depth','energy','wind','view'])$('worldControls').append($(id).closest('label'));
 $('worldControls').append($('panel').querySelector('.presets'));
 $('toggle').textContent=host.touch?'Pause':'Menu · Esc';
 function readSave(){try{const raw=localStorage.getItem(SAVE_KEY);if(!raw){save=null;return;}const candidate=JSON.parse(raw);if(!validSave(candidate))throw Error('invalid');save=candidate;}catch{save=null;storageIssue='The local save could not be read. Start a new game to create one.';}}
 function updateSave(){
  $('continueGame').disabled=!ready||!save||pending;$('newGame').classList.toggle('primary',!save);
  $('continueDetail').textContent=save?`${Math.floor(save.playTime/60)} min flown · ${new Date(save.savedAt).toLocaleDateString(undefined,{month:'short',day:'numeric'})}`:'No saved flight yet';
  $('saveStatus').textContent=storageIssue||(save?'Progress saved on this browser':'Your flight will save on this browser');
 }
 function focusFirst(){requestAnimationFrame(()=>root.querySelector('[data-screen="'+page+'"] button:not(:disabled), [data-screen="'+page+'"] input')?.focus({preventScroll:true}));}
 function show(next){
  page=next;root.hidden=false;root.append($('fullscreen'));document.body.classList.add('menu-open');document.body.classList.toggle('menu-main',page==='main');
  for(const el of root.querySelectorAll('[data-screen]'))el.hidden=el.dataset.screen!==page;
  $('menuContext').textContent=page==='main'?'FLIGHT SYSTEM / 6.1':kind==='normal'?'GAME / PAUSED':kind==='free'?'FREE ROAM / PAUSED':'FLIGHT SYSTEM / SETTINGS';
  $('menuFooterHint').textContent=page==='main'?'A whole planet. An open sky.':page==='settings'?'Changes are saved automatically.':page==='pause'?'Simulation paused · Take your time.':'Your existing flight is safe until you launch.';
  $('settingsWorldTab').hidden=kind!=='free';
  host.pause();updateSave();focusFirst();
 }
 function status(text,error=false){$('menuStatus').textContent=text;$('menuStatus').classList.toggle('warning',error);}
 function preferences(){
  prefs.quality=$('quality').value;prefs.exposure=$('exposure').value;prefs.sensitivity=$('sensitivity').value;prefs.invertY=$('invertY').checked;prefs.hud=$('showHUD').checked;
  $('sensitivityValue').textContent=Number(prefs.sensitivity).toFixed(2)+'×';
  document.body.classList.toggle('minimal-hud',!prefs.hud);
  try{localStorage.setItem(SETTINGS_KEY,JSON.stringify(prefs));}catch{status('Settings could not be saved on this browser.',true);}
  host.redraw();
 }
 for(const id of ['quality','exposure','sensitivity','invertY','showHUD'])$(id).addEventListener('input',preferences);
 $('sensitivityValue').textContent=Number(prefs.sensitivity).toFixed(2)+'×';
 $('restoreSettings').onclick=()=>{for(const [id,value] of Object.entries({quality:host.touch?'mobile':'1920',exposure:'1',sensitivity:'1'})){$(id).value=value;$(id).dispatchEvent(new Event('change'));} $('invertY').checked=false;$('showHUD').checked=true;preferences();host.labels();};
 root.addEventListener('input',()=>host.redraw());root.addEventListener('change',()=>host.redraw());
 function tab(name){for(const b of root.querySelectorAll('[data-tab]')){const active=b.dataset.tab===name;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;}for(const p of root.querySelectorAll('[data-tab-panel]'))p.hidden=p.dataset.tabPanel!==name;}
 for(const b of root.querySelectorAll('[data-tab]')){b.onclick=()=>tab(b.dataset.tab);b.onkeydown=e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...root.querySelectorAll('[data-tab]')].filter(t=>!t.hidden),i=tabs.indexOf(b),next=e.key==='Home'?tabs[0]:e.key==='End'?tabs.at(-1):tabs[(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];tab(next.dataset.tab);next.focus();};}
 function settings(from){origin=from;tab('graphics');show('settings');}
 function persist(){
  if(savePromise)return savePromise;
  if(kind!=='normal'||!ready)return Promise.resolve(true);
  saveBusy=true;lastSave=performance.now();
  savePromise=(async()=>{
   try{const snapshot=await host.snapshot();snapshot.version=1;snapshot.savedAt=Date.now();if(!validSave(snapshot))throw Error('Flight state is not ready');localStorage.setItem(SAVE_KEY,JSON.stringify(snapshot));save=snapshot;storageIssue='';updateSave();return true;}
   catch(e){storageIssue='Could not save this flight. Browser storage may be full or unavailable.';updateSave();status(storageIssue,true);host.notice(storageIssue);return false;}
   finally{saveBusy=false;savePromise=null;}
  })();return savePromise;
 }
 function resume(){if(pending||!kind)return;page='playing';root.hidden=true;document.body.classList.remove('menu-open','menu-main');document.body.append($('fullscreen'));host.resume();status('');if(storageIssue)host.notice(storageIssue);}
 function pause(){if(page!=='playing'||pending)return;show('pause');status('');void persist();}
 async function start(next,existing=null){
  if(!ready||pending)return;pending=true;root.setAttribute('aria-busy','true');for(const b of root.querySelectorAll('[data-launch]'))b.disabled=true;status(existing?'Restoring your flight…':'Preparing your spacecraft…');
  try{if(savePromise)await savePromise;await host.start(next,existing);kind=next;document.body.classList.toggle('normal-game',kind==='normal');$('pauseMode').textContent=kind==='normal'?'NORMAL GAME':'FREE ROAM';lastSave=performance.now();if(kind==='normal')await persist();pending=false;resume();}
  catch(e){status('Could not launch: '+e.message,true);}
  finally{pending=false;root.removeAttribute('aria-busy');for(const b of root.querySelectorAll('[data-launch]'))b.disabled=!ready;updateSave();}
 }
 $('continueGame').onclick=()=>start('normal',save);
 $('newGame').onclick=()=>{if(save)show('confirm');else void start('normal');};
 $('confirmNewGame').onclick=()=>start('normal');$('cancelNewGame').onclick=()=>show('main');
 $('freeRoam').onclick=()=>start('free');$('mainSettings').onclick=()=>settings('main');
 $('resumeGame').onclick=resume;$('pauseSettings').onclick=()=>settings('pause');$('settingsBack').onclick=()=>show(origin);
 $('returnMain').onclick=async()=>{if(pending)return;pending=true;status(kind==='normal'?'Saving flight…':'Returning to menu…');const ok=await persist();pending=false;if(!ok)return;show('main');status('');};
 $('toggle').onclick=()=>page==='playing'?pause():page==='pause'?resume():null;
 $('pause').onclick=pause;
 $('throttle').oninput=()=>host.throttle(Number($('throttle').value));
 // Escape may release pointer lock without producing a keyboard event.
 addEventListener('keydown',e=>{
  if(e.code==='Escape'){e.preventDefault();e.stopImmediatePropagation();if(e.repeat||performance.now()-escapeAt<180)return;if(pending)return;if(page==='playing')pause();else if(page==='settings')show(origin);else if(page==='pause')resume();else if(page==='confirm')show('main');return;}
  if(page==='playing')return;
  if(e.code==='Tab'){const nodes=[...root.querySelectorAll('button,input,select,a[href]')].filter(el=>!el.disabled&&el.tabIndex>=0&&el.getClientRects().length);if(nodes.length){const first=nodes[0],last=nodes.at(-1);if(e.shiftKey&&(document.activeElement===first||!root.contains(document.activeElement))){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}}
 },true);
 document.addEventListener('pointerlockchange',()=>{if(!document.pointerLockElement&&page==='playing'&&!pending){escapeAt=performance.now();pause();}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
 addEventListener('blur',()=>{if(page==='playing')pause();});
 readSave();show('main');
 return {
  get open(){return page!=='playing';},get normal(){return kind==='normal';},get sensitivity(){return Number(prefs.sensitivity);},get invertY(){return prefs.invertY;},
  get state(){return {page,kind,ready,pending,hasSave:!!save,saveBusy,saveError:storageIssue};},
  ready(){ready=true;document.body.classList.add('world-ready');$('bootStatus').textContent='SYSTEMS ONLINE';for(const b of root.querySelectorAll('[data-launch]'))b.disabled=false;updateSave();if(page==='main')focusFirst();},
  error(message){$('bootStatus').textContent='FLIGHT SYSTEM UNAVAILABLE';status(message,true);},
  async tick(now){if(page==='playing'&&kind==='normal'&&now-lastSave>30000)await persist();},
  updateThrottle(value){$('throttle').value=String(value);$('throttleValue').textContent=Math.round(value*100)+'%';},
  pause,preferences
 };
}
