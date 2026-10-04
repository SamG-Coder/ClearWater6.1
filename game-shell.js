// Browser UI and local persistence only. Flight and rendering stay in CUDA.
import {DEFAULT_COLOURS,colourKeys,validName,validColours,validIdentity,identityFromSave} from './ship-profile.js';
export const SAVE_KEY='clearwater6.1.save.v1';
const SETTINGS_KEY='clearwater6.1.settings.v1';
const $=id=>document.getElementById(id);
const finiteArray=(a,n)=>Array.isArray(a)&&a.length===n&&a.every(Number.isFinite);
export function validSave(s){
 if(!s||s.version!==1||!finiteArray(s.navigation,16)||!finiteArray(s.ship,256)||!finiteArray(s.camera,80))return false;
 if(s.identity!==undefined&&!validIdentity(s.identity))return false;
 const n=[0,2,4].map(i=>s.navigation[i]+s.navigation[i+1]),e=[6,8,10].map(i=>s.navigation[i]+s.navigation[i+1]);
 return Math.abs(Math.hypot(...n)-1)<.001&&Math.abs(Math.hypot(...e)-1)<.001&&Math.abs(n.reduce((v,x,i)=>v+x*e[i],0))<.001&&
  s.ship[3]===1&&s.ship[1]>=0&&s.ship[1]<1e12&&s.ship.every(v=>Math.abs(v)<1e12)&&s.camera.every(v=>Math.abs(v)<1e12)&&
  s.navigation.every(v=>Math.abs(v)<1e12)&&Number.isFinite(s.time)&&s.time>=0&&s.time<1e10&&Number.isFinite(s.clock)&&s.clock>=0&&s.clock<1e12&&Number.isFinite(s.playTime)&&s.playTime>=0&&s.playTime<1e10&&
  Number.isInteger(s.seed)&&s.seed>=1&&s.seed<=999999&&s.season>=1&&s.season<=366&&s.speed>=.1&&s.speed<=1&&Number.isFinite(s.savedAt);
}
export function createGameShell(host){
 const root=$('gameMenu');let page='boot',origin='boot',ready=false,loading=true,pending=false,kind=null,save=null,saveBusy=false,savePromise=null,lastSave=0,escapeAt=0,storageIssue='';
 let identity=identityFromSave(null),draft=null,previewing=false,colourSaveTimer=null,activeTab='graphics';
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
  for(const button of root.querySelectorAll('[data-launch]'))button.disabled=!ready||loading||pending;
  $('resumeGame').disabled=loading||pending;
  $('settingsShipTab').disabled=!ready||loading||pending;
  $('launchFlight').disabled=!ready||loading||pending||!validIdentity(draft);
  $('continueGame').disabled=!ready||loading||!save||pending;$('newGame').classList.toggle('primary',!save);
  $('continueDetail').textContent=save?`${save.identity?.shipName||'Saved flight'} · ${Math.floor(save.playTime/60)} min flown`:'No saved flight yet';
  $('saveStatus').textContent=storageIssue||(save?'Progress saved on this browser':'Your flight will save on this browser');
 }
 function focusFirst(){requestAnimationFrame(()=>root.querySelector('[data-screen="'+page+'"] button:not(:disabled), [data-screen="'+page+'"] input')?.focus({preventScroll:true}));}
 function show(next){
  if(previewing&&!['setup','confirm'].includes(next)&&!(next==='settings'&&activeTab==='ship'))stopPreview();
  page=next;root.hidden=false;root.append($('fullscreen'));document.body.classList.add('menu-open');document.body.classList.toggle('menu-main',page==='main'||page==='boot');root.dataset.screen=page;
  $(page==='settings'?'settingsLoading':page==='pause'?'pauseLoading':page==='main'?'mainLoading':'bootLoading').append($('flightLoading'));
  $('flightLoading').classList.toggle('compact-loading',page!=='boot');
  for(const el of root.querySelectorAll('[data-screen]'))el.hidden=el.dataset.screen!==page;
  $('menuContext').textContent=page==='main'?'FLIGHT SYSTEM / 6.1':page==='setup'?'FLIGHT SYSTEM / REGISTRATION':page==='boot'?'FLIGHT SYSTEM / PRE-FLIGHT':kind==='normal'?'GAME / PAUSED':kind==='free'?'FREE ROAM / PAUSED':'FLIGHT SYSTEM / SETTINGS';
  $('menuFooterHint').textContent=page==='main'||page==='boot'?'A whole planet. An open sky.':page==='settings'?'Changes are saved automatically.':page==='pause'?'Simulation paused · Take your time.':page==='defeat'?'Last checkpoint preserved.':'Your existing flight is safe until you launch.';
  $('settingsWorldTab').hidden=kind!=='free';
  $('settingsShipTab').hidden=!kind;
  if(page==='setup')$('setupColours').append($('shipColours'));
  host.pause();updateSave();focusFirst();
 }
 function status(text,error=false){$('menuStatus').textContent=text;$('menuStatus').classList.toggle('warning',error);}
 function identityLabels(){
  $('pilotHUD').textContent=`CLEARWATER / ${identity.pilotName}`;$('shipNameHUD').textContent=identity.shipName;
  $('registeredShipName').textContent=identity.shipName;$('registeredPilotName').textContent=`Registered to ${identity.pilotName}`;
 }
 function previewLabels(value){$('previewShipName').textContent=value.shipName||'Your next horizon.';$('previewPilotName').textContent=value.pilotName?`PILOT / ${value.pilotName}`:'LIVE SHIP PREVIEW';}
 function setColourInputs(colours){for(const key of colourKeys){$(key+'Colour').value=colours[key];$(key+'ColourValue').textContent=colours[key].toUpperCase();}root.style.setProperty('--booster-colour',colours.booster);}
 async function beginPreview(value){
  pending=true;updateSave();for(const input of $('shipColours').querySelectorAll('input,button'))input.disabled=true;
  try{await host.beginCraftPreview(value.colours);previewing=true;document.body.classList.add('craft-preview');$('craftPreviewCaption').hidden=false;$('craftOrbitZone').hidden=false;root.classList.toggle('preview-settings',page==='settings');previewLabels(value);}
  catch(e){status('Could not prepare the ship preview: '+e.message,true);}
  finally{pending=false;for(const input of $('shipColours').querySelectorAll('input,button'))input.disabled=false;updateSave();}
 }
 function stopPreview(){
  clearTimeout(colourSaveTimer);colourSaveTimer=null;
  const restoreColours=page==='setup'||page==='confirm';host.endCraftPreview(restoreColours);previewing=false;document.body.classList.remove('craft-preview');root.classList.remove('preview-settings');$('craftPreviewCaption').hidden=true;$('craftOrbitZone').hidden=true;
  if(!restoreColours&&kind==='normal')void persist();
 }
 function registrationChanged(){
  if(!draft)return;draft.pilotName=$('pilotName').value.trim();draft.shipName=$('newShipName').value.trim();
  for(const [id,key] of [['pilotName','pilotName'],['newShipName','shipName']])$(id).setCustomValidity(draft[key]&&!validName(draft[key])?'Use up to 32 characters without control characters.':'');
  $('registrationHint').textContent=validIdentity(draft)?'Your flight and ship colours will save on this browser.':'Enter both names to launch your flight.';
  previewLabels(draft);updateSave();
 }
 for(const id of ['pilotName','newShipName'])$(id).addEventListener('input',registrationChanged);
 function changeColours(colours){
  if(pending||!validColours(colours))return;const editingDraft=page==='setup',target=editingDraft?draft:identity;if(!target)return;
  target.colours={...colours};setColourInputs(colours);host.setCraftColours(colours);
  if(!editingDraft&&kind==='normal'){clearTimeout(colourSaveTimer);colourSaveTimer=setTimeout(()=>void persist(),350);}
 }
 for(const key of colourKeys)$(key+'Colour').addEventListener('input',()=>changeColours(Object.fromEntries(colourKeys.map(k=>[k,$(k+'Colour').value]))));
 const liveries={original:DEFAULT_COLOURS,solar:{primary:'#e8dfc7',secondary:'#c65825',booster:'#ff793a'},aurora:{primary:'#d8e3e3',secondary:'#7757ad',booster:'#b167ff'},stealth:{primary:'#283544',secondary:'#658b97',booster:'#46ffbc'}};
 for(const button of root.querySelectorAll('[data-livery]'))button.onclick=()=>changeColours(liveries[button.dataset.livery]);
 let orbitPointer=null,orbitX=0,orbitY=0;
 root.addEventListener('pointerdown',e=>{if(!previewing||pending||e.button!==0||e.target.closest('button,input,select,a,.setup-screen,.settings-screen'))return;orbitPointer=e.pointerId;orbitX=e.clientX;orbitY=e.clientY;root.setPointerCapture(e.pointerId);e.preventDefault();});
 root.addEventListener('pointermove',e=>{if(e.pointerId!==orbitPointer)return;host.orbitCraft(e.clientX-orbitX,e.clientY-orbitY);orbitX=e.clientX;orbitY=e.clientY;e.preventDefault();});
 for(const event of ['pointerup','pointercancel','lostpointercapture'])root.addEventListener(event,e=>{if(e.pointerId===orbitPointer)orbitPointer=null;});
 function preferences(){
  prefs.quality=$('quality').value;prefs.exposure=$('exposure').value;prefs.sensitivity=$('sensitivity').value;prefs.invertY=$('invertY').checked;prefs.hud=$('showHUD').checked;
  $('sensitivityValue').textContent=Number(prefs.sensitivity).toFixed(2)+'×';
  document.body.classList.toggle('minimal-hud',!prefs.hud);
  try{localStorage.setItem(SETTINGS_KEY,JSON.stringify(prefs));}catch{status('Settings could not be saved on this browser.',true);}
  host.prepareQuality();host.redraw();
 }
 for(const id of ['quality','exposure','sensitivity','invertY','showHUD'])$(id).addEventListener('input',preferences);
 $('sensitivityValue').textContent=Number(prefs.sensitivity).toFixed(2)+'×';
 $('restoreSettings').onclick=()=>{for(const [id,value] of Object.entries({quality:host.touch?'mobile':'1920',exposure:'1',sensitivity:'1'})){$(id).value=value;$(id).dispatchEvent(new Event('change'));} $('invertY').checked=false;$('showHUD').checked=true;preferences();host.labels();};
 root.addEventListener('input',e=>{if(!e.target.matches('#pilotName,#newShipName'))host.redraw();});root.addEventListener('change',()=>host.redraw());
 function tab(name){activeTab=name;if(previewing)stopPreview();for(const b of root.querySelectorAll('[data-tab]')){const active=b.dataset.tab===name;b.classList.toggle('active',active);b.setAttribute('aria-selected',String(active));b.tabIndex=active?0:-1;}for(const p of root.querySelectorAll('[data-tab-panel]'))p.hidden=p.dataset.tabPanel!==name;if(name==='ship'){ $('settingsColours').append($('shipColours'));setColourInputs(identity.colours);void beginPreview(identity);}}
 for(const b of root.querySelectorAll('[data-tab]')){b.onclick=()=>{if(!pending)tab(b.dataset.tab);};b.onkeydown=e=>{if(pending||!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...root.querySelectorAll('[data-tab]')].filter(t=>!t.hidden&&!t.disabled),i=tabs.indexOf(b),next=e.key==='Home'?tabs[0]:e.key==='End'?tabs.at(-1):tabs[(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];tab(next.dataset.tab);next.focus();};}
 function settings(from){origin=from;tab('graphics');show('settings');}
 function persist(){
  if(savePromise)return savePromise;
  if(kind!=='normal'||!ready||host.canSave?.()===false)return Promise.resolve(true);
  saveBusy=true;lastSave=performance.now();
  savePromise=(async()=>{
   try{const snapshot=await host.snapshot();if(snapshot.ship[67]===61&&snapshot.ship[56]<=0)return true;snapshot.version=1;snapshot.savedAt=Date.now();snapshot.identity=structuredClone(identity);if(!validSave(snapshot))throw Error('Flight state is not ready');localStorage.setItem(SAVE_KEY,JSON.stringify(snapshot));save=snapshot;storageIssue='';updateSave();return true;}
   catch(e){storageIssue='Could not save this flight. Browser storage may be full or unavailable.';updateSave();status(storageIssue,true);host.notice(storageIssue);return false;}
   finally{saveBusy=false;savePromise=null;}
  })();return savePromise;
 }
 function resume(){if(pending||loading||!kind||page==='defeat')return;if(previewing)stopPreview();page='playing';root.hidden=true;document.body.classList.remove('menu-open','menu-main');document.body.append($('fullscreen'));host.resume();status('');if(storageIssue)host.notice(storageIssue);}
 function pause(){if(page!=='playing'||pending)return;show('pause');status('');void persist();}
 async function start(next,existing=null,registration=null){
  if(!ready||loading||pending)return;pending=true;root.setAttribute('aria-busy','true');for(const b of root.querySelectorAll('[data-launch]'))b.disabled=true;status(existing?'Restoring your flight…':'Preparing your spacecraft…');
  try{if(savePromise)await savePromise;if(previewing)stopPreview();const chosen=registration||identityFromSave(existing);await host.start(next,existing,chosen);identity=structuredClone(chosen);kind=next;identityLabels();document.body.classList.toggle('normal-game',kind==='normal');$('pauseMode').textContent=kind==='normal'?'NORMAL GAME':'FREE ROAM';lastSave=performance.now();if(kind==='normal')await persist();pending=false;resume();}
  catch(e){status('Could not launch: '+e.message,true);}
  finally{pending=false;root.removeAttribute('aria-busy');for(const b of root.querySelectorAll('[data-launch]'))b.disabled=!ready;updateSave();}
 }
 $('recoverCraft').onclick=async()=>{if(pending)return;pending=true;$('recoverCraft').disabled=true;status('Recovering your craft…');try{await host.recover();page='pause';await persist();pending=false;resume();}catch(e){status('Recovery failed: '+e.message,true);}finally{pending=false;$('recoverCraft').disabled=false;}};
 $('defeatMain').onclick=()=>{if(!pending){show('main');status('');}};
 $('continueGame').onclick=()=>start('normal',save);
 $('newGame').onclick=async()=>{if(!ready||loading||pending)return;draft={pilotName:'',shipName:'',colours:{...DEFAULT_COLOURS}};$('pilotName').value='';$('newShipName').value='';setColourInputs(draft.colours);show('setup');registrationChanged();await beginPreview(draft);};
 $('newFlightForm').onsubmit=e=>{e.preventDefault();registrationChanged();if(pending||!validIdentity(draft))return;if(save){$('newFlightSummary').textContent=`${draft.pilotName} · ${draft.shipName}`;show('confirm');}else void start('normal',null,draft);};
 $('confirmNewGame').onclick=()=>{if(validIdentity(draft))void start('normal',null,draft);};$('cancelNewGame').onclick=()=>show('setup');
 $('cancelSetup').onclick=()=>{if(!pending)show('main');};
 $('freeRoam').onclick=()=>start('free');$('mainSettings').onclick=()=>settings('main');
 $('bootSettings').onclick=()=>settings('boot');
 $('resumeGame').onclick=resume;$('pauseSettings').onclick=()=>settings('pause');$('settingsBack').onclick=()=>{if(!pending)show(origin);};
 $('returnMain').onclick=async()=>{if(pending)return;pending=true;status(kind==='normal'?'Saving flight…':'Returning to menu…');const ok=await persist();pending=false;if(!ok)return;show('main');status('');};
 $('toggle').onclick=()=>page==='playing'?pause():page==='pause'?resume():null;
 $('pause').onclick=pause;
 $('throttle').oninput=()=>host.throttle(Number($('throttle').value));
 // Escape may release pointer lock without producing a keyboard event.
 addEventListener('keydown',e=>{
  if(e.code==='Escape'){e.preventDefault();e.stopImmediatePropagation();if(e.repeat||performance.now()-escapeAt<180)return;if(pending)return;if(page==='playing'){if(host.closeDialogue?.())return;pause();}else if(page==='settings')show(origin);else if(page==='pause')resume();else if(page==='confirm')show('setup');else if(page==='setup')show('main');return;}
  if(page==='playing')return;
  if(e.code==='Tab'){const nodes=[...root.querySelectorAll('button,input,select,a[href]')].filter(el=>!el.disabled&&el.tabIndex>=0&&el.getClientRects().length);if(nodes.length){const first=nodes[0],last=nodes.at(-1);if(e.shiftKey&&(document.activeElement===first||!root.contains(document.activeElement))){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}}
 },true);
 document.addEventListener('pointerlockchange',()=>{if(!document.pointerLockElement&&page==='playing'&&!pending&&!host.dialogueOpen?.()){escapeAt=performance.now();pause();}});
 document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});
 addEventListener('blur',()=>{if(page==='playing')pause();});
 readSave();show('boot');
 return {
  get open(){return page!=='playing';},get normal(){return kind==='normal';},get sensitivity(){return Number(prefs.sensitivity);},get invertY(){return prefs.invertY;},
  get state(){return {page,kind,ready,loading,pending,hasSave:!!save,saveBusy,saveError:storageIssue,identity:structuredClone(identity)};},
  loading(state){loading=state.active;$('bootStatus').textContent=state.error?'PRE-FLIGHT INTERRUPTED':loading?(state.mode==='profile'?'UPDATING GRAPHICS':'PRE-FLIGHT IN PROGRESS'):'SYSTEMS ONLINE';root.classList.toggle('systems-loading',loading);updateSave();},
  ready(){ready=true;document.body.classList.add('world-ready');if(origin==='boot')origin='main';updateSave();if(page==='boot')show('main');else if(page==='main')focusFirst();},
  error(message){if(page==='playing')show('pause');$('bootStatus').textContent='FLIGHT SYSTEM UNAVAILABLE';},
  async tick(now){if(page==='playing'&&kind==='normal'&&now-lastSave>30000)await persist();},
  updateThrottle(value){$('throttle').value=String(value);$('throttleValue').textContent=Math.round(value*100)+'%';},
  defeat(){if(kind!=='normal')return;show('defeat');status('');},pause,preferences
 };
}
