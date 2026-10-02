// DOM/input adapter only. Simulation, targeting and projection are CUDA-owned.
export function createCombatHUD(host){
 const $=id=>document.getElementById(id),root=$('combatHUD'),buttons=[...root.querySelectorAll('[data-weapon]')];
 const names=['AUTOCANNON','PULSE LASER','HOMING ROCKETS'];let weapon=0;
 const markers=Array.from({length:5},(_,i)=>{const el=document.createElement('div');el.className='hostile-marker';el.hidden=true;el.innerHTML='<span></span><i class="target-shield"></i><i class="target-hull"></i>';el.setAttribute('aria-hidden','true');$('combatMarkers').append(el);return el;});
 function select(value){weapon=value;host.select(value);buttons.forEach((b,i)=>{b.classList.toggle('selected',i===value);b.setAttribute('aria-pressed',String(i===value));});$('weaponName').textContent=names[value];}
 buttons.forEach(b=>b.onclick=()=>select(Number(b.dataset.weapon)));
 const fire=$('fireWeapon');let pointer=null;
 fire.addEventListener('pointerdown',e=>{if(!host.active()||pointer!==null)return;e.preventDefault();pointer=e.pointerId;fire.setPointerCapture(pointer);host.fire(true);fire.classList.add('firing');});
 function stopFire(){const id=pointer;pointer=null;if(id!==null&&fire.hasPointerCapture(id))fire.releasePointerCapture(id);host.fire(false);fire.classList.remove('firing');}
 for(const event of ['pointerup','pointercancel','lostpointercapture'])fire.addEventListener(event,e=>{if(pointer===e.pointerId)stopFire();});
 function reset(){stopFire();markers.forEach(el=>el.hidden=true);select(0);}
 function update(data){
  const hull=data[4],shield=data[5],target=data[6],alive=data[7],ammo=data[10],dead=data[11],flash=data[12],kills=data[15],sinceHit=data[16];
  $('hullValue').textContent=Math.ceil(hull);$('shieldValue').textContent=Math.ceil(shield);$('hullBar').style.width=hull+'%';$('shieldBar').style.width=shield/75*100+'%';
  $('hullMeter').setAttribute('aria-valuenow',String(Math.round(hull)));$('shieldMeter').setAttribute('aria-valuenow',String(Math.round(shield)));
  $('combatStatus').textContent=dead?'CRAFT DISABLED':shield<75&&sinceHit>4?'SHIELDS RECHARGING':hull<100&&sinceHit>10?'HULL REPAIR ACTIVE':alive?`${alive} HOSTILES IN RANGE`:'SCANNING FOR HOSTILES';
  $('combatKills').textContent=String(kills).padStart(2,'0');$('rocketCount').textContent=String(Math.floor(ammo));
  $('weaponHint').textContent=weapon===2?(ammo<1?'REARMING · 3s per rocket':target>=0?'TARGET LOCKED · FIRE':'POINT AT A HOSTILE TO LOCK'):weapon===1?'INSTANT BEAM · HOLD TO FIRE':'TRAVELLING ROUNDS · HOLD TO FIRE';
  root.classList.toggle('target-locked',target>=0);root.classList.toggle('hull-critical',hull<30);
  $('combatDamage').style.opacity=String(flash*.38);$('combatDamage').classList.toggle('hull-hit',shield<=0);
  const reticle=$('weaponReticle');reticle.hidden=data[67]===0;reticle.style.left=data[64]*100+'%';reticle.style.top=data[65]*100+'%';reticle.classList.toggle('locked',target>=0);
  markers.forEach((el,i)=>{const p=(17+i*3)*4,x=data[p],y=data[p+1],distance=data[p+2],visible=data[p+3]>0&&x>.025&&x<.975&&y>.05&&y<.90&&distance<2000;el.hidden=!visible;if(!visible)return;el.style.left=x*100+'%';el.style.top=y*100+'%';el.classList.toggle('locked',i===target);el.querySelector('span').textContent=`${i===target?'LOCK / ':''}${Math.round(distance)} m`;el.querySelector('.target-shield').style.setProperty('--fill',data[p+5]/60*100+'%');el.querySelector('.target-hull').style.setProperty('--fill',data[p+4]/90*100+'%');});
 }
 select(0);return {select,update,reset,stopFire};
}
