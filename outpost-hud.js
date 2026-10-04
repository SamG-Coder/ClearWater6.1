// Presentation and input only. CUDA owns proximity, state transitions and rewards.
export function createOutpostHUD(host){
 const root=document.createElement('div');root.id='outpostHUD';root.hidden=true;
 root.innerHTML=`<section class="mission-card"><small id="missionChapter">SURFACE OPERATIONS / 01</small><h2 id="missionTitle"></h2><p id="missionObjective"></p><div id="outpostVitals"><span>OUTPOST <b id="outpostHealth"></b></span><meter id="outpostMeter" min="0" max="2000" value="2000"></meter></div><div id="missionCredits"></div></section><div id="outpostMarker" aria-hidden="true"><span>◇</span><b>OUTPOST</b><small></small></div><div id="crewVale" class="crew-label">VALE<span>COMMAND</span></div><div id="crewRen" class="crew-label">REN<span>ENGINEERING</span></div><div class="surface-controls"><p id="surfaceNotice" role="status"></p><div><button id="landCraft" type="button"></button><button id="interactWorld" type="button"></button></div><small id="surfaceHelp"></small></div><div id="outpostDialogue" role="dialog" aria-modal="true" aria-labelledby="dialogueSpeaker" hidden><div class="crew-badge" aria-hidden="true">✦</div><section><small id="dialogueRole"></small><h2 id="dialogueSpeaker"></h2><p id="dialogueText"></p><div id="dialogueOptions"></div></section></div>`;
 document.body.append(root);const $=id=>root.querySelector('#'+id);let opened=false,key='',latest=null;
 const titles=['A place to return to','Prepare for contact','Incoming transmission','Defend the outpost','A second signal','Hold the line','All clear','Outpost secured','Outpost disabled'];
 const objectives=['Disembark and speak to Commander Vale by the command building.','Board your ship and take off. The crew will track the incoming pirates.','Pirates are approaching. Stay close to the outpost.','Destroy the first pirate wing. Protect the outpost.','A second pirate wing is approaching. Recharge and prepare.','Destroy the reinforcements before they disable the outpost.','Land and report to Commander Vale for your reward.','Defence complete. Explore the planet, or speak to Engineer Ren to service your craft.','Return to Commander Vale to regroup and retry the defence.'];
 const reasons=['','Find dry terrain to land.','The terrain is too steep. Find flatter ground.','Brake to below 22 m/s before landing.','Descend to within 48 m of the ground.','Landing area obstructed. Use the pad or open terrain.'];
 function send(command){host.command(command);}
 function action(button,command){let touchAt=-Infinity;button.addEventListener('pointerdown',event=>{if(event.pointerType==='touch'){event.preventDefault();touchAt=performance.now();send(command);}});button.onclick=event=>{if(event.detail!==0&&performance.now()-touchAt<700)return;send(command);};}
 action($('landCraft'),1);action($('interactWorld'),2);
 function option(label,command){const b=document.createElement('button');b.type='button';b.textContent=label;action(b,command);$('dialogueOptions').append(b);}
 function dialogue(npc,stage,normal){
  const signature=`${npc}:${stage}:${normal}`;if(signature===key)return;key=signature;
  const was=opened;opened=npc>0;$('outpostDialogue').hidden=!opened;document.body.classList.toggle('in-dialogue',opened);
  if(!opened){if(was)host.dialogue(false);return;}
  const commander=npc===2;$('dialogueSpeaker').textContent=commander?'Commander Vale':'Engineer Ren';$('dialogueRole').textContent=commander?'OUTPOST COMMAND / LOCAL CHANNEL':'FLIGHT ENGINEERING / LOCAL CHANNEL';$('dialogueOptions').replaceChildren();
  let text='Good to see you back. This outpost is yours to return to. Take a look around, then follow the next horizon.';
  if(commander&&normal&&stage===0){text='You arrived just in time. Two pirate wings are heading for our power plant. We can hold the shields for a while, but we need your ship in the air. Clear both wings and bring her home. There are 750 credits in it for you.';option('I’ll defend the outpost.',3);}
  else if(commander&&stage===6){text='The skies are clear. You kept our people safe. Your 750 credits are ready, and Ren will get your ship ready for the next flight.';option('Collect 750 credits',4);}
  else if(commander&&stage===8){text='They disabled our power plant. The crew made it to shelter. We can restore the shields for another attempt when you are ready.';option('Restore the outpost and try again',6);}
  else if(commander&&stage>0&&stage<6)text='Our beacon marks the outpost. Keep the pirates away from it. Use the autocannon up close, the laser for a clear shot, and homing rockets when you have a target lock. We are counting on you.';
  else if(!commander){text='She is a good ship. L lowers the landing gear over clear, level terrain. Brake first, then descend. E gets you out and back aboard. I can restore your hull, shields and rocket rack here whenever the skies are clear.';if(stage!==3&&stage!==5)option('Service my ship',5);}
  $('dialogueText').textContent=text;option('Back',7);if(!was)host.dialogue(true);$('dialogueOptions button')?.focus({preventScroll:true});
 }
 function update(ship,normal){
  latest=ship;const enabled=ship[87]===62;root.hidden=!enabled;if(!enabled){dialogue(0,0,false);return;}
  const mode=ship[84],stage=ship[85],candidate=ship[91],foot=mode===3,parked=mode===2;
  document.body.classList.toggle('on-foot',foot);document.getElementById('touchBoost').textContent=foot?'SPRINT':'BOOST';document.body.classList.toggle('craft-grounded',mode!==0);
  $('missionTitle').textContent=normal?titles[stage]||titles[0]:'Explore the outpost';
  $('missionObjective').textContent=normal?objectives[stage]||objectives[0]:'Land, meet the crew and explore on foot. Free Roam keeps unrestricted flight and peaceful skies.';
  $('missionChapter').textContent=stage===3||stage===5?`DEFENCE / WING ${stage===3?'01':'02'} / ${ship[136]} REMAINING`:stage===2||stage===4?`CONTACT IN ${Math.ceil(ship[86])} SECONDS`:'SURFACE OPERATIONS / 01';
  $('outpostVitals').hidden=!normal||stage<2||stage>5;$('outpostHealth').textContent=`${Math.ceil(ship[132])} HULL · ${Math.ceil(ship[133])} SHIELD`;$('outpostMeter').value=ship[132]+ship[133];
  $('missionCredits').textContent=normal?`${Math.floor(ship[108])} CREDITS · ${stage===7?'CONTRACT COMPLETE':'OUTPOST DEFENCE'}`:'FREE ROAM';
  $('landCraft').hidden=foot;$('landCraft').disabled=mode===1||mode===4;$('landCraft').textContent=parked?'L · Take off':mode===1?'Landing…':mode===4?'Lifting off…':'L · Land';
  $('interactWorld').hidden=!(parked||foot&&candidate>0);$('interactWorld').textContent=parked?'E · Disembark':candidate===1?'E · Board ship':candidate===2?'E · Talk to Vale':'E · Talk to Ren';
  $('surfaceHelp').textContent=foot?'WASD / joystick · Walk    Shift · Sprint    Mouse / drag · Look':parked?'Your ship is parked. Disembark to meet the crew.':'Brake with S · Space / Q changes altitude · 1 / 2 / 3 weapons';
  $('surfaceNotice').textContent=ship[89]>0?reasons[ship[90]]||'':foot&&candidate===0?'Walk to the crew outside the buildings.':'';
  const marker=$('outpostMarker'),distance=ship[146];marker.hidden=foot||distance<45;
  const mx=ship[144],my=ship[145],behind=ship[147]===0;marker.classList.toggle('offscreen',behind||Math.abs(mx)>.9||Math.abs(my)>.8);marker.style.left=`${50+Math.max(-.90,Math.min(.90,behind?Math.sign(mx||1):mx))*50}%`;marker.style.top=`${50+Math.max(-.70,Math.min(.70,behind?.5:my))*50}%`;marker.querySelector('small').textContent=distance>1000?`${(distance/1000).toFixed(1)} km`:`${Math.round(distance)} m`;
  for(const [i,id] of ['crewVale','crewRen'].entries()){const j=176+i*4,el=$(id);el.hidden=!foot||ship[j+3]===0||ship[j+2]>45||Math.abs(ship[j])>.95||Math.abs(ship[j+1])>.90;el.style.left=`${50+ship[j]*50}%`;el.style.top=`${50+ship[j+1]*50}%`;}
  dialogue(ship[152],stage,normal);
 }
 return {update,get open(){return opened;},get mode(){return latest?.[84]??0;},close(){if(!opened)return false;send(7);return true;},reset(){latest=null;key='';dialogue(0,0,false);root.hidden=true;document.body.classList.remove('on-foot','craft-grounded');}};
}
