import {readFile,writeFile,mkdir,cp} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {compile,serializableArtifact} from '../vendor/webcuda/compiler/compiler.js';
const [waterSource,shipSource,combatSource,outpostSource]=await Promise.all(['src/water.cu','src/ship.cu','src/combat.cu','src/outpost.cu'].map(file=>readFile(file,'utf8')));
const source=waterSource+'\n'+shipSource+'\n'+outpostSource+'\n'+combatSource;
await mkdir('dist/kernels',{recursive:true});
await mkdir('kernels',{recursive:true});
for(const m of source.matchAll(/__global__ void (\w+)/g)){
 // Keep 64 lanes, but render horizontal strips for coherent pixel accesses.
 const name=m[1],artifact=compile((name.startsWith('ship_')||name.startsWith('combat_')||name.startsWith('outpost_'))?source:waterSource,{entry:name,workgroupSize:['render','render_pc','render_pc_single','ship_render','combat_render','outpost_render'].includes(name)?[32,2,1]:['outpost_build','outpost_bounds','combat_prepare','ship_enemy_mesh','ship_probe','ship_effect_probe','ship_mesh','ship_bounds','fft_local','sample_quality_probe','appearance_probe','celestial_probe','bed_quality_probe','domain_probe','planet_probe','weather_probe','geology_probe','terrain_probe','terrain_screen_probe'].includes(name)?[64,1,1]:['outpost_step','outpost_init','combat_step','ship_step','ship_wash_pick','camera_step','brush_pick','weather_update','weather_visit','geology_seed','geology_update','geology_visit','terrain_cache_setup','terrain_camera_frame'].includes(name)?[1,1,1]:[8,8,1]});
 await writeFile(`dist/kernels/${name}.json`,JSON.stringify(serializableArtifact(artifact)));
 await writeFile(`kernels/${name}.json`,JSON.stringify(serializableArtifact(artifact)));
 console.log(`Compiled ${name}`);
}
for(const file of ['index.html','style.css','app.js','game-shell.js','flight-loading.js','ship-profile.js','combat-hud.js','outpost-hud.js','src','vendor','LICENSE','THIRD_PARTY_NOTICES.md'])await cp(file,`dist/${file}`,{recursive:true});
const host=await readFile('app.js','utf8'),shell=await readFile('game-shell.js','utf8'),loading=await readFile('flight-loading.js','utf8'),profile=await readFile('ship-profile.js','utf8'),combatHUD=await readFile('combat-hud.js','utf8'),outpostHUD=await readFile('outpost-hud.js','utf8'),css=await readFile('style.css','utf8'),html=await readFile('index.html','utf8');
const version=createHash('sha256').update(source).update(host).update(shell).update(loading).update(profile).update(combatHUD).update(outpostHUD).update(css).update(html).digest('hex').slice(0,12);
await writeFile('dist/app.js',host.replace("'./game-shell.js'",`'./game-shell.js?v=${version}'`).replace("'./flight-loading.js'",`'./flight-loading.js?v=${version}'`).replace("'./ship-profile.js'",`'./ship-profile.js?v=${version}'`).replace("'./combat-hud.js'",`'./combat-hud.js?v=${version}'`).replace("'./outpost-hud.js'",`'./outpost-hud.js?v=${version}'`));
await writeFile('dist/game-shell.js',shell.replace("'./ship-profile.js'",`'./ship-profile.js?v=${version}'`));
await writeFile('dist/index.html',html.replace('src="app.js"',`src="app.js?v=${version}"`).replace('href="style.css"',`href="style.css?v=${version}"`));
await writeFile('dist/.nojekyll','');
