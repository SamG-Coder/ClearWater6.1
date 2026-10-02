import {readFile,writeFile,mkdir,cp} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {compile,serializableArtifact} from '../vendor/webcuda/compiler/compiler.js';
const source=await readFile('src/water.cu','utf8');
await mkdir('dist/kernels',{recursive:true});
await mkdir('kernels',{recursive:true});
for(const m of source.matchAll(/__global__ void (\w+)/g)){
 // Keep 64 lanes, but render horizontal strips for coherent pixel accesses.
 const name=m[1],artifact=compile(source,{entry:name,workgroupSize:['render','render_pc','render_pc_single'].includes(name)?[32,2,1]:['fft_local','sample_quality_probe','bed_quality_probe','domain_probe','planet_probe','weather_probe','geology_probe','terrain_probe','terrain_screen_probe'].includes(name)?[64,1,1]:['camera_step','brush_pick','weather_update','weather_visit','geology_seed','geology_update','geology_visit','terrain_cache_setup'].includes(name)?[1,1,1]:[8,8,1]});
 await writeFile(`dist/kernels/${name}.json`,JSON.stringify(serializableArtifact(artifact)));
 await writeFile(`kernels/${name}.json`,JSON.stringify(serializableArtifact(artifact)));
 console.log(`Compiled ${name}`);
}
for(const file of ['index.html','style.css','app.js','src','vendor','LICENSE','THIRD_PARTY_NOTICES.md'])await cp(file,`dist/${file}`,{recursive:true});
const host=await readFile('app.js','utf8'),css=await readFile('style.css','utf8'),html=await readFile('index.html','utf8');
const version=createHash('sha256').update(source).update(host).update(css).update(html).digest('hex').slice(0,12);
await writeFile('dist/index.html',html.replace('src="app.js"',`src="app.js?v=${version}"`).replace('href="style.css"',`href="style.css?v=${version}"`));
await writeFile('dist/.nojekyll','');
