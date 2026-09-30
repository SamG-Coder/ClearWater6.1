import {readFile,writeFile,mkdir,cp} from 'node:fs/promises';
import {compile,serializableArtifact} from '../vendor/webcuda/compiler/compiler.js';
const source=await readFile('src/water.cu','utf8');
await mkdir('dist/kernels',{recursive:true});
await mkdir('kernels',{recursive:true});
for(const m of source.matchAll(/__global__ void (\w+)/g)){
 const name=m[1],artifact=compile(source,{entry:name,workgroupSize:name==='fft_local'?[64,1,1]:['camera_step','brush_pick'].includes(name)?[1,1,1]:[8,8,1]});
 await writeFile(`dist/kernels/${name}.json`,JSON.stringify(serializableArtifact(artifact)));
 await writeFile(`kernels/${name}.json`,JSON.stringify(serializableArtifact(artifact)));
 console.log(`Compiled ${name}`);
}
for(const file of ['index.html','style.css','app.js','src','vendor','LICENSE','THIRD_PARTY_NOTICES.md'])await cp(file,`dist/${file}`,{recursive:true});
await writeFile('dist/.nojekyll','');
