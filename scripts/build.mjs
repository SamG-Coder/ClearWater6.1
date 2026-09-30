import {readFile,writeFile,mkdir,cp} from 'node:fs/promises';
import {createHash} from 'node:crypto';
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
const host=await readFile('app.js','utf8'),css=await readFile('style.css','utf8'),html=await readFile('index.html','utf8');
const version=createHash('sha256').update(source).update(host).update(css).update(html).digest('hex').slice(0,12);
await writeFile('dist/index.html',html.replace('src="app.js"',`src="app.js?v=${version}"`).replace('href="style.css"',`href="style.css?v=${version}"`));
await writeFile('dist/.nojekyll','');
