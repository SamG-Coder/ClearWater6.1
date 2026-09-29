import http from 'node:http';
import {readFile} from 'node:fs/promises';
import path from 'node:path';
const root=path.resolve(import.meta.dirname,'..');
const mime={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.cu':'text/plain'};
http.createServer(async(req,res)=>{
 try {let p=decodeURIComponent(new URL(req.url,'http://localhost').pathname);if(p.endsWith('/'))p+='index.html';const file=path.resolve(root,'.'+p);if(!file.startsWith(root+path.sep))throw Error('Path');res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream','Cache-Control':'no-cache'});res.end(await readFile(file));}
 catch{res.writeHead(404);res.end('Not found');}
}).listen(5191,'127.0.0.1',()=>console.log('ClearWater6.1: http://127.0.0.1:5191'));
