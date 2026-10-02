// Save/UI metadata only. Colour conversion, materials and emitted light are CUDA.
export const DEFAULT_COLOURS=Object.freeze({primary:'#c4cdcb',secondary:'#d37635',booster:'#3c95ff'});
export const colourKeys=Object.keys(DEFAULT_COLOURS);
export function validName(value){return typeof value==='string'&&value===value.trim()&&value.length>0&&value.length<=32&&!/[\u0000-\u001f\u007f]/u.test(value);}
export function validColours(colours){return !!colours&&colourKeys.every(key=>typeof colours[key]==='string'&&/^#[0-9a-f]{6}$/i.test(colours[key]));}
export function validIdentity(identity){return !!identity&&validName(identity.pilotName)&&validName(identity.shipName)&&validColours(identity.colours);}
export function identityFromSave(save){return validIdentity(save?.identity)?structuredClone(save.identity):{pilotName:'Pilot',shipName:'Exploration craft 01',colours:{...DEFAULT_COLOURS}};}
export function paletteBytes(colours){return new Float32Array(colourKeys.flatMap(key=>{const hex=colours[key];return [1,3,5].map(offset=>parseInt(hex.slice(offset,offset+2),16)/255).concat(1);}));}
