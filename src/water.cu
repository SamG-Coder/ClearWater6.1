// ClearWater6.1 - original spectral water and optical renderer.
// CUDA subset compiled by CUDA WebShader. No external image assets.
__device__ float clamp01(float a){return fminf(1,fmaxf(0,a));}
__device__ float mixf(float a,float b,float t){return a+(b-a)*t;}
__device__ float fract(float a){return a-floorf(a);}
__device__ float3 vec(float x,float y,float z){return make_float3(x,y,z);}
__device__ float3 plus(float3 a,float3 b){return vec(a.x+b.x,a.y+b.y,a.z+b.z);}
__device__ float3 minus(float3 a,float3 b){return vec(a.x-b.x,a.y-b.y,a.z-b.z);}
__device__ float3 scale(float3 a,float b){return vec(a.x*b,a.y*b,a.z*b);}
__device__ float dotv(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ float3 unit(float3 a){return scale(a,rsqrtf(fmaxf(.0000001f,dotv(a,a))));}
__device__ float3 blend(float3 a,float3 b,float t){return plus(scale(a,1-t),scale(b,t));}
__device__ float3 crossv(float3 a,float3 b){return vec(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float eased(float a,float b,float x){float t=clamp01((x-a)/(b-a));return t*t*(3-2*t);}
__device__ float earth_radius(){return 6371000.0f;}
// Stable ray/sphere roots use altitude explicitly, avoiding R+h-R cancellation.
// The camera is at (0, altitude, 0); the planet centre is (0, -R, 0).
__device__ float2 sphere_roots(float altitude,float3 ray,float shell){
 float R=earth_radius()+shell,h=altitude-shell,b=(R+h)*ray.y,c=h*(2*R+h),disc=b*b-c;
 if(disc<0)return make_float2(-1,-1);
 float root=sqrtf(disc),q=-b+(b<0?root:-root);
 float a=fabsf(q)>.00001f?c/q:0;return make_float2(fminf(a,q),fmaxf(a,q));
}
__device__ float globe_hit(float altitude,float3 ray){float2 roots=sphere_roots(altitude,ray,0);return roots.x>0?roots.x:-1;}
__device__ float3 to_world(const float4 *camera,float3 v){
 float4 e=camera[5],n=camera[6],b=camera[7];return vec(e.x*v.x+n.x*v.y+b.x*v.z,e.y*v.x+n.y*v.y+b.y*v.z,e.z*v.x+n.z*v.y+b.z*v.z);
}
__device__ unsigned scramble(unsigned a){a^=a>>16;a*=2246822519u;a^=a>>13;a*=3266489917u;a^=a>>16;return a;}
// Scaling these bounded positive floats by 2^-24 is exact in binary32.
__device__ float randf(unsigned a){return ((float)(scramble(a)&16777215u)+.5f)*.000000059604644775390625f;}
__device__ float cell(float x,float z){return randf((unsigned)((int)x*92837111+(int)z*689287499));}
__device__ int wrap128(int a){return a&127;}
__device__ int reverse7(int a){int b=0;for(int i=0;i<7;i++){b=b*2+(a&1);a=a>>1;}return b;}
__device__ float patch(int c){return c==0?6.0f:(c==1?96.0f:24.0f);}
__device__ float2 cmul(float2 a,float2 b){return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
__device__ float2 initial(int x,int z,int c,float wind){
 int fx=x<64?x:x-128,fz=z<64?z:z-128;
 float kx=6.2831853f*(float)fx/patch(c),kz=6.2831853f*(float)fz/patch(c);
 float kk=kx*kx+kz*kz;if(kk<.00001f)return make_float2(0,0);
 float alignment=(kx*.6f-kz*.8f)*(kx*.6f-kz*.8f)/kk;
 float L=wind*wind/9.81f;
 float P=expf(-1/(kk*L*L))*expf(-kk*.006f)/(kk*kk)*(.12f+.88f*alignment);
 float band=c==0?1-expf(-kk*1.5f):expf(-kk*.8f);
 unsigned id=(unsigned)(c*16384+z*128+x+17);
 float radius=sqrtf(-2*logf(fmaxf(.000001f,randf(id*2u))));
 float angle=6.2831853f*randf(id*2u+1u);
 float amp=sqrtf(P*band)*(c==0?.023f:.018f)*6.2831853f/patch(c);
 return make_float2(radius*cosf(angle)*amp,radius*sinf(angle)*amp);
}
// Wind-dependent Gaussian spectra are cached; no log/exp/random generation
// is repeated in the per-frame spectrum kernel.
__global__ void seed_modes(float4 *seed,float2 *twiddles,float wind){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 if(c==0&&z==0&&x<64){float angle=6.2831853f*(float)x/128;twiddles[x]=make_float2(cosf(angle),sinf(angle));}
 if(c==2){seed[c*16384+z*128+x]=make_float4(0,0,0,0);return;}
 float2 a=initial(x,z,c,wind),b=initial(wrap128(-x),wrap128(-z),c,wind);
 seed[c*16384+z*128+x]=make_float4(a.x,a.y,b.x,b.y);
}
// Hermitian time spectrum: h0(k)e^iwt + conjugate(h0(-k))e^-iwt.
// Bit-reversal on both axes prepares the in-place-order radix-2 inverse FFT.
// Frequencies and force envelopes depend on depth, not animation time.
__global__ void prepare_modes(float *motion,const float4 *camera,float depth,float time){
 float waterDepth=depth;
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 if(camera[21].w!=0){if(camera[23].y==0)return;waterDepth=camera[21].x;}
 int fx=x<64?x:x-128,fz=z<64?z:z-128;
 float kx=6.2831853f*(float)fx/24,kz=6.2831853f*(float)fz/24,kk=kx*kx+kz*kz;
 float k=c==2?sqrtf(kk):6.2831853f*sqrtf((float)(fx*fx+fz*fz))/patch(c);
 float e=expf(-2*fminf(20,k*waterDepth)),tanhd=(1-e)/(1+e);
 float omega=c==2?sqrtf(9.81f*k*(1-e)/(1+e)):sqrtf(9.81f*k*tanhd);
 int idx=z*128+x;
 if(c<2){float old=motion[c*16384+idx];motion[81920+c*16384+idx]=camera[21].w!=0&&old>0?motion[81920+c*16384+idx]+(old-omega)*time:0;}
 motion[c*16384+idx]=omega;
 if(c==2){motion[49152+idx]=expf(-kk*.24f*.24f*.5f);motion[65536+idx]=.32f+.009f*kk;}
}
__global__ void spectrum(float2 *output,const float4 *seed,const float4 *disturbance,const float *motion,const float4 *camera,float *seaMemory,float time,float energy){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 if(c==2){float4 d=disturbance[z*128+x];output[c*16384+reverse7(z)*128+reverse7(x)]=make_float2(d.x,d.y);return;}
 float4 h=seed[c*16384+z*128+x];float2 a=make_float2(h.x,h.y),b=make_float2(h.z,h.w);

 float amplitude=1;
 if(camera[13].w!=0){
  int fx=x==64?0:(x<64?x:x-128),fz=z==64?0:(z<64?z:z-128);float kk=(float)(fx*fx+fz*fz);
  float4 wind=camera[10];float projection=((float)fx*wind.x+(float)fz*wind.y)/fmaxf(.1f,wind.z);
  float original=(float)fx*.6f-(float)fz*.8f;
  float directional=(.20f*kk+.80f*projection*projection)/fmaxf(.001f,.12f*kk+.88f*original*original);
  float ratio=wind.z/fmaxf(2,camera[15].x),strength=fminf(4.0f,fmaxf(.30f,ratio));
  float target=fminf(5,fmaxf(.18f,directional*strength)),old=seaMemory[c*16384+z*128+x];if(old<=0)old=1;
  float4 lag=camera[14];float response=c==0?(target<old?lag.y:lag.x):(target<old?lag.w:lag.z);
  float next=mixf(old,target,response);seaMemory[c*16384+z*128+x]=next;amplitude=sqrtf(next);
 }else seaMemory[c*16384+z*128+x]=1;
 float waveEnergy=energy*amplitude;
 float phase=motion[c*16384+z*128+x]*time+motion[81920+c*16384+z*128+x];
 float2 p=make_float2(cosf(phase),sinf(phase));
 float2 u=cmul(a,p),v=cmul(make_float2(b.x,-b.y),make_float2(p.x,-p.y));
 output[c*16384+reverse7(z)*128+reverse7(x)]=make_float2((u.x+v.x)*waveEnergy,(u.y+v.y)*waveEnergy);
}
__global__ void fft_stage(const float2 *input,float2 *output,int span,int axis){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 int pos=axis==0?x:z,half=span/2,j=pos%half,start=(pos/span)*span;
 int ia=axis==0?c*16384+z*128+start+j:c*16384+(start+j)*128+x;
 int ib=axis==0?ia+half:ia+half*128;
 float angle=6.2831853f*(float)j/(float)span;
 float2 a=input[ia],b=cmul(input[ib],make_float2(cosf(angle),sinf(angle)));
 float sign=pos%span<half?1.0f:-1.0f;
 output[c*16384+z*128+x]=make_float2(a.x+sign*b.x,a.y+sign*b.y);
}
// One 64-lane workgroup transforms a complete 128-value row or column.
// Each lane owns one disjoint butterfly, so stages synchronize in shared
// memory instead of rereading and rewriting global buffers fourteen times.
__global__ void fft_local(const float2 *input,float2 *output,const float2 *twiddles,int axis){
 __shared__ float2 values[128];
 __shared__ float2 rotation[64];
 int lane=threadIdx.x,line=blockIdx.x,base=blockIdx.z*16384;
 int first=axis==0?base+line*128+lane:base+lane*128+line;
 int second=axis==0?first+64:first+8192;
 values[lane]=input[first];values[lane+64]=input[second];rotation[lane]=twiddles[lane];
 __syncthreads();
 for(int span=2;span<=128;span*=2){
  int half=span/2,j=lane&(half-1),a=(lane/half)*span+j,b=a+half;
  float2 left=values[a],right=cmul(values[b],rotation[j*(128/span)]);
  values[a]=make_float2(left.x+right.x,left.y+right.y);
  values[b]=make_float2(left.x-right.x,left.y-right.y);
  __syncthreads();
 }
 output[first]=values[lane];output[second]=values[lane+64];
}
__global__ void resolve(const float2 *input,float4 *surface){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 int base=c*16384,idx=base+z*128+x;float dx=patch(c)/128;
 float h=input[idx].x;
 float sx=(input[base+z*128+wrap128(x+1)].x-input[base+z*128+wrap128(x-1)].x)/(2*dx);
 float sz=(input[base+wrap128(z+1)*128+x].x-input[base+wrap128(z-1)*128+x].x)/(2*dx);
 surface[idx]=make_float4(h,sx,sz,input[idx].y);
}
__device__ float4 sample(const float4 *s,float x,float z,int c){
 float u=x*128/patch(c),v=z*128/patch(c);int ix=(int)floorf(u),iz=(int)floorf(v);
 float a=fract(u),b=fract(v);int base=c*16384;
 float4 p=s[base+wrap128(iz)*128+wrap128(ix)],q=s[base+wrap128(iz)*128+wrap128(ix+1)];
 float4 r=s[base+wrap128(iz+1)*128+wrap128(ix)],t=s[base+wrap128(iz+1)*128+wrap128(ix+1)];
 return make_float4(mixf(mixf(p.x,q.x,a),mixf(r.x,t.x,a),b),mixf(mixf(p.y,q.y,a),mixf(r.y,t.y,a),b),mixf(mixf(p.z,q.z,a),mixf(r.z,t.z,a),b),0);
}
// PC reconstruction uses a periodic cubic B-spline with continuous curvature.
// Prefiltered coefficients retain the Fourier heights rather than blurring them.
__device__ float4 cubic_weights(float t){
 float t2=t*t,t3=t2*t,one=1-t;
 return make_float4(one*one*one/6,(4-6*t2+3*t3)/6,(1+3*t+3*t2-3*t3)/6,t3/6);
}
__device__ float4 cubic_derivatives(float t){
 float t2=t*t,one=1-t;
 return make_float4(-.5f*one*one,-2*t+1.5f*t2,.5f+t-1.5f*t2,.5f*t2);
}
// Separable inverse of the B-spline smoothing operator, expanded through
// the third power of the discrete Laplacian. Low-frequency residual is O(k^8).
// Two short passes prepare coefficients once per frame, shared by all rays.
__global__ void surface_coefficients(const float4 *input,float4 *output,int axis){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 int base=c*16384;float value=0;
 for(int j=-3;j<=3;j++){
  int d=j<0?-j:j;float weight=d==0?1.5925925926f:(d==1?-.3472222222f:(d==2?.0555555556f:-.0046296296f));
  int idx=axis==0?base+z*128+wrap128(x+j):base+wrap128(z+j)*128+x;
  value+=input[idx].x*weight;
 }
 output[base+z*128+x]=make_float4(value,0,0,0);
}
__device__ float dot4(float4 a,float4 b){return a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;}
__device__ float4 sample_pc(const float4 *s,float x,float z,int c,int slopes){
 float factor=128/patch(c),u=x*factor,v=z*factor;
 int ix=(int)floorf(u),iz=(int)floorf(v),base=c*16384;
 float4 wx=cubic_weights(fract(u)),wz=cubic_weights(fract(v));
 float4 dx=cubic_derivatives(fract(u)),dz=cubic_derivatives(fract(v));
 float4 rows=make_float4(0,0,0,0),derivatives=make_float4(0,0,0,0);
 for(int j=0;j<4;j++){
  int row=base+wrap128(iz+j-1)*128;
  float4 h=make_float4(s[row+wrap128(ix-1)].x,s[row+wrap128(ix)].x,s[row+wrap128(ix+1)].x,s[row+wrap128(ix+2)].x);
  float value=dot4(h,wx),gradient=slopes!=0?dot4(h,dx):0;
  if(j==0){rows.x=value;derivatives.x=gradient;}if(j==1){rows.y=value;derivatives.y=gradient;}
  if(j==2){rows.z=value;derivatives.z=gradient;}if(j==3){rows.w=value;derivatives.w=gradient;}
 }
 return make_float4(dot4(rows,wz),slopes!=0?dot4(derivatives,wz)*factor:0,slopes!=0?dot4(rows,dz)*factor:0,0);
}
// A seeded C2 field, including its analytic gradient. Wrapping the lattice
// at the navigation chart's 6144 m boundary keeps the wave warp continuous.
__device__ float4 pattern_corners(int ix,int iz,int seed,int period){
 int mask=period-1;
 return make_float4(cell((ix&mask)+seed,iz&mask),cell(((ix+1)&mask)+seed,iz&mask),cell((ix&mask)+seed,(iz+1)&mask),cell(((ix+1)&mask)+seed,(iz+1)&mask));
}
__device__ float4 pattern_field(float x,float z,int seed,int period){
 int ix=(int)floorf(x),iz=(int)floorf(z);float u=fract(x),v=fract(z);
 float du=30*u*u*(u-1)*(u-1),dv=30*v*v*(v-1)*(v-1);
 float a=u*u*u*(u*(u*6-15)+10),b=v*v*v*(v*(v*6-15)+10);
 float4 corners=pattern_corners(ix,iz,seed,period);float p=corners.x,q=corners.y,r=corners.z,t=corners.w;
 return make_float4(mixf(mixf(p,q,a),mixf(r,t,a),b),mixf(q-p,t-r,b)*du,mixf(r-p,t-q,a)*dv,0);
}
__device__ float4 wave_warp(float x,float z,int c,int axis){
 float size=c==0?24:384,amplitude=c==0?4.2f:67.2f;
 float4 n=pattern_field(x/size,z/size,axis==0?173:719,c==0?256:16);
 return make_float4((n.x-.5f)*amplitude,n.y*amplitude/size,n.z*amplitude/size,0);
}
// Both cascades vary in two dimensions. The full Jacobian keeps normals
// attached to the displaced surface; the pressure domain stays world-local.
__device__ float4 region_wave(const float4 *s,float x,float z,int smooth,int slopes,int c){
 float4 a=wave_warp(x,z,c,0),b=wave_warp(x,z,c,1);
 float4 w=smooth!=0?sample_pc(s,x+a.x,z+b.x,c,slopes):sample(s,x+a.x,z+b.x,c);
 return make_float4(w.x,w.y*(1+a.y)+w.z*b.y,w.y*a.z+w.z*(1+b.z),0);
}
// The periodic solver is sampled only inside its one world-space domain.
// A C2 taper suppresses wraparound and includes its derivative in the normal.
__device__ float4 local_pressure(const float4 *s,const float4 *brush,float x,float z,int smooth,int slopes){
 float4 domain=brush[2];x-=domain.x;z-=domain.y;
 if(domain.w==0||fabsf(x)>=12||fabsf(z)>=12)return make_float4(0,0,0,0);
 float ux=clamp01((fabsf(x)-7)/5),uz=clamp01((fabsf(z)-7)/5);
 float ax=1-ux*ux*ux*(ux*(ux*6-15)+10),az=1-uz*uz*uz*(uz*(uz*6-15)+10);
 float dx=-6*ux*ux*(ux-1)*(ux-1)*(x<0?-1:1),dz=-6*uz*uz*(uz-1)*(uz-1)*(z<0?-1:1);
 float4 w=smooth!=0?sample_pc(s,x,z,2,slopes):sample(s,x,z,2);
 return make_float4(w.x*ax*az,(w.y*ax+w.x*dx)*az,(w.z*az+w.x*dz)*ax,0);
}
__device__ float4 wave_pc(const float4 *s,const float4 *brush,float x,float z,float fade,int pressureActive,int slopes){
 float4 a=region_wave(s,x,z,1,slopes,0),b=region_wave(s,x,z,1,slopes,1),d=pressureActive!=0?local_pressure(s,brush,x,z,1,slopes):make_float4(0,0,0,0);
 return make_float4((a.x+d.x)*fade+b.x,(a.y+d.y)*fade+b.y,(a.z+d.z)*fade+b.z,0);
}
// Diagnostic only: compare reconstruction against an independent Fourier oracle.
__global__ void sample_quality_probe(const float4 *surface,const float4 *coefficients,const float4 *points,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float4 p=points[i];output[i*2]=sample(surface,p.x,p.y,0);output[i*2+1]=sample_pc(coefficients,p.x,p.y,0,1);
}
// An untouched pressure cascade is zero. Skip its bilinear reads until the
// first gesture; preserve the same additions and fading in both paths.
__device__ float4 wave(const float4 *s,const float4 *brush,float x,float z,float fade,int pressureActive){
 float4 a=region_wave(s,x,z,0,1,0),b=region_wave(s,x,z,0,1,1),d=pressureActive!=0?local_pressure(s,brush,x,z,0,1):make_float4(0,0,0,0);
 // Fade unresolved short displacement as well as its normal near the horizon.
 return make_float4((a.x+d.x)*fade+b.x,(a.y+d.y)*fade+b.y,(a.z+d.z)*fade+b.z,0);
}
__device__ float sample_height(const float4 *s,float x,float z,int c){
 float u=x*128/patch(c),v=z*128/patch(c);int ix=(int)floorf(u),iz=(int)floorf(v),base=c*16384;
 float a=fract(u),b=fract(v);
 return mixf(mixf(s[base+wrap128(iz)*128+wrap128(ix)].x,s[base+wrap128(iz)*128+wrap128(ix+1)].x,a),mixf(s[base+wrap128(iz+1)*128+wrap128(ix)].x,s[base+wrap128(iz+1)*128+wrap128(ix+1)].x,a),b);
}
// The six surface-intersection iterations normally stay in one noise cell.
// Cache its random corners once, but re-evaluate the exact quintic at every
// position. Crossing a cell takes the uncached path: no linear approximation.
struct WaveRegionCache {float4 cells;float4 shortX;float4 shortZ;float4 longX;float4 longZ;};
__device__ WaveRegionCache wave_cache(float x,float z){
 WaveRegionCache c;c.cells=make_float4(floorf(x/24),floorf(z/24),floorf(x/384),floorf(z/384));
 c.shortX=pattern_corners((int)c.cells.x,(int)c.cells.y,173,256);c.shortZ=pattern_corners((int)c.cells.x,(int)c.cells.y,719,256);
 c.longX=pattern_corners((int)c.cells.z,(int)c.cells.w,173,16);c.longZ=pattern_corners((int)c.cells.z,(int)c.cells.w,719,16);return c;
}
__device__ float2 cached_warp(float x,float z,int cascade,WaveRegionCache cache){
 float size=cascade==0?24:384,amplitude=cascade==0?4.2f:67.2f;float u=x/size,v=z/size;
 float ix=floorf(u),iz=floorf(v);u=fract(u);v=fract(v);
 float a=u*u*u*(u*(u*6-15)+10),b=v*v*v*(v*(v*6-15)+10);
 float4 cx=cascade==0?cache.shortX:cache.longX,cz=cascade==0?cache.shortZ:cache.longZ;
 if(ix!=(cascade==0?cache.cells.x:cache.cells.z)||iz!=(cascade==0?cache.cells.y:cache.cells.w)){
  cx=pattern_corners((int)ix,(int)iz,173,cascade==0?256:16);cz=pattern_corners((int)ix,(int)iz,719,cascade==0?256:16);
 }
 return make_float2((mixf(mixf(cx.x,cx.y,a),mixf(cx.z,cx.w,a),b)-.5f)*amplitude,(mixf(mixf(cz.x,cz.y,a),mixf(cz.z,cz.w,a),b)-.5f)*amplitude);
}
__device__ float cached_wave_height(const float4 *s,const float4 *coefficients,const float4 *brush,float x,float z,float fade,int pressureActive,int smooth,WaveRegionCache cache){
 float2 a=cached_warp(x,z,0,cache),b=cached_warp(x,z,1,cache);
 float h0=smooth!=0?sample_pc(coefficients,x+a.x,z+a.y,0,0).x:sample_height(s,x+a.x,z+a.y,0);
 float h1=smooth!=0?sample_pc(coefficients,x+b.x,z+b.y,1,0).x:sample_height(s,x+b.x,z+b.y,1);
 float pressure=0;if(pressureActive!=0)pressure=smooth!=0?local_pressure(coefficients,brush,x,z,1,0).x:local_pressure(s,brush,x,z,0,0).x;
 return (h0+pressure)*fade+h1;
}
// Diagnostics: query the actual interaction and regional background fields.
__global__ void domain_probe(const float4 *surface,const float4 *coefficients,const float4 *brush,const float4 *points,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float4 p=points[i];output[i*4]=local_pressure(surface,brush,p.x,p.y,0,1);
 output[i*4+1]=local_pressure(coefficients,brush,p.x,p.y,1,1);
 output[i*4+2]=region_wave(coefficients,p.x,p.y,1,1,0);
 output[i*4+3]=sample(surface,p.x-brush[2].x,p.y-brush[2].y,2);
}
// Slow, bounded sediment transport proxy driven by the resolved FFT, not a
// separate animated noise field. Long-wave orbital forcing and local pressure
// redistribute the ripple phase; short waves are attenuated at the bed.
// State is retained when paused or when the water becomes deep.
__global__ void sand_transport(const float4 *brush,const float4 *surface,const float4 *camera,float4 *sandState,float dt,float depth,int pressureActive,int useGlobeDepth){
 float waterDepth=depth;
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=128||z>=128)return;
 int i=z*128+x;float4 old=sandState[i];
 if(useGlobeDepth!=0&&camera[21].w!=0&&camera[21].x>=3)return;
 if(useGlobeDepth!=0&&camera[21].w!=0)waterDepth=camera[21].x;
 float shallow=clamp01((3-waterDepth)/2);shallow=shallow*shallow*(3-2*shallow);
 float4 w=region_wave(surface,(float)x*.75f,(float)z*.75f,0,1,1);
 float shortBed=expf(-6.2831853f*waterDepth/6);
 float4 a=region_wave(surface,(float)x*.75f,(float)z*.75f,0,1,0);
 float4 d=pressureActive!=0?local_pressure(surface,brush,(float)x*.75f,(float)z*.75f,0,1):make_float4(0,0,0,0);
 float forcing=5*(w.x+shortBed*a.x+d.x*expf(-waterDepth*.8f));
 float rate=shallow*(forcing+8*(w.y*fabsf(w.y)+w.z*fabsf(w.z)));
 float step=fminf(.05f,fmaxf(0,dt));
 // Smooth saturation keeps the bed bounded without a hard phase clipping edge.
 float next=old.x+step*(rate/(1+old.x*old.x*.25f)-shallow*.015f*old.x);
 sandState[i]=make_float4(next,0,0,0);
}
__device__ float wave_height(const float4 *s,const float4 *brush,float x,float z,float distance,int pressureActive){
 float4 ax=wave_warp(x,z,0,0),az=wave_warp(x,z,0,1),bx=wave_warp(x,z,1,0),bz=wave_warp(x,z,1,1);
 float a=sample_height(s,x+ax.x,z+az.x,0),b=sample_height(s,x+bx.x,z+bz.x,1),d=pressureActive!=0?local_pressure(s,brush,x,z,0,0).x:0;
 float fade=1/(1+distance*distance*.0008f);return (a+d)*fade+b;
}
// Geometry oracle entry: tested independently with double-precision ray equations.
__global__ void planet_probe(const float4 *points,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float4 p=points[i];float3 ray=unit(vec(p.y,p.z,p.w));float2 roots=sphere_roots(p.x,ray,0);
 output[i]=make_float4(globe_hit(p.x,ray),roots.x,roots.y,earth_radius());
}
// Navigation uses software double precision in one GPU invocation. Rendering
// uses camera-relative metres, with altitude stored separately from Earth radius.
// navigation: global unit up, transported unit east, accumulated local UV metres.
__device__ void camera_advance(float4 *camera,float2 *navigationState,float dt,float forward,float side,float up,float lookX,float lookY,float speed,float zoom,int reset,float depth){
 // Most frames change only the waves. Preserve the full camera state and
 // bypass navigation arithmetic until input, a dolly, or a diagnostic pose edit.
 if(reset==0&&forward==0&&side==0&&up==0&&lookX==0&&lookY==0&&zoom==0&&camera[1].z==0&&camera[5].w==camera[1].x&&camera[6].w==camera[1].y&&camera[8].x==camera[0].y){
  camera[2].w=depth/.86f;camera[3].w=expf(-depth*.055f);camera[8].y=fminf(30000000.0f,speed*fmaxf(1,camera[0].y*.06f));return;
 }
 double nav0=(double)navigationState[0].x+(double)navigationState[0].y;
 double nav1=(double)navigationState[1].x+(double)navigationState[1].y;
 double nav2=(double)navigationState[2].x+(double)navigationState[2].y;
 double nav3=(double)navigationState[3].x+(double)navigationState[3].y;
 double nav4=(double)navigationState[4].x+(double)navigationState[4].y;
 double nav5=(double)navigationState[5].x+(double)navigationState[5].y;
 double nav6=(double)navigationState[6].x+(double)navigationState[6].y;
 double nav7=(double)navigationState[7].x+(double)navigationState[7].y;
 float4 p=camera[0],r=camera[1];
 if(reset==3&&camera[8].z>0){
  // Space is a radial move above the observer, not a geographic reset. Keep
  // the transported frame and precise navigation so local solar time, weather
  // and the land beneath us agree with the surface after any speed of flight.
  p.y=earth_radius()*1.25f;r=make_float4(r.x,-1.5707963f,0,0);
 }else if(reset>0){
  // Also handles Space selected before the first camera frame is initialized.
  p=reset==2?make_float4(0,8,16,0):make_float4(0,2.6f,4,0);
  if(reset==3)p=make_float4(0,earth_radius()*1.25f,4,0);
  r=make_float4(0,reset==3?-1.5707963f:(reset==2?-.4f:-.32f),0,0);
  nav0=0.0;nav1=0.0;nav2=1.0;
  nav3=1.0;nav4=0.0;nav5=0.0;
  nav6=(double)p.x;nav7=(double)p.z;
 }else{
  r.x+=lookX;r.x-=floorf((r.x+3.14159265f)/6.2831853f)*6.2831853f;r.y=fminf(1.5707963f,fmaxf(-1.5707963f,r.y+lookY));
  r.z+=zoom;float zoomStep=r.z*(1-expf(-dt*10));r.z-=zoomStep;if(fabsf(r.z)<.00001f)r.z=0;
  if(zoomStep!=0){p.y=fminf(500000000.0f,fmaxf(.45f,(p.y+20)*expf(zoomStep)-20));if(zoomStep>0)r.y=mixf(r.y,-1.5707963f,eased(100,50000,p.y)*(1-expf(-dt*5)));}
  float flight=fminf(30000000.0f,speed*fmaxf(1,p.y*.06f));
  float length=fmaxf(1,sqrtf(forward*forward+side*side+up*up)),d=dt*flight/length;
  float dx=d*(sinf(r.x)*cosf(r.y)*forward+cosf(r.x)*side),dz=d*(-cosf(r.x)*cosf(r.y)*forward+sinf(r.x)*side);
  p.y=fminf(500000000.0f,fmaxf(.45f,p.y+d*(sinf(r.y)*forward+up)));
  nav6+=(double)dx;nav7+=(double)dz;
  float lengthXZ=sqrtf(dx*dx+dz*dz);
  if(lengthXZ>0){
   double nx=nav0,ny=nav1,nz=nav2,ex=nav3,ey=nav4,ez=nav5;
   double bx=ey*nz-ez*ny,by=ez*nx-ex*nz,bz=ex*ny-ey*nx;
   double tx=(ex*(double)dx+bx*(double)dz)/(double)lengthXZ,ty=(ey*(double)dx+by*(double)dz)/(double)lengthXZ,tz=(ez*(double)dx+bz*(double)dz)/(double)lengthXZ;
   double angle=(double)lengthXZ/(6371000.0+(double)p.y);
   angle-=(double)floorf((float)((angle+3.141592653589793)/6.283185307179586))*6.283185307179586;
   double a2=angle*angle;
   // Native shader trig has enough angular error to drift hundreds of metres
   // over a circumnavigation. Evaluate this one navigation rotation in double.
   double si=angle*(1.0+a2*(-1.0/6.0+a2*(1.0/120.0+a2*(-1.0/5040.0+a2*(1.0/362880.0+a2*(-1.0/39916800.0+a2*(1.0/6227020800.0+a2*(-1.0/1307674368000.0+a2*(1.0/355687428096000.0+a2*(-1.0/121645100408832000.0))))))))));
   double co=1.0+a2*(-1.0/2.0+a2*(1.0/24.0+a2*(-1.0/720.0+a2*(1.0/40320.0+a2*(-1.0/3628800.0+a2*(1.0/479001600.0+a2*(-1.0/87178291200.0+a2*(1.0/20922789888000.0+a2*(-1.0/6402373705728000.0+a2*(1.0/2432902008176640000.0))))))))));
   double xx=nx*co+tx*si,yy=ny*co+ty*si,zz=nz*co+tz*si;
   double inverse=1.0/sqrt(xx*xx+yy*yy+zz*zz);xx*=inverse;yy*=inverse;zz*=inverse;
   // Exact parallel transport of east along the current great-circle step.
   double along=ex*tx+ey*ty+ez*tz;
   ex+=along*(tx*(co-1.0)-nx*si);ey+=along*(ty*(co-1.0)-ny*si);ez+=along*(tz*(co-1.0)-nz*si);
   double projection=ex*xx+ey*yy+ez*zz;ex-=projection*xx;ey-=projection*yy;ez-=projection*zz;
   inverse=1.0/sqrt(ex*ex+ey*ey+ez*ez);
   nav0=xx;nav1=yy;nav2=zz;nav3=ex*inverse;nav4=ey*inverse;nav5=ez*inverse;
  }
  // The shared spectral patch stays within millimetre precision at any altitude.
  p.x=(float)(nav6-(double)floorf((float)((nav6+3072.0)/6144.0))*6144.0);
  p.z=(float)(nav7-(double)floorf((float)((nav7+3072.0)/6144.0))*6144.0);
 }
 camera[0]=p;camera[1]=r;
 float yawSin=sinf(r.x),yawCos=cosf(r.x),pitchSin=sinf(r.y),pitchCos=cosf(r.y);
 camera[2]=make_float4(yawSin*pitchCos,pitchSin,-yawCos*pitchCos,depth/.86f);
 camera[3]=make_float4(yawCos,0,yawSin,expf(-depth*.055f));
 camera[4]=make_float4(-yawSin*pitchSin,pitchCos,yawCos*pitchSin,0);
 float3 normal=vec((float)nav0,(float)nav1,(float)nav2),east=vec((float)nav3,(float)nav4,(float)nav5),back=crossv(east,normal);
 camera[5]=make_float4(east.x,east.y,east.z,r.x);camera[6]=make_float4(normal.x,normal.y,normal.z,r.y);camera[7]=make_float4(back.x,back.y,back.z,0);
 camera[8]=make_float4(p.y,fminf(30000000.0f,speed*fmaxf(1,p.y*.06f)),earth_radius(),r.z);
 float3 sun=unit(vec(-.42f,.63f,.66f));camera[9]=make_float4(dotv(sun,east),dotv(sun,normal),dotv(sun,back),0);
 {float hi=(float)nav0;navigationState[0]=make_float2(hi,(float)(nav0-(double)hi));}
 {float hi=(float)nav1;navigationState[1]=make_float2(hi,(float)(nav1-(double)hi));}
 {float hi=(float)nav2;navigationState[2]=make_float2(hi,(float)(nav2-(double)hi));}
 {float hi=(float)nav3;navigationState[3]=make_float2(hi,(float)(nav3-(double)hi));}
 {float hi=(float)nav4;navigationState[4]=make_float2(hi,(float)(nav4-(double)hi));}
 {float hi=(float)nav5;navigationState[5]=make_float2(hi,(float)(nav5-(double)hi));}
 {float hi=(float)nav6;navigationState[6]=make_float2(hi,(float)(nav6-(double)hi));}
 {float hi=(float)nav7;navigationState[7]=make_float2(hi,(float)(nav7-(double)hi));}
}
__global__ void camera_step(float4 *camera,float2 *navigationState,float dt,float forward,float side,float up,float lookX,float lookY,float speed,float zoom,int reset,float depth){
 camera_advance(camera,navigationState,dt,forward,side,up,lookX,lookY,speed,zoom,reset,depth);
}
// Project the pointer to the actual FFT surface on the GPU, retaining the
// previous hit so a held drag injects force along its world-space path.
__global__ void brush_pick(const float4 *surface,const float4 *camera,float4 *brush,float pointerX,float pointerY,float aspect,int held,int moving,int pressureActive){
 float4 domain=brush[2];domain.z=0;brush[2]=domain;
 float4 old=brush[1];brush[0]=make_float4(0,0,0,0);
 if(held==0||camera[0].y>500){brush[1]=make_float4(old.x,old.y,0,0);return;}
 float4 p=camera[0],r=camera[1];
 float3 f=vec(sinf(r.x)*cosf(r.y),sinf(r.y),-cosf(r.x)*cosf(r.y));
 float3 right=vec(cosf(r.x),0,sinf(r.x)),up=vec(-sinf(r.x)*sinf(r.y),cosf(r.y),cosf(r.x)*sinf(r.y));
 float3 ray=unit(plus(f,plus(scale(right,pointerX*aspect*.65f),scale(up,pointerY*.65f))));
 if(ray.y>-.06f){brush[1]=make_float4(old.x,old.y,0,0);return;}
 float t=globe_hit(p.y,ray);if(t<0||t>2000){brush[1]=make_float4(old.x,old.y,0,0);return;}
 for(int j=0;j<4;j++){float h=wave_height(surface,brush,p.x+ray.x*t,p.z+ray.z*t,t,pressureActive)-(ray.x*ray.x+ray.z*ray.z)*t*t/(2*earth_radius());t=mixf(t,(h-p.y)/ray.y,.75f);}
 float x=p.x+ray.x*t,z=p.z+ray.z*t;
 if(camera[21].w!=0&&terrain_height(camera,to_world(camera,unit(vec(ray.x*t,earth_radius()+p.y+ray.y*t,ray.z*t))))>0){brush[1]=make_float4(old.x,old.y,0,0);return;}
 if(domain.w==0||fabsf(x-domain.x)>6||fabsf(z-domain.y)>6){domain=make_float4(x,z,1,1);brush[2]=domain;}

 float dx=old.z>.5f&&moving!=0?x-old.x:0,dz=old.z>.5f&&moving!=0?z-old.y:0;
 // Bound a single event's travel so a camera teleport cannot create an explosion.
 float len=sqrtf(dx*dx+dz*dz),limit=fminf(1,.7f/fmaxf(.00001f,len));dx*=limit;dz*=limit;
 brush[0]=make_float4(x,z,dx,dz);brush[1]=make_float4(x,z,1,0);
}
// Analytic damped spectral oscillator: h'' + omega? h = moving pressure.
// Both the height and vertical velocity are complex Fourier coefficients.
__global__ void force_modes(float4 *disturbance,const float4 *brush,const float *motion,float dt,int clear){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;if(x>=128||z>=128)return;
 int idx=z*128+x;float4 state=brush[2].z!=0?make_float4(0,0,0,0):disturbance[idx];
 if(clear!=0||x==64||z==64||(x==0&&z==0)){disturbance[idx]=make_float4(0,0,0,0);return;}
 float4 b=brush[0];float distance=sqrtf(b.z*b.z+b.w*b.w);
 if(distance==0&&state.x==0&&state.y==0&&state.z==0&&state.w==0)return;
 int fx=x<64?x:x-128,fz=z<64?z:z-128;
 float kx=6.2831853f*(float)fx/24,kz=6.2831853f*(float)fz/24;
 float omega=motion[32768+idx],envelope=motion[49152+idx],decay=motion[65536+idx];
 float radius=.24f;
 float amplitude=-4.0f*fminf(.6f,distance)*6.2831853f*radius*radius/576*envelope;
 float pr=0,pi=0;
 if(distance>0)for(int j=0;j<4;j++){
  float t=((float)j+.5f)/4,px=b.x-b.z*(1-t),pz=b.y-b.w*(1-t);
  float phase=kx*(px-brush[2].x)+kz*(pz-brush[2].y);pr+=cosf(phase)*.25f;pi-=sinf(phase)*.25f;
 }
 // Displacement impulse from the moving pressure brush; exact free evolution
 // after release provides propagating wakes instead of a drawn height mask.
 state.z+=amplitude*pr;state.w+=amplitude*pi;
 float co=cosf(omega*dt),si=sinf(omega*dt),damping=expf(-decay*dt);
 disturbance[idx]=make_float4((state.x*co+state.z/omega*si)*damping,(state.y*co+state.w/omega*si)*damping,(state.z*co-state.x*omega*si)*damping,(state.w*co-state.y*omega*si)*damping);
}
__device__ float3 refract_cosine(float3 d,float3 n,float eta,float c){return minus(scale(d,eta),scale(n,eta*c+sqrtf(fmaxf(0,1-eta*eta*(1-c*c)))));}
__device__ float3 refractv(float3 d,float3 n,float eta){return refract_cosine(d,n,eta,dotv(d,n));}
__device__ float3 sunDir(){return unit(vec(-.42f,.66f,-.63f));}
__device__ float positive_power(float value,float exponent){return value>0?exp2f(log2f(value)*exponent):0;}
__device__ float3 sky(float3 d,float3 sunDirection){
 float v=positive_power(clamp01(d.y),.45f);float3 col=blend(vec(.38f,.55f,.68f),vec(.045f,.16f,.36f),v);
 // Explicit fixed powers avoid the compiler's software-f64 integer pow path.
 float sun=fmaxf(0,dotv(d,sunDirection)),s2=sun*sun,s4=s2*s2,s8=s4*s4,s16=s8*s8;
 // Below .99 the disk power is smaller than the minimum float value.
 float sunDisk=sun>.99f?positive_power(sun,16000):0;
 col=plus(col,scale(vec(1,.83f,.55f),sunDisk*20+s16*s16*s16*.07f));
 float cloud=clamp01(.5f+.25f*sinf(d.x*28+d.z*17)+.25f*sinf(d.z*43-d.x*12));
 float cloud2=cloud*cloud,cloud4=cloud2*cloud2;
 float cirrus=cloud4*cloud4*clamp01(d.y*3)*.22f;
 return scale(blend(col,vec(.94f,.96f,1),cirrus),.015f+.985f*eased(-.08f,.3f,sunDirection.y));
}
__device__ float4 bed_shape(float x,float z){
 float4 a=pattern_field(x*.125f,z*.125f,331,1024),b=pattern_field(x*.43f+17,z*.43f-31,997,1024);
 return make_float4((a.x-.5f)*.20f+(b.x-.5f)*.05f,a.y*.025f+b.y*.0215f,a.z*.025f+b.z*.0215f,0);
}
__device__ float bottom(float x,float z,float depth){return -depth+bed_shape(x,z).x;}
struct BedRegionCache {float4 cells;float4 broad;float4 fine;};
__device__ BedRegionCache bed_cache(float x,float z){
 BedRegionCache c;c.cells=make_float4(floorf(x*.125f),floorf(z*.125f),floorf(x*.43f+17),floorf(z*.43f-31));
 c.broad=pattern_corners((int)c.cells.x,(int)c.cells.y,331,1024);c.fine=pattern_corners((int)c.cells.z,(int)c.cells.w,997,1024);return c;
}
__device__ float cached_bottom(float x,float z,float depth,BedRegionCache cache){
 float u=x*.125f,v=z*.125f,s=x*.43f+17,t=z*.43f-31;
 float4 a=cache.broad,b=cache.fine;
 if(floorf(u)!=cache.cells.x||floorf(v)!=cache.cells.y)a=pattern_corners((int)floorf(u),(int)floorf(v),331,1024);
 if(floorf(s)!=cache.cells.z||floorf(t)!=cache.cells.w)b=pattern_corners((int)floorf(s),(int)floorf(t),997,1024);
 u=fract(u);v=fract(v);s=fract(s);t=fract(t);
 u=u*u*u*(u*(u*6-15)+10);v=v*v*v*(v*(v*6-15)+10);s=s*s*s*(s*(s*6-15)+10);t=t*t*t*(t*(t*6-15)+10);
 float h=(mixf(mixf(a.x,a.y,u),mixf(a.z,a.w,u),v)-.5f)*.20f+(mixf(mixf(b.x,b.y,s),mixf(b.z,b.w,s),t)-.5f)*.05f;
 return -depth+h;
}
__device__ float materialNoise(float x,float z){
 float ix=floorf(x),iz=floorf(z),u=fract(x),v=fract(z);u=u*u*(3-2*u);v=v*v*(3-2*v);
 return mixf(mixf(cell(ix,iz),cell(ix+1,iz),u),mixf(cell(ix,iz+1),cell(ix+1,iz+1),u),v);
}
__device__ float3 seabed(float x,float z,float footprint){
 float broad=materialNoise(x*.42f,z*.42f),mid=materialNoise(x*3.1f+17,z*3.1f);
 float bend=materialNoise(x*.65f+13,z*.65f-7)*16,branch=materialNoise(x*.31f-29,z*.31f+11);
 float phase=z*31+x*2.4f+bend;
 float ridge=sinf(phase),slope=cosf(phase);
 float detail=1/(1+footprint*footprint*1100);
 // Subpixel grain and gravel would flicker at reduced screen resolution.
 float grainDetail=1/(1+footprint*footprint*80000);
 float grain=grainDetail>.03f?materialNoise(x*170,z*170)*grainDetail:0;
 float illumination=.76f+detail*((.045f+.12f*branch)*ridge-(.04f+.13f*branch)*slope+.1f*grain);
 float3 sand=scale(blend(vec(.30f,.235f,.135f),vec(.49f,.41f,.26f),broad),illumination*(.85f+.2f*mid));
 // Sparse organic gravel pockets; most of the floor remains rippled sand.
 float pocket=clamp01((materialNoise(x*.7f+83,z*.7f-19)-.55f)*5);
 if(pocket>.01f&&footprint<.075f){
  float gx=floorf(x*7),gz=floorf(z*7),nearest=5,tone=0;
  for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
   float cx=gx+(float)i,cz=gz+(float)j,seed=cell(cx,cz);
   float px=(cx+cell(cx+23,cz-87))/7,pz=(cz+cell(cx-71,cz+53))/7;
   float dx=(x-px),dz=(z-pz),radius=.012f+.04f*seed;
   // The largest possible deformed ellipse fits inside this circle.
   // Outside it the original coverage is exactly zero, so skip the expensive
   // angle/edge shading without changing any visible gravel.
   if(dx*dx+dz*dz<=radius*radius*2.34f){
   float ang=seed*6.2831853f;
   float rx=dx*cosf(ang)-dz*sinf(ang),rz=dx*sinf(ang)+dz*cosf(ang);
   float angle=atan2f(rz,rx);
   float edge=1+.14f*sinf(angle*5+seed*13)+.09f*sinf(angle*9);
   float dist=sqrtf(rx*rx*1.6f+rz*rz*.65f)/(radius*edge);
   if(dist<nearest){nearest=dist;tone=seed;}
   }
  }
  float coverage=clamp01((1-nearest)/fmaxf(.10f,footprint*35))*pocket*clamp01((.075f-footprint)/.035f);
  float3 rock=blend(vec(.055f,.063f,.054f),vec(.22f,.16f,.09f),tone);
  rock=scale(rock,.65f+.6f*sqrtf(clamp01(1-nearest*nearest)));
  sand=blend(sand,rock,coverage);
 }
 return sand;
}
// PC sand is a shallow geometric relief with analytic derivatives. Slowly
// varying direction, spacing and amplitude break up the uniform stripe field.
__device__ float4 sand_relief(float x,float z,float footprint){
 float4 bend=pattern_field(x*.65f+13,z*.65f-7,541,1024),branch=pattern_field(x*.31f-29,z*.31f+11,811,1024);
 float phase=z*31+x*2.4f+bend.x*16,phaseX=2.4f+bend.y*10.4f,phaseZ=31+bend.z*10.4f;
 float detail=1/(1+footprint*footprint*1100),amp=(.0018f+.007f*branch.x)*detail;
 float ampX=.00217f*branch.y*detail,ampZ=.00217f*branch.z*detail;
 float ridge=cosf(phase)+.18f*cosf(2*phase+.6f),derivative=-sinf(phase)-.36f*sinf(2*phase+.6f);
 // A weaker oblique family creates crest joins and sheltered gaps instead of
 // one unbroken set of parallel lines. All normals differentiate this height.
 float other=z*26-x*10+bend.x*9,weight=.0024f*(1-branch.x)*detail;
 float co=cosf(other),si=-sinf(other);
 return make_float4(amp*ridge+weight*co,ampX*ridge+amp*derivative*phaseX-.000744f*branch.y*detail*co+weight*si*(-10+bend.y*5.85f),ampZ*ridge+amp*derivative*phaseZ-.000744f*branch.z*detail*co+weight*si*(26+bend.z*5.85f),phase);
}
__device__ float4 sand_moving(float x,float z,float footprint,float4 drift,float anchorX,float anchorZ){
 // Phase is locally linear over the millimetre-scale bed intersection correction.
 float phaseShift=drift.x+drift.y*(x-anchorX)+drift.z*(z-anchorZ);
 float dz=phaseShift/31;
 float4 relief=sand_relief(x,z+dz,footprint);
 relief.y+=relief.z*drift.y/31;relief.z*=1+drift.z/31;
 return relief;
}
// Rounded joins between irregular rock faces, carrying shape derivatives.
__device__ float3 rock_join(float3 a,float3 b){
 float h=clamp01(.5f+.5f*(a.x-b.x)/.09f);
 return vec(mixf(b.x,a.x,h)+.09f*h*(1-h),mixf(b.y,a.y,h),mixf(b.z,a.z,h));
}
// A rounded pebble cap is real bed height, with derivatives matching its
// geometry. w carries stone tone + 1, or a negative contact-shadow amount.
__device__ float4 stone_relief(float x,float z,float footprint,float3 sun){
 float fade=clamp01((.065f-footprint)/.035f);
 if(fade<=0)return make_float4(0,0,0,0);
 // This bound includes the maximum cap and shadow extent.
 if(materialNoise(x*.7f+83,z*.7f-19)<.32f)return make_float4(0,0,0,0);
 float gx=floorf(x*7),gz=floorf(z*7),height=0,gradientX=0,gradientZ=0,tone=-1,shadow=0;
 float vertical=sqrtf(1-.7502f*.7502f*(1-sun.y*sun.y));
 float shadowX=.7502f*sun.x/vertical,shadowZ=.7502f*sun.z/vertical;
 for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
  float cx=gx+(float)i,cz=gz+(float)j,seed=cell(cx,cz);
  float px=(cx+cell(cx+23,cz-87))/7,pz=(cz+cell(cx-71,cz+53))/7;
  float dx=x-px,dz=z-pz,radius=.018f+.036f*seed,cap=(.002f+.007f*seed)*fade;
  // Reject candidates before evaluating density or cap trigonometry.
  if(dx*dx+dz*dz>radius*radius*3)continue;
  // Density is evaluated at the stone centre, preserving whole objects.
  float density=clamp01((materialNoise(px*.7f+83,pz*.7f-19)-.55f)*5);
  if(cell(cx+131,cz-211)>density)continue;
  float angle=seed*6.2831853f,co=cosf(angle),si=sinf(angle),ax=radius*(.82f+.25f*seed),az=radius*(.74f+.17f*cell(cx-83,cz+29));
  float rx=dx*co-dz*si,rz=dx*si+dz*co,invX=1/(ax*ax),invZ=1/(az*az),u=rx/ax,v=rz/az;
  // Unequal convex faces form stone chips, rather than identical ellipses.
  float3 face=rock_join(vec(u,1,0),vec(-u*.91f+v*.16f,-.91f,.16f));
  face=rock_join(face,vec(v*(.92f+.13f*seed)+u*.14f,.14f,.92f+.13f*seed));
  face=rock_join(face,vec(-v*.88f+u*.21f,.21f,-.88f));
  face=rock_join(face,vec(u*.69f+v*.71f,.69f,.71f));
  face=rock_join(face,vec(-u*.74f-v*.62f,-.74f,-.62f));
  float q=face.x;
  if(q<1){
   // Low, partly buried stones with irregular shoulders and a tilted top.
   // The cubic shoulder joins the top and sand with continuous normals.
   float edge=clamp01((1-q)/.65f),profile=edge*edge*(3-2*edge),tilt=1+.12f*u+.07f*v;
   float derivative=edge>0&&edge<1?-6*edge*(1-edge)/.65f:0;
   float qu=face.y,qv=face.z;
   float du=cap*(derivative*qu*tilt+profile*.12f)/ax,dv=cap*(derivative*qv*tilt+profile*.07f)/az;
   float h=cap*profile*tilt;
   if(h>height){height=h;gradientX=du*co+dv*si;gradientZ=-du*si+dv*co;tone=seed;}
  }
  // Approximate projected contact shadows using the refracted mean sun ray.
  float sx=dx+shadowX*cap,sz=dz+shadowZ*cap,tx=sx*co-sz*si,tz=sx*si+sz*co;
  float sq=tx*tx*invX+tz*tz*invZ;
  shadow=fmaxf(shadow,clamp01((1.12f-sq)*4)*fade);
 }
 return make_float4(height,gradientX,gradientZ,tone>=0?tone+1:-shadow);
}
__device__ float bottom_pc(float x,float z,float depth,float footprint,float4 drift,float anchorX,float anchorZ,BedRegionCache cache){
 return cached_bottom(x,z,depth,cache)+sand_moving(x,z,footprint,drift,anchorX,anchorZ).x+stone_relief(x,z,footprint,sunDir()).x;
}
__device__ float3 seabed_pc(float x,float z,float footprint,float4 sand,float4 stone,float3 sun,float3 view){
 float broad=materialNoise(x*.42f,z*.42f),mid=materialNoise(x*3.1f+17,z*3.1f),fine=1/(1+footprint*footprint*80000);
 float grain=fine>.03f?materialNoise(x*170,z*170)*fine:0;
 float3 color=scale(blend(vec(.34f,.265f,.16f),vec(.56f,.465f,.31f),broad),.83f+.17f*mid+.055f*grain);
 float coverage=stone.w>=1?clamp01(stone.x/fmaxf(.00025f,footprint*.11f)):0;
 if(coverage>0){
  float tone=stone.w-1;
  float3 rock=blend(vec(.20f,.175f,.135f),vec(.40f,.31f,.205f),tone);
  float speckle=materialNoise(x*110+7,z*110-13);
  float flecks=1/(1+footprint*footprint*180000);
  float mineral=flecks>.03f?materialNoise(x*380-11,z*380+29)*flecks:0;
  rock=scale(rock,.76f+.3f*speckle+.07f*mineral);color=blend(color,rock,coverage);
 }
 float4 macro=bed_shape(x,z);float macroX=macro.y,macroZ=macro.z;
 float3 normal=unit(vec(-macroX-sand.y-stone.y,1,-macroZ-sand.z-stone.z));
 float shade=.26f+.84f*fmaxf(0,dotv(normal,sun));
 if(stone.w<0)shade*=1+.35f*stone.w;
 color=scale(color,shade);
 // Rough submerged mineral highlights follow the cap normal.
 float highlight=fmaxf(0,dotv(normal,unit(minus(sun,view))));
 float h2=highlight*highlight,h4=h2*h2,h8=h4*h4,h16=h8*h8;
 return plus(color,scale(vec(1,.94f,.81f),h16*h16*.003f*coverage));
}
// Diagnostics only: verify material slopes against independent finite steps
// through the actual bed relief, including sparse stone patches.
__global__ void bed_quality_probe(float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float x=((float)(i%64)-32)*.0913f,z=((float)(i/64)-32)*.0971f,e=.0001f;
 float4 drift=make_float4(.7f,.2f,-.3f,0);
 float4 sand=sand_moving(x,z,.001f,drift,0,0),stone=stone_relief(x,z,.001f,sunDir());
 float sx=(sand_moving(x+e,z,.001f,drift,0,0).x-sand_moving(x-e,z,.001f,drift,0,0).x)/(2*e),sz=(sand_moving(x,z+e,.001f,drift,0,0).x-sand_moving(x,z-e,.001f,drift,0,0).x)/(2*e);
 float4 a=stone_relief(x+e,z,.001f,sunDir()),b=stone_relief(x-e,z,.001f,sunDir()),c=stone_relief(x,z+e,.001f,sunDir()),d=stone_relief(x,z-e,.001f,sunDir());
 output[i*3]=sand;output[i*3+1]=stone;
 output[i*3+2]=make_float4(sx,sz,(a.x-b.x)/(2*e),(c.x-d.x)/(2*e));
}
// Forward sunlight transport. Bilinear photon splats accumulate all ray
// branches at folds; fixed-point atomics preserve energy on WebGPU.
__global__ void caustic_clear(unsigned *photons,int dispersion,int lightSize){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=lightSize||z>=lightSize)return;
 if(dispersion==0){photons[z*lightSize+x]=0;return;}
 int id=(z*lightSize+x)*4;photons[id]=0;photons[id+1]=0;photons[id+2]=0;photons[id+3]=0;
}
__global__ void caustic_map(const float4 *surface,const float4 *camera,unsigned *photons,float depth,int rays,int dispersion,int lightSize){
 float waterDepth=depth;
 if(camera[21].w!=0)waterDepth=camera[21].x;
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=rays||z>=rays)return;
 float wx=((float)x+.5f)*6/(float)rays,wz=((float)z+.5f)*6/(float)rays;
 float4 w=lightSize>256?sample_pc(surface,wx,wz,0,1):sample(surface,wx,wz,0);
 float3 n=unit(vec(-w.y,1,-w.z)),incident=scale(camera[13].w!=0?unit(vec(camera[9].x,fmaxf(.02f,camera[9].y),camera[9].z)):sunDir(),-1);
 int channels=dispersion!=0?3:1;
 for(int c=0;c<channels;c++){
  float eta=dispersion==0?.7502f:(c==0?.7524f:(c==1?.7502f:.7480f));
  float3 d=refractv(incident,n,eta);float distance=(-waterDepth-w.x)/d.y;
  float2 hit=make_float2(wx+d.x*distance,wz+d.z*distance);
  float u=hit.x*(float)lightSize/6-.5f,v=hit.y*(float)lightSize/6-.5f;
  int ix=(int)floorf(u),iz=(int)floorf(v);float fu=fract(u),fv=fract(v);
  for(int j=0;j<2;j++)for(int i=0;i<2;i++){
   int px=(ix+i)&(lightSize-1),pz=(iz+j)&(lightSize-1);
   float weight=(i==0?1-fu:fu)*(j==0?1-fv:fv);
   int target=dispersion==0?pz*lightSize+px:(pz*lightSize+px)*4+c;
   atomicAdd(&photons[target],(unsigned)(weight*4096+.5f));
  }
 }
}
__global__ void caustic_resolve(const unsigned *photons,float4 *light,float *monoLight,float normalization,int dispersion,int lightSize){
 // This kernel always launches complete 8 x 8 groups at either map size.
 __shared__ unsigned tile[100];
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 int id=dispersion==0?z*lightSize+x:(z*lightSize+x)*4;
 float r=0,green=0,blue=0;
 // At one mobile ray per light texel, regular splat gaps reveal a grid.
 // A periodic tent reconstruction removes that sampling pattern and preserves
 // total light energy without increasing ray count or adding a GPU pass.
 if(dispersion==0){
  int lane=threadIdx.y*8+threadIdx.x;
  for(int t=lane;t<100;t+=64){
   int px=((int)blockIdx.x*8+t%10-1)&255,pz=((int)blockIdx.y*8+t/10-1)&255;
   tile[t]=photons[pz*256+px];
  }
  __syncthreads();
  float filtered=0;
  for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
   float weight=(i==0?2.0f:1.0f)*(j==0?2.0f:1.0f);
   int index=((int)threadIdx.y+j+1)*10+(int)threadIdx.x+i+1;
   filtered+=(float)tile[index]*weight;
  }
  // Supported ray grids give a power-of-two denominator; division is exact.
  r=__fdividef(filtered,16*normalization);
 }else if(lightSize>256){
  // Reconstruct the photon lattice at the higher PC map resolution. Sharing
  // this tent tile suppresses splat bands while retaining focused light energy.
  int lane=threadIdx.y*8+threadIdx.x;
  for(int c=0;c<3;c++){
   for(int t=lane;t<100;t+=64){
    int px=((int)blockIdx.x*8+t%10-1)&(lightSize-1),pz=((int)blockIdx.y*8+t/10-1)&(lightSize-1);
    tile[t]=photons[(pz*lightSize+px)*4+c];
   }
   __syncthreads();float filtered=0;
   for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
    float weight=(i==0?2.0f:1.0f)*(j==0?2.0f:1.0f);
    int index=((int)threadIdx.y+j+1)*10+(int)threadIdx.x+i+1;filtered+=(float)tile[index]*weight;
   }
   float value=__fdividef(filtered,16*normalization);
   if(c==0)r=value;if(c==1)green=value;if(c==2)blue=value;
   __syncthreads();
  }
 }else{r=__fdividef((float)photons[id],normalization);}
 float g=dispersion!=0?(lightSize>256?green:__fdividef((float)photons[id+1],normalization)):r,b=dispersion!=0?(lightSize>256?blue:__fdividef((float)photons[id+2],normalization)):r;
 if(dispersion==0)monoLight[z*lightSize+x]=r;else light[z*lightSize+x]=make_float4(r,g,b,1);
}
__device__ float3 caustic(const float4 *light,const float *monoLight,float x,float z,int dispersion,int lightSize){
 float4 wx=wave_warp(x,z,0,0),wz=wave_warp(x,z,0,1);x+=wx.x;z+=wz.x;
 float offset=lightSize>256?.5f:0;
 float u=fract(x/6)*(float)lightSize-offset,v=fract(z/6)*(float)lightSize-offset;int ix=((int)floorf(u))&(lightSize-1),iz=((int)floorf(v))&(lightSize-1);float a=fract(u),b=fract(v);
 // Mobile sunlight has identical RGB channels. Read and interpolate it once.
 if(dispersion==0){
  float p=monoLight[iz*256+ix],q=monoLight[iz*256+(ix+1)%256],r=monoLight[((iz+1)%256)*256+ix],t=monoLight[((iz+1)%256)*256+(ix+1)%256];
  float value=mixf(mixf(p,q,a),mixf(r,t,a),b);return vec(value,value,value);
 }
 float4 p=light[iz*lightSize+ix],q=light[iz*lightSize+(ix+1)%lightSize],r=light[((iz+1)%lightSize)*lightSize+ix],t=light[((iz+1)%lightSize)*lightSize+(ix+1)%lightSize];
 return vec(mixf(mixf(p.x,q.x,a),mixf(r.x,t.x,a),b),mixf(mixf(p.y,q.y,a),mixf(r.y,t.y,a),b),mixf(mixf(p.z,q.z,a),mixf(r.z,t.z,a),b));
}
__device__ float film(float a){return clamp01((a*(2.51f*a+.03f))/(a*(2.43f*a+.59f)+.14f));}
// Seam-free 3D cloud density on the spherical normal; no latitude texture seam.
__device__ float globe_noise_shifted(float3 origin,float3 p){
 float ix=origin.x+floorf(p.x),iy=origin.y+floorf(p.y),iz=origin.z+floorf(p.z),x=fract(p.x),y=fract(p.y),z=fract(p.z);
 x=x*x*(3-2*x);y=y*y*(3-2*y);z=z*z*(3-2*z);float value=0;
 for(int k=0;k<2;k++)for(int j=0;j<2;j++)for(int i=0;i<2;i++){
  unsigned h=(unsigned)((int)(ix+(float)i)*92837111+(int)(iy+(float)j)*689287499+(int)(iz+(float)k)*283923481);
  value+=randf(h)*(i==0?1-x:x)*(j==0?1-y:y)*(k==0?1-z:z);
 }return value;
}
__device__ float globe_noise(float3 p){return globe_noise_shifted(vec(0,0,0),p);}
// Value and gradient from the same eight corners. Distant water uses this
// filtered field instead of two planet-wide sine trains.
__device__ float4 globe_gradient(float3 p){
 float ix=floorf(p.x),iy=floorf(p.y),iz=floorf(p.z),x=fract(p.x),y=fract(p.y),z=fract(p.z);
 float dx=30*x*x*(x-1)*(x-1),dy=30*y*y*(y-1)*(y-1),dz=30*z*z*(z-1)*(z-1);
 x=x*x*x*(x*(x*6-15)+10);y=y*y*y*(y*(y*6-15)+10);z=z*z*z*(z*(z*6-15)+10);
 float4 out=make_float4(0,0,0,0);
 for(int k=0;k<2;k++)for(int j=0;j<2;j++)for(int i=0;i<2;i++){
  unsigned h=(unsigned)((int)(ix+(float)i)*92837111+(int)(iy+(float)j)*689287499+(int)(iz+(float)k)*283923481);
  float a=i==0?1-x:x,b=j==0?1-y:y,c=k==0?1-z:z,v=randf(h);
  out.x+=v*a*b*c;out.y+=v*(i==0?-dx:dx)*b*c;out.z+=v*a*(j==0?-dy:dy)*c;out.w+=v*a*b*(k==0?-dz:dz);
 }return out;
}


// Seeded spherical plates. These are synthetic geography and kinematic
// landform proxies, not Earth's measured plates or a mantle/erosion solver.
// Best-candidate sites make irregular cells without a latitude-row lattice.
__device__ float3 random_sphere(unsigned id){
 float y=randf(id)*2-1,angle=randf(id+1u)*6.2831853f,r=sqrtf(fmaxf(0,1-y*y));
 return vec(sinf(angle)*r,y,cosf(angle)*r);
}
__global__ void geology_seed(float4 *plates,float4 *camera,int seedValue,int mapWidth,int offset){
 for(int i=0;i<28;i++){
  float best=-1;float3 chosen=vec(0,0,1);
  for(int candidate=0;candidate<16;candidate++){
   float3 n=random_sphere((unsigned)seedValue*49157u+(unsigned)(i*71+candidate*2));float separation=4;
   for(int j=0;j<i;j++){float4 q=plates[j*2];separation=fminf(separation,2-2*dotv(n,vec(q.x,q.y,q.z)));}
   if(separation>best){best=separation;chosen=n;}
  }
  unsigned h=(unsigned)seedValue*179u+(unsigned)i*8317u;
  float crust=randf(h)>.61f?1.0f:.0f;
  float3 spin=scale(random_sphere(h+21u),1.4f+randf(h+25u)*6.2f);
  plates[i*2]=make_float4(chosen.x,chosen.y,chosen.z,crust);
  plates[i*2+1]=make_float4(spin.x,spin.y,spin.z,30+randf(h+29u)*110);
 }
 camera[22]=make_float4((float)offset,(float)mapWidth,(float)seedValue,28);
 camera[23]=make_float4(-1,0,0,0);camera[31].y=(float)(offset+mapWidth*mapWidth/2);
}
__device__ float4 geology_cell(const float4 *plates,float3 n,int seedValue,int tectonics){
 float best=-2;int owner=0;
 for(int i=0;i<28;i++){float4 p=plates[i*2];float d=dotv(n,vec(p.x,p.y,p.z));if(d>best){best=d;owner=i;}}
 float4 site=plates[owner*2],rotation=plates[owner*2+1];float3 centre=vec(site.x,site.y,site.z);
 float3 velocity=crossv(vec(rotation.x,rotation.y,rotation.z),n);
 float total=0,crust=0,age=180,ridge=0,trench=0,uplift=0,boundary=0;
 for(int i=0;i<28;i++){
  float4 p=plates[i*2],w=plates[i*2+1];float3 other=vec(p.x,p.y,p.z);
  float weight=expf((dotv(n,other)-best)*24);total+=weight;crust+=p.w*weight;
  if(i==owner)continue;
  float3 delta=minus(other,centre);float inv=rsqrtf(dotv(delta,delta));
  // Signed relative motion normal to the bisector: positive opens a ridge.
  float3 toward=scale(delta,inv),relative=minus(crossv(vec(w.x,w.y,w.z),n),velocity);
  float opening=dotv(relative,toward),km=fmaxf(0,(best-dotv(n,other))*inv)*6371;
  float separating=fmaxf(0,opening),closing=fmaxf(0,-opening);
  if(separating>.3f)age=fminf(age,km/fmaxf(3,separating*5));
  float near=expf(-km*km/(160*160));ridge+=near*clamp01(separating/4);
  float collision=expf(-km*km/(230*230))*clamp01(closing/4);
  trench+=collision*(1-site.w)*(1-p.w*.5f);uplift+=collision*(site.w+p.w)*.5f;
  boundary+=near*opening;
 }
 crust/=total;
 float3 shift=vec(randf((unsigned)seedValue+43u)*11,randf((unsigned)seedValue+47u)*11,randf((unsigned)seedValue+53u)*11);
 float broad=globe_noise(plus(scale(n,3.7f),shift)),medium=globe_noise(plus(scale(n,11.3f),shift));
 float coastal=globe_noise(plus(scale(n,37),shift))-.5f;
 float continental=clamp01(crust*.57f+broad*.43f+(medium-.5f)*.12f+coastal*.065f);
 float relief=globe_noise(plus(scale(n,49),shift))-.5f;
 float ocean=-2600-320*sqrtf(fminf(180,age));
 if(tectonics!=0)ocean+=ridge*650-trench*4200;
 ocean+=relief*330;
 float shelf=eased(.35f,.57f,continental),interior=eased(.57f,.82f,continental);
 float h=mixf(ocean,-130,shelf)+interior*2100;
 if(tectonics!=0)h+=uplift*3600*eased(.36f,.62f,continental);
 h+=relief*interior*950;
 return make_float4(fmaxf(-11000,fminf(6500,h)),continental,boundary,(float)owner);
}
__global__ void geology_map(const float4 *plates,float4 *camera,int mapWidth,int offset,int seedValue){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=mapWidth||y>=mapWidth/2)return;
 float lon=(((float)x+.5f)/(float)mapWidth-.5f)*6.2831853f,lat=(.5f-((float)y+.5f)/(float)(mapWidth/2))*3.14159265f;
 float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
 camera[offset+y*mapWidth+x]=geology_cell(plates,n,seedValue,1);
}

// A static maximum-height pyramid bounds whole ray intervals. It is built
// once per world seed, independently of camera position or output resolution.
__global__ void geology_mip(float4 *camera,int width,int level){
 int w=width>>level,h=(width/2)>>level,x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=w||y>=h)return;
 int previous=(int)camera[22].x,offset=previous;
 for(int j=0;j<level;j++){previous=offset;offset+=(width>>j)*((width/2)>>j);}
 float peak=-12000;
 for(int j=0;j<2;j++)for(int i=0;i<2;i++)peak=fmaxf(peak,camera[previous+(y*2+j)*(w*2)+x*2+i].x);
 camera[offset+y*w+x]=make_float4(peak,0,0,0);
}
__device__ float4 geology_sample(const float4 *camera,float3 n){
 int offset=(int)camera[22].x,width=(int)camera[22].y,height=width/2;
 float u=(atan2f(n.x,n.z)/6.2831853f+.5f)*(float)width-.5f;
 float v=(.5f-atan2f(n.y,sqrtf(n.x*n.x+n.z*n.z))/3.14159265f)*(float)height-.5f;
 int x=(int)floorf(u),y=(int)floorf(v);float a=fract(u),b=fract(v);float4 out=make_float4(0,0,0,0);
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){
  int yy=y+j,xx=x+i;if(yy<0){yy=-yy-1;xx+=width/2;}if(yy>=height){yy=2*height-yy-1;xx+=width/2;}
  float4 q=camera[offset+yy*width+(xx&(width-1))];float weight=(i==0?1-a:a)*(j==0?1-b:b);
  out.x+=q.x*weight;out.y+=q.y*weight;out.z+=q.z*weight;out.w+=q.w*weight;
 }return out;
}

// The small cache stores broad geography. World-space detail bands add
// terrain relief without allocating metre-resolution data around the planet.
// The detail vanishes at the shoreline, preserving one shared coastline.
__device__ float3 terrain_rotate(float3 p){return vec(.8f*p.x+.6f*p.z,.36f*p.x+.8f*p.y-.48f*p.z,-.48f*p.x+.6f*p.y+.64f*p.z);}
__device__ float terrain_direct(const float4 *camera,float3 n){
 float h=geology_sample(camera,n).x;
 if(h>0){
  float3 shift=vec(camera[22].z*.0031f,17,31),p=plus(scale(n,713),shift);
  float coarse=globe_noise(p)*.6f+globe_noise(terrain_rotate(scale(p,2.17f)))*.4f;
  float ridge=1-fabsf(coarse*2-1);
  float mountain=(ridge*ridge-.55f)*920;
  float middle=(globe_noise(plus(terrain_rotate(scale(n,2311)),shift))-.5f)*190;
  float fine=(globe_noise(plus(terrain_rotate(scale(n,7517)),shift))-.5f)*68;
  float crags=(globe_noise(plus(terrain_rotate(scale(n,21373)),shift))-.5f)*20;
  h+=(mountain*eased(20,600,h)+(middle+fine+crags)*eased(10,250,h));
 }return h;
}

// A camera-local spherical cache amortizes terrain evaluation. Its frame is
// fixed between recentres, so fly-camera rotation never moves the land texture.
__global__ void terrain_cache_setup(float4 *camera,int width,int offset,int enabled,int nearWidth,int nearOffset){
 float3 n=vec(camera[6].x,camera[6].y,camera[6].z),old=vec(camera[27].x,camera[27].y,camera[27].z);
 float3 delta=minus(n,old);int update=enabled!=0&&camera[0].y<40000&&((int)camera[27].w!=width||camera[30].x!=camera[22].z||dotv(delta,delta)*earth_radius()*earth_radius()>16000000)?1:0;
 camera[30].y=(float)update;camera[30].z=4;
 camera[31].y=(float)nearOffset;camera[31].z=(float)nearWidth;
 float3 nearOld=vec(camera[nearOffset].x,camera[nearOffset].y,camera[nearOffset].z),nearDelta=minus(n,nearOld);
 int nearUpdate=enabled!=0&&camera[0].y<4000&&(update!=0||camera[nearOffset].w==0||camera[nearOffset+3].x!=camera[22].z||dotv(nearDelta,nearDelta)*earth_radius()*earth_radius()>65536)?1:0;
 camera[nearOffset+3].y=(float)nearUpdate;
 if(nearUpdate!=0){
  camera[nearOffset]=make_float4(n.x,n.y,n.z,(float)nearWidth);
  camera[nearOffset+1]=make_float4(camera[5].x,camera[5].y,camera[5].z,2048);
  camera[nearOffset+2]=make_float4(camera[7].x,camera[7].y,camera[7].z,0);
  camera[nearOffset+3].x=camera[22].z;camera[nearOffset+3].z+=1;
 }
 if(update==0)return;
 camera[27]=make_float4(n.x,n.y,n.z,(float)width);
 camera[28]=make_float4(camera[5].x,camera[5].y,camera[5].z,(float)offset);
 camera[29]=make_float4(camera[7].x,camera[7].y,camera[7].z,32768);
 camera[30].x=camera[22].z;
}
__global__ void terrain_camera_frame(float4 *camera,const float2 *navigationState){
 int nearOffset=(int)camera[31].y;
 // Split the exact eye position once. Fine terrain uses small metre offsets,
 // so its texture and intersections avoid Earth-radius float quantization.
 double nx=(double)navigationState[0].x+(double)navigationState[0].y,ny=(double)navigationState[1].x+(double)navigationState[1].y,nz=(double)navigationState[2].x+(double)navigationState[2].y;
 double ex=(double)navigationState[3].x+(double)navigationState[3].y,ey=(double)navigationState[4].x+(double)navigationState[4].y,ez=(double)navigationState[5].x+(double)navigationState[5].y;
 double navX=(double)navigationState[6].x+(double)navigationState[6].y,navZ=(double)navigationState[7].x+(double)navigationState[7].y;
 double ox=camera[19].z!=0?(double)camera[0].x-(navX-(double)floorf((float)((navX+3072.0)/6144.0))*6144.0):0.0;
 double oz=camera[19].z!=0?(double)camera[0].z-(navZ-(double)floorf((float)((navZ+3072.0)/6144.0))*6144.0):0.0;
 double radial=6371000.0+(double)camera[0].y,cy=sqrt(radial*radial-ox*ox-oz*oz);
 double px=nx*cy+ex*ox+(ey*nz-ez*ny)*oz,py=ny*cy+ey*ox+(ez*nx-ex*nz)*oz,pz=nz*cy+ez*ox+(ex*ny-ey*nx)*oz;
 float3 ne=vec(camera[nearOffset+1].x,camera[nearOffset+1].y,camera[nearOffset+1].z),nb=vec(camera[nearOffset+2].x,camera[nearOffset+2].y,camera[nearOffset+2].z);
 float3 ce=vec(camera[5].x,camera[5].y,camera[5].z),cn=vec(camera[6].x,camera[6].y,camera[6].z),cb=vec(camera[7].x,camera[7].y,camera[7].z);
 camera[nearOffset+4]=make_float4(dotv(ce,ne),dotv(cn,ne),dotv(cb,ne),(float)(px*(double)ne.x+py*(double)ne.y+pz*(double)ne.z));
 camera[nearOffset+5]=make_float4(dotv(ce,nb),dotv(cn,nb),dotv(cb,nb),(float)(px*(double)nb.x+py*(double)nb.y+pz*(double)nb.z));
 double gx=(.8*px+.6*pz)*.19,gy=(.36*px+.8*py-.48*pz)*.19,gz=(-.48*px+.6*py+.64*pz)*.19;
 float ix=floorf((float)gx),iy=floorf((float)gy),iz=floorf((float)gz);
 camera[nearOffset+6]=make_float4(ix,iy,iz,0);camera[nearOffset+7]=make_float4((float)(gx-(double)ix),(float)(gy-(double)iy),(float)(gz-(double)iz),0);
}
__global__ void terrain_cache(float4 *camera,int width){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=width||y>=width||camera[30].y==0)return;
 float u=(((float)x+.5f)/(float)width-.5f)*32768/earth_radius(),v=(((float)y+.5f)/(float)width-.5f)*32768/earth_radius();
 float c=sqrtf(fmaxf(0,1-u*u-v*v));float4 a=camera[27],b=camera[28],d=camera[29];
 float3 n=vec(a.x*c+b.x*u+d.x*v,a.y*c+b.y*u+d.y*v,a.z*c+b.z*u+d.z*v);
 float h=terrain_direct(camera,n);camera[(int)b.w+y*width+x]=make_float4(h,0,0,0);
}
__global__ void terrain_cache_mip(float4 *camera,int width,int level){
 int w=width>>level,x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=w||y>=w||camera[30].y==0)return;
 int previous=(int)camera[28].w,offset=previous;
 for(int j=0;j<level;j++){previous=offset;int side=width>>j;offset+=side*side;}
 float peak=-12000;
 for(int j=0;j<2;j++)for(int i=0;i<2;i++)peak=fmaxf(peak,camera[previous+(y*2+j)*(w*2)+x*2+i].x);
 camera[offset+y*w+x]=make_float4(peak,0,0,0);
}
__device__ float4 terrain_weights(float a){
 float a2=a*a,a3=a2*a,b=1-a;
 return make_float4(b*b*b/6,(4-6*a2+3*a3)/6,(1+3*a+3*a2-3*a3)/6,a3/6);
}

__device__ float terrain_coarse_height(const float4 *camera,float3 n){
 float fade=1-eased(20000,180000,camera[0].y);
 if(fade==0)return geology_sample(camera,n).x;
 float4 centre=camera[27],east=camera[28],back=camera[29];int width=(int)centre.w;
 float u=dotv(n,vec(east.x,east.y,east.z))*earth_radius()/32768+.5f,v=dotv(n,vec(back.x,back.y,back.z))*earth_radius()/32768+.5f;
 float h=0;
 if(width>0&&camera[31].x==0&&u>.01f&&u<.99f&&v>.01f&&v<.99f&&dotv(n,vec(centre.x,centre.y,centre.z))>.99f){
  float x=u*(float)width-.5f,y=v*(float)width-.5f,a=fract(x),b=fract(y);int ix=(int)floorf(x),iy=(int)floorf(y),offset=(int)east.w;
  // C2 reconstruction avoids a visible cell lattice in the terrain normal.
  int taps=(int)camera[30].z;float4 wx=terrain_weights(a),wy=terrain_weights(b);
  for(int j=0;j<taps;j++){
   int row=offset+(iy+j-1)*width+ix-1;float w=j==0?wy.x:(j==1?wy.y:(j==2?wy.z:wy.w));
   h+=(camera[row].x*wx.x+camera[row+1].x*wx.y+camera[row+2].x*wx.z+camera[row+3].x*wx.w)*w;
  }
  float edge=fminf(fminf(u,1-u),fminf(v,1-v));if(edge<.06f)h=mixf(terrain_direct(camera,n),h,eased(.01f,.06f,edge));
 }else h=terrain_direct(camera,n);
 return fade<1?mixf(geology_sample(camera,n).x,h,fade):h;
}
__device__ float4 terrain_coarse_gradient(const float4 *camera,float3 n){
 float4 centre=camera[27],east=camera[28],back=camera[29];int width=(int)centre.w;
 float u=dotv(n,vec(east.x,east.y,east.z))*earth_radius()/32768+.5f,v=dotv(n,vec(back.x,back.y,back.z))*earth_radius()/32768+.5f;
 float h=0,gx=0,gz=0;
 if(width>0&&camera[31].x==0&&u>.06f&&u<.94f&&v>.06f&&v<.94f&&dotv(n,vec(centre.x,centre.y,centre.z))>.99f){
  float x=u*(float)width-.5f,y=v*(float)width-.5f,a=fract(x),b=fract(y);int ix=(int)floorf(x),iy=(int)floorf(y),offset=(int)east.w;
  // C2 reconstruction avoids a visible cell lattice in the terrain normal.
  int taps=(int)camera[30].z;float4 wx=terrain_weights(a),wy=terrain_weights(b);
  float4 dx=make_float4(-.5f*(1-a)*(1-a),-2*a+1.5f*a*a,.5f+a-1.5f*a*a,.5f*a*a);
  float4 dz=make_float4(-.5f*(1-b)*(1-b),-2*b+1.5f*b*b,.5f+b-1.5f*b*b,.5f*b*b);
  for(int j=0;j<taps;j++){
   int row=offset+(iy+j-1)*width+ix-1;float v0=camera[row].x,v1=camera[row+1].x,v2=camera[row+2].x,v3=camera[row+3].x;
   float w=j==0?wy.x:(j==1?wy.y:(j==2?wy.z:wy.w)),d=j==0?dz.x:(j==1?dz.y:(j==2?dz.z:dz.w));
   float value=v0*wx.x+v1*wx.y+v2*wx.z+v3*wx.w;
   h+=value*w;gx+=(v0*dx.x+v1*dx.y+v2*dx.z+v3*dx.w)*w;gz+=value*d;
  }

 }else return make_float4(0,0,0,-100000);

 float factor=(float)width/32768;
 float3 grad=vec(east.x*gx+back.x*gz,east.y*gx+back.y*gz,east.z*gx+back.z*gz);
 grad=scale(minus(grad,scale(n,dotv(n,grad))),factor);
 return make_float4(grad.x,grad.y,grad.z,h);
}
// A nested 2 km patch resolves close terrain without raising the global cache
// resolution. Build only after 256 m travel, and keep the same frame on rotation.
// Hermite samples store height and both derivatives: four vector loads replace
// the sixteen scalar loads used by the broad terrain spline.
__global__ void terrain_near_cache(float4 *camera,int width){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,offset=(int)camera[31].y;
 if(x>=width||y>=width||camera[offset+3].y==0)return;
 float4 c=camera[offset],e=camera[offset+1],b=camera[offset+2];
 float u=(((float)x+.5f)/(float)width-.5f)*2048/earth_radius(),v=(((float)y+.5f)/(float)width-.5f)*2048/earth_radius();
 float f=sqrtf(fmaxf(0,1-u*u-v*v));float3 n=vec(c.x*f+e.x*u+b.x*v,c.y*f+e.y*u+b.y*v,c.z*f+e.z*u+b.z*v);
 float h=terrain_coarse_height(camera,n),amplitude=2.4f*eased(0,12,h),seed=camera[22].z*.013f;
 float3 p=plus(terrain_rotate(scale(n,earth_radius()/38)),vec(seed,21,49));
 float relief=(globe_noise(p)-.5f)*1.45f+(globe_noise(terrain_rotate(scale(p,2.71f)))-.5f)*.55f;
 camera[offset+8+y*width+x]=make_float4(h+amplitude*relief,0,0,0);
}
__global__ void terrain_near_coefficients(float4 *camera,int width){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,offset=(int)camera[31].y;
 if(x>=width||y>=width||camera[offset+3].y==0)return;
 int xm=x>0?x-1:x,xp=x+1<width?x+1:x,ym=y>0?y-1:y,yp=y+1<width?y+1:y,source=offset+8;
 float h=camera[source+y*width+x].x,dx=(camera[source+y*width+xp].x-camera[source+y*width+xm].x)*.5f,dz=(camera[source+yp*width+x].x-camera[source+ym*width+x].x)*.5f;
 float mixed=(camera[source+yp*width+xp].x-camera[source+yp*width+xm].x-camera[source+ym*width+xp].x+camera[source+ym*width+xm].x)*.25f;
 camera[source+width*width+y*width+x]=make_float4(h,dx,dz,mixed);
}
struct TerrainNear { float h; float3 gradient; float weight; float3 weightGradient; };
__device__ float2 terrain_cubic_pair(float4 a,float4 b,float4 w){return make_float2(a.x*w.x+a.y*w.y+b.x*w.z+b.y*w.w,a.z*w.x+a.w*w.y+b.z*w.z+b.w*w.w);}
__device__ TerrainNear terrain_near_sample_uv(const float4 *camera,float3 n,float u,float v){
 TerrainNear out;out.h=0;out.gradient=vec(0,0,0);out.weight=0;out.weightGradient=vec(0,0,0);
 int offset=(int)camera[31].y;if(offset==0||camera[31].x!=0||camera[0].y>=20000)return out;
 float4 c=camera[offset],e=camera[offset+1],b=camera[offset+2];int width=(int)c.w;
 if(width==0||dotv(n,vec(c.x,c.y,c.z))<.999f)return out;
 float edge=fminf(fminf(u,1-u),fminf(v,1-v));if(edge<=.06f)return out;
 float x=u*(float)width-.5f,y=v*(float)width-.5f,a=fract(x),d=fract(y),a2=a*a,d2=d*d;int ix=(int)floorf(x),iy=(int)floorf(y),source=offset+8+width*width;
 float4 wx=make_float4(2*a2*a-3*a2+1,a2*a-2*a2+a,-2*a2*a+3*a2,a2*a-a2),wz=make_float4(2*d2*d-3*d2+1,d2*d-2*d2+d,-2*d2*d+3*d2,d2*d-d2);
 float4 dx=make_float4(6*a2-6*a,3*a2-4*a+1,-6*a2+6*a,3*a2-2*a),dz=make_float4(6*d2-6*d,3*d2-4*d+1,-6*d2+6*d,3*d2-2*d);
 int row=source+iy*width+ix;float4 q00=camera[row],q10=camera[row+1],q01=camera[row+width],q11=camera[row+width+1];
 float2 r0=terrain_cubic_pair(q00,q10,wx),r1=terrain_cubic_pair(q01,q11,wx),s0=terrain_cubic_pair(q00,q10,dx),s1=terrain_cubic_pair(q01,q11,dx);
 out.h=r0.x*wz.x+r0.y*wz.y+r1.x*wz.z+r1.y*wz.w;
 float gx=s0.x*wz.x+s0.y*wz.y+s1.x*wz.z+s1.y*wz.w,gz=r0.x*dz.x+r0.y*dz.y+r1.x*dz.z+r1.y*dz.w;
 float3 g=scale(vec(e.x*gx+b.x*gz,e.y*gx+b.y*gz,e.z*gx+b.z*gz),(float)width/2048);out.gradient=minus(g,scale(n,dotv(g,n)));
 // Relief is already zero with zero slope at sea level in the source field.
 // Keep the inexpensive reconstruction in submerged parts of the patch too.
 float t=clamp01((edge-.06f)/.14f);out.weight=t*t*(3-2*t);
 float sign=u<.5f?1:-1;float3 axis=vec(e.x,e.y,e.z);if(fminf(v,1-v)<fminf(u,1-u)){sign=v<.5f?1:-1;axis=vec(b.x,b.y,b.z);}
 axis=minus(axis,scale(n,dotv(axis,n)));
 out.weightGradient=scale(axis,sign*6*t*(1-t)/(.14f*2048));
 return out;
}
__device__ TerrainNear terrain_near_sample(const float4 *camera,float3 n){
 int offset=(int)camera[31].y;float4 e=camera[offset+1],b=camera[offset+2];
 float u=dotv(n,vec(e.x,e.y,e.z))*earth_radius()/2048+.5f,v=dotv(n,vec(b.x,b.y,b.z))*earth_radius()/2048+.5f;
 return terrain_near_sample_uv(camera,n,u,v);
}
__device__ float terrain_height(const float4 *camera,float3 n){
 TerrainNear near=terrain_near_sample(camera,n);if(near.weight>=1)return near.h;
 return mixf(terrain_coarse_height(camera,n),near.h,near.weight);
}
__device__ float4 terrain_cached_gradient(const float4 *camera,float3 n){
 TerrainNear near=terrain_near_sample(camera,n);if(near.weight>=1)return make_float4(near.gradient.x,near.gradient.y,near.gradient.z,near.h);
 float4 coarse=terrain_coarse_gradient(camera,n);if(near.weight<=0||coarse.w< -90000)return coarse;
 float3 gradient=plus(blend(vec(coarse.x,coarse.y,coarse.z),near.gradient,near.weight),scale(near.weightGradient,near.h-coarse.w));
 return make_float4(gradient.x,gradient.y,gradient.z,mixf(coarse.w,near.h,near.weight));
}
__device__ float3 terrain_world(const float4 *camera,float3 ray,float t){
 return to_world(camera,unit(vec(ray.x*t,earth_radius()+camera[0].y+ray.y*t,ray.z*t)));
}
__device__ TerrainNear terrain_near_ray(const float4 *camera,float3 ray,float t){
 int offset=(int)camera[31].y;float4 e=camera[offset+4],b=camera[offset+5];
 float radius=earth_radius()+terrain_ray_altitude(camera[0].y,ray,t),factor=earth_radius()/(radius*2048);
 float u=(e.w+dotv(vec(e.x,e.y,e.z),ray)*t)*factor+.5f,v=(b.w+dotv(vec(b.x,b.y,b.z),ray)*t)*factor+.5f;
 return terrain_near_sample_uv(camera,terrain_world(camera,ray,t),u,v);
}
__device__ float terrain_ray_height(const float4 *camera,float3 ray,float t){
 TerrainNear near=terrain_near_ray(camera,ray,t);if(near.weight>=1)return near.h;
 return mixf(terrain_coarse_height(camera,terrain_world(camera,ray,t)),near.h,near.weight);
}
__device__ float4 terrain_ray_gradient(const float4 *camera,float3 ray,float t){
 TerrainNear near=terrain_near_ray(camera,ray,t);if(near.weight>=1)return make_float4(near.gradient.x,near.gradient.y,near.gradient.z,near.h);
 float4 coarse=terrain_coarse_gradient(camera,terrain_world(camera,ray,t));if(near.weight<=0||coarse.w< -90000)return coarse;
 float3 g=plus(blend(vec(coarse.x,coarse.y,coarse.z),near.gradient,near.weight),scale(near.weightGradient,near.h-coarse.w));
 return make_float4(g.x,g.y,g.z,mixf(coarse.w,near.h,near.weight));
}
__device__ float terrain_clearance(const float4 *camera,float3 ray,float t){
 float hy=camera[0].y+ray.y*t,x=ray.x*t,z=ray.z*t,R=earth_radius();
 // Rationalized altitude keeps metre precision beside a 6371 km radius.
 float altitude=hy+(x*x+z*z)/(sqrtf((R+hy)*(R+hy)+x*x+z*z)+R+hy);
 return altitude-terrain_ray_height(camera,ray,t);
}
__device__ float terrain_ray_altitude(float altitude,float3 ray,float t){
 float y=altitude+ray.y*t,x=ray.x*t,z=ray.z*t,R=earth_radius();
 return y+(x*x+z*z)/(sqrtf((R+y)*(R+y)+x*x+z*z)+R+y);
}
// Reuse the spline height and derivative in one fetch. A screen-tile hit is
// only a starting estimate; every subpixel still solves its own surface ray.
__device__ float terrain_refine(const float4 *camera,float3 ray,float t,float radius){
 float initial=t,altitude=camera[0].y;float3 worldRay=to_world(camera,ray);
 for(int j=0;j<5;j++){
  if(t<=0||fabsf(t-initial)>radius)return -1;
  float3 n=terrain_world(camera,ray,t);float4 g=terrain_ray_gradient(camera,ray,t);
  if(altitude>=20000)return -1;
  float h=g.w> -90000?g.w:terrain_ray_height(camera,ray,t);if(h<=0)return -1;
  float height=terrain_ray_altitude(altitude,ray,t),error=height-h;
  // The world normal is float32: below this range-scaled residual, another
  // refinement mainly changes coordinate rounding rather than the hit.
  if(fabsf(error)<.004f+t*.0000008f)return t;
  float3 radial=unit(vec(ray.x*t,earth_radius()+altitude+ray.y*t,ray.z*t));
  float derivative=dotv(radial,ray);
  if(g.w> -90000)derivative-=dotv(vec(g.x,g.y,g.z),worldRay)*earth_radius()/(earth_radius()+height);
  else derivative=(terrain_clearance(camera,ray,t+64)-error)/64;
  if(derivative>=-.02f)return -1;
  t-=error/derivative;
 }return -1;
}


__device__ float2 geology_uv(float3 n,int width){
 return make_float2((atan2f(n.x,n.z)/6.2831853f+.5f)*(float)width-.5f,(.5f-atan2f(n.y,sqrtf(n.x*n.x+n.z*n.z))/3.14159265f)*(float)(width/2)-.5f);
}
__device__ float terrain_local_upper(const float4 *camera,float3 a,float3 b){
 float4 c=camera[27],e=camera[28],d=camera[29];int width=(int)c.w;
 if(width==0||camera[31].x!=0||camera[0].y>=20000||dotv(a,vec(c.x,c.y,c.z))<.999f||dotv(b,vec(c.x,c.y,c.z))<.999f)return -20000;
 float au=dotv(a,vec(e.x,e.y,e.z))*earth_radius()/32768+.5f,av=dotv(a,vec(d.x,d.y,d.z))*earth_radius()/32768+.5f;
 float bu=dotv(b,vec(e.x,e.y,e.z))*earth_radius()/32768+.5f,bv=dotv(b,vec(d.x,d.y,d.z))*earth_radius()/32768+.5f;
 float lowX=fminf(au,bu),highX=fmaxf(au,bu),lowY=fminf(av,bv),highY=fmaxf(av,bv);
 if(lowX<.08f||lowY<.08f||highX>.92f||highY>.92f)return -20000;
 // Three source cells enclose the spline support and the short chart arc.
 int x0=(int)floorf(lowX*(float)width-.5f)-3,x1=(int)ceilf(highX*(float)width-.5f)+3,y0=(int)floorf(lowY*(float)width-.5f)-3,y1=(int)ceilf(highY*(float)width-.5f)+3;
 int span=x1-x0>y1-y0?x1-x0:y1-y0,level=0,offset=(int)e.w;
 while((1<<level)<span){int side=width>>level;offset+=side*side;level++;}
 int w=width>>level;float peak=-12000;
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){int x=(i==0?x0:x1)>>level,y=(j==0?y0:y1)>>level;peak=fmaxf(peak,camera[offset+y*w+x].x);}
 return peak>0?peak+4.01f:peak+.01f;
}
__device__ float geology_height_uv(const float4 *camera,float u,float v){
 int width=(int)camera[22].y,offset=(int)camera[22].x,x=(int)floorf(u),y=(int)floorf(v);float a=fract(u),b=fract(v);
 return mixf(mixf(camera[offset+y*width+(x&(width-1))].x,camera[offset+y*width+((x+1)&(width-1))].x,a),mixf(camera[offset+(y+1)*width+(x&(width-1))].x,camera[offset+(y+1)*width+((x+1)&(width-1))].x,a),b);
}
__device__ float terrain_upper(const float4 *camera,float3 ray,float start,float end,float padding){
 int width=(int)camera[22].y;float3 a=terrain_world(camera,ray,start),b=terrain_world(camera,ray,end);
 float local=terrain_local_upper(camera,a,b);if(local> -15000)return fmaxf(0,local);
 if(fabsf(a.y)>.985f||fabsf(b.y)>.985f)return 7200;
 float2 p=geology_uv(a,width),q=geology_uv(b,width);q.x-=floorf((q.x-p.x)/(float)width+.5f)*(float)width;
 float lowY=fminf(p.y,q.y),highY=fmaxf(p.y,q.y);
 // Latitude can turn along a great circle; include its analytic extremum.
 float3 direction=to_world(camera,ray);float ny=camera[6].y,denominator=direction.y*ray.y-ny;
 if(fabsf(denominator)>.0000001f){
  float critical=-(earth_radius()+camera[0].y)*(direction.y-ny*ray.y)/denominator;
  if(critical>start&&critical<end){float3 n=terrain_world(camera,ray,critical);if(fabsf(n.y)>.985f)return 7200;float y=geology_uv(n,width).y;lowY=fminf(lowY,y);highY=fmaxf(highY,y);}
 }
 // Small ray segments use the actual bilinear rectangle, not the maximum of
 // an entire 39 km cell. Padding also encloses local spline source positions.
 float pad=padding/fmaxf(.15f,sqrtf(1-fmaxf(a.y*a.y,b.y*b.y)));
 float lowX=fminf(p.x,q.x)-pad,highX=fmaxf(p.x,q.x)+pad;lowY-=padding;highY+=padding;
 if(floorf(lowX)==floorf(highX)&&floorf(lowY)==floorf(highY)&&lowY>=0&&highY<(float)(width/2-1)){
  float peak=fmaxf(fmaxf(geology_height_uv(camera,lowX,lowY),geology_height_uv(camera,highX,lowY)),fmaxf(geology_height_uv(camera,lowX,highY),geology_height_uv(camera,highX,highY)));
  return peak<=0?0:peak+414*eased(20,600,peak)+139*eased(10,250,peak)+4.1f;
 }
 int x0=(int)floorf(lowX),x1=(int)ceilf(highX),y0=(int)floorf(lowY),y1=(int)ceilf(highY);
 y0=y0<0?0:y0;y1=y1>=width/2?width/2-1:y1;
 int span=x1-x0>y1-y0?x1-x0:y1-y0,level=0,offset=(int)camera[22].x;if(span>=width/2)return 7200;
 while((1<<level)<span&&((width/2)>>level)>1){offset+=(width>>level)*((width/2)>>level);level++;}
 int w=width>>level,h=(width/2)>>level,xa=(int)floorf((float)x0/(float)(1<<level)),xb=(int)floorf((float)x1/(float)(1<<level)),ya=y0>>level,yb=y1>>level;
 float peak=-12000;
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){int x=i==0?xa:xb,y=j==0?ya:yb;y=y>=h?h-1:y;peak=fmaxf(peak,camera[offset+y*w+(x&(w-1))].x);}
 if(peak<=0)return 0;
 // Exact amplitude bounds for the procedural land detail, also enclosing
 // its positive, normalized B-spline reconstruction.
 return peak+414*eased(20,600,peak)+139*eased(10,250,peak)+4.1f;
}
__device__ float terrain_trace(const float4 *camera,float3 ray,float sea,float upperHint){
 if(camera[21].w==0)return -1;
 float altitude=camera[0].y;
 // Bound an entirely submerged local segment using the global bilinear
 // cell's maximum gradient, including the local spline support radius.
 float4 wet=camera[25];
 if(camera[31].w==0&&altitude<1000&&sea>0&&sea<2500&&wet.z>sea+(wet.w>0?0:160)&&wet.x+wet.y*(sea+(wet.w>0?0:160))<-.01f)return -1;
 if(upperHint==0)return -1;float2 bound=sphere_roots(altitude,ray,upperHint>0?upperHint:7200);if(bound.y<=0)return -1;
 // From orbit, solve the displaced spherical height field with bounded
 // fixed-point refinements. Close flight uses a first-crossing bracket.
 if(altitude>30000&&sea>0&&ray.y<-.15f){
  float t=sea,h=terrain_height(camera,terrain_world(camera,ray,t));
  if(h<=0)return -1;
  for(int j=0;j<7;j++){float2 root=sphere_roots(altitude,ray,fmaxf(0,h));if(root.x<=0)return -1;t=root.x;h=terrain_height(camera,terrain_world(camera,ray,t));}
  return h>0?t:-1;
 }
 float start=fmaxf(0,bound.x),end=sea>0?fminf(sea,bound.y):bound.y;
 if(end<=start)return -1;
 if(camera[31].w==0&&end<wet.z){
  float nearest=fminf(end,fmaxf(0,-(earth_radius()+altitude)*ray.y));
  if(terrain_ray_altitude(altitude,ray,nearest)>wet.x+wet.y*end+4.01f)return -1;
 }
 // A local tangent-plane seed resolves nearby shores/hills without spending
 // the horizon budget on metre-sized steps. Refine against spherical height.
 float slope=camera[24].y*ray.x+camera[24].z*ray.z-ray.y;
 if(slope>.00001f){
  float t=(altitude-camera[24].x)/slope;
  if(t>0&&t<fminf(2500,end)){
   float refined=terrain_refine(camera,ray,t,2500);
   if(refined>0&&refined<end)return refined;
  }
 }
 if(ray.y>.65f)return -1;
 // Keep the fallback sample positions independent of screen-tile bounds.
 // Moving a silhouette across a tile must not change which thin ridge is hit.
 bound=sphere_roots(altitude,ray,7200);start=fmaxf(0,bound.x);end=sea>0?fminf(sea,bound.y):bound.y;
 float previous=start;int detailSteps=fabsf(ray.y)<.02f?64:16;
 for(int group=0;group<16;group++){
  float f0=(float)group/16,f1=(float)(group+1)/16;
  float a=start+(end-start)*f0*f0,b=start+(end-start)*f1*f1;
  if(camera[31].w==0){
   float peak=terrain_upper(camera,ray,a,b,.02f),nearest=fminf(b,fmaxf(a,-(earth_radius()+altitude)*ray.y));
   if(peak==0||terrain_ray_altitude(altitude,ray,nearest)>peak+.01f){previous=b;continue;}
  }
  for(int k=1;k<=detailSteps;k++){
   int i=group*detailSteps+k;
   float f=(float)i/(float)(16*detailSteps),t=start+(end-start)*f*f;
   if(terrain_clearance(camera,ray,t)<0){
    float lo=previous,hi=t;
    for(int j=0;j<10;j++){float mid=(lo+hi)*.5f;if(terrain_clearance(camera,ray,mid)>0)lo=mid;else hi=mid;}
    float hit=(lo+hi)*.5f;return terrain_ray_height(camera,ray,hit)>0?hit:-1;
   }previous=t;
  }
 }return -1;
}
__device__ float4 underwater_bound(const float4 *camera,float3 n,float h){
 int width=(int)camera[22].y,height=width/2,offset=(int)camera[22].x;
 float latitude=atan2f(n.y,sqrtf(n.x*n.x+n.z*n.z));
 float u=(atan2f(n.x,n.z)/6.2831853f+.5f)*(float)width-.5f,v=(.5f-latitude/3.14159265f)*(float)height-.5f;
 int x=(int)floorf(u),y=(int)floorf(v);if(y<0||y>=height-1)return make_float4(h,1,0,0);
 float a=camera[offset+y*width+(x&(width-1))].x,b=camera[offset+y*width+((x+1)&(width-1))].x;
 float c=camera[offset+(y+1)*width+(x&(width-1))].x,d=camera[offset+(y+1)*width+((x+1)&(width-1))].x;
 float cell=earth_radius()*6.2831853f/(float)width,dx=cell*fmaxf(.001f,cosf(fabsf(latitude)+6.2831853f/(float)width));
 float gx=fmaxf(fabsf(b-a),fabsf(d-c))/dx,gz=fmaxf(fabsf(c-a),fabsf(d-b))/cell;
 float margin=fminf(fminf(fract(u),1-fract(u))*dx,fminf(fract(v),1-fract(v))*cell);
 return make_float4(h,sqrtf(gx*gx+gz*gz)*1.05f,margin,0);
}


// Share an expanded geographic height bound across each 8x8 pixel tile.
// The maximum pyramid queries include two extra globe cells on every side.
__global__ void terrain_tile_heights(float4 *camera,float *hits,int width,int height,int samples,int offset){
 int tx=blockIdx.x*blockDim.x+threadIdx.x,ty=blockIdx.y*blockDim.y+threadIdx.y,tw=(width+7)/8,th=(height+7)/8;
 if(tx>=tw||ty>=th)return;
 if(tx==0&&ty==0)camera[26]=make_float4((float)offset,(float)samples,(float)width,(float)height);
 float maximum=0;
 if(camera[0].y>30000||camera[31].w!=0)maximum=7200;
 else for(int i=0;i<5;i++){
  float px=(float)(tx*8)+(i==4?4:(i&1)*8),py=(float)(ty*8)+(i==4?4:(i>>1)*8);
  float sx=(2*px/(float)width-1)*(float)width/(float)height,sy=1-2*py/(float)height;
  float3 f=vec(camera[2].x,camera[2].y,camera[2].z),r=vec(camera[3].x,camera[3].y,camera[3].z),u=vec(camera[4].x,camera[4].y,camera[4].z);
  float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));float sea=globe_hit(camera[0].y,ray),peak=7200;
  for(int j=0;j<3;j++){
   float2 bounds=sphere_roots(camera[0].y,ray,peak);if(bounds.y<=0){peak=0;break;}
   float start=fmaxf(0,bounds.x),end=sea>0?fminf(sea,bounds.y):bounds.y;
   peak=terrain_upper(camera,ray,start,end,2);if(peak<=0)break;
  }
  maximum=fmaxf(maximum,peak);
 }
 float seed=-1,furthest=0;int found=0;
 if(camera[0].y<20000&&camera[31].w==0&&maximum>0)for(int i=0;i<5;i++){
  float px=(float)(tx*8)+(i==4?4:(i&1)*8),py=(float)(ty*8)+(i==4?4:(i>>1)*8);
  float sx=(2*px/(float)width-1)*(float)width/(float)height,sy=1-2*py/(float)height;
  float3 f=vec(camera[2].x,camera[2].y,camera[2].z),r=vec(camera[3].x,camera[3].y,camera[3].z),u=vec(camera[4].x,camera[4].y,camera[4].z);
  float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));
  float t=terrain_trace(camera,ray,globe_hit(camera[0].y,ray),maximum);
  if(t>0){seed=seed<0?t:fminf(seed,t);furthest=fmaxf(furthest,t);found++;}
 }
 // Discontinuous silhouette tiles trace independently. Smooth tiles start
 // from the nearest of five hits so a rear ridge cannot seed a nearer face.
 if(found!=5||furthest-seed>fmaxf(32,seed*.08f))seed=-1;
 int tile=offset+width*height*samples+2*(ty*tw+tx);hits[tile]=maximum;hits[tile+1]=seed;
}
__device__ float terrain_tile_upper(const float *hits,const float4 *camera,int x,int y){
 int width=(int)camera[26].z,height=(int)camera[26].w,samples=(int)camera[26].y,offset=(int)camera[26].x;
 return hits[offset+width*height*samples+2*((y/8)*((width+7)/8)+x/8)];
}
// Keep terrain intersection work out of the optical/material shader. Every
// PC subpixel gets its own exact ray hit; this is not a reduced-resolution pass.
// PC reuses the otherwise unused monochrome-light binding. Mobile appends
// one hit plane after its light map, preserving the eight-buffer render limit.
__global__ void terrain_intersections(float4 *camera,float *hits,int width,int height,int samples,int offset,int view){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,sample=blockIdx.z;
 if(x>=width||y>=height||sample>=samples)return;
 float upper=terrain_tile_upper(hits,camera,x,y);
 if(view>=3||upper==0){hits[offset+sample*width*height+y*width+x]=-1;return;}
 float px=(float)x+(samples>1?((float)(sample&1)+.5f)*.5f:.5f),py=(float)y+(samples>1?((float)(sample>>1)+.5f)*.5f:.5f);
 float sx=(2*px/(float)width-1)*(float)width/(float)height,sy=1-2*py/(float)height;
 float3 f=vec(camera[2].x,camera[2].y,camera[2].z),r=vec(camera[3].x,camera[3].y,camera[3].z),u=vec(camera[4].x,camera[4].y,camera[4].z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));
 float hit=-1,seed=hits[offset+width*height*samples+2*((y/8)*((width+7)/8)+x/8)+1];
 if(view<3){
  if(seed>0&&camera[31].w==0)hit=terrain_refine(camera,ray,seed,fmaxf(16,seed*.015f));
  if(hit<=0)hit=terrain_trace(camera,ray,globe_hit(camera[0].y,ray),upper);
 }
 hits[offset+sample*width*height+y*width+x]=hit;
}
__device__ float terrain_pixel_hit(const float *hits,const float4 *camera,int width,int height,float x,float y,int view){
 if(camera[21].w==0||view>=3)return -1;
 int sample=camera[26].y>1?((int)(fract(y)*2))*2+(int)(fract(x)*2):0;
 return hits[(int)camera[26].x+sample*width*height+(int)y*width+(int)x];
}
__device__ float4 cached_water_bound(const float4 *camera,float3 n,float h,float4 fallback){
 if(camera[27].w==0||camera[31].x!=0)return fallback;
 float4 east=camera[28],back=camera[29];int width=(int)camera[27].w,offset=(int)east.w;
 float u=(dotv(n,vec(east.x,east.y,east.z))*earth_radius()/32768+.5f)*(float)width-.5f,v=(dotv(n,vec(back.x,back.y,back.z))*earth_radius()/32768+.5f)*(float)width-.5f;
 int x=(int)floorf(u),y=(int)floorf(v),radius=(int)camera[30].z+2;
 if(x<radius+2||y<radius+2||x>=width-radius-2||y>=width-radius-2)return fallback;
 float dx=0,dz=0;
 for(int j=-radius;j<=radius;j++)for(int i=-radius;i<=radius;i++){
  float a=camera[offset+(y+j)*width+x+i].x,b=camera[offset+(y+j)*width+x+i+1].x,c=camera[offset+(y+j+1)*width+x+i].x;
  dx=fmaxf(dx,fabsf(b-a));dz=fmaxf(dz,fabsf(c-a));
 }
 float cell=32768/(float)width,margin=((float)(radius-2)-fmaxf(fract(u),fract(v)))*cell;
 return make_float4(h,sqrtf(dx*dx+dz*dz)/cell*1.01f,margin,1);
}
__global__ void geology_update(float4 *camera,float depth,int enabled){
 float3 n=vec(camera[6].x,camera[6].y,camera[6].z);float4 g=geology_sample(camera,n);
 float ground=enabled!=0?terrain_height(camera,n):0;
 float actual=enabled!=0?fmaxf(.15f,-ground):depth,previous=camera[23].x;
 camera[21]=make_float4(actual,ground,g.y,(float)enabled);
 camera[23]=make_float4(actual,fabsf(actual-previous)>.001f?1.0f:0.0f,g.z,g.w);
 camera[2].w=actual/.86f;camera[3].w=expf(-actual*.055f);
 float3 east=vec(camera[5].x,camera[5].y,camera[5].z),back=vec(camera[7].x,camera[7].y,camera[7].z);
 float e=128/earth_radius(),sx=(terrain_height(camera,unit(plus(n,scale(east,e))))-terrain_height(camera,unit(minus(n,scale(east,e)))))/256;
 float sz=(terrain_height(camera,unit(plus(n,scale(back,e))))-terrain_height(camera,unit(minus(n,scale(back,e)))))/256;
 camera[24]=make_float4(ground,sx,sz,camera[0].y-ground);camera[25]=cached_water_bound(camera,n,ground,underwater_bound(camera,n,g.x));
 float eyeClearance=camera[20].z==1?1.70f:2;
 if(enabled!=0&&ground>0&&camera[0].y<ground+eyeClearance){camera[0].y=ground+eyeClearance;camera[8].x=ground+eyeClearance;camera[24].w=eyeClearance;}
}
__global__ void terrain_probe(const float4 *points,const float4 *camera,float4 *output,int count,int rays){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float4 p=points[i];float3 n=unit(vec(p.x,p.y,p.z));
 if(rays==2){output[i]=terrain_cached_gradient(camera,n);}
 else if(rays!=0){float t=terrain_trace(camera,n,globe_hit(camera[0].y,n),-1);output[i]=make_float4(t,t>0?terrain_clearance(camera,n,t):0,t>0?terrain_height(camera,terrain_world(camera,n,t)):0,0);}
 else output[i]=make_float4(terrain_height(camera,n),geology_sample(camera,n).x,0,0);
}
// Test-only oracle: a dense first-crossing trace without screen seeds, cached
// height bounds or Newton shortcuts. It never enters the normal frame graph.
__global__ void terrain_screen_probe(const float4 *points,const float4 *camera,const float *hits,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 int width=(int)camera[26].z,height=(int)camera[26].w,samples=(int)camera[26].y;
 float4 p=points[i];int x=(int)p.x,y=(int)p.y,sample=(int)p.z,steps=p.w>0?(int)p.w:1024;
 float px=(float)x+(samples>1?((float)(sample&1)+.5f)*.5f:.5f),py=(float)y+(samples>1?((float)(sample>>1)+.5f)*.5f:.5f);
 float sx=(2*px/(float)width-1)*(float)width/(float)height,sy=1-2*py/(float)height;
 float3 f=vec(camera[2].x,camera[2].y,camera[2].z),r=vec(camera[3].x,camera[3].y,camera[3].z),u=vec(camera[4].x,camera[4].y,camera[4].z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));
 float actual=hits[(int)camera[26].x+sample*width*height+y*width+x],reference=-1;
 float2 bound=sphere_roots(camera[0].y,ray,7200);float sea=globe_hit(camera[0].y,ray);
 float start=fmaxf(0,bound.x),end=sea>0?fminf(sea,bound.y):bound.y,previous=start;
 if(end>start)for(int j=1;j<=steps;j++){
  float q=(float)j/(float)steps,t=start+(end-start)*q*q;
  if(terrain_clearance(camera,ray,t)<0){
   float lo=previous,hi=t;
   for(int k=0;k<18;k++){float mid=(lo+hi)*.5f;if(terrain_clearance(camera,ray,mid)>0)lo=mid;else hi=mid;}
   float hit=(lo+hi)*.5f;if(terrain_ray_height(camera,ray,hit)>0)reference=hit;break;
  }previous=t;
 }
 output[i]=make_float4(actual,reference,actual>0?terrain_clearance(camera,ray,actual):0,terrain_tile_upper(hits,camera,x,y));
}
__global__ void geology_probe(const float4 *points,const float4 *plates,const float4 *camera,float4 *output,int count,int seedValue){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float4 p=points[i];float3 n=unit(vec(p.x,p.y,p.z));
 output[i*3]=geology_cell(plates,n,seedValue,1);output[i*3+1]=geology_sample(camera,n);output[i*3+2]=geology_cell(plates,n,seedValue,0);
}
__device__ float3 geology_color(float4 g,int mode){
 if(mode==4){unsigned id=(unsigned)((int)(g.w+.5f));float edge=clamp01(fabsf(g.z)/5);return blend(vec(.18f+randf(id*17u)*.5f,.18f+randf(id*17u+1u)*.5f,.18f+randf(id*17u+2u)*.5f),g.z>0?vec(.90f,.30f,.10f):vec(.15f,.65f,.85f),edge*.65f);}
 float d=fmaxf(1.4f,-g.x);float3 c=blend(vec(.02f,.12f,.30f),vec(.004f,.008f,.035f),eased(2000,10000,d));
 c=blend(c,vec(.025f,.42f,.44f),1-eased(80,2500,d));return blend(c,vec(.61f,.80f,.61f),expf(-d/24));
}

// Global weather is a deterministic, driven circulation approximation, not a
// forecast. All coordinates are Earth-fixed unit vectors: no longitude seam.
// Wind-memory and rain-plane ideas are adapted from ../ClearWater, with the
// planar passing front replaced by moving spherical pressure systems.
__device__ float3 rotate_y(float3 p,float angle){float c=cosf(angle),v=sinf(angle);return vec(c*p.x+v*p.z,p.y,-v*p.x+c*p.z);}
__device__ float3 globe_sun(float clock,float season){
 float day=clock/86400,declination=.3977885f*sinf((season+day-81)*.01720242f);
 float angle=(.5f-fract(day))*6.2831853f,c=sqrtf(1-declination*declination);
 return vec(sinf(angle)*c,declination,cosf(angle)*c);
}
// Mean-radius circular Moon, not an astronomical ephemeris. Physical scale:
// https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
__device__ float3 globe_moon(float clock){
 float days=clock/86400,orbit=days*6.2831853f/27.32166f+1.6f;
 float angle=orbit-days*6.2831853f,latitude=.44f*sinf(orbit);
 return scale(vec(sinf(angle)*cosf(latitude),sinf(latitude),cosf(angle)*cosf(latitude)),384400000);
}
__device__ float4 weather_cell(float3 n,float clock,float season,int component){
 float days=clock/86400,ay=fabsf(n.y),summer=.07f*sinf((season+days-81)*.01720242f);
 float tropics=expf(-(n.y-summer)*(n.y-summer)*120),mid=expf(-(ay-.70f)*(ay-.70f)*40);
 float dry=expf(-(ay-.43f)*(ay-.43f)*90),polar=eased(.80f,.98f,ay);
 float3 east=vec(n.z,0,-n.x),north=minus(vec(0,1,0),scale(n,n.y));
 float zonal=-7+24*mid-4*polar;
 float meridional=-4*sinf(n.y*6.2831853f);
 float3 velocity=plus(scale(east,zonal),scale(north,meridional));
 float depression=0,front=0;
 for(int i=0;i<6;i++){
  float hemisphere=i<3?1.0f:-1.0f,fi=(float)i;
  float latitude=hemisphere*(.46f+.15f*sinf(fi*2.3f+days*.19f));
  float longitude=fi*2.3999632f+.7f+days*(.14f+.03f*sinf(fi*7));
  float co=sqrtf(1-latitude*latitude);float3 centre=vec(sinf(longitude)*co,latitude,cosf(longitude)*co);
  float cosine=dotv(n,centre),radius=.105f+.028f*sinf(fi*3.1f+1);
  float r2=fmaxf(0,2*(1-cosine))/(radius*radius);
  if(r2>36)continue;
  float life=.52f+.48f*sinf(days*.67f+fi*1.73f),strength=.40f+.60f*life*life;
  float core=expf(-r2*.5f)*strength;
  float3 inward=minus(centre,scale(n,cosine));
  // Opposite circulation in each hemisphere; vanishes continuously at a pole.
  velocity=plus(velocity,scale(crossv(n,inward),-hemisphere*260*core));
  velocity=plus(velocity,scale(inward,32*core));
  depression+=core;
  float3 tangentEast=vec(centre.z,0,-centre.x),tangentNorth=minus(vec(0,1,0),scale(centre,centre.y));
  float x=dotv(n,tangentEast)/radius,z=dotv(n,tangentNorth)/radius;
  float angle=atan2f(z,x)*hemisphere+sqrtf(r2)*2.3f-days*.4f;
  float band=.5f+.5f*cosf(angle);
  front+=expf(-r2*.12f)*strength*(.25f+.75f*band*band);
 }
 if(component!=0)return make_float4(velocity.x,velocity.y,velocity.z,clamp01(depression));
 // Differential advection follows the prevailing east/west wind belts.
 float3 q=rotate_y(n,-days*zonal*.01356f);
 float broad=globe_noise(plus(scale(q,9),vec(days*.018f,3,7)));
 float detail=globe_noise(plus(scale(q,27),vec(11,days*.024f,-9)));
 float cover=clamp01(.16f+.36f*tropics+.20f*mid-.26f*dry+.62f*front+(broad-.5f)*1.1f+(detail-.5f)*.32f);
 float rain=eased(.57f,.93f,cover)*clamp01(.30f*tropics+depression*.9f+front*.35f);
 float heat=dotv(n,globe_sun(clock,season));
 float temperature=29-53*n.y*n.y+3*heat-cover*2;
 return make_float4(cover,rain,temperature,1016+9*dry-30*clamp01(depression));
}
// Padded storage after the small camera state keeps render bindings within
// WebGPU's portable eight-storage-buffer limit, including the PC sand renderer.
// [32, 65568): two global maps; [65568, 131104): cached hemispherical sky.
__device__ float4 weather_map_sample(const float4 *camera,float3 n,int component){
 int width=(int)camera[13].z,height=width/2,count=width*height;
 float u=(atan2f(n.x,n.z)/6.2831853f+.5f)*(float)width-.5f;
 float v=(.5f-atan2f(n.y,sqrtf(n.x*n.x+n.z*n.z))/3.14159265f)*(float)height-.5f;
 int x=(int)floorf(u),y=(int)floorf(v);float a=fract(u),b=fract(v);float4 out=make_float4(0,0,0,0);
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){
  int yy=y+j,xx=x+i;
  // Reflect over a pole and turn longitude by 180 degrees, never clamp a row.
  if(yy<0){yy=-yy-1;xx+=width/2;}if(yy>=height){yy=2*height-yy-1;xx+=width/2;}
  float4 p=camera[32+component*count+yy*width+(xx&(width-1))];
  float weight=(i==0?1-a:a)*(j==0?1-b:b);
  out.x+=p.x*weight;out.y+=p.y*weight;out.z+=p.z*weight;out.w+=p.w*weight;
 }return out;
}
__global__ void weather_map(float4 *camera,float clock,float season,int mapSize){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=mapSize||y>=mapSize/2)return;
 float lon=(((float)x+.5f)/(float)mapSize-.5f)*6.2831853f,lat=(.5f-((float)y+.5f)/(float)(mapSize/2))*3.14159265f;
 float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));int id=y*mapSize+x;
 camera[32+id]=weather_cell(n,clock,season,0);camera[32+mapSize*mapSize/2+id]=weather_cell(n,clock,season,1);
}
__device__ float cloud_filtered_column(const float4 *camera,float3 normal,int layer,float cover,float footprint){
 if(cover<.04f)return 0;
 float days=camera[13].x/86400,ay=fabsf(normal.y);
 float mid=expf(-(ay-.7f)*(ay-.7f)*40),zonal=-7+24*mid-4*eased(.8f,.98f,ay);
 float3 q=rotate_y(normal,-days*zonal*.01356f);
 // Rotate and bend successive scales so their Cartesian lattice axes cannot
 // line up into repeated blocks. Density is independent of camera altitude.
 float coarse=globe_noise(plus(scale(q,31),vec(11,17,31)));
 float medium=globe_noise(plus(terrain_rotate(scale(q,83)),vec(3+coarse*3,29,7-coarse*2)));
 float fine=globe_noise(plus(terrain_rotate(terrain_rotate(scale(q,211))),vec(medium*2,17,31)));
 float detail=globe_noise(terrain_rotate(scale(q,layer==0?823:397)));
 float shape=coarse*.48f+medium*.32f+fine*.20f;
 float threshold=.61f-cover*.32f;
 float density=fmaxf(0,shape-threshold)*3.2f;
 // Kilometre-scale formations open actual gaps within the large weather
 // systems. Weak detail on top of a positive column only made a flat sheet.
 float resolved=1/(1+footprint*footprint*.0000000625f);
 float formation=eased(.62f-cover*.35f,.83f-cover*.35f,mixf(.5f,detail,resolved)*.8f+fine*.2f);
 return clamp01(density)*formation*eased(.015f,.14f,density)*(.60f+.40f*medium)*(layer==0?1.0f:.35f)*eased(.03f,.35f,cover);
}
__device__ float cloud_column(const float4 *camera,float3 normal,int layer,float cover){
 return cloud_filtered_column(camera,normal,layer,cover,0);
}
__device__ float cloud_density(const float4 *camera,float3 normal,int layer){
 return cloud_column(camera,normal,layer,weather_map_sample(camera,normal,0).x);
}
__global__ void weather_update(float4 *camera,float clock,float season,float time,float dt,float baseWind,int enabled,int mapSize,int skyWidth,int refresh){
 float3 n=vec(camera[6].x,camera[6].y,camera[6].z),east=vec(camera[5].x,camera[5].y,camera[5].z),back=vec(camera[7].x,camera[7].y,camera[7].z);
 camera[13]=make_float4(clock,time,(float)mapSize,(float)enabled);
 camera[16]=make_float4((float)skyWidth,(float)(skyWidth/4),65568,(float)(mapSize*mapSize/2));
 float3 sun=enabled!=0?globe_sun(clock,season):unit(vec(-.42f,.63f,.66f));
 camera[9]=make_float4(dotv(sun,east),dotv(sun,n),dotv(sun,back),0);
 camera[12]=make_float4(sun.x,sun.y,sun.z,(.5f-fract(clock/86400))*6.2831853f);
 float4 field=weather_map_sample(camera,n,0),flow=weather_map_sample(camera,n,1);
 float3 v=vec(flow.x,flow.y,flow.z);float vx=dotv(v,east),vz=dotv(v,back),wind=sqrtf(vx*vx+vz*vz);
 float4 old=camera[10],previous=camera[18];float change=1-dotv(n,vec(previous.x,previous.y,previous.z));
 float amount=camera[19].w==0||change>.01f||refresh!=0?1:1-expf(-dt/25);
 float x=mixf(old.x,vx,amount),z=mixf(old.y,vz,amount),speed=sqrtf(x*x+z*z);
 camera[10]=make_float4(x,z,speed,enabled!=0?field.x:0);
 float direct=1;
 if(enabled!=0){
  float mu=fmaxf(.12f,dotv(sun,n));
  float3 projected=unit(plus(n,scale(minus(sun,scale(n,dotv(n,sun))),2500/(earth_radius()*mu))));
  direct=expf(-(cloud_density(camera,projected,0)*2.6f+cloud_density(camera,projected,1)*2.0f));
 }
 camera[11]=make_float4(enabled!=0?field.y:0,direct,field.z,field.w);
 camera[14]=make_float4(1-expf(-dt/45),1-expf(-dt/160),1-expf(-dt/240),1-expf(-dt/800));
 camera[15]=make_float4(baseWind,v.x,v.y,v.z);
 float localHour=fract(clock/86400+atan2f(n.x,n.z)/6.2831853f)*24;
 camera[17]=make_float4(wind,field.y,atan2f(n.y,sqrtf(n.x*n.x+n.z*n.z))*57.29578f,localHour);
 camera[18]=make_float4(n.x,n.y,n.z,0);camera[19].w=1;
 float3 moon=globe_moon(clock);camera[7].w=moon.x;camera[9].w=moon.y;camera[18].w=moon.z;
}
__global__ void weather_probe(const float4 *points,float4 *output,const float4 *camera,int count,float season){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float4 p=points[i];float3 n=unit(vec(p.x,p.y,p.z)),sun=globe_sun(p.w,season);
 output[i*4]=weather_cell(n,p.w,season,0);output[i*4+1]=weather_cell(n,p.w,season,1);
 output[i*4+2]=make_float4(sun.x,sun.y,sun.z,dotv(n,sun));output[i*4+3]=weather_map_sample(camera,n,0);
}
// A reviewable way to visit an actual rainy cell, not a local storm override.
__global__ void weather_visit(float4 *camera,float2 *navigationState,float latitude,float longitude,float altitude,int findRain,int mapSize){
 float lat=latitude*.0174532925f,lon=longitude*.0174532925f;
 if(findRain!=0){
  float best=-1;int bx=0,by=0;float3 sun=vec(camera[12].x,camera[12].y,camera[12].z);
  for(int y=1;y<mapSize/2;y+=2)for(int x=0;x<mapSize;x+=2){
   float la=(.5f-((float)y+.5f)/(float)(mapSize/2))*3.14159265f,lo=(((float)x+.5f)/(float)mapSize-.5f)*6.2831853f;
   float3 n=vec(sinf(lo)*cosf(la),sinf(la),cosf(lo)*cosf(la));
   float score=camera[32+y*mapSize+x].y*(.15f+.85f*clamp01(dotv(n,sun)));
   if(score>best){best=score;bx=x;by=y;}
  }
  lat=(.5f-((float)by+.5f)/(float)(mapSize/2))*3.14159265f;lon=(((float)bx+.5f)/(float)mapSize-.5f)*6.2831853f;
 }
 float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat)),east=vec(cosf(lon),0,-sinf(lon));
 navigationState[0]=make_float2(n.x,0);navigationState[1]=make_float2(n.y,0);navigationState[2]=make_float2(n.z,0);
 navigationState[3]=make_float2(east.x,0);navigationState[4]=make_float2(east.y,0);navigationState[5]=make_float2(east.z,0);
 navigationState[6]=make_float2(0,0);navigationState[7]=make_float2(4,0);
 camera[0]=make_float4(0,altitude,4,0);camera[1]=make_float4(0,-.23f,0,0);camera[5].w=-1000;camera[19].w=0;
}

// Locate a daylight coast or deep basin in the generated field. Interpolating
// a sea-level crossing gives a shallow spawn instead of an arbitrary deep cell.
__global__ void geology_visit(float4 *camera,float2 *navigationState,float targetDepth,float clock,float season){
 int width=(int)camera[22].y,height=width/2,offset=(int)camera[22].x;
 float best=-100000000,bx=.5f*(float)width,by=.5f*(float)height;float3 sun=globe_sun(clock,season);
 for(int y=2;y<height-2;y+=2)for(int x=0;x<width;x+=2){
  float h=camera[offset+y*width+x].x,h2=camera[offset+y*width+((x+2)&(width-1))].x;
  float fraction=0;int crossing=(h+targetDepth)*(h2+targetDepth)<0?1:0;
  if(crossing!=0)fraction=clamp01((-targetDepth-h)/(h2-h))*2;
  if(targetDepth>0&&targetDepth<20&&crossing==0)continue;
  if(targetDepth<0&&h<300)continue;
  float lon=(((float)x+.5f+fraction)/(float)width-.5f)*6.2831853f,lat=(.5f-((float)y+.5f)/(float)height)*3.14159265f;
  float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
  float lighting=targetDepth<0?1-fabsf(dotv(n,sun)-.55f):dotv(n,sun);
  float score=lighting*20-fabsf(h+targetDepth)*(crossing!=0&&targetDepth>0?0:.004f)-fabsf(n.y)*2;
  if(targetDepth<0)score+=fminf(4,fmaxf(0,-camera[offset+y*width+x].z))*1.8f;
  if(score>best){best=score;bx=(float)x+.5f+fraction;by=(float)y+.5f;}
 }
 float lon=(bx/(float)width-.5f)*6.2831853f,lat=(.5f-by/(float)height)*3.14159265f;
 float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
 // Correct the two-cell coarse bracket against the actual bilinear sampler.
 if(targetDepth>0&&targetDepth<20){
  float lo=lon-6.2831853f*2/(float)width,hi=lon+6.2831853f*2/(float)width;
  float sign=geology_sample(camera,vec(sinf(lo)*cosf(lat),sinf(lat),cosf(lo)*cosf(lat))).x+targetDepth;
  for(int j=0;j<18;j++){float mid=(lo+hi)*.5f;float h=geology_sample(camera,vec(sinf(mid)*cosf(lat),sinf(lat),cosf(mid)*cosf(lat))).x+targetDepth;if(h*sign>0)lo=mid;else hi=mid;}
  lon=(lo+hi)*.5f;n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
 }
 float3 east=vec(cosf(lon),0,-sinf(lon));
 navigationState[0]=make_float2(n.x,0);navigationState[1]=make_float2(n.y,0);navigationState[2]=make_float2(n.z,0);
 navigationState[3]=make_float2(east.x,0);navigationState[4]=make_float2(east.y,0);navigationState[5]=make_float2(east.z,0);
 navigationState[6]=make_float2(0,0);navigationState[7]=make_float2(4,0);
 camera[0]=make_float4(0,targetDepth<0?terrain_direct(camera,n)+900:(targetDepth<20?2.6f:8),4,0);camera[1]=make_float4(0,-.32f,0,0);camera[5].w=-1000;camera[19].w=0;
}
// Two curved cloud decks share the exact density used by near-water shadows.
// Bounded shell integration avoids a full-screen volumetric march on phones.
__device__ float4 globe_clouds(float3 ray,const float4 *camera,float ground){
 float altitude=camera[0].y,R=earth_radius(),trans=1;float3 color=vec(0,0,0),sun=vec(camera[9].x,camera[9].y,camera[9].z);
 for(int j=0;j<2;j++){
  int layer=altitude>4000?1-j:j;float shell=layer==0?1600:4400;
  float2 roots=sphere_roots(altitude,ray,shell);float t=roots.x>0?roots.x:roots.y;
  if(t<=0||(ground>0&&t>ground))continue;
  float3 local=unit(vec(ray.x*t,R+altitude+ray.y*t,ray.z*t)),n=to_world(camera,local);
  float density=cloud_density(camera,n,layer);if(density<.001f)continue;
  float viewing=fmaxf(.15f,fabsf(dotv(local,ray))),opacity=1-expf(-density*(layer==0?2.8f:1.7f)/viewing);
  float daylight=eased(-.08f,.26f,dotv(local,sun)),upper=dotv(local,ray)<0?1.0f:.0f;
  float forward=positive_power(fmaxf(0,dotv(ray,sun)),12);
  float silver=forward*positive_power(1-density,3)*.9f;
  float light=.008f+daylight*(.07f+upper*.22f+.60f*expf(-density*1.8f)+silver);
  float3 lit=scale(blend(vec(.56f,.65f,.75f),vec(1,.97f,.90f),clamp01(upper+silver)),light);
  color=plus(color,scale(lit,trans*opacity));trans*=1-opacity;
 }return make_float4(color.x,color.y,color.z,1-trans);
}
// View-dependent cloud volume, cached below output resolution. Shadowing is a
// bounded vertical optical-depth approximation; the density is truly 3D.
__device__ float4 cloud_volume(float3 ray,const float4 *camera,float ground,float pixelAngle,float jitter){
 float altitude=camera[0].y,R=earth_radius();float2 outer=sphere_roots(altitude,ray,8200),inner=sphere_roots(altitude,ray,800);
 if(outer.y<=0)return make_float4(0,0,0,0);
 float start=fmaxf(0,outer.x),end=outer.y;
 if(altitude<800)start=fmaxf(start,inner.y);
 else if(inner.x>0)end=fminf(end,inner.x);
 if(ground>0)end=fminf(end,ground);
 if(end<=start)return make_float4(0,0,0,0);
 // In atmosphere, optical haze hides the far tail. Spend the fixed march
 // budget on nearby formations instead of kilometres of empty horizon.
 end=fminf(end,start+mixf(90000,1000000,eased(16000,90000,altitude)));
 int steps=camera[16].x>256?32:12;float span=end-start,trans=1;float3 color=vec(0,0,0),sun=vec(camera[9].x,camera[9].y,camera[9].z);
 float concentrate=1-eased(12000,60000,altitude);
 float phase=.45f+.65f*positive_power(fmaxf(0,dotv(ray,sun)),8);
 for(int i=0;i<steps;i++){
  if(trans<.015f)break;
  float a=(float)i/(float)steps,b=(float)(i+1)/(float)steps;a=mixf(a,a*a,concentrate);b=mixf(b,b*b,concentrate);
  // Stratification decorrelates ray samples from equal-height density slices.
  // Keep the pattern fixed in screen space: no time-varying noise or history.
  float step=(b-a)*span,t=start+(a+(b-a)*fract(jitter+(float)i*.61803399f))*span;float3 p=vec(ray.x*t,R+altitude+ray.y*t,ray.z*t);
  float radius=sqrtf(dotv(p,p)),h=terrain_ray_altitude(altitude,ray,t);float3 local=scale(p,1/radius),normal=to_world(camera,local);
  float cover=weather_map_sample(camera,normal,0).x;
  float top=2200+cover*5300,vertical=(h-800)/(top-800);
  if(vertical<=0||vertical>=1||cover<.08f)continue;
  float column=cloud_filtered_column(camera,normal,0,cover,t*pixelAngle);if(column<.008f)continue;
  // Denser formations grow taller; sparse edges taper into separate puffs.
  top=2200+cover*6000*(.32f+.68f*eased(.025f,.40f,column));vertical=(h-800)/(top-800);
  if(vertical>=1)continue;
  float days=camera[13].x/86400,ay=fabsf(normal.y),zonal=-7+24*expf(-(ay-.7f)*(ay-.7f)*40)-4*eased(.8f,.98f,ay);
  float3 q=terrain_rotate(rotate_y(scale(normal,radius*.00085f),-days*zonal*.01356f));
  float billow=globe_noise(q),detail=globe_noise(plus(terrain_rotate(scale(q,2.37f)),vec(billow*2,11,7)));
  // Resolved billows erode the top, not just the opacity of a flat slab.
  // Integrating wide orbital footprints averages the smallest erosion band.
  float erosion=mixf(.5f,detail,1/(1+step*step*.000001f));
  float ceiling=.38f+.48f*billow+.14f*erosion;
  float envelope=eased(0,.10f,vertical)*(1-eased(ceiling-.26f,ceiling,vertical));
  float density=fmaxf(0,column*(.55f+billow*.90f)-(1-erosion)*.08f)*envelope;
  if(density<.003f)continue;
  float mu=dotv(local,sun),daylight=eased(-.08f,.22f,mu);
  float optical=fmaxf(0,800+(top-800)*ceiling-h)*column*.0017f/fmaxf(.15f,mu);
  float direct=expf(-optical),fill=.10f+.18f*expf(-density*3);
  float3 ambient=blend(vec(.055f,.080f,.13f),vec(.34f,.42f,.52f),vertical);
  float3 lit=plus(scale(ambient,.015f+daylight*(.4f+fill)),scale(vec(1,.96f,.86f),daylight*(direct*.9f+fill*.45f)*phase));
  float opacity=1-expf(-density*step*.0023f);
  color=plus(color,scale(lit,trans*opacity));trans*=1-opacity;
 }
 return make_float4(color.x,color.y,color.z,1-trans);
}
__global__ void weather_cloud_view(float4 *camera,const float *hits,int cloudWidth,int cloudHeight,float aspect){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=cloudWidth||y>=cloudHeight)return;
 if(x==0&&y==0)camera[20]=make_float4((float)cloudWidth,(float)cloudHeight,camera[20].z,0);
 float sx=(2*((float)x+.5f)/(float)cloudWidth-1)*aspect,sy=1-2*((float)y+.5f)/(float)cloudHeight;
 float3 f=vec(camera[2].x,camera[2].y,camera[2].z),r=vec(camera[3].x,camera[3].y,camera[3].z),u=vec(camera[4].x,camera[4].y,camera[4].z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));
 float ground=globe_hit(camera[0].y,ray),land=terrain_trace(camera,ray,ground,camera[21].w!=0?terrain_tile_upper(hits,camera,(int)(((float)x+.5f)/(float)cloudWidth*camera[26].z),(int)(((float)y+.5f)/(float)cloudHeight*camera[26].w)):-1);
 if(land>0)ground=land;
 camera[131104+y*cloudWidth+x]=cloud_volume(ray,camera,ground,1.3f/(float)cloudHeight,randf((unsigned)(x+y*cloudWidth)+7919u));
}
// Filter the low-resolution premultiplied cloud cache once, rather than adding
// expensive noise-removal taps to every full-resolution optical sample.
__global__ void weather_cloud_filter(float4 *camera,float4 *scratch,int width,int height,int axis){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 int radius=camera[16].x>256?2:1;float4 sum=make_float4(0,0,0,0);
 for(int i=-radius;i<=radius;i++){
  int xx=x+(axis==0?i:0),yy=y+(axis==1?i:0);xx=xx<0?0:(xx>=width?width-1:xx);yy=yy<0?0:(yy>=height?height-1:yy);
  float weight=radius==1?(i==0?.5f:.25f):(i==0?.375f:(i==1||i==-1?.25f:.0625f));
  float4 p=axis==0?camera[131104+yy*width+xx]:scratch[yy*width+xx];sum.x+=p.x*weight;sum.y+=p.y*weight;sum.z+=p.z*weight;sum.w+=p.w*weight;
 }
 if(axis==0)scratch[y*width+x]=sum;else camera[131104+y*width+x]=sum;
}
__device__ float4 cloud_view_sample(const float4 *camera,float sx,float sy){
 int width=(int)camera[20].x,height=(int)camera[20].y;
 float u=fminf((float)width-1.001f,fmaxf(0,sx*(float)width-.5f)),v=fminf((float)height-1.001f,fmaxf(0,sy*(float)height-.5f));
 int x=(int)floorf(u),y=(int)floorf(v);float a=fract(u),b=fract(v);float4 out=make_float4(0,0,0,0);
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){float4 p=camera[131104+(y+j)*width+x+i];float w=(i==0?1-a:a)*(j==0?1-b:b);out.x+=p.x*w;out.y+=p.y*w;out.z+=p.z*w;out.w+=p.w*w;}return out;
}
// World-space rain planes and expanding impact normals, adapted from the
// reference project's approach. Drops run on wave time, independent of clock rate.
__device__ float3 rain_surface(float x,float z,float time,float rain,float footprint){
 if(rain<.01f||footprint>.08f)return vec(0,0,0);
 float u=x*.92387953f+z*.38268343f,v=z*.92387953f-x*.38268343f;
 float cx=floorf(u/.7f),cz=floorf(v/.7f),nx=0,nz=0;
 for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
  float gx=cx+(float)i,gz=cz+(float)j,period=.58f+.45f*cell(gx+791,gz-381),clock=time+cell(gx,gz)*period;
  float tick=floorf(clock/period),age=clock-tick*period;if(cell(gx+tick*13,gz-tick*7)>rain*.8f)continue;
  float dx=u-(gx+cell(gx+tick*19,gz+17))*.7f,dz=v-(gz+cell(gx+53,gz+tick*23))*.7f,r=sqrtf(dx*dx+dz*dz),q=r-age*.55f;
  float width=.0007f+footprint*footprint*2,slope=-q/width*.002f*expf(-q*q/width-age*7)/fmaxf(.008f,r);
  nx+=slope*dx;nz+=slope*dz;
 }return vec(nx*.92387953f-nz*.38268343f,0,nx*.38268343f+nz*.92387953f);
}
__device__ float rain_volume(float3 origin,float3 ray,float distance,const float4 *camera){
 float rain=camera[11].x*(1-eased(900,1600,origin.y)),time=camera[13].y;if(rain<.01f)return 0;
 float result=0,windX=camera[10].x*.16f,windZ=camera[10].y*.16f;
 for(int axis=0;axis<2;axis++){
  float dir=axis==0?ray.z:ray.x,along=axis==0?origin.z:origin.x,velocity=axis==0?windZ:windX,transverse=axis==0?windX:windZ;
  if(fabsf(dir)<.1f)continue;
  float base=floorf((along-velocity*time)/5),sgn=dir<0?-1.0f:1.0f;
  for(int i=1;i<=3;i++){
   float plane=base+sgn*(float)i,t=(plane*5+velocity*time-along)/dir;if(t<.3f||t>distance||t>30)continue;
   float3 p=plus(origin,scale(ray,t));float u=((axis==0?p.x:p.z)-transverse*time)/.8f,column=floorf(u),v=(p.y+12*time)/2.6f+cell(column,plane)*7,row=floorf(v);
   if(cell(column+row*17,plane+axis*59)>rain*.35f)continue;
   float dy=(fract(v)-.5f)*2.6f,dx=(fract(u)-(.2f+.6f*cell(column+row,plane+23)))*.8f+dy*transverse/12,width=.003f+t*.0006f;
   result+=eased(width*2,0,fabsf(dx))*eased(.23f,.04f,fabsf(dy))*fabsf(dir)*(1-t/40);
  }
 }return result;
}

__device__ float3 space_stars(float3 d){
 float3 q=scale(d,1300);float ix=floorf(q.x),iy=floorf(q.y),iz=floorf(q.z);
 unsigned seed=(unsigned)((int)ix*92837111+(int)iy*689287499+(int)iz*283923481);float chance=randf(seed);
 float glow=chance>.99965f?(.15f+randf(seed+17u)*.7f):0;
 return vec(glow*.84f,glow*.91f,glow);
}
__device__ float3 moon_position(const float4 *camera){return vec(camera[7].w,camera[9].w,camera[18].w);}
__device__ float3 moon_material(float3 n,float3 sun,float3 view,float3 centre){
 // A tidally locked procedural surface: the maria/craters rotate with the
 // Earth-facing lunar frame, independently of the observer and lighting.
 float3 front=scale(unit(centre),-1),right=unit(crossv(vec(0,1,0),front)),up=crossv(front,right);
 float3 p=vec(dotv(n,right),dotv(n,up),dotv(n,front));
 float mare=globe_noise(plus(scale(p,4.7f),vec(13,7,19))),detail=globe_noise(terrain_rotate(scale(p,29)));
 float albedo=mixf(.085f,.23f,eased(.35f,.63f,mare))*(.84f+detail*.28f);
 float relief=0;
 for(int i=0;i<12;i++){
  unsigned seed=(unsigned)i*317u+721u;float3 c=unit(vec(randf(seed)*2-1,randf(seed+7u)*2-1,randf(seed+13u)*2-1));
  float radius=.027f+.064f*randf(seed+19u),d=sqrtf(fmaxf(0,2*(1-dotv(p,c))))/radius;
  if(d>1.4f)continue;float rim=expf(-(d-1)*(d-1)*60),bowl=1-eased(.65f,.96f,d);albedo*=1-bowl*.23f+rim*.32f;
  relief+=(dotv(c,vec(dotv(sun,right),dotv(sun,up),dotv(sun,front)))-dotv(n,sun))*rim*3;
 }
 float nl=fmaxf(0,dotv(n,sun)),nv=fmaxf(.015f,dotv(n,view));
 // Lambert + lunar opposition response. The dark hemisphere receives only
 // faint Earthshine; it is never an emissive full disk.
 float illumination=nl>0?(.65f*nl+.35f*nl/(nl+nv))*(1+relief):0;
 float behind=-dotv(centre,sun),shadowDistance=sqrtf(fmaxf(0,dotv(centre,centre)-behind*behind));
 float solarVisibility=behind>0?eased(earth_radius()-800000,earth_radius()+800000,shadowDistance):1;
 float earthshine=.0025f*fmaxf(0,dotv(n,front));
 return scale(vec(.95f,.94f,.91f),albedo*(illumination*solarVisibility*2.6f+earthshine));
}
__device__ float4 moon_ray(float3 worldRay,const float4 *camera){
 float3 centre=moon_position(camera),eye=scale(vec(camera[6].x,camera[6].y,camera[6].z),earth_radius()+camera[0].y),delta=minus(centre,eye);
 float projection=dotv(delta,worldRay);if(projection<=0)return make_float4(0,0,0,-1);
 // Perpendicular-distance form avoids subtracting almost equal distance^2
 // terms when a 1737 km disk is seen from 384400 km away.
 float3 nearest=minus(delta,scale(worldRay,projection));float radius=1737400,disc=radius*radius-dotv(nearest,nearest);
 if(disc<0)return make_float4(0,0,0,-1);float root=sqrtf(disc),t=projection-root;if(t<=0)return make_float4(0,0,0,-1);
 float3 n=scale(minus(scale(worldRay,-root),nearest),1/radius),sun=vec(camera[12].x,camera[12].y,camera[12].z);
 float3 color=moon_material(n,sun,scale(worldRay,-1),centre);return make_float4(color.x,color.y,color.z,t);
}
__global__ void celestial_probe(const float4 *camera,const float4 *points,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float4 p=points[i];float3 n=unit(vec(p.x,p.y,p.z));
 if(p.w==0){output[i]=moon_ray(n,camera);return;}
 float3 sun=vec(camera[12].x,camera[12].y,camera[12].z),col=moon_material(n,sun,n,moon_position(camera));output[i]=make_float4(col.x,col.y,col.z,fmaxf(0,dotv(n,sun)));
}

__device__ float3 terrain_radiance(const float4 *camera,float3 ray,float t){
 float3 n=terrain_world(camera,ray,t),axis=fabsf(n.y)>.95f?vec(1,0,0):vec(0,1,0),east=unit(crossv(axis,n)),north=crossv(n,east);
 float h=0,spacing=fmaxf(32,t*.00045f),step=spacing/earth_radius(),dx=0,dz=0;
 float4 cached=camera[0].y<20000&&t<30000?terrain_ray_gradient(camera,ray,t):make_float4(0,0,0,-100000);float3 normal=n;
 if(cached.w> -90000&&camera[0].y<20000&&t<30000){
  float3 gradient=vec(cached.x,cached.y,cached.z);h=cached.w;dx=dotv(gradient,east);dz=dotv(gradient,north);normal=unit(minus(n,gradient));
 }else{
  h=terrain_height(camera,n);
  dx=(terrain_height(camera,unit(plus(n,scale(east,step))))-terrain_height(camera,unit(minus(n,scale(east,step)))))/(2*spacing);
  dz=(terrain_height(camera,unit(plus(n,scale(north,step))))-terrain_height(camera,unit(minus(n,scale(north,step)))))/(2*spacing);
  normal=unit(minus(n,plus(scale(east,dx),scale(north,dz))));
 }
 float3 sun=vec(camera[12].x,camera[12].y,camera[12].z);
 float3 climateP=plus(scale(n,9),vec(17,5,camera[22].z*.01f));
 float latitude=fabsf(n.y),patches=globe_noise(climateP)*.55f+globe_noise(terrain_rotate(scale(climateP,2.73f)))*.30f+globe_noise(terrain_rotate(scale(climateP,7.31f)))*.15f,texture=globe_noise(terrain_rotate(scale(n,1351)));
 // Material coordinates are Earth-fixed too; the chase camera chart must not
 // drag the grain across the beach when circling the ship.
 float grain=.5f;
 if(t<2000){int o=(int)camera[31].y;float4 base=camera[o+6],fraction=camera[o+7];float3 delta=terrain_rotate(scale(to_world(camera,ray),t*.19f));
  grain=mixf(.5f,globe_noise_shifted(vec(base.x,base.y,base.z),plus(vec(fraction.x,fraction.y,fraction.z),delta)),1-eased(1500,2000,t));
 }
 float desert=expf(-(latitude-.42f)*(latitude-.42f)*65)*(.28f+.72f*patches);
 float3 grass=blend(vec(.032f,.075f,.027f),vec(.12f,.14f,.05f),patches);
 float3 albedo=blend(grass,vec(.38f,.27f,.13f),desert);
 float rocky=eased(.045f,.19f,sqrtf(dx*dx+dz*dz))*eased(200,1600,h);
 albedo=blend(albedo,blend(vec(.13f,.115f,.095f),vec(.27f,.24f,.20f),texture),rocky);
 float sandGrain=mixf(.5f,grain,1/(1+t*t*.000003f));
 albedo=blend(albedo,blend(vec(.27f,.215f,.125f),vec(.49f,.415f,.27f),sandGrain),1-eased(2,18,h));
 albedo=scale(albedo,mixf(.48f,1,eased(.05f,1.5f,h)));
 float snowline=3900-3300*latitude*latitude+(patches-.5f)*700,snow=eased(snowline-180,snowline+250,h);
 snow=fmaxf(snow,eased(.91f,.99f,latitude));albedo=blend(albedo,vec(.74f,.80f,.84f),snow);
 albedo=scale(albedo,.83f+.25f*texture+(grain-.5f)*.16f/(1+t*t*.0001f));
 float day=clamp01(dotv(n,sun)),direct=clamp01(dotv(normal,sun));
 float cover=camera[13].w!=0?weather_map_sample(camera,n,0).x:0;
 float light=.008f+day*.19f+direct*(.95f-cover*.33f);
 return scale(albedo,light);
}
__device__ float cloud_visibility(float altitude,float3 ray){
 // An observer inside the volume has no clear-air gap in front of its entry.
 // Intersecting an arbitrary middle shell here creates a discontinuity at that
 // shell's tangent and paints a bright horizontal strip across the horizon.
 float distance=0;
 if(altitude<800){float2 roots=sphere_roots(altitude,ray,800);distance=fmaxf(0,roots.y);}
 else if(altitude>8200){float2 roots=sphere_roots(altitude,ray,8200);distance=fmaxf(0,roots.x);}
 return mixf(expf(-distance*.000055f),1,eased(5000,20000,altitude));
}
__device__ float3 planet_radiance(float3 ray,const float4 *camera,float hit,int cachedCloud,float pixelU,float pixelV,float landT,float pixelAngle){
 float altitude=camera[0].y,R=earth_radius();float4 sl=camera[9];float3 sun=vec(sl.x,sl.y,sl.z),worldRay=to_world(camera,ray);
 float visibility=fmaxf(eased(25000,90000,altitude),1-eased(-.12f,.08f,sun.y));
 float3 color=scale(space_stars(camera[13].w!=0?rotate_y(worldRay,camera[13].x*.00007292115f):worldRay),visibility);float sunDot=dotv(ray,sun);
 if(sunDot>.99996f)color=plus(color,vec(18,15,11));
 float moonDistance=-1;
 if((hit<0&&landT<0)||altitude>80000){float4 moon=moon_ray(worldRay,camera);float nearest=landT>0?landT:hit;if(moon.w>0&&(nearest<0||moon.w<nearest)){color=vec(moon.x,moon.y,moon.z);moonDistance=moon.w;hit=moon.w;landT=-1;}}
 if(landT>0){color=terrain_radiance(camera,ray,landT);hit=landT;}
 else if(hit>0&&moonDistance<0){
  float3 localNormal=unit(vec(ray.x*hit,R+altitude+ray.y*hit,ray.z*hit)),normal=to_world(camera,localNormal);
  float day=clamp01(dotv(localNormal,sun)),limb=1-clamp01(-dotv(localNormal,ray));
  float basin=globe_noise(scale(normal,7));
  float3 ocean=blend(vec(.002f,.010f,.037f),vec(.004f,.030f,.050f),basin);
  if(camera[21].w!=0){float4 geology=geology_sample(camera,normal);float d=fmaxf(1.4f,-geology.x);ocean=blend(ocean,vec(.11f,.30f,.25f),expf(-d*.032f));}
  float3 shadingNormal=localNormal;float wind=7;
  if(camera[13].w!=0){
   float4 flow=weather_map_sample(camera,normal,1);wind=sqrtf(flow.x*flow.x+flow.y*flow.y+flow.z*flow.z);
   // Resolved metre-scale glints fade to a statistical rough surface as the
   // footprint grows. This avoids drawing a smooth plastic sheet at flight height.
   float footprint=hit*pixelAngle,detail=1/(1+footprint*footprint*.0018f),time=camera[13].y;
   float3 variation=vec(0,0,0);
   if(detail>.002f){
    float3 p=scale(normal,R/38);p=plus(p,vec(time*.023f,-time*.017f,time*.031f));
    float4 a=globe_gradient(p),b=globe_gradient(plus(terrain_rotate(scale(p,2.73f)),vec(31,17,11)));
    // Transpose rotation transforms the second field's derivative back.
    float3 g=vec(.8f*b.y+.36f*b.z-.48f*b.w,.8f*b.z+.6f*b.w,.6f*b.y-.48f*b.z+.64f*b.w);
    variation=plus(scale(vec(a.y,a.z,a.w),.055f),scale(g,.026f/(1+footprint*footprint*.06f)));
   }
   variation=scale(variation,detail);
   float swellDetail=1/(1+footprint*footprint*.000055f);
   if(swellDetail>.005f){
    float4 swell=globe_gradient(plus(terrain_rotate(scale(normal,R/220)),vec(time*.003f,11,-time*.005f)));
    float3 g=vec(.8f*swell.y+.36f*swell.z-.48f*swell.w,.8f*swell.z+.6f*swell.w,.6f*swell.y-.48f*swell.z+.64f*swell.w);
    variation=plus(variation,scale(g,.038f*swellDetail));
   }
   variation=minus(variation,scale(normal,dotv(normal,variation)));
   float3 worldNormal=unit(minus(normal,variation));
   shadingNormal=vec(dotv(worldNormal,vec(camera[5].x,camera[5].y,camera[5].z)),dotv(worldNormal,vec(camera[6].x,camera[6].y,camera[6].z)),dotv(worldNormal,vec(camera[7].x,camera[7].y,camera[7].z)));
  }
  float nv=clamp01(-dotv(shadingNormal,ray)),grazing=1-nv,g2=grazing*grazing,fresnel=.02037f+.97963f*g2*g2*grazing;
  float3 reflectedRay=minus(ray,scale(shadingNormal,2*dotv(ray,shadingNormal)));
  float skyElevation=clamp01(dotv(reflectedRay,localNormal));
  float3 reflected=scale(blend(vec(.36f,.50f,.63f),vec(.035f,.10f,.23f),sqrtf(skyElevation)),.012f+day);
  if(camera[13].w!=0&&altitude<1800)reflected=blend(weather_sky_sample(reflectedRay,camera),reflected,eased(600,1800,altitude));
  color=blend(scale(ocean,.04f+day*1.2f),reflected,fresnel);
  float3 halfv=unit(minus(sun,ray));float nh=fmaxf(0,dotv(halfv,shadingNormal)),alpha=.035f+.0045f*fminf(25,wind),a2=alpha*alpha;
  float denominator=nh*nh*(a2-1)+1;
  float spec=a2/(3.14159265f*denominator*denominator)*.02037f*day*.85f;
  color=plus(color,scale(vec(1,.87f,.68f),spec));
  if(camera[13].w==0){
  float3 cloudP=plus(scale(normal,19),vec(2.8f,1.2f,-1.7f));
  float low=globe_noise(cloudP),detail=globe_noise(scale(cloudP,2.73f)),fine=globe_noise(scale(cloudP,7.1f));
  float bands=.055f*sinf(normal.y*31+normal.x*11);
  float cover=eased(.48f,.68f,low*.64f+detail*.26f+fine*.10f+bands);
  float shadow=cover*.3f;color=scale(color,1-shadow);
  float3 cloudColor=scale(vec(.84f,.88f,.92f),.025f+day*1.2f);
  color=blend(color,cloudColor,cover*.92f);
  }else{float cover=weather_map_sample(camera,normal,0).x;color=scale(color,1-cover*.38f);}
  color=plus(color,scale(vec(.006f,.017f,.027f),limb*limb*day));
 }
 // Eight samples through a 100 km exponential atmosphere. Rayleigh extinction
 // and an approximate slant sunlight path produce the blue limb and terminator.
 float2 atmosphere=sphere_roots(altitude,ray,100000);
 if(atmosphere.y>0){
  float start=fmaxf(0,atmosphere.x),end=hit>0?fminf(hit,atmosphere.y):atmosphere.y;
  int steps=camera[16].x>256?12:8;float span=fmaxf(0,end-start);float3 transmission=vec(1,1,1),scatter=vec(0,0,0);
  float phase=.0596831f*(1+sunDot*sunDot);
  for(int i=0;i<steps;i++){
   float a=(float)i/(float)steps,b=(float)(i+1)/(float)steps;
   if(hit>0&&altitude>100000){a=1-(1-a)*(1-a);b=1-(1-b)*(1-b);}
   float step=(b-a)*span,t=start+(a+b)*.5f*span;float3 p=vec(ray.x*t,R+altitude+ray.y*t,ray.z*t);
   float radius=sqrtf(dotv(p,p)),h=fmaxf(0,radius-R),density=expf(-h/8500)*step;
   float mu=dotv(scale(p,1/radius),sun),horizon=-sqrtf(fmaxf(0,2*h/R));
   float lit=eased(horizon-.025f,horizon+.025f,mu),slant=8500*expf(-h/8500)/(fmaxf(.04f,mu)+.035f);
   float3 extinction=vec(expf(-density*.0000058f),expf(-density*.0000135f),expf(-density*.0000331f));
   float3 light=vec(expf(-slant*.0000058f),expf(-slant*.0000135f),expf(-slant*.0000331f));
   float strength=lit*phase*mixf(18,2.5f,eased(20000,150000,altitude));
   scatter=plus(scatter,vec(transmission.x*(1-extinction.x)*light.x*strength,transmission.y*(1-extinction.y)*light.y*strength,transmission.z*(1-extinction.z)*light.z*strength));
   transmission=vec(transmission.x*extinction.x,transmission.y*extinction.y,transmission.z*extinction.z);
  }
  color=plus(vec(color.x*transmission.x,color.y*transmission.y,color.z*transmission.z),scatter);
 }
 if(camera[13].w!=0&&(moonDistance<0||(atmosphere.y>0&&fmaxf(0,atmosphere.x)<moonDistance))){
  float4 clouds=cachedCloud!=0?cloud_view_sample(camera,pixelU,pixelV):globe_clouds(ray,camera,hit);
  // In the lower atmosphere, haze lies between the eye and the clouds. Do not
  // integrate an entire clear-sky column over an opaque cloud base.
  float visible=cloud_visibility(altitude,ray);
  color=plus(scale(color,1-clouds.w*visible),scale(vec(clouds.x,clouds.y,clouds.z),visible));
 }
 return color;
}

__global__ void weather_sky(float4 *camera,int skyWidth){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,height=skyWidth/4;
 if(x>=skyWidth||y>=height||camera[0].y>=1800)return;
 float angle=(((float)x+.5f)/(float)skyWidth)*6.2831853f,e=((float)y+.5f)/(float)height;e=e*e*1.5707963f;
 float3 ray=vec(sinf(angle)*cosf(e),sinf(e),-cosf(angle)*cosf(e));
 float3 col=planet_radiance(ray,camera,-1,0,0,0,-1,1.3f/(float)skyWidth);camera[65568+y*skyWidth+x]=make_float4(col.x,col.y,col.z,1);
}
__device__ float3 weather_sky_sample(float3 d,const float4 *camera){
 int width=(int)camera[16].x,height=width/4;
 float u=atan2f(d.x,-d.z)*(float)width/6.2831853f-.5f;
 float v=sqrtf(atan2f(fmaxf(0,d.y),sqrtf(d.x*d.x+d.z*d.z))/1.5707963f)*(float)height-.5f;
 v=fminf((float)height-1.001f,fmaxf(0,v));int x=(int)floorf(u),y=(int)floorf(v);float a=fract(u),b=fract(v);float3 col=vec(0,0,0);
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){float4 p=camera[65568+(y+j)*width+((x+i)&(width-1))];col=plus(col,scale(vec(p.x,p.y,p.z),(i==0?1-a:a)*(j==0?1-b:b)));}return col;
}
__device__ float3 surface_weather(float3 col,float3 ray,const float4 *camera,float distance){
 float rain=camera[11].x*(1-eased(800,2400,camera[0].y));if(rain<.01f)return col;
 float daylight=.015f+.985f*eased(-.08f,.3f,camera[9].y);
 float haze=(1-expf(-fminf(distance,3000)*rain*.0012f));
 col=blend(col,scale(vec(.17f,.23f,.28f),daylight),haze);
 float streak=rain_volume(vec(camera[0].x,camera[0].y,camera[0].z),ray,distance,camera);
 return plus(col,scale(vec(.45f,.55f,.62f),streak*daylight*.4f));
}
__device__ float3 craft_water(float3 col,float3 p,float3 n,float3 ray,const float4 *camera,const float4 *brush){
 if(camera[19].z==0||camera[19].x>80)return col;
 float foam=0,time=brush[10].w;
 for(int i=0;i<2;i++){
  float4 jet=brush[4+i];if(jet.w<.001f)continue;float dx=p.x-jet.x,dz=p.z-jet.y,r=sqrtf(dx*dx+dz*dz),radius=jet.z;
  if(r>radius*4)continue;
  float noise=globe_noise(vec(dx*1.9f,dz*1.9f,time*.9f));
  float ring=(r-radius*(1.25f+.45f*noise))/fmaxf(.15f,radius*.40f);
  float rim=expf(-ring*ring*1.4f),crest=eased(.035f,.28f,1-n.y);
  foam+=jet.w*rim*(.38f+.62f*noise)+jet.w*crest*expf(-r*r/(radius*radius*5));
 }
 float day=.025f+.975f*eased(-.08f,.25f,camera[9].y);col=blend(col,scale(vec(.50f,.67f,.71f),day),clamp01(foam*.8f));
 for(int i=0;i<4;i++){
  float4 source=brush[6+i];if(source.w<=0)continue;float3 delta=minus(vec(source.x,source.y,source.z),p);float d2=dotv(delta,delta);if(d2>3600)continue;
  float3 light=unit(delta),halfv=unit(minus(light,ray));float nl=fmaxf(0,dotv(n,light)),nh=fmaxf(0,dotv(n,halfv));
  float cone=i<2?1:eased(.68f,.94f,-dotv(light,vec(brush[10].x,brush[10].y,brush[10].z)));
  float rough=.08f+.18f*clamp01(foam),a2=rough*rough,den=nh*nh*(a2-1)+1;
  float spec=a2/(3.14159265f*den*den+.00001f)*.035f;
  float amount=source.w/(4+d2)*(1-eased(1600,3600,d2))*cone*(nl*(.025f+foam*.3f)+spec);
  float3 hue=i<2?vec(brush[11].x,brush[11].y,brush[11].z):vec(.68f,.84f,1);col=plus(col,scale(hue,amount));
 }
 for(int i=0;i<2;i++){
  if((float)i>=brush[12].x)break;float4 source=brush[13+i*2];float3 delta=minus(vec(source.x,source.y,source.z),p);float d2=dotv(delta,delta);if(d2>6400)continue;
  float3 l=unit(delta),h=unit(minus(l,ray));float nh=fmaxf(0,dotv(n,h)),den=nh*nh*(.018f-1)+1;
  float amount=source.w/(8+d2)*(1-eased(2500,6400,d2))*(fmaxf(0,dotv(n,l))*.1f+.00025f/(den*den+.00001f));
  float4 hue=brush[14+i*2];col=plus(col,scale(vec(hue.x,hue.y,hue.z),amount));
 }
 return col;
}
__device__ float3 craft_spray(float3 col,float3 ray,const float4 *camera,const float4 *brush,float limit){
 if(camera[19].z==0||camera[19].x>32)return col;
 float time=brush[10].w,day=.015f+.985f*eased(-.08f,.25f,camera[9].y);
 for(int i=0;i<2;i++){
  float4 jet=brush[4+i];if(jet.w<.001f)continue;float radius=jet.z*2.8f,height=.6f+jet.w*.9f;
  float3 origin=vec((camera[0].x-jet.x)/radius,(camera[0].y-.45f)/height,(camera[0].z-jet.y)/radius),dir=vec(ray.x/radius,ray.y/height,ray.z/radius);
  float a=dotv(dir,dir),b=dotv(origin,dir),disc=b*b-a*(dotv(origin,origin)-1);if(disc<=0)continue;
  float root=sqrtf(disc),lo=fmaxf(0,(-b-root)/a),hi=fminf(limit>0?limit:1000,(-b+root)/a);if(hi<=lo)continue;
  float density=0;
  for(int j=0;j<3;j++){float t=lo+(hi-lo)*((float)j+.5f)/3;float3 q=plus(origin,scale(dir,t));float radius2=q.x*q.x+q.z*q.z;
   float noise=globe_noise(vec(q.x*5+time*.7f,q.y*4-time*1.2f,q.z*5));density+=eased(.08f,.50f,radius2)*(1-dotv(q,q))*(.35f+noise*.65f);
  }
  float opacity=1-expf(-fmaxf(0,density)*(hi-lo)*jet.w*.20f);
  float3 light=plus(scale(vec(.43f,.59f,.64f),day),scale(vec(.015f,.075f,.24f),jet.w));col=blend(col,light,opacity);
 }
 return col;
}
__device__ float3 shade_pixel(const float4 *brush,const float4 *surface,const float4 *coefficients,const float4 *light,const float *monoLight,const float4 *camera,int width,int height,float depth,float exposure,int view,int pressureActive,int dispersion,int lightSize,float pixelX,float pixelY,float sampleScale){
 if(camera[21].w!=0)depth=camera[21].x;
 int smooth=lightSize>256?1:0;
 float4 pos=camera[0],forward=camera[2],right=camera[3],up=camera[4];float sx=(2*pixelX/(float)width-1)*(float)width/(float)height;
 float sy=1-2*pixelY/(float)height;float3 f=vec(forward.x,forward.y,forward.z);
 float3 r=vec(right.x,right.y,right.z),u=vec(up.x,up.y,up.z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));float3 col=vec(0,0,0);
 float4 solar=camera[9];float3 localSun=vec(solar.x,solar.y,solar.z);
 float globeT=globe_hit(pos.y,ray),planetMix=fmaxf(eased(250,1200,globeT),eased(60,400,pos.y));
 float vignette=1-.10f*(sx*sx+sy*sy);
 if(view>=3&&globeT>0){float3 normal=to_world(camera,unit(vec(ray.x*globeT,earth_radius()+pos.y+ray.y*globeT,ray.z*globeT)));return geology_color(geology_sample(camera,normal),view);}
 float landT=terrain_pixel_hit(monoLight,camera,width,height,pixelX,pixelY,view);
 if(globeT<0||planetMix>=1||landT>0){float3 far=planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height,landT,1.3f/((float)height*sampleScale));if(camera[13].w!=0)far=surface_weather(far,ray,camera,landT>0?landT:(globeT>0?globeT:3000));return scale(far,exposure*vignette);}
 if(globeT>0){
 float t=globeT;float4 w=make_float4(0,0,0,0);WaveRegionCache regions=wave_cache(pos.x+ray.x*t,pos.z+ray.z*t);
 for(int i=0;i<(smooth!=0?6:4);i++){float h=cached_wave_height(surface,coefficients,brush,pos.x+ray.x*t,pos.z+ray.z*t,1/(1+t*t*.0008f),pressureActive,smooth,regions);h-=(ray.x*ray.x+ray.z*ray.z)*t*t/(2*earth_radius());t=mixf(t,(h-pos.y)/ray.y,.75f);}
 float3 p=vec(pos.x+ray.x*t,pos.y+ray.y*t,pos.z+ray.z*t);
 float causticDetail=1/(1+t*t*.0008f),pixelFootprint=t/((float)height*sampleScale);
 w=smooth!=0?wave_pc(coefficients,brush,p.x,p.z,causticDetail,pressureActive,1):wave(surface,brush,p.x,p.z,causticDetail,pressureActive);float3 rain=camera[13].w!=0?rain_surface(p.x,p.z,camera[13].y,camera[11].x,pixelFootprint):vec(0,0,0);float3 n=unit(vec(-w.y-rain.x+ray.x*t/earth_radius(),1,-w.z-rain.z+ray.z*t/earth_radius()));float viewCosine=dotv(n,ray),nv=fmaxf(.02f,-viewCosine);
 float grazing=1-clamp01(nv),grazing2=grazing*grazing;
 float fresnel=.02037f+.97963f*grazing2*grazing2*grazing;
 float3 reflection=minus(ray,scale(n,2*viewCosine));float3 reflected=camera[13].w!=0?weather_sky_sample(reflection,camera):sky(reflection,localSun);
 if(camera[21].w!=0)depth=fmaxf(.15f,-terrain_ray_height(camera,ray,t));
 float3 transmitted=refract_cosine(ray,n,.7502f,viewCosine);float vertical=fminf(-.1f,transmitted.y);float travel=(-depth-p.y)/vertical;
 float bx=p.x+transmitted.x*travel,bz=p.z+transmitted.z*travel;BedRegionCache bedRegions=bed_cache(bx,bz);
 for(int j=0;j<(smooth!=0?4:2);j++){travel=(cached_bottom(bx,bz,depth,bedRegions)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 travel=fmaxf(0,travel);float3 bed=seabed(bx,bz,pixelFootprint),ca=caustic(light,monoLight,bx,bz,dispersion,lightSize);
 // Light travels down through the water before returning along the view ray.
 float opticalDistance=travel+(camera[21].w!=0?depth/.86f:forward.w);
 float shadow=camera[13].w!=0?camera[11].y:1;ca=scale(blend(vec(1,1,1),ca,causticDetail*right.w*shadow),.32f+.68f*shadow);
 float attenR=expf(-opticalDistance*.19f),attenG=expf(-opticalDistance*.09f),attenB=expf(-opticalDistance*.055f);
 float3 through=vec(bed.x*(.12f+.95f*ca.x)*attenR+.008f*(1-attenR),bed.y*(.12f+.95f*ca.y)*attenG+.042f*(1-attenG),bed.z*(.12f+.95f*ca.z)*attenB+.075f*(1-attenB));
 col=blend(scale(through,.015f+.985f*eased(-.08f,.3f,localSun.y)),reflected,fresnel);
 float3 halfv=unit(minus(localSun,ray));
 float specPower=mixf(320,8000,1/(1+pixelFootprint*pixelFootprint*800));
 float spec=positive_power(fmaxf(0,dotv(n,halfv)),specPower)*3.5f*(specPower/8000);
 col=plus(col,scale(vec(1,.89f,.68f),spec*(camera[13].w!=0?camera[11].y*eased(-.02f,.08f,localSun.y):1)));
 float haze=1-expf(-t*.00025f);col=blend(col,scale(vec(.38f,.55f,.68f),.015f+.985f*eased(-.08f,.3f,localSun.y)),haze);
 if(view==0)col=craft_water(col,p,n,ray,camera,brush);
 if(view==1)col=scale(ca,.35f);if(view==2)col=plus(scale(n,.5f),vec(.5f,.5f,.5f));
 }else{col=sky(ray,localSun);}
 if(planetMix>0)col=blend(col,planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height,landT,1.3f/((float)height*sampleScale)),planetMix);
 // Apply the same air path after mixing near and distant water.
 if(camera[13].w!=0)col=surface_weather(col,ray,camera,globeT>0?globeT:3000);
 if(view==0)col=craft_spray(col,ray,camera,brush,globeT);
 col=scale(col,exposure*vignette);
 return col;
}
__device__ float3 shade_pixel_pc(const float4 *brush,const float4 *sandState,const float4 *surface,const float4 *coefficients,const float4 *light,const float *monoLight,const float4 *camera,int width,int height,float depth,float exposure,int view,int pressureActive,int dispersion,int lightSize,float pixelX,float pixelY,float sampleScale){
 if(camera[21].w!=0)depth=camera[21].x;
 int smooth=lightSize>256?1:0;
 float4 pos=camera[0],forward=camera[2],right=camera[3],up=camera[4];float sx=(2*pixelX/(float)width-1)*(float)width/(float)height;
 float sy=1-2*pixelY/(float)height;float3 f=vec(forward.x,forward.y,forward.z);
 float3 r=vec(right.x,right.y,right.z),u=vec(up.x,up.y,up.z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));float3 col=vec(0,0,0);
 float4 solar=camera[9];float3 localSun=vec(solar.x,solar.y,solar.z);
 float globeT=globe_hit(pos.y,ray),planetMix=fmaxf(eased(250,1200,globeT),eased(60,400,pos.y));
 float vignette=1-.10f*(sx*sx+sy*sy);
 if(view>=3&&globeT>0){float3 normal=to_world(camera,unit(vec(ray.x*globeT,earth_radius()+pos.y+ray.y*globeT,ray.z*globeT)));return geology_color(geology_sample(camera,normal),view);}
 float landT=terrain_pixel_hit(monoLight,camera,width,height,pixelX,pixelY,view);
 if(globeT<0||planetMix>=1||landT>0){float3 far=planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height,landT,1.3f/((float)height*sampleScale));if(camera[13].w!=0)far=surface_weather(far,ray,camera,landT>0?landT:(globeT>0?globeT:3000));return scale(far,exposure*vignette);}
 if(globeT>0){
 float t=globeT;float4 w=make_float4(0,0,0,0);WaveRegionCache regions=wave_cache(pos.x+ray.x*t,pos.z+ray.z*t);
 for(int i=0;i<(smooth!=0?6:4);i++){float h=cached_wave_height(surface,coefficients,brush,pos.x+ray.x*t,pos.z+ray.z*t,1/(1+t*t*.0008f),pressureActive,smooth,regions);h-=(ray.x*ray.x+ray.z*ray.z)*t*t/(2*earth_radius());t=mixf(t,(h-pos.y)/ray.y,.75f);}
 float3 p=vec(pos.x+ray.x*t,pos.y+ray.y*t,pos.z+ray.z*t);
 float causticDetail=1/(1+t*t*.0008f),pixelFootprint=t/((float)height*sampleScale);
 w=smooth!=0?wave_pc(coefficients,brush,p.x,p.z,causticDetail,pressureActive,1):wave(surface,brush,p.x,p.z,causticDetail,pressureActive);float3 rain=camera[13].w!=0?rain_surface(p.x,p.z,camera[13].y,camera[11].x,pixelFootprint):vec(0,0,0);float3 n=unit(vec(-w.y-rain.x+ray.x*t/earth_radius(),1,-w.z-rain.z+ray.z*t/earth_radius()));float viewCosine=dotv(n,ray),nv=fmaxf(.02f,-viewCosine);
 float grazing=1-clamp01(nv),grazing2=grazing*grazing;
 float fresnel=.02037f+.97963f*grazing2*grazing2*grazing;
 float3 reflection=minus(ray,scale(n,2*viewCosine));float3 reflected=camera[13].w!=0?weather_sky_sample(reflection,camera):sky(reflection,localSun);
 if(camera[21].w!=0)depth=fmaxf(.15f,-terrain_ray_height(camera,ray,t));
 float3 transmitted=refract_cosine(ray,n,.7502f,viewCosine);float vertical=fminf(-.1f,transmitted.y);float travel=(-depth-p.y)/vertical;
 float bx=p.x+transmitted.x*travel,bz=p.z+transmitted.z*travel;BedRegionCache bedRegions=bed_cache(bx,bz);
 for(int j=0;j<2;j++){travel=(cached_bottom(bx,bz,depth,bedRegions)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 float anchorX=bx,anchorZ=bz;float4 drift=make_float4(0,0,0,0);
 if(depth<3){float shallow=clamp01((3-depth)/2);shallow=shallow*shallow*(3-2*shallow);drift=sample_pc(sandState,bx/16,bz/16,0,1);drift.x*=shallow;drift.y*=shallow/16;drift.z*=shallow/16;}
 if(smooth!=0)for(int j=0;j<2;j++){travel=(bottom_pc(bx,bz,depth,pixelFootprint,drift,anchorX,anchorZ,bedRegions)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 travel=fmaxf(0,travel);float3 bed=vec(0,0,0);
 if(smooth!=0){float4 sand=sand_moving(bx,bz,pixelFootprint,drift,anchorX,anchorZ),stone=stone_relief(bx,bz,pixelFootprint,localSun);float3 bedSun=scale(refractv(scale(localSun,-1),n,.7502f),-1);bed=seabed_pc(bx,bz,pixelFootprint,sand,stone,bedSun,transmitted);}
 else bed=seabed(bx,bz,pixelFootprint);
 float3 ca=caustic(light,monoLight,bx,bz,dispersion,lightSize);
 // Light travels down through the water before returning along the view ray.
 float opticalDistance=travel+(camera[21].w!=0?depth/.86f:forward.w);
 float shadow=camera[13].w!=0?camera[11].y:1;ca=scale(blend(vec(1,1,1),ca,causticDetail*right.w*shadow),.32f+.68f*shadow);
 float attenR=expf(-opticalDistance*.19f),attenG=expf(-opticalDistance*.09f),attenB=expf(-opticalDistance*.055f);
 float3 through=vec(bed.x*(.12f+.95f*ca.x)*attenR+.008f*(1-attenR),bed.y*(.12f+.95f*ca.y)*attenG+.042f*(1-attenG),bed.z*(.12f+.95f*ca.z)*attenB+.075f*(1-attenB));
 col=blend(scale(through,.015f+.985f*eased(-.08f,.3f,localSun.y)),reflected,fresnel);
 float3 halfv=unit(minus(localSun,ray));
 float specPower=mixf(320,8000,1/(1+pixelFootprint*pixelFootprint*800));
 float spec=positive_power(fmaxf(0,dotv(n,halfv)),specPower)*3.5f*(specPower/8000);
 col=plus(col,scale(vec(1,.89f,.68f),spec*(camera[13].w!=0?camera[11].y*eased(-.02f,.08f,localSun.y):1)));
 float haze=1-expf(-t*.00025f);col=blend(col,scale(vec(.38f,.55f,.68f),.015f+.985f*eased(-.08f,.3f,localSun.y)),haze);
 if(view==0)col=craft_water(col,p,n,ray,camera,brush);
 if(view==1)col=scale(ca,.35f);if(view==2)col=plus(scale(n,.5f),vec(.5f,.5f,.5f));
 }else{col=sky(ray,localSun);}
 if(planetMix>0)col=blend(col,planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height,landT,1.3f/((float)height*sampleScale)),planetMix);
 // Apply the same air path after mixing near and distant water.
 if(camera[13].w!=0)col=surface_weather(col,ray,camera,globeT>0?globeT:3000);
 if(view==0)col=craft_spray(col,ray,camera,brush,globeT);
 col=scale(col,exposure*vignette);
 return col;
}
__device__ unsigned pack_color(float3 col){
 unsigned rr=(unsigned)(positive_power(film(col.x),.454545f)*255),gg=(unsigned)(positive_power(film(col.y),.454545f)*255),bb=(unsigned)(positive_power(film(col.z),.454545f)*255);
 return rr|(gg<<8)|(bb<<16)|4278190080u;
}
// Keep the one-sample shader separate from the PC supersampling path.
__global__ void render(const float4 *brush,const float4 *surface,const float4 *coefficients,const float4 *light,const float *monoLight,const float4 *camera,unsigned *image,int width,int height,float depth,float exposure,int view,int pressureActive,int dispersion,int lightSize){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 float3 col=shade_pixel(brush,surface,coefficients,light,monoLight,camera,width,height,depth,exposure,view,pressureActive,dispersion,lightSize,(float)x+.5f,(float)y+.5f,1);
 image[y*width+x]=pack_color(col);
}
// PC spatial supersampling: average linear radiance before tone mapping.
// FFT and lighting are shared across all four samples; no extra framebuffer.
__global__ void render_pc(const float4 *brush,const float4 *sandState,const float4 *surface,const float4 *coefficients,const float4 *light,const float *monoLight,const float4 *camera,unsigned *image,int width,int height,float depth,float exposure,int view,int pressureActive,int dispersion,int lightSize){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 float3 col=vec(0,0,0);
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){
  float3 sample=shade_pixel_pc(brush,sandState,surface,coefficients,light,monoLight,camera,width,height,depth,exposure,view,pressureActive,dispersion,lightSize,(float)x+((float)i+.5f)*.5f,(float)y+((float)j+.5f)*.5f,2);
  col=plus(col,sample);
 }
 image[y*width+x]=pack_color(scale(col,.25f));
}

// Diagnostic PC single sampling retains the same materials and relief as PC
// multisampling. Mobile's entry never depends on these material functions.
__global__ void render_pc_single(const float4 *brush,const float4 *sandState,const float4 *surface,const float4 *coefficients,const float4 *light,const float *monoLight,const float4 *camera,unsigned *image,int width,int height,float depth,float exposure,int view,int pressureActive,int dispersion,int lightSize){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 float3 col=shade_pixel_pc(brush,sandState,surface,coefficients,light,monoLight,camera,width,height,depth,exposure,view,pressureActive,dispersion,lightSize,(float)x+.5f,(float)y+.5f,1);
 image[y*width+x]=pack_color(col);
}

// Diagnostic-only sampling; the test computes finite differences and tile
// correlations outside the renderer, including the original periodic control.
__global__ void appearance_probe(const float4 *coefficients,const float4 *camera,const float4 *points,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float4 p=points[i];int mode=(int)p.z;
 if(mode<2)output[i]=region_wave(coefficients,p.x,p.y,1,1,mode);
 else if(mode<4)output[i]=sample_pc(coefficients,p.x,p.y,mode-2,1);
 else if(mode==4)output[i]=sand_relief(p.x,p.y,.001f);
 else if(mode==5)output[i]=bed_shape(p.x,p.y);
 else if(mode==6)output[i]=globe_gradient(vec(p.x,p.y,p.w));
 else if(mode==7){WaveRegionCache cache=wave_cache(p.x+p.w,p.y-p.w*.37f);output[i]=make_float4(cached_wave_height(coefficients,coefficients,coefficients,p.x,p.y,1,0,1,cache),0,0,0);}
 else if(mode==9){BedRegionCache cache=bed_cache(p.x+p.w,p.y-p.w*.37f);output[i]=make_float4(cached_bottom(p.x,p.y,0,cache),0,0,0);}
 else if(mode==10)output[i]=make_float4(cloud_visibility(p.x,vec(sqrtf(fmaxf(0,1-p.y*p.y)),p.y,0)),0,0,0);
 else {
  float lat=p.x*.0174532925f,lon=p.y*.0174532925f;float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
  float cover=weather_map_sample(camera,n,0).x;output[i]=make_float4(cloud_column(camera,n,0,cover),cover,dotv(n,vec(camera[12].x,camera[12].y,camera[12].z)),0);
 }
}
