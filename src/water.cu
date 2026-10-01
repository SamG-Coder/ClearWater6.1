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
// World-space seeded phase variation. Quintic blending gives continuous
// derivatives at region boundaries, without extra spectra or FFT passes.
__device__ float4 wave_region(float x,float z){
 float u=x/24,v=z/24,ix=floorf(u),iz=floorf(v);u=fract(u);v=fract(v);
 float du=30*u*u*(u-1)*(u-1)/24,dv=30*v*v*(v-1)*(v-1)/24;
 float a=u*u*u*(u*(u*6-15)+10),b=v*v*v*(v*(v*6-15)+10);
 float p=cell(ix,iz),q=cell(ix+1,iz),r=cell(ix,iz+1),t=cell(ix+1,iz+1);
 return make_float4((mixf(mixf(p,q,a),mixf(r,t,a),b)-.5f)*6,mixf(q-p,t-r,b)*du*6,mixf(r-p,t-q,a)*dv*6,0);
}
__device__ float4 region_wave(const float4 *s,float x,float z,int smooth,int slopes){
 float4 region=wave_region(x,z);
 float4 w=smooth!=0?sample_pc(s,x,z+region.x,0,slopes):sample(s,x,z+region.x,0);
 return make_float4(w.x,w.y+w.z*region.y,w.z*(1+region.z),0);
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
 float4 a=region_wave(s,x,z,1,slopes),b=sample_pc(s,x,z,1,slopes),d=pressureActive!=0?local_pressure(s,brush,x,z,1,slopes):make_float4(0,0,0,0);
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
 float4 a=region_wave(s,x,z,0,1),b=sample(s,x,z,1),d=pressureActive!=0?local_pressure(s,brush,x,z,0,1):make_float4(0,0,0,0);
 // Fade unresolved short displacement as well as its normal near the horizon.
 return make_float4((a.x+d.x)*fade+b.x,(a.y+d.y)*fade+b.y,(a.z+d.z)*fade+b.z,0);
}
__device__ float sample_height(const float4 *s,float x,float z,int c){
 float u=x*128/patch(c),v=z*128/patch(c);int ix=(int)floorf(u),iz=(int)floorf(v),base=c*16384;
 float a=fract(u),b=fract(v);
 return mixf(mixf(s[base+wrap128(iz)*128+wrap128(ix)].x,s[base+wrap128(iz)*128+wrap128(ix+1)].x,a),mixf(s[base+wrap128(iz+1)*128+wrap128(ix)].x,s[base+wrap128(iz+1)*128+wrap128(ix+1)].x,a),b);
}
// Diagnostics: query the actual interaction and regional background fields.
__global__ void domain_probe(const float4 *surface,const float4 *coefficients,const float4 *brush,const float4 *points,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
 float4 p=points[i];output[i*4]=local_pressure(surface,brush,p.x,p.y,0,1);
 output[i*4+1]=local_pressure(coefficients,brush,p.x,p.y,1,1);
 output[i*4+2]=region_wave(coefficients,p.x,p.y,1,1);
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
 float4 w=surface[16384+i];
 float shortBed=expf(-6.2831853f*waterDepth/6);
 float4 a=region_wave(surface,(float)x*.75f,(float)z*.75f,0,1);
 float4 d=pressureActive!=0?local_pressure(surface,brush,(float)x*.75f,(float)z*.75f,0,1):make_float4(0,0,0,0);
 float forcing=5*(w.x+shortBed*a.x+d.x*expf(-waterDepth*.8f));
 float rate=shallow*(forcing+8*(w.y*fabsf(w.y)+w.z*fabsf(w.z)));
 float step=fminf(.05f,fmaxf(0,dt));
 // Smooth saturation keeps the bed bounded without a hard phase clipping edge.
 float next=old.x+step*(rate/(1+old.x*old.x*.25f)-shallow*.015f*old.x);
 sandState[i]=make_float4(next,0,0,0);
}
__device__ float wave_height(const float4 *s,const float4 *brush,float x,float z,float distance,int pressureActive){
 float4 region=wave_region(x,z);float a=sample_height(s,x,z+region.x,0),b=sample_height(s,x,z,1),d=pressureActive!=0?local_pressure(s,brush,x,z,0,0).x:0;
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
__global__ void camera_step(float4 *camera,float2 *navigationState,float dt,float forward,float side,float up,float lookX,float lookY,float speed,float zoom,int reset,float depth){
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
 if(reset>0){
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
__device__ float bottom(float x,float z,float depth){return -depth+.10f*sinf(x*.19f)*sinf(z*.23f)+.04f*sinf(x*.63f+z*.31f);}
__device__ float materialNoise(float x,float z){
 float ix=floorf(x),iz=floorf(z),u=fract(x),v=fract(z);u=u*u*(3-2*u);v=v*v*(3-2*v);
 return mixf(mixf(cell(ix,iz),cell(ix+1,iz),u),mixf(cell(ix,iz+1),cell(ix+1,iz+1),u),v);
}
__device__ float3 seabed(float x,float z,float footprint){
 float broad=materialNoise(x*.42f,z*.42f),mid=materialNoise(x*3.1f+17,z*3.1f);
 float bend=materialNoise(x*.8f,z*.8f)*2;
 float phase=z*38+x*5+bend;
 float ridge=sinf(phase),slope=cosf(phase);
 float detail=1/(1+footprint*footprint*1100);
 // Subpixel grain and gravel would flicker at reduced screen resolution.
 float grainDetail=1/(1+footprint*footprint*80000);
 float grain=grainDetail>.03f?materialNoise(x*170,z*170)*grainDetail:0;
 float illumination=.76f+detail*(.13f*ridge-.15f*slope+.1f*grain);
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
 float bend=x*.41f+z*.23f,branch=z*.77f-x*.17f,modulation=x*.29f+z*.37f;
 float phase=z*31+x*2.4f+2.8f*sinf(bend)+1.1f*sinf(branch);
 float phaseX=2.4f+1.148f*cosf(bend)-.187f*cosf(branch),phaseZ=31+.644f*cosf(bend)+.847f*cosf(branch);
 float detail=1/(1+footprint*footprint*1100),amp=(.0065f+.0025f*sinf(modulation))*detail;
 float ampX=.000725f*cosf(modulation)*detail,ampZ=.000925f*cosf(modulation)*detail;
 float ridge=cosf(phase)+.18f*cosf(2*phase+.6f),derivative=-sinf(phase)-.36f*sinf(2*phase+.6f);
 return make_float4(amp*ridge,ampX*ridge+amp*derivative*phaseX,ampZ*ridge+amp*derivative*phaseZ,phase);
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
__device__ float bottom_pc(float x,float z,float depth,float footprint,float4 drift,float anchorX,float anchorZ){
 return bottom(x,z,depth)+sand_moving(x,z,footprint,drift,anchorX,anchorZ).x+stone_relief(x,z,footprint,sunDir()).x;
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
 float macroX=.019f*cosf(x*.19f)*sinf(z*.23f)+.0252f*cosf(x*.63f+z*.31f);
 float macroZ=.023f*sinf(x*.19f)*cosf(z*.23f)+.0124f*cosf(x*.63f+z*.31f);
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
 z+=wave_region(x,z).x;
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
__device__ float globe_noise(float3 p){
 float ix=floorf(p.x),iy=floorf(p.y),iz=floorf(p.z),x=fract(p.x),y=fract(p.y),z=fract(p.z);
 x=x*x*(3-2*x);y=y*y*(3-2*y);z=z*z*(3-2*z);float value=0;
 for(int k=0;k<2;k++)for(int j=0;j<2;j++)for(int i=0;i<2;i++){
  unsigned h=(unsigned)((int)(ix+(float)i)*92837111+(int)(iy+(float)j)*689287499+(int)(iz+(float)k)*283923481);
  value+=randf(h)*(i==0?1-x:x)*(j==0?1-y:y)*(k==0?1-z:z);
 }return value;
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
 camera[23]=make_float4(-1,0,0,0);
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
 float continental=clamp01(crust*.57f+broad*.43f+(medium-.5f)*.12f);
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
// The first bathymetry stage keeps continental crust submerged. The same
// signed elevation field supplies raised land in the following stage.
__global__ void geology_update(float4 *camera,float depth,int enabled){
 float3 n=vec(camera[6].x,camera[6].y,camera[6].z);float4 g=geology_sample(camera,n);
 float actual=enabled!=0?fmaxf(1.4f,-g.x):depth;
 float previous=camera[23].x;
 camera[21]=make_float4(actual,g.x,g.y,(float)enabled);
 camera[23]=make_float4(actual,fabsf(actual-previous)>.001f?1.0f:0.0f,g.z,g.w);
 camera[2].w=actual/.86f;camera[3].w=expf(-actual*.055f);
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
__device__ float cloud_density(const float4 *camera,float3 normal,int layer){
 float cover=weather_map_sample(camera,normal,0).x;
 if(cover<.04f)return 0;
 float days=camera[13].x/86400,ay=fabsf(normal.y);
 float mid=expf(-(ay-.7f)*(ay-.7f)*40),zonal=-7+24*mid-4*eased(.8f,.98f,ay);
 float3 q=rotate_y(normal,-days*zonal*.01356f);
 // Four filtered bands retain wisps and structure within large systems.
 float coarse=globe_noise(plus(scale(q,31),vec(11,17,31)));
 float medium=globe_noise(plus(scale(q,83),vec(3,29,7)));
 float fine=globe_noise(scale(q,211));
 float detail=globe_noise(scale(q,layer==0?823:397));
 float resolved=1-eased(30000,300000,camera[0].y);
 float shape=coarse*.42f+medium*.30f+fine*.20f+mixf(.5f,detail,resolved)*.08f;
 float threshold=.59f-cover*.31f;
 float density=fmaxf(0,shape-threshold)*3.2f;
 return clamp01(density)*(.60f+.40f*medium)*(layer==0?1.0f:.35f)*eased(.03f,.35f,cover);
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
  if(targetDepth<20&&crossing==0)continue;
  float lon=(((float)x+.5f+fraction)/(float)width-.5f)*6.2831853f,lat=(.5f-((float)y+.5f)/(float)height)*3.14159265f;
  float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
  float score=dotv(n,sun)*20-fabsf(h+targetDepth)*(crossing!=0?0:.001f)-fabsf(n.y)*2;
  if(score>best){best=score;bx=(float)x+.5f+fraction;by=(float)y+.5f;}
 }
 float lon=(bx/(float)width-.5f)*6.2831853f,lat=(.5f-by/(float)height)*3.14159265f;
 float3 n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
 // Correct the two-cell coarse bracket against the actual bilinear sampler.
 if(targetDepth<20){
  float lo=lon-6.2831853f*2/(float)width,hi=lon+6.2831853f*2/(float)width;
  float sign=geology_sample(camera,vec(sinf(lo)*cosf(lat),sinf(lat),cosf(lo)*cosf(lat))).x+targetDepth;
  for(int j=0;j<18;j++){float mid=(lo+hi)*.5f;float h=geology_sample(camera,vec(sinf(mid)*cosf(lat),sinf(lat),cosf(mid)*cosf(lat))).x+targetDepth;if(h*sign>0)lo=mid;else hi=mid;}
  lon=(lo+hi)*.5f;n=vec(sinf(lon)*cosf(lat),sinf(lat),cosf(lon)*cosf(lat));
 }
 float3 east=vec(cosf(lon),0,-sinf(lon));
 navigationState[0]=make_float2(n.x,0);navigationState[1]=make_float2(n.y,0);navigationState[2]=make_float2(n.z,0);
 navigationState[3]=make_float2(east.x,0);navigationState[4]=make_float2(east.y,0);navigationState[5]=make_float2(east.z,0);
 navigationState[6]=make_float2(0,0);navigationState[7]=make_float2(4,0);
 camera[0]=make_float4(0,targetDepth<20?2.6f:8,4,0);camera[1]=make_float4(0,-.32f,0,0);camera[5].w=-1000;camera[19].w=0;
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
__device__ float4 cloud_volume(float3 ray,const float4 *camera,float ground){
 float altitude=camera[0].y,R=earth_radius();float2 outer=sphere_roots(altitude,ray,8200),inner=sphere_roots(altitude,ray,800);
 if(outer.y<=0)return make_float4(0,0,0,0);
 float start=fmaxf(0,outer.x),end=outer.y;
 if(altitude<800)start=fmaxf(start,inner.y);
 else if(inner.x>0)end=fminf(end,inner.x);
 if(ground>0)end=fminf(end,ground);
 if(end<=start)return make_float4(0,0,0,0);
 int steps=camera[16].x>256?24:12;float step=(end-start)/(float)steps,trans=1;float3 color=vec(0,0,0),sun=vec(camera[9].x,camera[9].y,camera[9].z);
 float phase=.45f+.65f*positive_power(fmaxf(0,dotv(ray,sun)),8);
 for(int i=0;i<steps;i++){
  if(trans<.015f)break;
  float t=start+((float)i+.5f)*step;float3 p=vec(ray.x*t,R+altitude+ray.y*t,ray.z*t);
  float radius=sqrtf(dotv(p,p)),h=radius-R;float3 local=scale(p,1/radius),normal=to_world(camera,local);
  float cover=weather_map_sample(camera,normal,0).x;
  float top=2200+cover*5300,vertical=(h-800)/(top-800);
  if(vertical<=0||vertical>=1||cover<.08f)continue;
  float column=cloud_density(camera,normal,0);if(column<.008f)continue;
  float days=camera[13].x/86400,ay=fabsf(normal.y),zonal=-7+24*expf(-(ay-.7f)*(ay-.7f)*40)-4*eased(.8f,.98f,ay);
  float3 q=rotate_y(scale(normal,radius*.00065f),-days*zonal*.01356f);
  float detail=globe_noise(q)*.67f+globe_noise(scale(q,2.37f))*.33f;
  float envelope=eased(0,.12f,vertical)*(1-eased(.48f,1,vertical));
  float density=fmaxf(0,column-(1-detail)*.20f)*envelope;
  if(density<.003f)continue;
  float mu=dotv(local,sun),daylight=eased(-.08f,.22f,mu);
  float optical=(top-h)*column*.0017f/fmaxf(.15f,mu);
  float direct=expf(-optical),fill=.10f+.18f*expf(-density*3);
  float3 ambient=blend(vec(.055f,.080f,.13f),vec(.34f,.42f,.52f),vertical);
  float3 lit=plus(scale(ambient,.015f+daylight*(.4f+fill)),scale(vec(1,.96f,.86f),daylight*(direct*.9f+fill*.45f)*phase));
  float opacity=1-expf(-density*step*.0023f);
  color=plus(color,scale(lit,trans*opacity));trans*=1-opacity;
 }
 return make_float4(color.x,color.y,color.z,1-trans);
}
__global__ void weather_cloud_view(float4 *camera,int cloudWidth,int cloudHeight,float aspect){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=cloudWidth||y>=cloudHeight)return;
 if(x==0&&y==0)camera[20]=make_float4((float)cloudWidth,(float)cloudHeight,0,0);
 float sx=(2*((float)x+.5f)/(float)cloudWidth-1)*aspect,sy=1-2*((float)y+.5f)/(float)cloudHeight;
 float3 f=vec(camera[2].x,camera[2].y,camera[2].z),r=vec(camera[3].x,camera[3].y,camera[3].z),u=vec(camera[4].x,camera[4].y,camera[4].z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));
 camera[131104+y*cloudWidth+x]=cloud_volume(ray,camera,globe_hit(camera[0].y,ray));
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
 float rain=camera[11].x,time=camera[13].y;if(rain<.01f||origin.y>1400)return 0;
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
__device__ float3 planet_radiance(float3 ray,const float4 *camera,float hit,int cachedCloud,float pixelU,float pixelV){
 float altitude=camera[0].y,R=earth_radius();float4 sl=camera[9];float3 sun=vec(sl.x,sl.y,sl.z),worldRay=to_world(camera,ray);
 float visibility=fmaxf(eased(25000,90000,altitude),1-eased(-.12f,.08f,sun.y));
 float3 color=scale(space_stars(camera[13].w!=0?rotate_y(worldRay,camera[13].x*.00007292115f):worldRay),visibility);float sunDot=dotv(ray,sun);
 if(sunDot>.99996f)color=plus(color,vec(18,15,11));
 if(hit>0){
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
   float footprint=hit*.001f,detail=1/(1+footprint*footprint*.012f),time=camera[13].y;
   float warp=detail>.01f?globe_noise(scale(normal,R/240))*4:0;
   float phase1=dotv(normal,vec(.042f,.011f,.028f))*R-time*.72f+warp;
   float phase2=dotv(normal,vec(-.017f,.031f,.006f))*R-time*.53f-warp*.7f;
   float3 variation=plus(scale(vec(.83f,.22f,.55f),cosf(phase1)*.028f),scale(vec(-.47f,.86f,.17f),cosf(phase2)*.023f));
   variation=scale(minus(variation,scale(normal,dotv(normal,variation))),detail);
   float3 worldNormal=unit(minus(normal,variation));
   shadingNormal=vec(dotv(worldNormal,vec(camera[5].x,camera[5].y,camera[5].z)),dotv(worldNormal,vec(camera[6].x,camera[6].y,camera[6].z)),dotv(worldNormal,vec(camera[7].x,camera[7].y,camera[7].z)));
  }
  float nv=clamp01(-dotv(shadingNormal,ray)),grazing=1-nv,g2=grazing*grazing,fresnel=.02037f+.97963f*g2*g2*grazing;
  float3 reflectedRay=minus(ray,scale(shadingNormal,2*dotv(ray,shadingNormal)));
  float skyElevation=clamp01(dotv(reflectedRay,localNormal));
  float3 reflected=scale(blend(vec(.36f,.50f,.63f),vec(.035f,.10f,.23f),sqrtf(skyElevation)),.012f+day);
  if(camera[13].w!=0&&altitude<400)reflected=weather_sky_sample(reflectedRay,camera);
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
 if(camera[13].w!=0){
  float4 clouds=cachedCloud!=0?cloud_view_sample(camera,pixelU,pixelV):globe_clouds(ray,camera,hit);
  // In the lower atmosphere, haze lies between the eye and the clouds. Do not
  // integrate an entire clear-sky column over an opaque cloud base.
  float2 roots=sphere_roots(altitude,ray,1600);float cloudDistance=roots.x>0?roots.x:roots.y;
  float visible=mixf(expf(-fmaxf(0,cloudDistance)*.000055f),1,eased(5000,20000,altitude));
  color=plus(scale(color,1-clouds.w*visible),scale(vec(clouds.x,clouds.y,clouds.z),visible));
 }
 return color;
}

__global__ void weather_sky(float4 *camera,int skyWidth){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,height=skyWidth/4;
 if(x>=skyWidth||y>=height||camera[0].y>=400)return;
 float angle=(((float)x+.5f)/(float)skyWidth)*6.2831853f,e=((float)y+.5f)/(float)height;e=e*e*1.5707963f;
 float3 ray=vec(sinf(angle)*cosf(e),sinf(e),-cosf(angle)*cosf(e));
 float3 col=planet_radiance(ray,camera,-1,0,0,0);camera[65568+y*skyWidth+x]=make_float4(col.x,col.y,col.z,1);
}
__device__ float3 weather_sky_sample(float3 d,const float4 *camera){
 int width=(int)camera[16].x,height=width/4;
 float u=atan2f(d.x,-d.z)*(float)width/6.2831853f-.5f;
 float v=sqrtf(atan2f(fmaxf(0,d.y),sqrtf(d.x*d.x+d.z*d.z))/1.5707963f)*(float)height-.5f;
 v=fminf((float)height-1.001f,fmaxf(0,v));int x=(int)floorf(u),y=(int)floorf(v);float a=fract(u),b=fract(v);float3 col=vec(0,0,0);
 for(int j=0;j<2;j++)for(int i=0;i<2;i++){float4 p=camera[65568+(y+j)*width+((x+i)&(width-1))];col=plus(col,scale(vec(p.x,p.y,p.z),(i==0?1-a:a)*(j==0?1-b:b)));}return col;
}
__device__ float3 surface_weather(float3 col,float3 ray,const float4 *camera,float distance){
 float rain=camera[11].x;if(rain<.01f)return col;
 float daylight=.015f+.985f*eased(-.08f,.3f,camera[9].y);
 float haze=(1-expf(-fminf(distance,3000)*rain*.0012f));
 col=blend(col,scale(vec(.17f,.23f,.28f),daylight),haze);
 float streak=rain_volume(vec(camera[0].x,camera[0].y,camera[0].z),ray,distance,camera);
 return plus(col,scale(vec(.45f,.55f,.62f),streak*daylight*.4f));
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
 if(globeT<0||planetMix>=1){float3 far=planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height);if(camera[13].w!=0&&pos.y<400)far=surface_weather(far,ray,camera,globeT>0?globeT:3000);return scale(far,exposure*vignette);}
 if(globeT>0){
 float t=globeT;float4 w=make_float4(0,0,0,0);
 for(int i=0;i<(smooth!=0?6:4);i++){float h=smooth!=0?wave_pc(coefficients,brush,pos.x+ray.x*t,pos.z+ray.z*t,1/(1+t*t*.0008f),pressureActive,0).x:wave_height(surface,brush,pos.x+ray.x*t,pos.z+ray.z*t,t,pressureActive);h-=(ray.x*ray.x+ray.z*ray.z)*t*t/(2*earth_radius());t=mixf(t,(h-pos.y)/ray.y,.75f);}
 float3 p=vec(pos.x+ray.x*t,pos.y+ray.y*t,pos.z+ray.z*t);
 float causticDetail=1/(1+t*t*.0008f),pixelFootprint=t/((float)height*sampleScale);
 w=smooth!=0?wave_pc(coefficients,brush,p.x,p.z,causticDetail,pressureActive,1):wave(surface,brush,p.x,p.z,causticDetail,pressureActive);float3 rain=camera[13].w!=0?rain_surface(p.x,p.z,camera[13].y,camera[11].x,pixelFootprint):vec(0,0,0);float3 n=unit(vec(-w.y-rain.x+ray.x*t/earth_radius(),1,-w.z-rain.z+ray.z*t/earth_radius()));float viewCosine=dotv(n,ray),nv=fmaxf(.02f,-viewCosine);
 float grazing=1-clamp01(nv),grazing2=grazing*grazing;
 float fresnel=.02037f+.97963f*grazing2*grazing2*grazing;
 float3 reflection=minus(ray,scale(n,2*viewCosine));float3 reflected=camera[13].w!=0?weather_sky_sample(reflection,camera):sky(reflection,localSun);
 float3 transmitted=refract_cosine(ray,n,.7502f,viewCosine);float vertical=fminf(-.1f,transmitted.y);float travel=(-depth-p.y)/vertical;
 float bx=p.x+transmitted.x*travel,bz=p.z+transmitted.z*travel;
 for(int j=0;j<(smooth!=0?4:2);j++){travel=(bottom(bx,bz,depth)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 travel=fmaxf(0,travel);float3 bed=seabed(bx,bz,pixelFootprint),ca=caustic(light,monoLight,bx,bz,dispersion,lightSize);
 // Light travels down through the water before returning along the view ray.
 float opticalDistance=travel+forward.w;
 float shadow=camera[13].w!=0?camera[11].y:1;ca=scale(blend(vec(1,1,1),ca,causticDetail*right.w*shadow),.32f+.68f*shadow);
 float attenR=expf(-opticalDistance*.19f),attenG=expf(-opticalDistance*.09f),attenB=expf(-opticalDistance*.055f);
 float3 through=vec(bed.x*(.12f+.95f*ca.x)*attenR+.008f*(1-attenR),bed.y*(.12f+.95f*ca.y)*attenG+.042f*(1-attenG),bed.z*(.12f+.95f*ca.z)*attenB+.075f*(1-attenB));
 col=blend(scale(through,.015f+.985f*eased(-.08f,.3f,localSun.y)),reflected,fresnel);
 float3 halfv=unit(minus(localSun,ray));
 float specPower=mixf(320,8000,1/(1+pixelFootprint*pixelFootprint*800));
 float spec=positive_power(fmaxf(0,dotv(n,halfv)),specPower)*3.5f*(specPower/8000);
 col=plus(col,scale(vec(1,.89f,.68f),spec*(camera[13].w!=0?camera[11].y*eased(-.02f,.08f,localSun.y):1)));
 float haze=1-expf(-t*.00025f);col=blend(col,scale(vec(.38f,.55f,.68f),.015f+.985f*eased(-.08f,.3f,localSun.y)),haze);if(camera[13].w!=0)col=surface_weather(col,ray,camera,t);
 if(view==1)col=scale(ca,.35f);if(view==2)col=plus(scale(n,.5f),vec(.5f,.5f,.5f));
 }else{col=sky(ray,localSun);}
 if(planetMix>0)col=blend(col,planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height),planetMix);
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
 if(globeT<0||planetMix>=1){float3 far=planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height);if(camera[13].w!=0&&pos.y<400)far=surface_weather(far,ray,camera,globeT>0?globeT:3000);return scale(far,exposure*vignette);}
 if(globeT>0){
 float t=globeT;float4 w=make_float4(0,0,0,0);
 for(int i=0;i<(smooth!=0?6:4);i++){float h=smooth!=0?wave_pc(coefficients,brush,pos.x+ray.x*t,pos.z+ray.z*t,1/(1+t*t*.0008f),pressureActive,0).x:wave_height(surface,brush,pos.x+ray.x*t,pos.z+ray.z*t,t,pressureActive);h-=(ray.x*ray.x+ray.z*ray.z)*t*t/(2*earth_radius());t=mixf(t,(h-pos.y)/ray.y,.75f);}
 float3 p=vec(pos.x+ray.x*t,pos.y+ray.y*t,pos.z+ray.z*t);
 float causticDetail=1/(1+t*t*.0008f),pixelFootprint=t/((float)height*sampleScale);
 w=smooth!=0?wave_pc(coefficients,brush,p.x,p.z,causticDetail,pressureActive,1):wave(surface,brush,p.x,p.z,causticDetail,pressureActive);float3 rain=camera[13].w!=0?rain_surface(p.x,p.z,camera[13].y,camera[11].x,pixelFootprint):vec(0,0,0);float3 n=unit(vec(-w.y-rain.x+ray.x*t/earth_radius(),1,-w.z-rain.z+ray.z*t/earth_radius()));float viewCosine=dotv(n,ray),nv=fmaxf(.02f,-viewCosine);
 float grazing=1-clamp01(nv),grazing2=grazing*grazing;
 float fresnel=.02037f+.97963f*grazing2*grazing2*grazing;
 float3 reflection=minus(ray,scale(n,2*viewCosine));float3 reflected=camera[13].w!=0?weather_sky_sample(reflection,camera):sky(reflection,localSun);
 float3 transmitted=refract_cosine(ray,n,.7502f,viewCosine);float vertical=fminf(-.1f,transmitted.y);float travel=(-depth-p.y)/vertical;
 float bx=p.x+transmitted.x*travel,bz=p.z+transmitted.z*travel;
 for(int j=0;j<2;j++){travel=(bottom(bx,bz,depth)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 float anchorX=bx,anchorZ=bz;float4 drift=make_float4(0,0,0,0);
 if(depth<3){float shallow=clamp01((3-depth)/2);shallow=shallow*shallow*(3-2*shallow);drift=sample_pc(sandState,bx/16,bz/16,0,1);drift.x*=shallow;drift.y*=shallow/16;drift.z*=shallow/16;}
 if(smooth!=0)for(int j=0;j<2;j++){travel=(bottom_pc(bx,bz,depth,pixelFootprint,drift,anchorX,anchorZ)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 travel=fmaxf(0,travel);float3 bed=vec(0,0,0);
 if(smooth!=0){float4 sand=sand_moving(bx,bz,pixelFootprint,drift,anchorX,anchorZ),stone=stone_relief(bx,bz,pixelFootprint,localSun);float3 bedSun=scale(refractv(scale(localSun,-1),n,.7502f),-1);bed=seabed_pc(bx,bz,pixelFootprint,sand,stone,bedSun,transmitted);}
 else bed=seabed(bx,bz,pixelFootprint);
 float3 ca=caustic(light,monoLight,bx,bz,dispersion,lightSize);
 // Light travels down through the water before returning along the view ray.
 float opticalDistance=travel+forward.w;
 float shadow=camera[13].w!=0?camera[11].y:1;ca=scale(blend(vec(1,1,1),ca,causticDetail*right.w*shadow),.32f+.68f*shadow);
 float attenR=expf(-opticalDistance*.19f),attenG=expf(-opticalDistance*.09f),attenB=expf(-opticalDistance*.055f);
 float3 through=vec(bed.x*(.12f+.95f*ca.x)*attenR+.008f*(1-attenR),bed.y*(.12f+.95f*ca.y)*attenG+.042f*(1-attenG),bed.z*(.12f+.95f*ca.z)*attenB+.075f*(1-attenB));
 col=blend(scale(through,.015f+.985f*eased(-.08f,.3f,localSun.y)),reflected,fresnel);
 float3 halfv=unit(minus(localSun,ray));
 float specPower=mixf(320,8000,1/(1+pixelFootprint*pixelFootprint*800));
 float spec=positive_power(fmaxf(0,dotv(n,halfv)),specPower)*3.5f*(specPower/8000);
 col=plus(col,scale(vec(1,.89f,.68f),spec*(camera[13].w!=0?camera[11].y*eased(-.02f,.08f,localSun.y):1)));
 float haze=1-expf(-t*.00025f);col=blend(col,scale(vec(.38f,.55f,.68f),.015f+.985f*eased(-.08f,.3f,localSun.y)),haze);if(camera[13].w!=0)col=surface_weather(col,ray,camera,t);
 if(view==1)col=scale(ca,.35f);if(view==2)col=plus(scale(n,.5f),vec(.5f,.5f,.5f));
 }else{col=sky(ray,localSun);}
 if(planetMix>0)col=blend(col,planet_radiance(ray,camera,globeT,1,pixelX/(float)width,pixelY/(float)height),planetMix);
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
