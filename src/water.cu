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
__global__ void prepare_modes(float *motion,float depth){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 int fx=x<64?x:x-128,fz=z<64?z:z-128;
 float kx=6.2831853f*(float)fx/24,kz=6.2831853f*(float)fz/24,kk=kx*kx+kz*kz;
 float k=c==2?sqrtf(kk):6.2831853f*sqrtf((float)(fx*fx+fz*fz))/patch(c);
 float e=expf(-2*fminf(20,k*depth)),tanhd=(1-e)/(1+e);
 float omega=c==2?sqrtf(9.81f*k*(1-e)/(1+e)):sqrtf(9.81f*k*tanhd);
 int idx=z*128+x;motion[c*16384+idx]=omega;
 if(c==2){motion[49152+idx]=expf(-kk*.24f*.24f*.5f);motion[65536+idx]=.32f+.009f*kk;}
}
__global__ void spectrum(float2 *output,const float4 *seed,const float4 *disturbance,const float *motion,float time,float energy){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.z;
 if(x>=128||z>=128||c>=3)return;
 if(c==2){float4 d=disturbance[z*128+x];output[c*16384+reverse7(z)*128+reverse7(x)]=make_float2(d.x,d.y);return;}
 float4 h=seed[c*16384+z*128+x];float2 a=make_float2(h.x,h.y),b=make_float2(h.z,h.w);
 float phase=motion[c*16384+z*128+x]*time;
 float2 p=make_float2(cosf(phase),sinf(phase));
 float2 u=cmul(a,p),v=cmul(make_float2(b.x,-b.y),make_float2(p.x,-p.y));
 output[c*16384+reverse7(z)*128+reverse7(x)]=make_float2((u.x+v.x)*energy,(u.y+v.y)*energy);
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
__global__ void sand_transport(const float4 *brush,const float4 *surface,float4 *sandState,float dt,float depth,int pressureActive){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=128||z>=128)return;
 int i=z*128+x;float4 old=sandState[i];
 float shallow=clamp01((3-depth)/2);shallow=shallow*shallow*(3-2*shallow);
 float4 w=surface[16384+i];
 float shortBed=expf(-6.2831853f*depth/6);
 float4 a=region_wave(surface,(float)x*.75f,(float)z*.75f,0,1);
 float4 d=pressureActive!=0?local_pressure(surface,brush,(float)x*.75f,(float)z*.75f,0,1):make_float4(0,0,0,0);
 float forcing=5*(w.x+shortBed*a.x+d.x*expf(-depth*.8f));
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
// Camera basis and depth-only optical factors are computed once per frame.
// Basis w components carry optical factors; xyz remain camera directions.
__global__ void camera_step(float4 *camera,float dt,float forward,float side,float up,float lookX,float lookY,float speed,int reset,float depth){
 float4 p=camera[0],r=camera[1];
 if(reset>0){p=reset==2?make_float4(0,8,16,0):make_float4(0,2.6f,4,0);r=make_float4(0,reset==2?-.4f:-.32f,0,0);}
 else{
 r.x+=lookX;r.y=fminf(1.5f,fmaxf(-1.5f,r.y+lookY));
 float length=fmaxf(1,sqrtf(forward*forward+side*side+up*up));float d=dt*speed/length;
 p.x+=d*(sinf(r.x)*cosf(r.y)*forward+cosf(r.x)*side);
 p.z+=d*(-cosf(r.x)*cosf(r.y)*forward+sinf(r.x)*side);
 p.y=fmaxf(.45f,p.y+d*(sinf(r.y)*forward+up));
 }
 camera[0]=p;camera[1]=r;
 float yawSin=sinf(r.x),yawCos=cosf(r.x),pitchSin=sinf(r.y),pitchCos=cosf(r.y);
 camera[2]=make_float4(yawSin*pitchCos,pitchSin,-yawCos*pitchCos,depth/.86f);
 camera[3]=make_float4(yawCos,0,yawSin,expf(-depth*.055f));
 camera[4]=make_float4(-yawSin*pitchSin,pitchCos,yawCos*pitchSin,0);
}
// Project the pointer to the actual FFT surface on the GPU, retaining the
// previous hit so a held drag injects force along its world-space path.
__global__ void brush_pick(const float4 *surface,const float4 *camera,float4 *brush,float pointerX,float pointerY,float aspect,int held,int moving,int pressureActive){
 float4 domain=brush[2];domain.z=0;brush[2]=domain;
 float4 old=brush[1];brush[0]=make_float4(0,0,0,0);
 if(held==0){brush[1]=make_float4(old.x,old.y,0,0);return;}
 float4 p=camera[0],r=camera[1];
 float3 f=vec(sinf(r.x)*cosf(r.y),sinf(r.y),-cosf(r.x)*cosf(r.y));
 float3 right=vec(cosf(r.x),0,sinf(r.x)),up=vec(-sinf(r.x)*sinf(r.y),cosf(r.y),cosf(r.x)*sinf(r.y));
 float3 ray=unit(plus(f,plus(scale(right,pointerX*aspect*.65f),scale(up,pointerY*.65f))));
 if(ray.y>-.06f){brush[1]=make_float4(old.x,old.y,0,0);return;}
 float t=-p.y/ray.y;
 for(int j=0;j<4;j++){float h=wave_height(surface,brush,p.x+ray.x*t,p.z+ray.z*t,t,pressureActive);t=mixf(t,(h-p.y)/ray.y,.75f);}
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
__device__ float3 sky(float3 d){
 float v=positive_power(clamp01(d.y),.45f);float3 col=blend(vec(.38f,.55f,.68f),vec(.045f,.16f,.36f),v);
 // Explicit fixed powers avoid the compiler's software-f64 integer pow path.
 float sun=fmaxf(0,dotv(d,sunDir())),s2=sun*sun,s4=s2*s2,s8=s4*s4,s16=s8*s8;
 // Below .99 the disk power is smaller than the minimum float value.
 float sunDisk=sun>.99f?positive_power(sun,16000):0;
 col=plus(col,scale(vec(1,.83f,.55f),sunDisk*20+s16*s16*s16*.07f));
 float cloud=clamp01(.5f+.25f*sinf(d.x*28+d.z*17)+.25f*sinf(d.z*43-d.x*12));
 float cloud2=cloud*cloud,cloud4=cloud2*cloud2;
 float cirrus=cloud4*cloud4*clamp01(d.y*3)*.22f;
 return blend(col,vec(.94f,.96f,1),cirrus);
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
__device__ float4 stone_relief(float x,float z,float footprint){
 float fade=clamp01((.065f-footprint)/.035f);
 if(fade<=0)return make_float4(0,0,0,0);
 // This bound includes the maximum cap and shadow extent.
 if(materialNoise(x*.7f+83,z*.7f-19)<.32f)return make_float4(0,0,0,0);
 float gx=floorf(x*7),gz=floorf(z*7),height=0,gradientX=0,gradientZ=0,tone=-1,shadow=0;
 float3 sun=sunDir();float vertical=sqrtf(1-.7502f*.7502f*(1-sun.y*sun.y));
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
 return bottom(x,z,depth)+sand_moving(x,z,footprint,drift,anchorX,anchorZ).x+stone_relief(x,z,footprint).x;
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
 float4 sand=sand_moving(x,z,.001f,drift,0,0),stone=stone_relief(x,z,.001f);
 float sx=(sand_moving(x+e,z,.001f,drift,0,0).x-sand_moving(x-e,z,.001f,drift,0,0).x)/(2*e),sz=(sand_moving(x,z+e,.001f,drift,0,0).x-sand_moving(x,z-e,.001f,drift,0,0).x)/(2*e);
 float4 a=stone_relief(x+e,z,.001f),b=stone_relief(x-e,z,.001f),c=stone_relief(x,z+e,.001f),d=stone_relief(x,z-e,.001f);
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
__global__ void caustic_map(const float4 *surface,unsigned *photons,float depth,int rays,int dispersion,int lightSize){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=rays||z>=rays)return;
 float wx=((float)x+.5f)*6/(float)rays,wz=((float)z+.5f)*6/(float)rays;
 float4 w=lightSize>256?sample_pc(surface,wx,wz,0,1):sample(surface,wx,wz,0);
 float3 n=unit(vec(-w.y,1,-w.z)),incident=scale(sunDir(),-1);
 int channels=dispersion!=0?3:1;
 for(int c=0;c<channels;c++){
  float eta=dispersion==0?.7502f:(c==0?.7524f:(c==1?.7502f:.7480f));
  float3 d=refractv(incident,n,eta);float distance=(-depth-w.x)/d.y;
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
__device__ float3 shade_pixel(const float4 *brush,const float4 *surface,const float4 *coefficients,const float4 *light,const float *monoLight,const float4 *camera,int width,int height,float depth,float exposure,int view,int pressureActive,int dispersion,int lightSize,float pixelX,float pixelY,float sampleScale){
 int smooth=lightSize>256?1:0;
 float4 pos=camera[0],forward=camera[2],right=camera[3],up=camera[4];float sx=(2*pixelX/(float)width-1)*(float)width/(float)height;
 float sy=1-2*pixelY/(float)height;float3 f=vec(forward.x,forward.y,forward.z);
 float3 r=vec(right.x,right.y,right.z),u=vec(up.x,up.y,up.z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));float3 col=vec(0,0,0);
 if(ray.y<-.0005f){
 float t=-pos.y/ray.y;float4 w=make_float4(0,0,0,0);
 for(int i=0;i<(smooth!=0?6:4);i++){float h=smooth!=0?wave_pc(coefficients,brush,pos.x+ray.x*t,pos.z+ray.z*t,1/(1+t*t*.0008f),pressureActive,0).x:wave_height(surface,brush,pos.x+ray.x*t,pos.z+ray.z*t,t,pressureActive);t=mixf(t,(h-pos.y)/ray.y,.75f);}
 float3 p=vec(pos.x+ray.x*t,pos.y+ray.y*t,pos.z+ray.z*t);
 float causticDetail=1/(1+t*t*.0008f),pixelFootprint=t/((float)height*sampleScale);
 w=smooth!=0?wave_pc(coefficients,brush,p.x,p.z,causticDetail,pressureActive,1):wave(surface,brush,p.x,p.z,causticDetail,pressureActive);float3 n=unit(vec(-w.y,1,-w.z));float viewCosine=dotv(n,ray),nv=fmaxf(.02f,-viewCosine);
 float grazing=1-clamp01(nv),grazing2=grazing*grazing;
 float fresnel=.02037f+.97963f*grazing2*grazing2*grazing;
 float3 reflected=sky(minus(ray,scale(n,2*viewCosine)));
 float3 transmitted=refract_cosine(ray,n,.7502f,viewCosine);float vertical=fminf(-.1f,transmitted.y);float travel=(-depth-p.y)/vertical;
 float bx=p.x+transmitted.x*travel,bz=p.z+transmitted.z*travel;
 for(int j=0;j<(smooth!=0?4:2);j++){travel=(bottom(bx,bz,depth)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 travel=fmaxf(0,travel);float3 bed=seabed(bx,bz,pixelFootprint),ca=caustic(light,monoLight,bx,bz,dispersion,lightSize);
 // Light travels down through the water before returning along the view ray.
 float opticalDistance=travel+forward.w;
 ca=blend(vec(1,1,1),ca,causticDetail*right.w);
 float attenR=expf(-opticalDistance*.19f),attenG=expf(-opticalDistance*.09f),attenB=expf(-opticalDistance*.055f);
 float3 through=vec(bed.x*(.12f+.95f*ca.x)*attenR+.008f*(1-attenR),bed.y*(.12f+.95f*ca.y)*attenG+.042f*(1-attenG),bed.z*(.12f+.95f*ca.z)*attenB+.075f*(1-attenB));
 col=blend(through,reflected,fresnel);
 float3 halfv=unit(minus(sunDir(),ray));
 float specPower=mixf(320,8000,1/(1+pixelFootprint*pixelFootprint*800));
 float spec=positive_power(fmaxf(0,dotv(n,halfv)),specPower)*3.5f*(specPower/8000);
 col=plus(col,scale(vec(1,.89f,.68f),spec));
 float haze=1-expf(-t*.00025f);col=blend(col,vec(.38f,.55f,.68f),haze);
 if(view==1)col=scale(ca,.35f);if(view==2)col=plus(scale(n,.5f),vec(.5f,.5f,.5f));
 }else{col=sky(ray);}
 float vignette=1-.10f*(sx*sx+sy*sy);col=scale(col,exposure*vignette);
 return col;
}
__device__ float3 shade_pixel_pc(const float4 *brush,const float4 *sandState,const float4 *surface,const float4 *coefficients,const float4 *light,const float *monoLight,const float4 *camera,int width,int height,float depth,float exposure,int view,int pressureActive,int dispersion,int lightSize,float pixelX,float pixelY,float sampleScale){
 int smooth=lightSize>256?1:0;
 float4 pos=camera[0],forward=camera[2],right=camera[3],up=camera[4];float sx=(2*pixelX/(float)width-1)*(float)width/(float)height;
 float sy=1-2*pixelY/(float)height;float3 f=vec(forward.x,forward.y,forward.z);
 float3 r=vec(right.x,right.y,right.z),u=vec(up.x,up.y,up.z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));float3 col=vec(0,0,0);
 if(ray.y<-.0005f){
 float t=-pos.y/ray.y;float4 w=make_float4(0,0,0,0);
 for(int i=0;i<(smooth!=0?6:4);i++){float h=smooth!=0?wave_pc(coefficients,brush,pos.x+ray.x*t,pos.z+ray.z*t,1/(1+t*t*.0008f),pressureActive,0).x:wave_height(surface,brush,pos.x+ray.x*t,pos.z+ray.z*t,t,pressureActive);t=mixf(t,(h-pos.y)/ray.y,.75f);}
 float3 p=vec(pos.x+ray.x*t,pos.y+ray.y*t,pos.z+ray.z*t);
 float causticDetail=1/(1+t*t*.0008f),pixelFootprint=t/((float)height*sampleScale);
 w=smooth!=0?wave_pc(coefficients,brush,p.x,p.z,causticDetail,pressureActive,1):wave(surface,brush,p.x,p.z,causticDetail,pressureActive);float3 n=unit(vec(-w.y,1,-w.z));float viewCosine=dotv(n,ray),nv=fmaxf(.02f,-viewCosine);
 float grazing=1-clamp01(nv),grazing2=grazing*grazing;
 float fresnel=.02037f+.97963f*grazing2*grazing2*grazing;
 float3 reflected=sky(minus(ray,scale(n,2*viewCosine)));
 float3 transmitted=refract_cosine(ray,n,.7502f,viewCosine);float vertical=fminf(-.1f,transmitted.y);float travel=(-depth-p.y)/vertical;
 float bx=p.x+transmitted.x*travel,bz=p.z+transmitted.z*travel;
 for(int j=0;j<2;j++){travel=(bottom(bx,bz,depth)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 float anchorX=bx,anchorZ=bz;float4 drift=make_float4(0,0,0,0);
 if(depth<3){float shallow=clamp01((3-depth)/2);shallow=shallow*shallow*(3-2*shallow);drift=sample_pc(sandState,bx/16,bz/16,0,1);drift.x*=shallow;drift.y*=shallow/16;drift.z*=shallow/16;}
 if(smooth!=0)for(int j=0;j<2;j++){travel=(bottom_pc(bx,bz,depth,pixelFootprint,drift,anchorX,anchorZ)-p.y)/vertical;bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 travel=fmaxf(0,travel);float3 bed=vec(0,0,0);
 if(smooth!=0){float4 sand=sand_moving(bx,bz,pixelFootprint,drift,anchorX,anchorZ),stone=stone_relief(bx,bz,pixelFootprint);float3 bedSun=scale(refractv(scale(sunDir(),-1),n,.7502f),-1);bed=seabed_pc(bx,bz,pixelFootprint,sand,stone,bedSun,transmitted);}
 else bed=seabed(bx,bz,pixelFootprint);
 float3 ca=caustic(light,monoLight,bx,bz,dispersion,lightSize);
 // Light travels down through the water before returning along the view ray.
 float opticalDistance=travel+forward.w;
 ca=blend(vec(1,1,1),ca,causticDetail*right.w);
 float attenR=expf(-opticalDistance*.19f),attenG=expf(-opticalDistance*.09f),attenB=expf(-opticalDistance*.055f);
 float3 through=vec(bed.x*(.12f+.95f*ca.x)*attenR+.008f*(1-attenR),bed.y*(.12f+.95f*ca.y)*attenG+.042f*(1-attenG),bed.z*(.12f+.95f*ca.z)*attenB+.075f*(1-attenB));
 col=blend(through,reflected,fresnel);
 float3 halfv=unit(minus(sunDir(),ray));
 float specPower=mixf(320,8000,1/(1+pixelFootprint*pixelFootprint*800));
 float spec=positive_power(fmaxf(0,dotv(n,halfv)),specPower)*3.5f*(specPower/8000);
 col=plus(col,scale(vec(1,.89f,.68f),spec));
 float haze=1-expf(-t*.00025f);col=blend(col,vec(.38f,.55f,.68f),haze);
 if(view==1)col=scale(ca,.35f);if(view==2)col=plus(scale(n,.5f),vec(.5f,.5f,.5f));
 }else{col=sky(ray);}
 float vignette=1-.10f*(sx*sx+sy*sy);col=scale(col,exposure*vignette);
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
