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
__device__ float randf(unsigned a){return ((float)(scramble(a)&16777215u)+.5f)/16777216;}
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
__device__ float4 wave(const float4 *s,float x,float z,float distance){
 float4 a=sample(s,x,z,0),b=sample(s,x,z,1),d=sample(s,x,z,2);
 float fade=1/(1+distance*distance*.0008f);
 // Fade unresolved short displacement as well as its normal near the horizon.
 return make_float4((a.x+d.x)*fade+b.x,(a.y+d.y)*fade+b.y,(a.z+d.z)*fade+b.z,0);
}
__device__ float sample_height(const float4 *s,float x,float z,int c){
 float u=x*128/patch(c),v=z*128/patch(c);int ix=(int)floorf(u),iz=(int)floorf(v),base=c*16384;
 float a=fract(u),b=fract(v);
 return mixf(mixf(s[base+wrap128(iz)*128+wrap128(ix)].x,s[base+wrap128(iz)*128+wrap128(ix+1)].x,a),mixf(s[base+wrap128(iz+1)*128+wrap128(ix)].x,s[base+wrap128(iz+1)*128+wrap128(ix+1)].x,a),b);
}
__device__ float wave_height(const float4 *s,float x,float z,float distance){
 float a=sample_height(s,x,z,0),b=sample_height(s,x,z,1),d=sample_height(s,x,z,2);
 float fade=1/(1+distance*distance*.0008f);return (a+d)*fade+b;
}
// The camera state is GPU resident. JavaScript supplies raw input axes only.
__global__ void camera_step(float4 *camera,float dt,float forward,float side,float up,float lookX,float lookY,float speed,int reset){
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
 camera[2]=make_float4(yawSin*pitchCos,pitchSin,-yawCos*pitchCos,0);
 camera[3]=make_float4(yawCos,0,yawSin,0);
 camera[4]=make_float4(-yawSin*pitchSin,pitchCos,yawCos*pitchSin,0);
}
// Project the pointer to the actual FFT surface on the GPU, retaining the
// previous hit so a held drag injects force along its world-space path.
__global__ void brush_pick(const float4 *surface,const float4 *camera,float4 *brush,float pointerX,float pointerY,float aspect,int held,int moving){
 float4 old=brush[1];brush[0]=make_float4(0,0,0,0);
 if(held==0){brush[1]=make_float4(old.x,old.y,0,0);return;}
 float4 p=camera[0],r=camera[1];
 float3 f=vec(sinf(r.x)*cosf(r.y),sinf(r.y),-cosf(r.x)*cosf(r.y));
 float3 right=vec(cosf(r.x),0,sinf(r.x)),up=vec(-sinf(r.x)*sinf(r.y),cosf(r.y),cosf(r.x)*sinf(r.y));
 float3 ray=unit(plus(f,plus(scale(right,pointerX*aspect*.65f),scale(up,pointerY*.65f))));
 if(ray.y>-.06f){brush[1]=make_float4(old.x,old.y,0,0);return;}
 float t=-p.y/ray.y;
 for(int j=0;j<4;j++){float h=wave_height(surface,p.x+ray.x*t,p.z+ray.z*t,t);t=mixf(t,(h-p.y)/ray.y,.75f);}
 float x=p.x+ray.x*t,z=p.z+ray.z*t;
 float dx=old.z>.5f&&moving!=0?x-old.x:0,dz=old.z>.5f&&moving!=0?z-old.y:0;
 // Bound a single event's travel so a camera teleport cannot create an explosion.
 float len=sqrtf(dx*dx+dz*dz),limit=fminf(1,.7f/fmaxf(.00001f,len));dx*=limit;dz*=limit;
 brush[0]=make_float4(x,z,dx,dz);brush[1]=make_float4(x,z,1,0);
}
// Analytic damped spectral oscillator: h'' + omega? h = moving pressure.
// Both the height and vertical velocity are complex Fourier coefficients.
__global__ void force_modes(float4 *disturbance,const float4 *brush,const float *motion,float dt,int clear){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;if(x>=128||z>=128)return;
 int idx=z*128+x;float4 state=disturbance[idx];
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
  float phase=kx*px+kz*pz;pr+=cosf(phase)*.25f;pi-=sinf(phase)*.25f;
 }
 // Displacement impulse from the moving pressure brush; exact free evolution
 // after release provides propagating wakes instead of a drawn height mask.
 state.z+=amplitude*pr;state.w+=amplitude*pi;
 float co=cosf(omega*dt),si=sinf(omega*dt),damping=expf(-decay*dt);
 disturbance[idx]=make_float4((state.x*co+state.z/omega*si)*damping,(state.y*co+state.w/omega*si)*damping,(state.z*co-state.x*omega*si)*damping,(state.w*co-state.y*omega*si)*damping);
}
__device__ float3 refractv(float3 d,float3 n,float eta){float c=dotv(d,n);return minus(scale(d,eta),scale(n,eta*c+sqrtf(fmaxf(0,1-eta*eta*(1-c*c)))));}
__device__ float3 sunDir(){return unit(vec(-.42f,.66f,-.63f));}
__device__ float positive_power(float value,float exponent){return value>0?exp2f(log2f(value)*exponent):0;}
__device__ float3 sky(float3 d){
 float v=positive_power(clamp01(d.y),.45f);float3 col=blend(vec(.38f,.55f,.68f),vec(.045f,.16f,.36f),v);
 // Explicit fixed powers avoid the compiler's software-f64 integer pow path.
 float sun=fmaxf(0,dotv(d,sunDir())),s2=sun*sun,s4=s2*s2,s8=s4*s4,s16=s8*s8;
 col=plus(col,scale(vec(1,.83f,.55f),positive_power(sun,16000)*20+s16*s16*s16*.07f));
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
   float edge=1+.14f*sinf(atan2f(rz,rx)*5+seed*13)+.09f*sinf(atan2f(rz,rx)*9);
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
__device__ float2 landing(const float4 *s,float x,float z,float depth,float eta){
 float4 w=sample(s,x,z,0);float3 n=unit(vec(-w.y,1,-w.z));
 // Cache focusing on the mean-depth plane; the view ray still intersects the
 // undulating seabed. A planar water surface must give uniform illumination.
 float3 d=refractv(scale(sunDir(),-1),n,eta);float t=(-depth-w.x)/d.y;
 return make_float2(x+d.x*t,z+d.z*t);
}
// Forward sunlight transport. Bilinear photon splats accumulate all ray
// branches at folds; fixed-point atomics preserve energy on WebGPU.
__global__ void caustic_clear(unsigned *photons,int dispersion){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=256||z>=256)return;
 if(dispersion==0){photons[z*256+x]=0;return;}
 int id=(z*256+x)*4;photons[id]=0;photons[id+1]=0;photons[id+2]=0;photons[id+3]=0;
}
__global__ void caustic_map(const float4 *surface,unsigned *photons,float depth,int rays,int dispersion){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=rays||z>=rays)return;
 float wx=((float)x+.5f)*6/(float)rays,wz=((float)z+.5f)*6/(float)rays;
 int channels=dispersion!=0?3:1;
 for(int c=0;c<channels;c++){
  float eta=dispersion==0?.7502f:(c==0?.7524f:(c==1?.7502f:.7480f));
  float2 hit=landing(surface,wx,wz,depth,eta);
  float u=hit.x*256/6-.5f,v=hit.y*256/6-.5f;
  int ix=(int)floorf(u),iz=(int)floorf(v);float fu=fract(u),fv=fract(v);
  for(int j=0;j<2;j++)for(int i=0;i<2;i++){
   int px=(ix+i)&255,pz=(iz+j)&255;
   float weight=(i==0?1-fu:fu)*(j==0?1-fv:fv);
   int target=dispersion==0?pz*256+px:(pz*256+px)*4+c;
   atomicAdd(&photons[target],(unsigned)(weight*4096+.5f));
  }
 }
}
__global__ void caustic_resolve(const unsigned *photons,float4 *light,float normalization,int dispersion){
 // This kernel always launches a complete 32 x 32 grid of 8 x 8 groups.
 __shared__ unsigned tile[100];
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;
 int id=dispersion==0?z*256+x:(z*256+x)*4;
 float r=0;
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
  r=filtered/(16*normalization);
 }else{r=(float)photons[id]/normalization;}
 float g=dispersion!=0?(float)photons[id+1]/normalization:r,b=dispersion!=0?(float)photons[id+2]/normalization:r;
 light[z*256+x]=make_float4(r,g,b,1);
}
__device__ float3 caustic(const float4 *light,float x,float z){
 float u=fract(x/6)*256,v=fract(z/6)*256;int ix=(int)floorf(u),iz=(int)floorf(v);float a=fract(u),b=fract(v);
 float4 p=light[iz*256+ix],q=light[iz*256+(ix+1)%256],r=light[((iz+1)%256)*256+ix],t=light[((iz+1)%256)*256+(ix+1)%256];
 return vec(mixf(mixf(p.x,q.x,a),mixf(r.x,t.x,a),b),mixf(mixf(p.y,q.y,a),mixf(r.y,t.y,a),b),mixf(mixf(p.z,q.z,a),mixf(r.z,t.z,a),b));
}
__device__ float film(float a){return clamp01((a*(2.51f*a+.03f))/(a*(2.43f*a+.59f)+.14f));}
__global__ void render(const float4 *surface,const float4 *light,const float4 *camera,unsigned *image,int width,int height,float depth,float exposure,int view){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 float4 pos=camera[0],forward=camera[2],right=camera[3],up=camera[4];float sx=(2*((float)x+.5f)/(float)width-1)*(float)width/(float)height;
 float sy=1-2*((float)y+.5f)/(float)height;float3 f=vec(forward.x,forward.y,forward.z);
 float3 r=vec(right.x,right.y,right.z),u=vec(up.x,up.y,up.z);
 float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(u,sy*.65f))));float3 col=vec(0,0,0);
 if(ray.y<-.0005f){
 float t=-pos.y/ray.y;float4 w=make_float4(0,0,0,0);
 for(int i=0;i<4;i++){float h=wave_height(surface,pos.x+ray.x*t,pos.z+ray.z*t,t);t=mixf(t,(h-pos.y)/ray.y,.75f);}
 float3 p=vec(pos.x+ray.x*t,pos.y+ray.y*t,pos.z+ray.z*t);
 w=wave(surface,p.x,p.z,t);float3 n=unit(vec(-w.y,1,-w.z));float nv=fmaxf(.02f,-dotv(n,ray));
 float grazing=1-clamp01(nv),grazing2=grazing*grazing;
 float fresnel=.02037f+.97963f*grazing2*grazing2*grazing;
 float3 reflected=sky(minus(ray,scale(n,2*dotv(ray,n))));
 float3 transmitted=refractv(ray,n,.7502f);float travel=(-depth-p.y)/fminf(-.1f,transmitted.y);
 float bx=p.x+transmitted.x*travel,bz=p.z+transmitted.z*travel;
 for(int j=0;j<2;j++){travel=(bottom(bx,bz,depth)-p.y)/fminf(-.1f,transmitted.y);bx=p.x+transmitted.x*travel;bz=p.z+transmitted.z*travel;}
 travel=fmaxf(0,travel);float3 bed=seabed(bx,bz,t/(float)height),ca=caustic(light,bx,bz);
 // Light travels down through the water before returning along the view ray.
 float opticalDistance=travel+depth/.86f;
 float causticDetail=1/(1+t*t*.0008f);
 ca=blend(vec(1,1,1),ca,causticDetail*expf(-depth*.055f));
 float attenR=expf(-opticalDistance*.19f),attenG=expf(-opticalDistance*.09f),attenB=expf(-opticalDistance*.055f);
 float3 through=vec(bed.x*(.12f+.95f*ca.x)*attenR+.008f*(1-attenR),bed.y*(.12f+.95f*ca.y)*attenG+.042f*(1-attenG),bed.z*(.12f+.95f*ca.z)*attenB+.075f*(1-attenB));
 col=blend(through,reflected,fresnel);
 float3 halfv=unit(minus(sunDir(),ray));
 float pixelFootprint=t/(float)height;
 float specPower=mixf(320,8000,1/(1+pixelFootprint*pixelFootprint*800));
 float spec=positive_power(fmaxf(0,dotv(n,halfv)),specPower)*3.5f*(specPower/8000);
 col=plus(col,scale(vec(1,.89f,.68f),spec));
 float haze=1-expf(-t*.00025f);col=blend(col,vec(.38f,.55f,.68f),haze);
 if(view==1)col=scale(ca,.35f);if(view==2)col=plus(scale(n,.5f),vec(.5f,.5f,.5f));
 }else{col=sky(ray);}
 float vignette=1-.10f*(sx*sx+sy*sy);col=scale(col,exposure*vignette);
 unsigned rr=(unsigned)(positive_power(film(col.x),.454545f)*255),gg=(unsigned)(positive_power(film(col.y),.454545f)*255),bb=(unsigned)(positive_power(film(col.z),.454545f)*255);
 image[y*width+x]=rr|(gg<<8)|(bb<<16)|4278190080u;
}
