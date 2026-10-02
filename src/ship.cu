// Original ClearWater spacecraft. Geometry, navigation, ray traversal and
// materials are CUDA; the browser only dispatches kernels and forwards input.
// 126 manufactured components; the hull and canopy get denser curved meshes.
// Static geometry and a
// two-level BVH are built once, then shared by primary and sunlight rays.
__device__ int ship_tri_base(int part){return part<4?part*2048:6144+part*512;}
__device__ int ship_node_base(int part){return 255+(part<4?part*341:1024+part*85);}
struct ShipPart { float3 centre; float3 size; float3 rotation; int shape; int material; int mirror; };
__device__ float3 ship_rotate(float3 p,float3 r){
 float a=cosf(r.x),b=sinf(r.x);p=vec(p.x,p.y*a-p.z*b,p.y*b+p.z*a);
 a=cosf(r.y);b=sinf(r.y);p=vec(p.x*a+p.z*b,p.y,-p.x*b+p.z*a);
 a=cosf(r.z);b=sinf(r.z);return vec(p.x*a-p.y*b,p.x*b+p.y*a,p.z);
}
__device__ ShipPart ship_component(int id){
 ShipPart p;p.centre=vec(0,0,0);p.size=vec(1,1,1);p.rotation=vec(0,0,0);p.shape=4;p.material=0;p.mirror=1;
 if(id<32){
  if(id==0){p.shape=0;p.size=vec(1.65f,.72f,6.0f);p.centre.z=-.5f;}
  if(id==1){p.shape=0;p.size=vec(1.28f,.40f,5.35f);p.centre=vec(0,-.61f,0);p.material=2;}
  if(id==2){p.shape=5;p.size=vec(.98f,.69f,2.3f);p.centre=vec(0,.89f,-1.95f);p.material=3;}
  if(id==3){p.shape=5;p.size=vec(.032f,.71f,2.35f);p.centre=vec(0,.89f,-1.95f);p.material=2;}
  if(id==4){p.size=vec(.48f,.27f,.08f);p.centre=vec(0,1.03f,-1.10f);p.rotation.x=.20f;p.material=9;}
  if(id==5){p.size=vec(.62f,.14f,.23f);p.centre=vec(0,.80f,-2.92f);p.rotation.x=-.28f;p.material=9;}
  if(id==6){p.shape=10;p.size=vec(.996f,.712f,2.30f);p.centre=vec(0,.89f,-1.95f);p.material=2;}
  if(id==7){p.shape=0;p.size=vec(.50f,.30f,1.48f);p.centre=vec(0,-.01f,-5.6f);p.material=1;}
  if(id==8){p.size=vec(.58f,.31f,1.88f);p.centre=vec(0,.82f,2.3f);p.material=0;}
  if(id==9){p.size=vec(.46f,.065f,.7f);p.centre=vec(0,1.17f,2.05f);p.material=2;}
  if(id==10){p.shape=1;p.size=vec(.6f,.10f,1.55f);p.centre=vec(0,.94f,3.2f);p.rotation.z=1.5707963f;p.material=2;}
  if(id==11){p.shape=3;p.size=vec(.55f,.10f,.75f);p.centre=vec(0,-.14f,5.90f);p.material=4;}
  if(id==12||id==13){p.size=vec(.28f,.05f,1.28f);p.centre=vec(id==12?-.97f:.97f,.7f,2.4f);p.material=5;}
  if(id==14||id==15){p.size=vec(.12f,.16f,2.10f);p.centre=vec(id==14?-1.34f:1.34f,.16f,.55f);p.material=1;}
  if(id==16){p.shape=2;p.size=vec(.49f,.49f,1.1f);p.centre=vec(0,-.14f,5.25f);p.material=2;}
  if(id==17){p.shape=3;p.size=vec(.40f,.05f,.7f);p.centre=vec(0,-.14f,5.98f);p.material=6;}
  if(id==18){p.shape=4;p.size=vec(.30f,.27f,.015f);p.centre=vec(0,-.14f,5.85f);p.material=6;}
  if(id==19){p.size=vec(.17f,.08f,.10f);p.centre=vec(0,.28f,-5.6f);p.material=3;}
  if(id>=20&&id<24){float s=(id%2)==0?-1:1;p.size=vec(.045f,.05f,.72f);p.centre=vec(s*.89f,.78f,-1.2f-(float)(id/2-10)*1.22f);p.material=2;}
  if(id>=24&&id<28){float s=(id%2)==0?-1:1;p.size=vec(.42f,.035f,.47f);p.centre=vec(s*.88f,.70f,-.05f+(float)(id/2-12)*1.22f);p.material=id<26?0:1;}
  if(id==28||id==29){p.size=vec(.095f,.045f,1.5f);p.centre=vec(id==28?-.70f:.70f,.22f,-3.7f);p.rotation.z=id==28?-.1f:.1f;p.material=2;}
  if(id==30){p.size=vec(.38f,.035f,.52f);p.centre=vec(0,.42f,-4.25f);p.rotation.x=-.20f;p.material=0;}
  if(id==31){p.size=vec(.10f,.055f,.37f);p.centre=vec(0,.49f,-4.26f);p.rotation.x=-.20f;p.material=1;}
 }else{
  int side=id<80?-1:1,k=(id-32)%48;float s=1;p.mirror=side;
  if(k==0){p.shape=1;p.size=vec(s*5.7f,.26f,3.15f);p.centre=vec(s*1.0f,-.13f,.7f);}
  // Retired winglet slots keep component IDs and the binary BVH layout stable.
  if(k==1){p.size=vec(0,0,0);p.material=-1;}
  if(k==2){p.shape=2;p.size=vec(1.0f,1.0f,2.65f);p.centre=vec(s*4.65f,.22f,2.25f);p.material=2;}
  if(k==3){p.shape=2;p.size=vec(1.13f,1.13f,1.40f);p.centre=vec(s*4.65f,.22f,.72f);p.material=0;}
  if(k==4){p.shape=3;p.size=vec(.87f,.17f,.75f);p.centre=vec(s*4.65f,.22f,-.72f);p.material=2;}
  if(k==5){p.shape=3;p.size=vec(1.02f,.07f,1);p.centre=vec(s*4.65f,.22f,-.83f);p.material=4;}
  if(k==6){p.shape=3;p.size=vec(.94f,.12f,1);p.centre=vec(s*4.65f,.22f,5.00f);p.material=4;}
  if(k==7){p.shape=2;p.size=vec(.76f,.76f,.52f);p.centre=vec(s*4.65f,.22f,4.63f);p.material=2;}
  if(k==8){p.size=vec(.64f,.62f,.018f);p.centre=vec(s*4.65f,.22f,4.57f);p.material=6;}
  if(k==9){p.size=vec(.50f,.13f,1.15f);p.centre=vec(s*4.65f,1.34f,.82f);p.material=0;}
  if(k==10){p.size=vec(.16f,.21f,1.2f);p.centre=vec(s*5.65f,.30f,.83f);p.material=1;}
  if(k==11){p.shape=1;p.size=vec(2.25f,.12f,1.20f);p.centre=vec(s*5.25f,.9f,2.35f);p.rotation.z=s*1.05f;p.material=0;}
  if(k==12){p.size=vec(.16f,.13f,1.08f);p.centre=vec(s*6.62f,.01f,2.15f);p.material=2;}
  if(k==13){p.size=vec(.16f,.065f,.15f);p.centre=vec(s*6.68f,.15f,1.35f);p.material=side<0?7:8;}
  if(k==14){p.shape=2;p.size=vec(.30f,.30f,.75f);p.centre=vec(s*2.12f,-.37f,2.15f);p.rotation.y=1.5707963f;p.material=4;}
  if(k==15){p.shape=2;p.size=vec(.065f,.065f,1.25f);p.centre=vec(s*2.8f,-.5f,2.20f);p.rotation.y=s*1.08f;p.material=4;}
  if(k>=16&&k<24){float a=(float)(k-16)*.78539816f;p.size=vec(.25f,.085f,.61f);p.centre=vec(s*4.65f+sinf(a)*.89f,.22f+cosf(a)*.89f,4.30f);p.rotation.z=-a;p.rotation.x=.09f;p.material=4;}
  if(k>=24&&k<32){float a=(float)(k-24)*.78539816f;p.size=vec(.07f,.38f,.045f);p.centre=vec(s*4.65f+sinf(a)*.46f,.22f+cosf(a)*.46f,-.77f);p.rotation.z=-a;p.rotation.y=.4f;p.material=4;}
  if(k>=32&&k<40){p.shape=3;p.size=vec(.94f,.032f,1);p.centre=vec(s*4.65f,.22f,2.05f+(float)(k-32)*.20f);p.material=k%3==0?1:4;}
  if(k>=40&&k<44){p.size=vec(.38f,.046f,.32f);p.centre=vec(s*(2.35f+(float)(k-40)*.64f),.14f,1.45f+(float)(k-40)*.12f);p.material=k==40?1:0;}
  if(k==44){p.size=vec(.15f,.25f,1.10f);p.centre=vec(s*1.86f,-.68f,1.27f);p.material=2;}
  if(k==45){p.shape=2;p.size=vec(.035f,.035f,.54f);p.centre=vec(s*6.62f,.15f,1.35f);p.material=4;}
  if(k==46){p.shape=2;p.size=vec(.062f,.062f,1.80f);p.centre=vec(s*5.36f,.85f,2.20f);p.material=1;}
  if(k==47){p.shape=3;p.size=vec(1.00f,.06f,1);p.centre=vec(s*4.65f,.22f,1.72f);p.material=1;}
 }
 return p;
}
__device__ float2 ship_octagon(float u){
 float a=fract(u)*8, t=fract(a);int k=(int)floorf(a);float2 p=make_float2(1,.55f),q=make_float2(.55f,1);
 if(k==1){p=q;q=make_float2(-.55f,1);}if(k==2){p=make_float2(-.55f,1);q=make_float2(-1,.55f);}if(k==3){p=make_float2(-1,.55f);q=make_float2(-1,-.55f);}
 if(k==4){p=make_float2(-1,-.55f);q=make_float2(-.55f,-1);}if(k==5){p=make_float2(-.55f,-1);q=make_float2(.55f,-1);}if(k==6){p=make_float2(.55f,-1);q=make_float2(1,-.55f);}if(k==7){p=make_float2(1,-.55f);q=make_float2(1,.55f);}
 return make_float2(mixf(p.x,q.x,t),mixf(p.y,q.y,t));
}
__device__ float3 ship_vertex(ShipPart p,float u,float v){
 float a=u*6.2831853f,cap=fminf(1,fmaxf(0,fminf(v,1-v)*32));float2 q=ship_octagon(u);float3 point=vec(q.x*cap,q.y*cap,2*v-1);
 if(p.shape==0){
  float ca=cosf(a),sa=sinf(a),qx=(ca<0?-1:1)*positive_power(fabsf(ca),.68f),qy=(sa<0?-1:1)*positive_power(fabsf(sa),.78f);
  float w=(.045f+.955f*eased(0,.43f,v))*(1-.28f*eased(.65f,1,v)),h=.20f+.80f*eased(.02f,.45f,v);float end=fminf(1,fmaxf(0,fminf(v,1-v)*64));
  point=vec(qx*w*end,qy*h*end+.06f*sinf(v*3.14159265f),2*v-1);
 }
 if(p.shape==1){float w=mixf(1,.36f,v);point=vec(v,q.y*mixf(1,.38f,v)*cap,q.x*w*cap+.48f*v);}
 if(p.shape==2){float radius=.82f+.18f*sinf(v*3.14159265f);point=vec(cosf(a)*radius,sinf(a)*radius,2*v-1);}
 if(p.shape==3){float t=v*6.2831853f;point=vec(cosf(a)*(p.size.x+p.size.y*cosf(t)),sinf(a)*(p.size.x+p.size.y*cosf(t)),sinf(t)*p.size.y*p.size.z);point=plus(p.centre,ship_rotate(point,p.rotation));point.x*=(float)p.mirror;return point;}
 if(p.shape==5||p.shape==10){float t=p.shape==10?.62f+(v-.5f)*.016f:v;float profile=positive_power(fmaxf(0,sinf(t*3.14159265f)),.62f);point=vec(cosf(a)*profile,sinf(a)*profile,2*t-1);}
 point=vec(point.x*p.size.x,point.y*p.size.y,point.z*p.size.z);point=plus(p.centre,ship_rotate(point,p.rotation));point.x*=(float)p.mirror;return point;
}
__global__ void ship_mesh(float4 *mesh,float4 *bounds){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=71680)return;int part=i<8192?i/2048:4+(i-8192)/512,local=i-ship_tri_base(part),leaf=local/8,triangle=i%8;ShipPart p=ship_component(part);
 int axis=part<4?16:8,nu=part<4?32:((p.shape==2||p.shape==3||p.shape==10)?32:16),nv=(part<4?1024:256)/nu,du=nu/axis,dv=nv/axis,quad=triangle/2;
 int ix=(leaf%axis)*du+quad%du,iy=(leaf/axis)*dv+quad/du;
 float u=(float)ix/(float)nu,v=(float)iy/(float)nv;int flip=triangle%2;
 for(int j=0;j<3;j++){
  int ox=flip==0?(j==1?1:0):(j==0?0:1),oy=flip==0?(j==2?1:0):(j==1?0:1);
  float uu=u+(float)ox/(float)nu,vv=v+(float)oy/(float)nv;float3 point=ship_vertex(p,uu,vv);
  // Keep manufactured octagonal creases sharp: sample the normal just inside
  // this face rather than averaging across adjoining faces at a mesh seam.
  float un=mixf(uu,u+.5f/(float)nu,.002f),vn=mixf(vv,v+.5f/(float)nv,.002f);
  float3 a=minus(ship_vertex(p,un+.0001f,vn),ship_vertex(p,un-.0001f,vn)),b=minus(ship_vertex(p,un,vn+.0001f),ship_vertex(p,un,vn-.0001f));
  float3 normal=unit(crossv(a,b));mesh[i*6+j]=make_float4(point.x,point.y,point.z,uu);mesh[i*6+3+j]=make_float4(normal.x,normal.y,normal.z,vv);
 }
 if(local==0){int node=ship_node_base(part);bounds[node*2].w=(float)p.material;}
}
__device__ float3 ship_min(float3 a,float3 b){return vec(fminf(a.x,b.x),fminf(a.y,b.y),fminf(a.z,b.z));}
__device__ float3 ship_max(float3 a,float3 b){return vec(fmaxf(a.x,b.x),fmaxf(a.y,b.y),fmaxf(a.z,b.z));}
__device__ float3 ship_xyz(float4 a){return vec(a.x,a.y,a.z);}
__global__ void ship_bounds(const float4 *mesh,float4 *bounds,int level){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(level==0){
  if(i>=8960)return;int part=i<1024?i/256:4+(i-1024)/64,leaf=i-(part<4?part*256:768+part*64),first=part<4?85:21;float3 lo=vec(1000,1000,1000),hi=vec(-1000,-1000,-1000);
  for(int t=0;t<8;t++)for(int j=0;j<3;j++){float3 p=ship_xyz(mesh[(ship_tri_base(part)+leaf*8+t)*6+j]);lo=ship_min(lo,p);hi=ship_max(hi,p);}
  int b=(ship_node_base(part)+first+leaf)*2;bounds[b]=make_float4(lo.x-.0001f,lo.y-.0001f,lo.z-.0001f,0);bounds[b+1]=make_float4(hi.x+.0001f,hi.y+.0001f,hi.z+.0001f,0);
 }else if(level<5){
  int stride=level==1?64:(level==2?16:(level==3?4:1));if(i>=128*stride)return;int part=i/stride,j=i%stride,count=part<4?stride:(stride/4);if(count==0||j>=count)return;
  int start=(count-1)/3,node=start+j,b=(ship_node_base(part)+node)*2;float3 lo=vec(1000,1000,1000),hi=vec(-1000,-1000,-1000);
  for(int k=0;k<4;k++){int c=(ship_node_base(part)+node*4+1+k)*2;lo=ship_min(lo,ship_xyz(bounds[c]));hi=ship_max(hi,ship_xyz(bounds[c+1]));}
  float material=bounds[b].w;bounds[b]=make_float4(lo.x,lo.y,lo.z,material);bounds[b+1]=make_float4(hi.x,hi.y,hi.z,0);
 }else{
  int count=128>>(level-5),start=count-1;if(i>=count)return;int b=(start+i)*2;float3 lo,hi;
  if(level==5){int c=ship_node_base(i)*2;lo=ship_xyz(bounds[c]);hi=ship_xyz(bounds[c+1]);}
  else{int c=(2*(start+i)+1)*2;lo=ship_min(ship_xyz(bounds[c]),ship_xyz(bounds[c+2]));hi=ship_max(ship_xyz(bounds[c+1]),ship_xyz(bounds[c+3]));}
  bounds[b]=make_float4(lo.x,lo.y,lo.z,0);bounds[b+1]=make_float4(hi.x,hi.y,hi.z,0);
 }
}
__device__ float3 ship_axis(const float4 *ship,float3 p){return plus(scale(ship_xyz(ship[4]),p.x),plus(scale(ship_xyz(ship[5]),p.y),scale(ship_xyz(ship[6]),p.z)));}
__device__ float3 ship_inverse(const float4 *ship,float3 p){return vec(dotv(p,ship_xyz(ship[4])),dotv(p,ship_xyz(ship[5])),dotv(p,ship_xyz(ship[6])));}
__global__ void ship_step(float4 *camera,float2 *navigationState,float4 *ship,float deltaTime,float forward,float turn,float rise,float lookX,float lookY,float speed,int boost,int action,int inspect,float depth,float aspect){
 float dt=fminf(.1f,fmaxf(0,deltaTime));
 if(ship[0].w==0||action==1){
  if(camera[8].z==0){float navLength=navigationState[0].x*navigationState[0].x+navigationState[1].x*navigationState[1].x+navigationState[2].x*navigationState[2].x;camera_advance(camera,navigationState,0,0,0,0,0,0,1,0,navLength>.5f?0:1,depth);}
  ship[0]=make_float4(camera[0].x,22,camera[0].z,1);ship[1]=make_float4(0,0,0,0);ship[2]=make_float4(.34f,.30f,22,0);ship[3]=make_float4(0,0,0,speed);
 }
 if(action==3){ship[0].y=earth_radius()*.70f;ship[1].y=-.40f;ship[2].y=.72f;ship[2].x=.32f;}
 float4 pose=ship[1];ship[2].w=(float)inspect;
 if(inspect!=0){ship[2].x+=lookX;ship[2].y=fminf(1.30f,fmaxf(-.15f,ship[2].y-lookY));}
 else{pose.x+=lookX+turn*dt*.85f;pose.y=fminf(1.45f,fmaxf(-1.45f,pose.y+lookY));if(forward>0||fabsf(turn)>.01f||fabsf(lookX)>.001f)ship[2].x*=expf(-dt*3);ship[2].y=mixf(ship[2].y,.24f,1-expf(-dt*3));}
 pose.x-=floorf((pose.x+3.14159265f)/6.2831853f)*6.2831853f;
 float targetBank=fminf(.6f,fmaxf(-.6f,-turn*.48f-lookX/fmaxf(.008f,dt)*.12f));pose.z=mixf(pose.z,targetBank,1-expf(-dt*4));
 float maxSpeed=fminf(30000000,speed*fmaxf(1,ship[0].y*.006f)*(boost!=0?6:1)),target=fmaxf(0,forward)*maxSpeed;
 pose.w=mixf(pose.w,target,1-expf(-dt*(forward<0?7:2.4f)));if(pose.w<.002f)pose.w=0;
 camera[0]=make_float4(ship[0].x,ship[0].y,ship[0].z,0);camera[1]=make_float4(pose.x,pose.y,0,0);
 float actual=fmaxf(pose.w,fabsf(rise)*maxSpeed*.35f),factor=fmaxf(1,camera[0].y*.06f);
 camera_advance(camera,navigationState,dt,actual>0?pose.w/actual:0,0,actual>0?rise*maxSpeed*.35f/actual:0,0,0,actual/factor,0,0,depth);
 float floorHeight=camera[21].w!=0?terrain_height(camera,ship_xyz(camera[6])):0;camera[0].y=fmaxf(camera[0].y,fmaxf(6,floorHeight+6));
 ship[0]=make_float4(camera[0].x,camera[0].y,camera[0].z,1);ship[1]=pose;ship[3]=make_float4(ship[3].x+dt,mixf(ship[3].y,forward>0?(boost!=0?1:.55f):.12f,1-expf(-dt*4)),camera[0].y-floorHeight,speed);
 float3 angles=vec(pose.y,-pose.x,pose.z),right=ship_rotate(vec(1,0,0),angles),up=ship_rotate(vec(0,1,0),angles),back=ship_rotate(vec(0,0,1),angles);
 // Positive mouse yaw points the nose to the right in the globe's tangent frame.
 right=ship_rotate(vec(1,0,0),vec(0,-pose.x,0));back=vec(-sinf(pose.x)*cosf(pose.y),-sinf(pose.y),cosf(pose.x)*cosf(pose.y));up=unit(crossv(back,right));
 float3 bankRight=plus(scale(right,cosf(pose.z)),scale(up,sinf(pose.z))),bankUp=minus(scale(up,cosf(pose.z)),scale(right,sinf(pose.z)));
 ship[4]=make_float4(bankRight.x,bankRight.y,bankRight.z,0);ship[5]=make_float4(bankUp.x,bankUp.y,bankUp.z,0);ship[6]=make_float4(back.x,back.y,back.z,0);
 float orbit=ship[2].x,elevation=ship[2].y,distance=ship[2].z*fmaxf(1,.88f/aspect);float3 offset=plus(scale(right,sinf(orbit)*cosf(elevation)*distance),plus(scale(up,sinf(elevation)*distance),scale(back,cosf(orbit)*cosf(elevation)*distance)));
 // Keep the model-to-eye transform in small local metres. Subtracting two
 // float32 orbital altitudes would quantize the canopy and tracking camera.
 offset.y=fmaxf(offset.y,floorHeight+2-ship[0].y);
 float3 eye=plus(ship_xyz(ship[0]),offset),f=unit(minus(scale(up,.50f),offset)),r=unit(crossv(f,up)),u=crossv(r,f);
 camera[0]=make_float4(eye.x,eye.y,eye.z,0);camera[2]=make_float4(f.x,f.y,f.z,depth/.86f);camera[3]=make_float4(r.x,r.y,r.z,expf(-depth*.055f));camera[4]=make_float4(u.x,u.y,u.z,0);camera[8].x=eye.y;camera[8].y=pose.w;
 float3 localEye=ship_inverse(ship,offset);ship[7]=make_float4(localEye.x,localEye.y,localEye.z,0);
 camera[19].x=ship[0].y;camera[19].y=pose.w;
}
__device__ int ship_next(int node,int arity){while(node>0&&node%arity==0)node=(node-1)/arity;return node==0?-1:node+1;}
__device__ int ship_box_hit(const float4 *bounds,int node,float3 origin,float3 inverse,float limit){
 float3 a=minus(ship_xyz(bounds[node*2]),origin),b=minus(ship_xyz(bounds[node*2+1]),origin);a=vec(a.x*inverse.x,a.y*inverse.y,a.z*inverse.z);b=vec(b.x*inverse.x,b.y*inverse.y,b.z*inverse.z);float3 lo=ship_min(a,b),hi=ship_max(a,b);
 return fmaxf(.001f,fmaxf(lo.x,fmaxf(lo.y,lo.z)))<=fminf(limit,fminf(hi.x,fminf(hi.y,hi.z)))?1:0;
}
struct ShipHit {float t;float u;float v;int triangle;int part;};
__device__ ShipHit ship_trace(const float4 *mesh,const float4 *bounds,float3 origin,float3 ray,int shadow){
 ShipHit hit;hit.t=100000;hit.triangle=-1;hit.part=-1;hit.u=0;hit.v=0;
 float3 inv=vec(1/(fabsf(ray.x)<.0000001f?.0000001f:ray.x),1/(fabsf(ray.y)<.0000001f?.0000001f:ray.y),1/(fabsf(ray.z)<.0000001f?.0000001f:ray.z));
 int top=0;
 while(top>=0){
  if(ship_box_hit(bounds,top,origin,inv,hit.t)==0){top=ship_next(top,2);continue;}
  if(top<127){top=top*2+1;continue;}
  int part=top-127,node=0,base=ship_node_base(part),firstLeaf=part<4?85:21;
  if(bounds[base*2].w<0||(shadow==2&&part==2)){top=ship_next(top,2);continue;}
  while(node>=0){
   if(ship_box_hit(bounds,base+node,origin,inv,hit.t)==0){node=ship_next(node,4);continue;}
   if(node<firstLeaf){node=node*4+1;continue;}
   int first=ship_tri_base(part)+(node-firstLeaf)*8;
   for(int j=0;j<8;j++){
    int tri=first+j;float3 a=ship_xyz(mesh[tri*6]),e1=minus(ship_xyz(mesh[tri*6+1]),a),e2=minus(ship_xyz(mesh[tri*6+2]),a),p=crossv(ray,e2);float det=dotv(e1,p);
    if(fabsf(det)<.00000001f)continue;float invDet=1/det;float3 d=minus(origin,a);float u=dotv(d,p)*invDet;if(u<0||u>1)continue;
    float3 q=crossv(d,e1);float v=dotv(ray,q)*invDet;if(v<0||u+v>1)continue;float t=dotv(e2,q)*invDet;
    if(t>.001f&&t<hit.t){hit.t=t;hit.u=u;hit.v=v;hit.triangle=tri;hit.part=part;if(shadow==1)return hit;}
   }
   node=ship_next(node,4);
  }
  top=ship_next(top,2);
 }
 return hit;
}
__device__ float ship_line(float x,float width,float footprint){return (1-eased(fmaxf(0,width-footprint*.5f),width+footprint*.5f,fabsf(x)))*fminf(1,2*width/fmaxf(.0001f,footprint));}
__global__ void ship_probe(const float4 *mesh,const float4 *bounds,const float4 *rays,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;ShipHit h=ship_trace(mesh,bounds,ship_xyz(rays[i*2]),unit(ship_xyz(rays[i*2+1])),0);output[i]=make_float4(h.t,(float)h.part,(float)h.triangle,h.u+h.v);
}
__device__ float ship_serial(float x,float y,float footprint){
 int digit=(int)floorf(x/.43f);float u=fract(x/.43f)*.43f,v=y;
 if(x<0||x>.86f||v<0||v>.62f)return 0;
 float mark=ship_line(u-.33f,.024f,footprint)*eased(0,.04f,v)*(1-eased(.58f,.62f,v));
 if(digit==0)mark=fmaxf(mark,fmaxf(ship_line(u-.08f,.024f,footprint),fmaxf(ship_line(v-.04f,.024f,footprint),ship_line(v-.58f,.024f,footprint)))*eased(.035f,.065f,u)*(1-eased(.345f,.37f,u)));
 return clamp01(mark);
}
__device__ float4 ship_radiance(const float4 *mesh,const float4 *bounds,const float4 *camera,const float4 *ship,ShipHit hit,float3 ray,float footprint,float knownShadow){
 int tri=hit.triangle,mat=(int)bounds[ship_node_base(hit.part)*2].w;float w=1-hit.u-hit.v;
 float3 n=unit(plus(scale(ship_xyz(mesh[tri*6+3]),w),plus(scale(ship_xyz(mesh[tri*6+4]),hit.u),scale(ship_xyz(mesh[tri*6+5]),hit.v))));if(dotv(n,ray)>0)n=scale(n,-1);
 float3 p=plus(ship_xyz(ship[7]),scale(ray,hit.t));float u=mesh[tri*6].w*w+mesh[tri*6+1].w*hit.u+mesh[tri*6+2].w*hit.v,v=mesh[tri*6+3].w*w+mesh[tri*6+4].w*hit.u+mesh[tri*6+5].w*hit.v;
 float3 color=vec(.55f,.61f,.60f);float rough=.29f,metal=.30f,emission=0;
 if(mat==1){color=vec(.65f,.145f,.035f);rough=.29f;metal=.25f;}
 if(mat==2){color=vec(.025f,.035f,.042f);rough=.44f;metal=.7f;}
 if(mat==3){color=vec(.015f,.055f,.075f);rough=.075f;metal=.22f;}
 if(mat==4){color=vec(.18f,.22f,.25f);rough=.28f;metal=.92f;}
 if(mat==5){color=vec(.016f,.021f,.025f);rough=.53f;metal=.72f;}
 if(mat==6){float cx=hit.part<32?0:(hit.part<80?-4.65f:4.65f),cy=hit.part<32?-.14f:.22f;float rr=(p.x-cx)*(p.x-cx)+(p.y-cy)*(p.y-cy);float core=expf(-rr*12);color=blend(vec(.012f,.08f,.7f),vec(.4f,.8f,1),core);emission=(.6f+core*7)*(1+ship[3].y*2);}
 if(mat==7){color=vec(1,.045f,.01f);emission=3;}
 if(mat==8){color=vec(.08f,1,.50f);emission=3;}
 if(mat==9){color=vec(.014f,.025f,.028f);rough=.65f;metal=.1f;if(hit.part==5&&n.y>.2f){color=vec(.015f,.17f,.22f);emission=1;}}
 // Metre-scale panel seams, recessed fasteners, warning chevrons and paint
 // wear are filtered by the projected pixel footprint, not repeating noise.
 float fp=fmaxf(.002f,footprint),seam=0,rivet=0;
 if(mat==0||mat==1){
  int component=hit.part<32?-1:(hit.part-32)%48;float ax=fabsf(p.x);
  if(hit.part==0||hit.part==1){
   float band=fminf(fabsf(p.z+4.6f-ax*.12f),fminf(fabsf(p.z+2.65f+ax*.14f),fminf(fabsf(p.z-.65f-ax*.08f),fabsf(p.z-3.75f+ax*.20f))));
   float shoulder=ax-(.40f+.42f*eased(-4.5f,-.5f,p.z));seam=fmaxf(ship_line(band,.008f,fp),ship_line(shoulder,.006f,fp));
   float rivetDistance=sqrtf((band-.10f)*(band-.10f)+(fabsf(shoulder)-.075f)*(fabsf(shoulder)-.075f));rivet=1-eased(.013f,.013f+fp,rivetDistance);
  }
  if(component==0){float band=fminf(fabsf(ax-3.10f),fabsf(ax-5.90f)),chord=fminf(fabsf(p.z-(.50f+ax*.23f)),fabsf(p.z-(2.05f+ax*.22f)));seam=fmaxf(ship_line(band,.008f,fp),ship_line(chord,.006f,fp));}
  if(component==3){seam=fmaxf(ship_line(fract(v*3)-.5f,.002f,fp*.3f),ship_line(fract(u*4)-.5f,.002f,fp*.6f));}
  color=scale(color,1-.45f*seam-.28f*rivet);
  float grain=globe_noise(scale(p,12)),brushed=globe_noise(vec(p.x*4,p.y*6,p.z*.7f));rough+=grain*.045f; color=scale(color,.96f+.04f*brushed);
  // Fine finish is visible only when the camera resolves it. Wear belongs at
  // exposed edges and seams; it is not a screen-wide scratched grid.
  float detail=1-eased(.008f,.028f,fp),micro=cell(floorf(p.x*180),floorf(p.z*180));color=scale(color,1+detail*(micro-.5f)*.022f);
  if(component==3){float soot=eased(.70f,1,v)*(.045f+.07f*globe_noise(scale(p,6)));color=blend(color,vec(.09f,.095f,.085f),soot);}
  if(component==9){float dx=fabsf(ax-4.65f)-.37f,dz=fabsf(p.z-.82f)-.96f;float fastener=1-eased(.021f,.021f+fp,sqrtf(dx*dx+dz*dz));color=blend(color,vec(.04f,.055f,.061f),fastener*.7f);}
  if(component==0&&ax>5.7f){float stripe=ship_line(p.z-2.1f,.12f,fp);color=blend(color,vec(.60f,.105f,.025f),stripe);}
 }
 if(mat==4){float heat=eased(2.8f,4.9f,p.z);color=blend(color,vec(.17f,.13f,.19f),heat*.32f);rough+=globe_noise(vec(p.x*25,p.y*25,p.z*3))*.08f;}
 if(mat==0&&hit.part>=32&&(hit.part-32)%48==0&&n.y>.4f){float serial=ship_serial(fabsf(p.x)-2.65f,p.z-.1f,fp);color=blend(color,vec(.02f,.035f,.04f),serial);}
 if(mat==1&&hit.part>=32&&(hit.part-32)%48==40){float warning=ship_line(fract(p.x*7+p.z*5)-.5f,.16f,fp*8);color=blend(color,vec(.025f,.027f,.023f),warning*.85f);}
 if(mat==5){float rib=ship_line(fract(p.z*14)-.5f,.13f,fp*14);color=scale(color,.5f+rib*1.5f);}
 if(mat==2&&hit.part>31){float band=ship_line(fract(p.z*6)-.5f,.08f,fp*6);color=blend(color,vec(.12f,.14f,.16f),band*.5f);}
 float3 light=unit(ship_inverse(ship,ship_xyz(camera[9]))),view=scale(ray,-1),halfv=unit(plus(light,view));float nl=fmaxf(0,dotv(n,light)),nv=fmaxf(.02f,dotv(n,view)),nh=fmaxf(0,dotv(n,halfv)),vh=fmaxf(0,dotv(view,halfv));
 float shadow=knownShadow;if(shadow<0){shadow=1;if(nl>.001f){ShipHit occ=ship_trace(mesh,bounds,plus(p,scale(n,.006f)),light,1);if(occ.triangle>=0)shadow=.08f;}}
 float3 worldN=ship_axis(ship,n),reflected=ship_axis(ship,minus(ray,scale(n,2*dotv(ray,n))));float day=eased(-.10f,.22f,camera[9].y),atmosphere=1-eased(15000,130000,camera[0].y);
 float3 ambient=blend(vec(.016f,.025f,.035f),vec(.12f,.19f,.27f),clamp01(worldN.y*.5f+.5f));ambient=scale(ambient,(.05f+day)*atmosphere+.035f);
 float3 env=blend(vec(.018f,.035f,.045f),vec(.13f,.26f,.43f),eased(-.2f,.8f,reflected.y));env=scale(env,(.02f+day)*atmosphere+.035f);
 if(mat==3&&camera[0].y<400)env=scale(weather_sky_sample(reflected,camera),1.8f);
 float a=rough*rough,a2=a*a,den=nh*nh*(a2-1)+1,D=a2/(3.14159265f*den*den+.00001f),k=(rough+1)*(rough+1)/8,G=nv/(nv*(1-k)+k)*nl/(nl*(1-k)+k);
 float f=1-vh,f2=f*f,fres=f2*f2*f;float3 f0=mat==3?vec(.16f,.22f,.27f):blend(vec(.04f,.04f,.04f),color,metal),F=blend(f0,vec(1,1,1),fres);
 float3 direct=plus(scale(color,(1-metal)*nl*.90f),scale(F,D*G/fmaxf(.04f,4*nv)*1.5f));direct=scale(direct,shadow*(.03f+1.7f*day));
 float3 indirect=vec(color.x*ambient.x,color.y*ambient.y,color.z*ambient.z);if(mat==3){float frame=ship_line(u-.125f,.002f,.003f)+ship_line(u-.375f,.002f,.003f);indirect=plus(indirect,scale(vec(.03f,.09f,.13f),1-clamp01(frame)));}float3 reflection=vec(env.x*F.x,env.y*F.y,env.z*F.z);reflection=scale(reflection,1.15f-rough*.6f);
 float ao=1-.24f*seam-.32f*rivet;if(hit.part>=32&&fabsf(p.x)>3.65f&&fabsf(p.x)<5.7f&&p.y<1.1f)ao*=.82f;
 float3 result=plus(scale(plus(direct,plus(indirect,reflection)),ao),scale(color,emission));
 if(mat==3){
  ShipHit interior=ship_trace(mesh,bounds,plus(p,scale(ray,.009f)),ray,2);
  float3 inside=vec(.008f,.015f,.019f);
  if(interior.part==5){float3 ip=plus(p,scale(ray,interior.t+.009f));float display=ship_line(fract(ip.x*10)-.5f,.07f,fp*10)*ship_line(fract(ip.z*12)-.5f,.11f,fp*12);inside=blend(vec(.012f,.028f,.034f),vec(.025f,.26f,.32f),display);}
  if(interior.part==4)inside=vec(.024f,.034f,.031f);
  float g=1-nv,g2=g*g,glassF=.12f+.88f*g2*g2*g;result=plus(scale(plus(direct,scale(env,glassF)),.8f),scale(inside,1-glassF));
 }
 return make_float4(result.x,result.y,result.z,shadow);
}
__device__ float3 ship_decode(unsigned color){return vec((float)(color&255u)/255,(float)((color>>8)&255u)/255,(float)((color>>16)&255u)/255);}
__device__ float ship_unfilm(float c){float y=positive_power(c,2.2f),a=2.51f-2.43f*y,b=.03f-.59f*y;return (-b+sqrtf(b*b+.56f*a*y))/(2*a);}
__device__ float3 ship_plume(const float4 *ship,float3 origin,float3 ray,float limit){
 float3 light=vec(0,0,0);float thrust=fmaxf(.12f,ship[3].y),length=1.5f+thrust*5.5f;
 for(int engine=0;engine<2;engine++){
  float cx=engine==0?-4.65f:4.65f,ox=origin.x-cx,oy=origin.y-.22f;
  float a=ray.x*ray.x+ray.y*ray.y,b=ox*ray.x+oy*ray.y,c=ox*ox+oy*oy-.75f*.75f,disc=b*b-a*c;
  if(disc<0||a<.00000001f)continue;float root=sqrtf(disc),lo=fmaxf(0,(-b-root)/a),hi=fminf(limit,(-b+root)/a);
  if(fabsf(ray.z)>.00001f){float z0=(5.03f-origin.z)/ray.z,z1=(5.03f+length-origin.z)/ray.z;lo=fmaxf(lo,fminf(z0,z1));hi=fminf(hi,fmaxf(z0,z1));}
  else if(origin.z<5.03f||origin.z>5.03f+length)continue;
  if(hi<=lo)continue;float step=(hi-lo)/8;
  for(int j=0;j<8;j++){
   float t=lo+((float)j+.5f)*step,z=origin.z+ray.z*t-5.03f,q=z/length,radius=.50f*(1-.84f*q),x=ox+ray.x*t,y=oy+ray.y*t,r=(x*x+y*y)/(radius*radius);
   float envelope=expf(-r*3.5f)*eased(1,0,q),diamonds=.72f+.28f*cosf(z*9-thrust*1.5f);
   float3 hue=blend(vec(.08f,.25f,1.8f),vec(.45f,1.4f,2.8f),expf(-r*8));light=plus(light,scale(hue,envelope*diamonds*step*(1+thrust)*1.2f));
  }
 }
 return light;
}
__global__ void ship_render(const float4 *mesh,const float4 *bounds,const float4 *camera,const float4 *ship,unsigned *image,int width,int height,int samples,float exposure){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 float3 origin=ship_xyz(ship[7]),f=ship_inverse(ship,ship_xyz(camera[2])),r=ship_inverse(ship,ship_xyz(camera[3])),up=ship_inverse(ship,ship_xyz(camera[4]));
 // Reject the combined hull / exhaust bounds once for all spatial samples.
 // The padding covers the four rays, including the narrow portrait viewport.
 float sx0=(2*((float)x+.5f)-(float)width)/(float)height,sy0=1-2*((float)y+.5f)/(float)height;
 float3 centreRay=unit(plus(f,plus(scale(r,sx0*.65f),scale(up,sy0*.65f))));float3 inverse=vec(1/(fabsf(centreRay.x)<.000001f?.000001f:centreRay.x),1/(fabsf(centreRay.y)<.000001f?.000001f:centreRay.y),1/(fabsf(centreRay.z)<.000001f?.000001f:centreRay.z));
 float padding=.01f+sqrtf(dotv(origin,origin))*2/(float)height;float3 low=minus(ship_xyz(bounds[0]),vec(padding,padding,padding)),high=plus(ship_xyz(bounds[1]),vec(padding,padding,padding));high.z=fmaxf(high.z,12.2f);
 float3 a=minus(low,origin),b=minus(high,origin);a=vec(a.x*inverse.x,a.y*inverse.y,a.z*inverse.z);b=vec(b.x*inverse.x,b.y*inverse.y,b.z*inverse.z);float3 near=ship_min(a,b),far=ship_max(a,b);
 if(fmaxf(0,fmaxf(near.x,fmaxf(near.y,near.z)))>fminf(far.x,fminf(far.y,far.z)))return;
 float3 sum=vec(0,0,0);unsigned background=image[y*width+x];int covered=0,previousPart=-1;float previousShadow=-1;float3 bg=ship_decode(background);
 for(int sample=0;sample<4;sample++){
  if(sample>=samples)break;float ox=samples==1?.5f:((float)(sample%2)+.5f)*.5f,oy=samples==1?.5f:((float)(sample/2)+.5f)*.5f;
  float sx=(2*((float)x+ox)-(float)width)/(float)height,sy=1-2*((float)y+oy)/(float)height;float3 ray=unit(plus(f,plus(scale(r,sx*.65f),scale(up,sy*.65f))));ShipHit hit=ship_trace(mesh,bounds,origin,ray,0);
  float3 plume=ship_plume(ship,origin,ray,hit.t);
  if(hit.triangle<0){if(dotv(plume,plume)<.000001f){sum=plus(sum,bg);continue;}covered++;float3 linear=vec(ship_unfilm(bg.x),ship_unfilm(bg.y),ship_unfilm(bg.z));sum=plus(sum,ship_decode(pack_color(plus(linear,plume))));continue;}covered++;
  float4 lighting=ship_radiance(mesh,bounds,camera,ship,hit,ray,hit.t*1.3f/(float)height,hit.part==previousPart?previousShadow:-1);previousPart=hit.part;previousShadow=lighting.w;
  sum=plus(sum,ship_decode(pack_color(plus(scale(ship_xyz(lighting),exposure),plume))));
 }
 if(covered==0)return;float3 result=scale(sum,1/(float)samples);image[y*width+x]=(unsigned)(clamp01(result.x)*255)|((unsigned)(clamp01(result.y)*255)<<8)|((unsigned)(clamp01(result.z)*255)<<16)|4278190080u;
}
