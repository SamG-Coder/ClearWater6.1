// Original ClearWater spacecraft. Geometry, navigation, ray traversal and
// materials are CUDA; the browser only dispatches kernels and forwards input.
// 128 manufactured components, including paired weapon mounts. The hull and
// chamfered canopy receive the densest meshes.
// Static geometry and a
// two-level BVH are built once, then shared by primary and sunlight rays.
__device__ int ship_tri_base(int part){return part<4?part*2048:6144+part*512;}
__device__ int ship_node_base(int part){return 255+(part<4?part*341:1024+part*85);}
struct ShipPart { float3 centre; float3 size; float3 rotation; int shape; int material; int mirror; };
// One source of truth for modeled apertures and projectile/beam origins.
__device__ float3 ship_weapon_mount(int weapon,int side,int tube){
 float s=(float)side;if(weapon==1)return vec(s*1.85f,.12f,-3.24f);
 if(weapon==2)return vec(s*(tube==0?6.02f:6.38f),-.20f,-1.02f);
 return vec(s*2.75f,.14f,-5.24f);
}
__device__ float3 ship_rotate(float3 p,float3 r){
 float a=cosf(r.x),b=sinf(r.x);p=vec(p.x,p.y*a-p.z*b,p.y*b+p.z*a);
 a=cosf(r.y);b=sinf(r.y);p=vec(p.x*a+p.z*b,p.y,-p.x*b+p.z*a);
 a=cosf(r.z);b=sinf(r.z);return vec(p.x*a-p.y*b,p.x*b+p.y*a,p.z);
}
__device__ ShipPart ship_component(int id){
 ShipPart p;p.centre=vec(0,0,0);p.size=vec(1,1,1);p.rotation=vec(0,0,0);p.shape=4;p.material=0;p.mirror=1;
 if(id<32){
  if(id==0){p.shape=12;p.size=vec(1.65f,.72f,5.825f);p.centre.z=-.325f;}
  if(id==1){p.shape=0;p.size=vec(1.28f,.40f,5.35f);p.centre=vec(0,-.61f,0);p.material=2;}
  if(id==2){p.shape=5;p.size=vec(.98f,.58f,2.3f);p.centre=vec(0,.86f,-1.95f);p.material=3;}
  if(id==3){p.shape=5;p.size=vec(.025f,.603f,2.32f);p.centre=vec(0,.86f,-1.95f);p.material=2;}
  if(id==4){p.size=vec(.48f,.27f,.08f);p.centre=vec(0,1.03f,-1.10f);p.rotation.x=.20f;p.material=9;}
  if(id==5){p.size=vec(.62f,.14f,.23f);p.centre=vec(0,.80f,-2.92f);p.rotation.x=-.28f;p.material=9;}
  if(id==6){p.shape=10;p.size=vec(.998f,.60f,2.30f);p.centre=vec(0,.86f,-1.95f);p.material=2;}
  // A recessed sensor fascia sits inside the blunt armoured nose.
  if(id==7){p.size=vec(.47f,.145f,.045f);p.centre=vec(0,-.035f,-6.068f);p.material=2;}
  if(id==8){p.size=vec(.58f,.31f,1.88f);p.centre=vec(0,.82f,2.3f);p.material=0;}
  if(id==9){p.size=vec(.46f,.065f,.7f);p.centre=vec(0,1.17f,2.05f);p.material=2;}
  if(id==10){p.shape=1;p.size=vec(.6f,.10f,1.55f);p.centre=vec(0,.94f,3.2f);p.rotation.z=1.5707963f;p.material=2;}
  if(id==11){p.shape=3;p.size=vec(.55f,.10f,.75f);p.centre=vec(0,-.14f,5.90f);p.material=4;}
  if(id==12||id==13){p.size=vec(.28f,.05f,1.28f);p.centre=vec(id==12?-.97f:.97f,.7f,2.4f);p.material=5;}
  // Load-bearing shoulders bridge the laser receivers into the main hull.
  if(id==14||id==15){p.size=vec(.60f,.26f,.72f);p.centre=vec(id==14?-1.53f:1.53f,.17f,-1.42f);p.material=0;}
  if(id==16){p.shape=2;p.size=vec(.49f,.49f,1.1f);p.centre=vec(0,-.14f,5.25f);p.material=2;}
  if(id==17){p.shape=3;p.size=vec(.40f,.05f,.7f);p.centre=vec(0,-.14f,5.98f);p.material=6;}
  if(id==18){p.shape=4;p.size=vec(.30f,.27f,.015f);p.centre=vec(0,-.14f,5.85f);p.material=6;}
  if(id==19){p.size=vec(.105f,.035f,.008f);p.centre=vec(0,-.015f,-6.116f);p.material=11;}
  // A continuous coaming overlaps the hull below and the glass seating above.
  // The dark gasket is geometry, not a line painted over an empty gap.
  if(id==20){p.shape=13;p.size=vec(1.11f,.24f,2.70f);p.centre=vec(0,.53f,-1.95f);p.material=0;}
  if(id==21){p.shape=13;p.size=vec(1.025f,.04f,2.32f);p.centre=vec(0,.746f,-1.95f);p.material=2;}
  if(id==22||id==23){p.size=vec(.08f,.12f,1.65f);p.centre=vec(id==22?-.95f:.95f,.66f,-1.25f);p.material=0;}
  if(id>=24&&id<28){float s=(id%2)==0?-1:1;p.size=vec(.42f,.035f,.47f);p.centre=vec(s*.88f,.70f,-.05f+(float)(id/2-12)*1.22f);p.material=id<26?0:1;}
  if(id==28||id==29){p.size=vec(.095f,.045f,1.5f);p.centre=vec(id==28?-.70f:.70f,.22f,-3.7f);p.rotation.z=id==28?-.1f:.1f;p.material=10;}
  if(id==30){p.size=vec(.45f,.10f,.58f);p.centre=vec(0,.45f,-4.85f);p.rotation.x=-.25f;p.material=0;}
  if(id==31){p.size=vec(.10f,.03f,.32f);p.centre=vec(0,.56f,-4.80f);p.rotation.x=-.25f;p.material=1;}
 }else{
  int side=id<80?-1:1,k=(id-32)%48;float s=1;p.mirror=side;
  if(k==0){p.shape=1;p.size=vec(s*5.7f,.26f,3.15f);p.centre=vec(s*1.0f,-.13f,.7f);}
  // Reuse former wing/decorative slots for hardware, without increasing the BVH.
  if(k==1){p.size=vec(.34f,.25f,1.08f);p.centre=plus(ship_weapon_mount(0,1,0),vec(0,0,3.52f));p.material=0;}
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
  if(k==14){p.shape=2;p.size=vec(.21f,.21f,.55f);p.centre=plus(ship_weapon_mount(0,1,0),vec(0,0,2.48f));p.material=2;}
  if(k==15){p.shape=2;p.size=vec(.11f,.11f,1.20f);p.centre=plus(ship_weapon_mount(0,1,0),vec(0,0,1.24f));p.material=4;}
  if(k>=16&&k<24){float a=(float)(k-16)*.78539816f;p.size=vec(.25f,.085f,.61f);p.centre=vec(s*4.65f+sinf(a)*.89f,.22f+cosf(a)*.89f,4.30f);p.rotation.z=-a;p.rotation.x=.09f;p.material=4;}
  if(k>=24&&k<32){float a=(float)(k-24)*.78539816f;p.size=vec(.07f,.38f,.045f);p.centre=vec(s*4.65f+sinf(a)*.46f,.22f+cosf(a)*.46f,-.77f);p.rotation.z=-a;p.rotation.y=.4f;p.material=4;}
  if(k>=32&&k<40){p.shape=3;p.size=vec(.94f,.032f,1);p.centre=vec(s*4.65f,.22f,2.05f+(float)(k-32)*.20f);p.material=k%3==0?1:4;}
  if(k==40){p.size=vec(.42f,.32f,1.28f);p.centre=vec(6.2f,-.20f,.62f);p.material=0;}
  if(k==41||k==42){p.shape=2;p.size=vec(.14f,.14f,.30f);p.centre=plus(ship_weapon_mount(2,1,k-41),vec(0,0,.34f));p.material=2;}
  if(k==43){p.size=vec(.45f,.045f,.70f);p.centre=vec(6.2f,.16f,.35f);p.material=2;}
  if(k==44){p.size=vec(.45f,.28f,.70f);p.centre=vec(2.75f,-.08f,-1.03f);p.material=2;}
  if(k==45){p.shape=3;p.size=vec(.14f,.038f,1);p.centre=plus(ship_weapon_mount(0,1,0),vec(0,0,.09f));p.material=2;}
  if(k==46){p.size=vec(.27f,.24f,.84f);p.centre=plus(ship_weapon_mount(1,1,0),vec(0,0,.90f));p.material=2;}
  if(k==47){p.size=vec(.16f,.16f,.025f);p.centre=plus(ship_weapon_mount(1,1,0),vec(0,0,.045f));p.material=11;}
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
 if(p.shape==0||p.shape==12){
  float ca=cosf(a),sa=sinf(a),qx=(ca<0?-1:1)*positive_power(fabsf(ca),.68f),qy=(sa<0?-1:1)*positive_power(fabsf(sa),.78f);
  float w=(.045f+.955f*eased(0,.43f,v))*(1-.28f*eased(.65f,1,v)),h=.20f+.80f*eased(.02f,.45f,v);float end=fminf(1,fmaxf(0,fminf(v,1-v)*64));
  point=vec(qx*w*end,qy*h*end+.06f*sinf(v*3.14159265f),2*v-1);
  if(p.shape==12){
   // Broad chisel section flows into the fuselage; the first ring forms a
   // recessed front face instead of stretching the armour to a pointed tip.
   float t=clamp01((v-.03125f)/.96875f),rounded=eased(.48f,.75f,t),front=clamp01(v*32);
   qx=mixf(q.x,qx,rounded);qy=mixf(q.y,qy,rounded);
   w=mixf(.38f,1,eased(0,.43f,t))*(1-.28f*eased(.65f,1,t));h=mixf(.44f,1,eased(0,.45f,t));
   point=vec(qx*w*front*end,qy*h*front*end+.06f*sinf(t*3.14159265f),2*t-1+.035f*(1-front));
  }
 }
 if(p.shape==1){float w=mixf(1,.36f,v);point=vec(v,q.y*mixf(1,.38f,v)*cap,q.x*w*cap+.48f*v);}
 if(p.shape==2){float radius=.82f+.18f*sinf(v*3.14159265f);point=vec(cosf(a)*radius,sinf(a)*radius,2*v-1);}
 if(p.shape==3){float t=v*6.2831853f;point=vec(cosf(a)*(p.size.x+p.size.y*cosf(t)),sinf(a)*(p.size.x+p.size.y*cosf(t)),sinf(t)*p.size.y*p.size.z);point=plus(p.centre,ship_rotate(point,p.rotation));point.x*=(float)p.mirror;return point;}
 if(p.shape==5||p.shape==10){
  // Planar windscreen, low roof and chamfered shoulders replace the bubble.
  // Breaks align with longitudinal vertices, keeping facet normals deliberate.
  float t=p.shape==10?.25f+(v-.5f)*.014f:v;
  float front=clamp01(t*4),rear=clamp01((t-.75f)*4),w=mixf(.60f,1,front)*(1-.30f*rear),h=mixf(.08f,1,front)*(1-.87f*rear),end=fminf(1,fmaxf(0,fminf(t,1-t)*64));
  point=vec(q.x*w*(1-.24f*fmaxf(0,q.y))*end,(fmaxf(0,q.y)*h+fminf(0,q.y)*.10f)*end-.20f,2*t-1);
 }
 if(p.shape==13){
  float front=clamp01(v*4),rear=clamp01((v-.75f)*4),w=mixf(.60f,1,front)*(1-.30f*rear);
  // A broad flat seating face supports the canopy's complete lower perimeter.
  // Taper the end caps down into the hull, rather than leaving a raised point.
  float qx=fabsf(q.x),qy=fabsf(q.y);qx=qx<=.55f?qx*(.90f/.55f):.90f+(qx-.55f)*(.10f/.45f);qy=qy<=.55f?qy*(.90f/.55f):.90f+(qy-.55f)*(.10f/.45f);
  point=vec((q.x<0?-qx:qx)*w*cap,(q.y<0?-qy:qy)*cap-(1-cap),2*v-1);
 }
 point=vec(point.x*p.size.x,point.y*p.size.y,point.z*p.size.z);point=plus(p.centre,ship_rotate(point,p.rotation));point.x*=(float)p.mirror;return point;
}
__device__ void ship_mesh_triangle(float4 *mesh,float4 *bounds,int i,int enemy){
 if(i>=71680)return;int part=i<8192?i/2048:4+(i-8192)/512,local=i-ship_tri_base(part),leaf=local/8,triangle=i%8;ShipPart p=ship_component(part);
 if(enemy!=0){
  if(part==0){p.size.z*=1.10f;p.size.y*=.88f;}
  if(part==7||part==19)p.centre.z-=.5825f;
  if(part>=32){int k=(part-32)%48;if(k==0){p.size.z*=.82f;p.rotation.y=-.10f;}if(k==11){p.size.x*=.86f;p.size.z*=1.42f;p.rotation.z=1.24f;p.material=1;}}
 }
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
__global__ void ship_mesh(float4 *mesh,float4 *bounds){ship_mesh_triangle(mesh,bounds,blockIdx.x*blockDim.x+threadIdx.x,0);}
__global__ void ship_enemy_mesh(float4 *mesh,float4 *bounds){ship_mesh_triangle(mesh,bounds,blockIdx.x*blockDim.x+threadIdx.x,1);}
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
__device__ float ship_angle(float a){return a-floorf((a+3.14159265f)/6.2831853f)*6.2831853f;}
// Exact critically damped response; the same feel at 30, 60 and 120 Hz.
__device__ float2 ship_spring(float position,float velocity,float target,float frequency,float dt){
 float error=position-target,j=velocity+frequency*error,decay=expf(-frequency*dt);
 return make_float2(target+(error+j*dt)*decay,(velocity-frequency*j*dt)*decay);
}
// Above the atmosphere, advance a Cartesian displacement instead of rotating
// around a constant-radius shell. Only this one navigation invocation needs
// double precision. The near-surface flight assistant fades out continuously.
__device__ void ship_space_advance(float4 *camera,float2 *navigationState,float3 travel,float dt,float freeFlight){
 double nx=(double)navigationState[0].x+(double)navigationState[0].y,ny=(double)navigationState[1].x+(double)navigationState[1].y,nz=(double)navigationState[2].x+(double)navigationState[2].y;
 double ex=(double)navigationState[3].x+(double)navigationState[3].y,ey=(double)navigationState[4].x+(double)navigationState[4].y,ez=(double)navigationState[5].x+(double)navigationState[5].y;
 double bx=ey*nz-ez*ny,by=ez*nx-ex*nz,bz=ex*ny-ey*nx;
 double dx=(double)travel.x*(double)dt,dy=(double)travel.y*(double)dt,dz=(double)travel.z*(double)dt,r=6371000.0+(double)camera[0].y;
 double x=nx*(r+dy)+ex*dx+bx*dz,y=ny*(r+dy)+ey*dx+by*dz,z=nz*(r+dy)+ez*dx+bz*dz;
 double radius=sqrt(x*x+y*y+z*z);x/=radius;y/=radius;z/=radius;
 double along=(ex*x+ey*y+ez*z)/(1.0+nx*x+ny*y+nz*z);ex-=along*(nx+x);ey-=along*(ny+y);ez-=along*(nz+z);
 double inv=1.0/sqrt(ex*ex+ey*ey+ez*ez);ex*=inv;ey*=inv;ez*=inv;
 double u=(double)navigationState[6].x+(double)navigationState[6].y+dx,v=(double)navigationState[7].x+(double)navigationState[7].y+dz;
 camera[0].x=(float)(u-(double)floorf((float)((u+3072.0)/6144.0))*6144.0);camera[0].z=(float)(v-(double)floorf((float)((v+3072.0)/6144.0))*6144.0);
 camera[0].y=fmaxf(6,mixf((float)(r+dy-6371000.0),(float)(radius-6371000.0),freeFlight));
 camera[5]=make_float4((float)ex,(float)ey,(float)ez,0);camera[6]=make_float4((float)x,(float)y,(float)z,0);camera[7]=make_float4((float)(ey*z-ez*y),(float)(ez*x-ex*z),(float)(ex*y-ey*x),0);
 {float hi=(float)x;navigationState[0]=make_float2(hi,(float)(x-(double)hi));}
 {float hi=(float)y;navigationState[1]=make_float2(hi,(float)(y-(double)hi));}
 {float hi=(float)z;navigationState[2]=make_float2(hi,(float)(z-(double)hi));}
 {float hi=(float)ex;navigationState[3]=make_float2(hi,(float)(ex-(double)hi));}
 {float hi=(float)ey;navigationState[4]=make_float2(hi,(float)(ey-(double)hi));}
 {float hi=(float)ez;navigationState[5]=make_float2(hi,(float)(ez-(double)hi));}
 {float hi=(float)u;navigationState[6]=make_float2(hi,(float)(u-(double)hi));}
 {float hi=(float)v;navigationState[7]=make_float2(hi,(float)(v-(double)hi));}
}
__device__ float3 ship_local(const float4 *camera,float3 p){return vec(dotv(p,ship_xyz(camera[5])),dotv(p,ship_xyz(camera[6])),dotv(p,ship_xyz(camera[7])));}
__device__ float3 ship_heading(float yaw,float pitch){return vec(sinf(yaw)*cosf(pitch),sinf(pitch),-cosf(yaw)*cosf(pitch));}
// A bounded game envelope, independent of the unrestricted exploration scale.
// Smooth stages leave room to enjoy the ocean, then cross planetary distances.
__device__ float ship_game_limit(float altitude,int boost){
 float low=eased(1500,12000,altitude),upper=eased(12000,80000,altitude),space=eased(80000,600000,altitude);
 float cruise=mixf(180,420,low);cruise=mixf(cruise,1200,upper);cruise=mixf(cruise,150000,space);
 float boosted=mixf(450,1050,low);boosted=mixf(boosted,3000,upper);boosted=mixf(boosted,600000,space);
 return boost!=0?boosted:cruise;
}
// Three persistent linear colours. Browser writes sRGB + w=1; conversion runs
// once per edit in this single-thread flight kernel, never per shaded pixel.
__device__ float ship_srgb(float c){return c<=.04045f?c/12.92f:positive_power((c+.055f)/1.055f,2.4f);}
__device__ void ship_prepare_colours(float4 *ship){
 if(ship[11].w==0){ship[11]=make_float4(.55f,.61f,.60f,2);ship[12]=make_float4(.65f,.145f,.035f,2);ship[13]=make_float4(.045f,.30f,1,2);}
 for(int i=11;i<=13;i++){float4 c=ship[i];if(c.w==1)ship[i]=make_float4(ship_srgb(c.x),ship_srgb(c.y),ship_srgb(c.z),2);}
}
__global__ void ship_step(float4 *camera,float2 *navigationState,float4 *ship,float deltaTime,float forward,float turn,float rise,float lookX,float lookY,float speed,int boost,int action,int inspect,int flightProfile,float depth,float aspect){
 float dt=fminf(.1f,fmaxf(0,deltaTime));
 ship_prepare_colours(ship);
 if(ship[0].w==0||action==1){
  if(camera[8].z==0){float navLength=navigationState[0].x*navigationState[0].x+navigationState[1].x*navigationState[1].x+navigationState[2].x*navigationState[2].x;camera_advance(camera,navigationState,0,0,0,0,0,0,1,0,navLength>.5f?0:1,depth);}
  ship[0]=make_float4(camera[0].x,22,camera[0].z,1);ship[1]=make_float4(0,0,0,0);ship[2]=make_float4(flightProfile!=0?0:.34f,.30f,22,0);ship[3]=make_float4(0,.16f,0,speed);
  ship[8]=make_float4(0,0,0,0);ship[9]=make_float4(0,0,0,0);ship[10]=make_float4(0,0,22,0);
 }
 // Navigation belongs to the ship; rendering belongs to the chase eye. Restore
 // the ship frame before integrating controls, rather than integrating in last
 // frame's offset camera frame (which would steer the planet as we orbit).
 float3 navN=unit(vec(navigationState[0].x,navigationState[1].x,navigationState[2].x)),navE=unit(vec(navigationState[3].x,navigationState[4].x,navigationState[5].x)),navB=crossv(navE,navN);
 camera[5]=make_float4(navE.x,navE.y,navE.z,camera[5].w);camera[6]=make_float4(navN.x,navN.y,navN.z,0);camera[7]=make_float4(navB.x,navB.y,navB.z,camera[7].w);
 if(action==3){ship[0].y=earth_radius()*.70f;ship[1].y=-.40f;ship[2].y=.72f;ship[2].x=.32f;ship[8]=make_float4(ship[1].x,-.40f,0,0);ship[9]=make_float4(0,0,0,0);ship[10].x=ship[1].x;ship[10].y=-.40f;}
 if(action==4){
  float3 moon=moon_position(camera),sun=unit(ship_xyz(camera[12])),side=unit(crossv(sun,vec(0,1,0))),out=unit(plus(scale(sun,.65f),scale(side,.76f)));
  float3 point=plus(moon,scale(out,6800000)),normal=unit(point),east=unit(crossv(vec(0,1,0),normal)),back=crossv(east,normal);
  navigationState[0]=make_float2(normal.x,0);navigationState[1]=make_float2(normal.y,0);navigationState[2]=make_float2(normal.z,0);navigationState[3]=make_float2(east.x,0);navigationState[4]=make_float2(east.y,0);navigationState[5]=make_float2(east.z,0);navigationState[6]=make_float2(0,0);navigationState[7]=make_float2(0,0);
  camera[5]=make_float4(east.x,east.y,east.z,0);camera[6]=make_float4(normal.x,normal.y,normal.z,0);camera[7]=make_float4(back.x,back.y,back.z,moon.x);
  float3 local=ship_local(camera,scale(out,-1));float yaw=atan2f(local.x,-local.z),pitch=atan2f(local.y,sqrtf(local.x*local.x+local.z*local.z));
  ship[0]=make_float4(0,sqrtf(dotv(point,point))-earth_radius(),0,1);ship[1]=make_float4(yaw,pitch,0,0);ship[2]=make_float4(0,.10f,22,0);ship[8]=make_float4(yaw,pitch,0,0);ship[9]=make_float4(0,0,0,0);ship[10]=make_float4(yaw,pitch,22,0);
 }
 float4 pose=ship[1];ship[2].w=(float)inspect;
 if(inspect!=0){ship[2].x+=lookX;ship[2].y=fminf(1.30f,fmaxf(-.15f,ship[2].y-lookY));ship[8]=make_float4(pose.x,pose.y,0,0);}
 else{
  ship[8].x=ship_angle(ship[8].x+lookX+turn*dt*1.12f);float pitchLimit=mixf(1.45f,1.5707f,eased(12000,80000,ship[0].y));ship[8].y=fminf(pitchLimit,fmaxf(-pitchLimit,ship[8].y+lookY));
  // Paused inspection/diagnostic edits should still respond to mouse input.
  if(dt==0&&(lookX!=0||lookY!=0)){pose.x=ship[8].x;pose.y=ship[8].y;ship[8].z=0;ship[8].w=0;}
  if(forward>0||fabsf(turn)>.01f||fabsf(lookX)>.001f)ship[2].x*=expf(-dt*4);
  ship[2].y=mixf(ship[2].y,.24f,1-expf(-dt*4));
 }
 if(inspect==0){
  float2 yaw=ship_spring(pose.x,ship[8].z,pose.x+ship_angle(ship[8].x-pose.x),14,dt),pitch=ship_spring(pose.y,ship[8].w,ship[8].y,14,dt);
  pose.x=ship_angle(yaw.x);pose.y=pitch.x;ship[8].z=yaw.y;ship[8].w=pitch.y;
 }
 float targetBank=inspect==0?fminf(.62f,fmaxf(-.62f,-ship[8].z*.48f)):0;pose.z=mixf(pose.z,targetBank,1-expf(-dt*7));
 float freeFlight=eased(12000,80000,ship[0].y);
 float maxSpeed=flightProfile!=0?ship_game_limit(ship[0].y,boost)*fminf(1,fmaxf(.1f,speed)):fminf(30000000,speed*fmaxf(1,ship[0].y*.006f)*(boost!=0?6:1)),target=fmaxf(0,forward)*maxSpeed;
 pose.w=mixf(pose.w,target,1-expf(-dt*(forward<0?7:forward>0?2.8f:.48f*(1-freeFlight))));if(pose.w<.002f)pose.w=0;
 float3 nose=vec(sinf(pose.x)*cosf(pose.y),sinf(pose.y),-cosf(pose.x)*cosf(pose.y));
 float3 lift=blend(vec(0,1,0),vec(-sinf(pose.x)*sinf(pose.y),cosf(pose.y),cosf(pose.x)*sinf(pose.y)),freeFlight);
 float3 previous=ship_xyz(ship[9]),desired=plus(scale(nose,pose.w),scale(lift,rise*maxSpeed*.35f));
 if(flightProfile!=0){
  // Clamp the complete vector: diagonal thrust and a fast atmospheric re-entry
  // must respect the same envelope as straight flight. Zero-dt pause is inert.
  desired=scale(desired,fminf(1,maxSpeed/fmaxf(.001f,sqrtf(dotv(desired,desired)))));
  if(dt>0){previous=scale(previous,fminf(1,maxSpeed/fmaxf(.001f,sqrtf(dotv(previous,previous)))));pose.w=fminf(pose.w,maxSpeed);}
 }
 float3 velocity=blend(previous,desired,1-expf(-dt*(forward<0?10:6))),travel=scale(plus(previous,velocity),.5f);
 float3 worldTravel=to_world(camera,scale(travel,dt));
 float actual=sqrtf(dotv(travel,travel)),factor=fmaxf(1,ship[0].y*.06f);
 camera[0]=make_float4(ship[0].x,ship[0].y,ship[0].z,0);camera[1]=make_float4(0,0,0,0);
 if(freeFlight>0&&actual>0&&dt>0){
  float3 worldNose=to_world(camera,nose),worldTarget=to_world(camera,ship_heading(ship[8].x,ship[8].y)),worldChase=to_world(camera,ship_heading(ship[10].x,ship[10].y)),worldVelocity=to_world(camera,velocity);
  ship_space_advance(camera,navigationState,travel,dt,freeFlight);
  float3 local=ship_local(camera,worldNose),aim=ship_local(camera,worldTarget),chase=ship_local(camera,worldChase);
  pose.x=ship_angle(pose.x+ship_angle(atan2f(local.x,-local.z)-pose.x)*freeFlight);pose.y=mixf(pose.y,atan2f(local.y,sqrtf(local.x*local.x+local.z*local.z)),freeFlight);
  ship[8].x=ship_angle(ship[8].x+ship_angle(atan2f(aim.x,-aim.z)-ship[8].x)*freeFlight);ship[8].y=mixf(ship[8].y,atan2f(aim.y,sqrtf(aim.x*aim.x+aim.z*aim.z)),freeFlight);
  ship[10].x=ship_angle(ship[10].x+ship_angle(atan2f(chase.x,-chase.z)-ship[10].x)*freeFlight);ship[10].y=mixf(ship[10].y,atan2f(chase.y,sqrtf(chase.x*chase.x+chase.z*chase.z)),freeFlight);
  velocity=blend(velocity,ship_local(camera,worldVelocity),freeFlight);
 }else camera_advance(camera,navigationState,dt,actual>0?-travel.z/actual:0,actual>0?travel.x/actual:0,actual>0?travel.y/actual:0,0,0,actual/factor,0,0,depth);
 float floorHeight=camera[21].w!=0?terrain_height(camera,ship_xyz(camera[6])):0;camera[0].y=fmaxf(camera[0].y,fmaxf(6,floorHeight+6));
 if(camera[0].y<=fmaxf(6,floorHeight+6)+.001f)velocity.y=fmaxf(0,velocity.y);
 if(flightProfile!=0&&dt>0){float cap=ship_game_limit(camera[0].y,boost)*fminf(1,fmaxf(.1f,speed));velocity=scale(velocity,fminf(1,cap/fmaxf(.001f,sqrtf(dotv(velocity,velocity)))));pose.w=fminf(pose.w,cap);}
 ship[9]=make_float4(velocity.x,velocity.y,velocity.z,sqrtf(dotv(velocity,velocity)));
 ship[0]=make_float4(camera[0].x,camera[0].y,camera[0].z,1);ship[1]=pose;ship[3]=make_float4(ship[3].x+dt,mixf(ship[3].y,forward>0?(boost!=0?1:.62f):fabsf(rise)>.01f?.38f:.16f,1-expf(-dt*5)),camera[0].y-floorHeight,speed);
 float3 angles=vec(pose.y,-pose.x,pose.z),right=ship_rotate(vec(1,0,0),angles),up=ship_rotate(vec(0,1,0),angles),back=ship_rotate(vec(0,0,1),angles);
 // Positive mouse yaw points the nose to the right in the globe's tangent frame.
 right=ship_rotate(vec(1,0,0),vec(0,-pose.x,0));back=vec(-sinf(pose.x)*cosf(pose.y),-sinf(pose.y),cosf(pose.x)*cosf(pose.y));up=unit(crossv(back,right));
 float3 bankRight=plus(scale(right,cosf(pose.z)),scale(up,sinf(pose.z))),bankUp=minus(scale(up,cosf(pose.z)),scale(right,sinf(pose.z)));
 ship[4]=make_float4(bankRight.x,bankRight.y,bankRight.z,0);ship[5]=make_float4(bankUp.x,bankUp.y,bankUp.z,0);ship[6]=make_float4(back.x,back.y,back.z,0);
 // Chase the heading with a short, bounded lag and keep most of the horizon level.
 ship[10].x=ship_angle(ship[10].x+ship_angle(pose.x-ship[10].x)*(1-expf(-dt*8)));ship[10].y=mixf(ship[10].y,pose.y,1-expf(-dt*8));
 float cameraYaw=inspect!=0?pose.x:pose.x+fminf(.24f,fmaxf(-.24f,ship_angle(ship[10].x-pose.x))),cameraPitch=inspect!=0?pose.y:mixf(pose.y,ship[10].y,.65f);
 float3 cameraRight=vec(cosf(cameraYaw),0,sinf(cameraYaw)),cameraBack=vec(-sinf(cameraYaw)*cosf(cameraPitch),-sinf(cameraPitch),cosf(cameraYaw)*cosf(cameraPitch)),cameraUp=unit(crossv(cameraBack,cameraRight));
 float extra=inspect!=0?0:.08f*clamp01(ship[9].w/fmaxf(1,flightProfile!=0?maxSpeed:speed))+.07f*eased(.62f,1,ship[3].y);
 float wantedDistance=ship[2].z*(1+extra);ship[10].z=dt>0?mixf(ship[10].z,wantedDistance,1-expf(-dt*5)):wantedDistance;
 float orbit=ship[2].x,elevation=ship[2].y,distance=ship[10].z*fmaxf(1,.88f/aspect);float3 offset=plus(scale(cameraRight,sinf(orbit)*cosf(elevation)*distance),plus(scale(cameraUp,sinf(elevation)*distance),scale(cameraBack,cosf(orbit)*cosf(elevation)*distance)));
 // Keep the model-to-eye transform in small local metres. Subtracting two
 // float32 orbital altitudes would quantize the canopy and tracking camera.
 offset.y=fmaxf(offset.y,floorHeight+2-ship[0].y);
 float3 viewUp=unit(plus(cameraUp,scale(cameraRight,sinf(pose.z)*.12f))),eye=plus(ship_xyz(ship[0]),offset),f=unit(minus(vec(0,.50f,0),offset)),r=unit(crossv(f,viewUp)),u=crossv(r,f);
 float3 localEye=ship_inverse(ship,offset);ship[7]=make_float4(localEye.x,localEye.y,localEye.z,0);
 // All world rays must start at the actual eye, including shoreline traces.
 // Parallel transport the frame to that eye without changing the ship's
 // precise navigation state or its small, unquantized model-to-eye transform.
 float3 oldN=ship_xyz(camera[6]),oldE=ship_xyz(camera[5]);
 double cnx=(double)navigationState[0].x+(double)navigationState[0].y,cny=(double)navigationState[1].x+(double)navigationState[1].y,cnz=(double)navigationState[2].x+(double)navigationState[2].y;
 double cex=(double)navigationState[3].x+(double)navigationState[3].y,cey=(double)navigationState[4].x+(double)navigationState[4].y,cez=(double)navigationState[5].x+(double)navigationState[5].y,cr=6371000.0+(double)ship[0].y+(double)offset.y;
 double cx=cnx*cr+cex*(double)offset.x+(cey*cnz-cez*cny)*(double)offset.z,cy=cny*cr+cey*(double)offset.x+(cez*cnx-cex*cnz)*(double)offset.z,cz=cnz*cr+cez*(double)offset.x+(cex*cny-cey*cnx)*(double)offset.z;
 double eyeRadius=sqrt(cx*cx+cy*cy+cz*cz);float3 eyeN=vec((float)(cx/eyeRadius),(float)(cy/eyeRadius),(float)(cz/eyeRadius));
 float3 eyeE=unit(minus(oldE,scale(plus(oldN,eyeN),dotv(oldE,eyeN)/(1+dotv(oldN,eyeN))))),eyeB=crossv(eyeE,eyeN);
 float3 wf=to_world(camera,f),wr=to_world(camera,r),wu=to_world(camera,u),sr=to_world(camera,bankRight),su=to_world(camera,bankUp),sb=to_world(camera,back);
 camera[5]=make_float4(eyeE.x,eyeE.y,eyeE.z,camera[5].w);camera[6]=make_float4(eyeN.x,eyeN.y,eyeN.z,0);camera[7]=make_float4(eyeB.x,eyeB.y,eyeB.z,camera[7].w);
 f=ship_local(camera,wf);r=ship_local(camera,wr);u=ship_local(camera,wu);sr=ship_local(camera,sr);su=ship_local(camera,su);sb=ship_local(camera,sb);
 ship[4]=make_float4(sr.x,sr.y,sr.z,0);ship[5]=make_float4(su.x,su.y,su.z,0);ship[6]=make_float4(sb.x,sb.y,sb.z,0);
 eye.y=(float)(eyeRadius-6371000.0);
 camera[0]=make_float4(eye.x,eye.y,eye.z,0);camera[1]=make_float4(pose.x,pose.y,0,0);camera[2]=make_float4(f.x,f.y,f.z,depth/.86f);camera[3]=make_float4(r.x,r.y,r.z,expf(-depth*.055f));camera[4]=make_float4(u.x,u.y,u.z,0);camera[8].x=eye.y;camera[8].y=ship[9].w;
 float3 combatTravel=ship_local(camera,worldTravel);ship[17]=make_float4(combatTravel.x,combatTravel.y,combatTravel.z,0);
 camera[19].x=ship[0].y;camera[19].y=ship[9].w;camera[19].z=1;camera[8].w=maxSpeed;
}
// Two vectored lift jets create pressure at their actual sea-plane footprints.
// brush[3] holds the moving-domain shift, clear flag and combined strength;
// brush[4..5] hold world X/Z, Gaussian radius and strength for each engine.
__global__ void ship_wash_pick(const float4 *camera,const float4 *ship,float4 *brush){
 float4 old=brush[2];brush[0]=make_float4(0,0,0,0);brush[1]=make_float4(0,0,0,0);brush[3]=make_float4(0,0,0,0);
 float3 jet=unit(minus(scale(ship_xyz(ship[6]),.28f),scale(ship_xyz(ship[5]),.96f)));
 for(int i=0;i<2;i++){
  float3 engine=plus(ship_xyz(ship[0]),ship_axis(ship,vec(i==0?-4.65f:4.65f,-.5f,2.35f)));
  float height=engine.y,hit=height/fmaxf(.2f,-jet.y),x=engine.x+jet.x*hit,z=engine.z+jet.z*hit;
  float nearWater=1-eased(5,28,height),strength=nearWater*nearWater*(.42f+.58f*ship[3].y)*eased(.2f,.65f,-jet.y)*(1-eased(100,300,ship[9].w));
  if(height<0||height>=28||jet.y>=-.2f)strength=0;
  if(strength>0&&camera[21].w!=0){float3 normal=to_world(camera,unit(vec(x-camera[0].x,earth_radius(),z-camera[0].z)));if(terrain_height(camera,normal)>-.12f)strength=0;}
  brush[4+i]=make_float4(x,z,.65f+fminf(28,fmaxf(0,height))*.075f,strength);
  float3 nozzle=plus(ship_xyz(ship[0]),ship_axis(ship,vec(i==0?-4.65f:4.65f,.22f,5.3f)));
  brush[6+i]=make_float4(nozzle.x,nozzle.y,nozzle.z,16+ship[3].y*65);
  float night=1-eased(-.04f,.18f,camera[9].y);
  float3 lamp=plus(ship_xyz(ship[0]),ship_axis(ship,vec(i==0?-.70f:.70f,.22f,-4.8f)));
  brush[8+i]=make_float4(lamp.x,lamp.y,lamp.z,night*160);
 }
 float3 lampDirection=ship_axis(ship,unit(vec(0,-.62f,-1)));brush[10]=make_float4(lampDirection.x,lampDirection.y,lampDirection.z,ship[3].x);
 brush[11]=ship[13]; // The water receives the same linear booster light as the hull.
 float strength=brush[4].w+brush[5].w;
 if(strength>.0001f){
  float x=(brush[4].x+brush[5].x)*.5f,z=(brush[4].y+brush[5].y)*.5f,dx=x-old.x,dz=z-old.y;
  int clear=old.w==0||fabsf(dx)>12||fabsf(dz)>12?1:0;
  brush[2]=make_float4(x,z,0,1);brush[3]=make_float4(clear!=0?0:dx,clear!=0?0:dz,(float)clear,strength);
 }else brush[2]=make_float4(old.x,old.y,0,old.w);
}
// Forced Fourier oscillators: height and velocity persist after leaving the
// footprint. Rephase the history when the domain moves, rather than restarting
// a periodic stamp. The existing C2 spatial taper prevents tiled copies.
__global__ void ship_wash_modes(float4 *disturbance,const float4 *brush,const float *motion,const float4 *ship,float dt){
 int x=blockIdx.x*blockDim.x+threadIdx.x,z=blockIdx.y*blockDim.y+threadIdx.y;if(x>=128||z>=128)return;
 int idx=z*128+x;float4 shift=brush[3],state=shift.z!=0?make_float4(0,0,0,0):disturbance[idx];
 if(x==64||z==64||(x==0&&z==0)){disturbance[idx]=make_float4(0,0,0,0);return;}
 if(shift.w==0&&state.x==0&&state.y==0&&state.z==0&&state.w==0)return;
 int fx=x<64?x:x-128,fz=z<64?z:z-128;float kx=6.2831853f*(float)fx/24,kz=6.2831853f*(float)fz/24,kk=kx*kx+kz*kz;
 float phase=kx*shift.x+kz*shift.y,rc=cosf(phase),rs=sinf(phase);
 state=make_float4(state.x*rc-state.y*rs,state.x*rs+state.y*rc,state.z*rc-state.w*rs,state.z*rs+state.w*rc);
 float targetRe=0,targetIm=0;
 if(dt>0)for(int i=0;i<2;i++){
  float4 engine=brush[4+i];float radius2=engine.z*engine.z;
  float pulse=.86f+.10f*sinf(ship[3].x*8+(float)i*1.7f)+.04f*sinf(ship[3].x*13-(float)i);
  float amplitude=-.90f*engine.w*pulse*6.2831853f*radius2/576*expf(-kk*radius2*.5f);
  float p=kx*(engine.x-brush[2].x)+kz*(engine.y-brush[2].y);targetRe+=amplitude*cosf(p);targetIm-=amplitude*sinf(p);
 }
 float omega=motion[32768+idx],co=cosf(omega*dt),si=sinf(omega*dt),decay=expf(-(.55f+motion[65536+idx])*dt-sqrtf(shift.x*shift.x+shift.y*shift.y)/8);
 disturbance[idx]=make_float4((state.x*co+state.z/omega*si+targetRe*(1-co))*decay,(state.y*co+state.w/omega*si+targetIm*(1-co))*decay,(state.z*co-state.x*omega*si+targetRe*omega*si)*decay,(state.w*co-state.y*omega*si+targetIm*omega*si)*decay);
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
// Finite local emitters put the selected booster light on nearby panels.
// Navigation lamps and landing strips remain readable on the night side.
__device__ float3 ship_local_light(float3 p,float3 n,float3 view,float3 albedo,const float4 *camera,const float4 *ship){
 float3 radiance=vec(0,0,0);float night=1-eased(-.04f,.18f,camera[9].y);
 for(int i=0;i<6;i++){
  float side=(i%2)==0?-1:1;float3 position=i<2?vec(side*4.65f,.22f,5.5f):i<4?vec(side*6.68f,.24f,1.35f):vec(side*.70f,.36f,-3.7f);
  float3 hue=i<2?ship_xyz(ship[13]):i<4?((i%2)==0?vec(1,.025f,.008f):vec(.025f,1,.22f)):vec(.62f,.80f,1);
  float power=i<2?8+ship[3].y*28:i<4?1.3f:night*2.5f;
  float3 delta=minus(position,p);float d2=dotv(delta,delta);if(d2>144)continue;float3 l=unit(delta),halfv=unit(plus(l,view));
  float diffuse=fmaxf(0,dotv(n,l)),spec=positive_power(fmaxf(0,dotv(n,halfv)),48)*.65f;
  float attenuation=power/(1+d2)*eased(144,81,d2);
  radiance=plus(radiance,scale(vec(hue.x*(albedo.x*diffuse+spec),hue.y*(albedo.y*diffuse+spec),hue.z*(albedo.z*diffuse+spec)),attenuation));
 }
 for(int i=0;i<2;i++){
  float flash=i==0?ship[20].x:ship[20].y;if(flash<=0)continue;float3 delta=minus(ship_weapon_mount((int)ship[20].z,i==0?-1:1,0),p);float d2=dotv(delta,delta);if(d2>36)continue;
  float3 l=unit(delta),h=unit(plus(l,view)),hue=ship[20].z==1?vec(.08f,.65f,1):vec(1,.48f,.09f);float diffuse=fmaxf(0,dotv(n,l)),spec=positive_power(fmaxf(0,dotv(n,h)),32);
  radiance=plus(radiance,scale(vec(hue.x*(albedo.x*diffuse+spec),hue.y*(albedo.y*diffuse+spec),hue.z*(albedo.z*diffuse+spec)),flash*2/(1+d2)));
 }
 return radiance;
}
__global__ void ship_effect_probe(const float4 *camera,const float4 *ship,const float4 *brush,const float4 *points,float4 *output,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float4 p=points[i];float3 color=vec(0,0,0);
 if(p.w==0)color=craft_water(color,vec(p.x,p.y,p.z),vec(0,1,0),unit(vec(.1f,-1,.1f)),camera,brush);
 else color=ship_local_light(vec(p.x,p.y,p.z),vec(0,0,1),vec(0,0,1),vec(.5f,.5f,.5f),camera,ship);
 output[i]=make_float4(color.x,color.y,color.z,1);
}
__device__ float4 ship_radiance(const float4 *mesh,const float4 *bounds,const float4 *camera,const float4 *ship,ShipHit hit,float3 ray,float footprint,float knownShadow){
 int tri=hit.triangle,mat=(int)bounds[ship_node_base(hit.part)*2].w;float w=1-hit.u-hit.v;
 float3 n=unit(plus(scale(ship_xyz(mesh[tri*6+3]),w),plus(scale(ship_xyz(mesh[tri*6+4]),hit.u),scale(ship_xyz(mesh[tri*6+5]),hit.v))));if(dotv(n,ray)>0)n=scale(n,-1);
 float3 p=plus(ship_xyz(ship[7]),scale(ray,hit.t));float u=mesh[tri*6].w*w+mesh[tri*6+1].w*hit.u+mesh[tri*6+2].w*hit.v,v=mesh[tri*6+3].w*w+mesh[tri*6+4].w*hit.u+mesh[tri*6+5].w*hit.v;
 float3 color=ship_xyz(ship[11]);float rough=.29f,metal=.30f,emission=0;
 if(mat==1){color=ship_xyz(ship[12]);rough=.29f;metal=.25f;}
 if(mat==2){color=vec(.025f,.035f,.042f);rough=.44f;metal=.7f;}
 if(mat==3){color=vec(.015f,.055f,.075f);rough=.075f;metal=.22f;}
 if(mat==4){color=vec(.18f,.22f,.25f);rough=.28f;metal=.92f;}
 if(mat==5){color=vec(.016f,.021f,.025f);rough=.53f;metal=.72f;}
 if(mat==6){float cx=hit.part<32?0:(hit.part<80?-4.65f:4.65f),cy=hit.part<32?-.14f:.22f;float rr=(p.x-cx)*(p.x-cx)+(p.y-cy)*(p.y-cy);float core=expf(-rr*12);float3 hue=ship_xyz(ship[13]);float peak=fmaxf(hue.x,fmaxf(hue.y,hue.z));color=blend(scale(hue,.75f),blend(hue,vec(peak,peak,peak),.32f),core);emission=(.6f+core*7)*(1+ship[3].y*2);}
 if(mat==7){color=vec(1,.045f,.01f);emission=3;}
 if(mat==8){color=vec(.08f,1,.50f);emission=3;}
 if(mat==9){color=vec(.014f,.025f,.028f);rough=.65f;metal=.1f;if(hit.part==5&&n.y>.2f){color=vec(.015f,.17f,.22f);emission=1;}}
 if(mat==10){color=vec(.55f,.78f,1);emission=.1f+4*(1-eased(-.04f,.18f,camera[9].y));rough=.18f;}
 if(mat==11){color=ship[16].w==2?vec(1,.045f,.015f):vec(.035f,.54f,1);emission=1.8f;rough=.12f;metal=.5f;}
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
  if(component==0&&ax>5.7f){float stripe=ship_line(p.z-2.1f,.12f,fp);color=blend(color,scale(ship_xyz(ship[12]),.9f),stripe);}
 }
 if(mat==4){float heat=eased(2.8f,4.9f,p.z);color=blend(color,vec(.17f,.13f,.19f),heat*.32f);rough+=globe_noise(vec(p.x*25,p.y*25,p.z*3))*.08f;}
 if(mat==0&&hit.part>=32&&(hit.part-32)%48==0&&n.y>.4f){float serial=ship_serial(fabsf(p.x)-2.65f,p.z-.1f,fp);color=blend(color,vec(.02f,.035f,.04f),serial);}
 if(mat==1&&hit.part>=32&&(hit.part-32)%48==40){float warning=ship_line(fract(p.x*7+p.z*5)-.5f,.16f,fp*8);color=blend(color,vec(.025f,.027f,.023f),warning*.85f);}
 if(mat==5){float rib=ship_line(fract(p.z*14)-.5f,.13f,fp*14);color=scale(color,.5f+rib*1.5f);}
 if(mat==2&&hit.part>31){float band=ship_line(fract(p.z*6)-.5f,.08f,fp*6);color=blend(color,vec(.12f,.14f,.16f),band*.5f);}
 float3 light=unit(ship_inverse(ship,ship_xyz(camera[9]))),view=scale(ray,-1),halfv=unit(plus(light,view));float nl=fmaxf(0,dotv(n,light)),nv=fmaxf(.02f,dotv(n,view)),nh=fmaxf(0,dotv(n,halfv)),vh=fmaxf(0,dotv(view,halfv));
 float shadow=knownShadow;if(shadow<0){shadow=1;if(nl>.001f){ShipHit occ=ship_trace(mesh,bounds,plus(p,scale(n,.006f)),light,1);if(occ.triangle>=0)shadow=.08f;}}
 float3 worldN=ship_axis(ship,n),reflected=ship_axis(ship,minus(ray,scale(n,2*dotv(ray,n))));float atmosphere=1-eased(15000,130000,camera[0].y),day=mixf(globe_hit(ship[0].y,ship_xyz(camera[9]))>0?0:1,eased(-.10f,.22f,camera[9].y),atmosphere);
 float3 ambient=blend(vec(.016f,.025f,.035f),vec(.12f,.19f,.27f),clamp01(worldN.y*.5f+.5f));ambient=scale(ambient,(.05f+day)*atmosphere+.035f);
 float3 env=blend(vec(.018f,.035f,.045f),vec(.13f,.26f,.43f),eased(-.2f,.8f,reflected.y));env=scale(env,(.02f+day)*atmosphere+.035f);
 if(mat==3&&camera[0].y<400)env=scale(weather_sky_sample(reflected,camera),1.8f);
 float a=rough*rough,a2=a*a,den=nh*nh*(a2-1)+1,D=a2/(3.14159265f*den*den+.00001f),k=(rough+1)*(rough+1)/8,G=nv/(nv*(1-k)+k)*nl/(nl*(1-k)+k);
 float f=1-vh,f2=f*f,fres=f2*f2*f;float3 f0=mat==3?vec(.16f,.22f,.27f):blend(vec(.04f,.04f,.04f),color,metal),F=blend(f0,vec(1,1,1),fres);
 float3 direct=plus(scale(color,(1-metal)*nl*.90f),scale(F,D*G/fmaxf(.04f,4*nv)*1.5f));direct=scale(direct,shadow*(.03f+1.7f*day));
 float3 indirect=vec(color.x*ambient.x,color.y*ambient.y,color.z*ambient.z);if(mat==3){float frame=ship_line(u-.125f,.002f,.003f)+ship_line(u-.375f,.002f,.003f);indirect=plus(indirect,scale(vec(.03f,.09f,.13f),1-clamp01(frame)));}float3 reflection=vec(env.x*F.x,env.y*F.y,env.z*F.z);reflection=scale(reflection,1.15f-rough*.6f);
 float ao=1-.24f*seam-.32f*rivet;if(hit.part>=32&&fabsf(p.x)>3.65f&&fabsf(p.x)<5.7f&&p.y<1.1f)ao*=.82f;
 float3 result=plus(scale(plus(direct,plus(indirect,reflection)),ao),scale(color,emission));
 result=plus(result,ship_local_light(p,n,view,color,camera,ship));
 // A short impact pulse follows the actual shield/hull damage timer.
 if(ship[14].w>0){float rim=positive_power(1-fmaxf(0,dotv(n,view)),2);float3 flash=ship[14].y>0?vec(.08f,.70f,2.2f):vec(2,.20f,.025f);result=plus(result,scale(flash,ship[14].w*(.12f+rim)));}
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
 float3 light=vec(0,0,0),booster=ship_xyz(ship[13]);float peak=fmaxf(booster.x,fmaxf(booster.y,booster.z));float thrust=fmaxf(.12f,ship[3].y),length=1.5f+thrust*5.5f;
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
   float3 hue=scale(blend(booster,vec(peak,peak,peak),expf(-r*8)*.16f),2.2f);light=plus(light,scale(hue,envelope*diamonds*step*(1+thrust)*1.2f));
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
