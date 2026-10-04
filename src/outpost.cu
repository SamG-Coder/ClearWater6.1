// A persistent, planet-anchored first mission. All movement, landing checks,
// geometry, collision, mission state and lighting execute in CUDA/WebGPU.
// ship[21..46] are part of the existing version-1 saved spacecraft record.
// 21: locomotion, mission stage, stage timer, schema=62. 28/29: split anchor.
// Modes: 0 flight, 1 landing, 2 parked, 3 on foot, 4 takeoff.
__device__ float3 outpost_axis(const float4 *ship,float3 p){return plus(scale(ship_xyz(ship[41]),p.x),plus(scale(ship_xyz(ship[42]),p.y),scale(ship_xyz(ship[43]),p.z)));}
__device__ float3 outpost_inverse(const float4 *ship,float3 p){return vec(dotv(p,ship_xyz(ship[41])),dotv(p,ship_xyz(ship[42])),dotv(p,ship_xyz(ship[43])));}
__device__ void outpost_frame(float4 *ship,const float4 *camera,const float2 *nav){
 double r=6371000.0+(double)ship[28].w,sr=6371000.0+(double)ship[0].y;
 float3 delta=vec((float)(((double)ship[28].x+(double)ship[29].x)*r-((double)nav[0].x+(double)nav[0].y)*sr),(float)(((double)ship[28].y+(double)ship[29].y)*r-((double)nav[1].x+(double)nav[1].y)*sr),(float)(((double)ship[28].z+(double)ship[29].z)*r-((double)nav[2].x+(double)nav[2].y)*sr));
 float3 rel=ship_local(camera,delta),e=ship_local(camera,ship_xyz(ship[30])),n=ship_local(camera,ship_xyz(ship[28])),b=ship_local(camera,crossv(ship_xyz(ship[30]),ship_xyz(ship[28])));
 ship[32]=make_float4(rel.x,rel.y,rel.z,sqrtf(dotv(rel,rel)));ship[41]=make_float4(e.x,e.y,e.z,0);ship[42]=make_float4(n.x,n.y,n.z,0);ship[43]=make_float4(b.x,b.y,b.z,0);
 float3 eye=outpost_inverse(ship,minus(ship_axis(ship,ship_xyz(ship[7])),rel));ship[40]=make_float4(eye.x,eye.y,eye.z,1);
}
__device__ float outpost_ground(const float4 *camera,const float4 *ship,float x,float z){
 float3 normal=to_world(camera,unit(vec(x,earth_radius()+ship[0].y,z)));float h=terrain_height(camera,normal);
 float3 local=outpost_inverse(ship,minus(vec(x,0,z),ship_xyz(ship[32])));
 if(fabsf(local.x)<32&&local.z>-39&&local.z<17)h=fmaxf(h,ship[28].w+.4f);
 return h;
}
__device__ int outpost_blocked(float3 p){
 if(p.x>-27.5f&&p.x<-7&&p.z>-37&&p.z<-22)return 1;
 if(p.x>9&&p.x<24&&p.z>-36&&p.z<-23)return 1;
 if(p.x>24&&p.x<28&&p.z>-23&&p.z<-19)return 1;
 if(p.x>19&&p.x<25&&p.z>-10&&p.z<-5)return 1;
 if(p.x<-23&&p.x>-30&&p.z>-15&&p.z<-3)return 1;
 if(dotv(vec(p.x+3,0,p.z+18),vec(p.x+3,0,p.z+18))<.55f)return 1;
 if(dotv(vec(p.x-10,0,p.z+18),vec(p.x-10,0,p.z+18))<.55f)return 1;
 return 0;
}
__device__ int outpost_landing_blocked(float3 p){
 // Expanded aircraft footprint, separate from the pilot's shoulder clearance.
 if(p.x>-35&&p.x<1&&p.z>-44&&p.z<-14)return 1;
 if(p.x>1&&p.x<32&&p.z>-44&&p.z<-15)return 1;
 if(p.x>17&&p.x<35&&p.z>-30&&p.z<-11)return 1;
 if(p.x>11&&p.x<33&&p.z>-18&&p.z<3)return 1;
 if(p.x<-15&&p.x>-38&&p.z>-23&&p.z<5)return 1;
 return 0;
}
__global__ void outpost_init(float4 *camera,float2 *navigationState,float4 *ship,int startParked){
 if(ship[21].w==62)return;
 float3 n=vec(navigationState[0].x,navigationState[1].x,navigationState[2].x),east=vec(navigationState[3].x,navigationState[4].x,navigationState[5].x),back=crossv(east,n),chosen=n;float score=1e20f;
 // Search the daylight coast already selected by geology_visit. A single
 // initialization query chooses land; no terrain generation is copied here.
 for(int i=0;i<128;i++){
  float angle=(float)i*2.399963f,dist=120+sqrtf((float)i)*260;float3 q=unit(plus(scale(n,earth_radius()),plus(scale(east,cosf(angle)*dist),scale(back,sinf(angle)*dist))));
  float h=terrain_direct(camera,q);if(h<3||h>220)continue;
  float hx=terrain_direct(camera,unit(plus(q,scale(east,32/earth_radius())))),hz=terrain_direct(camera,unit(plus(q,scale(back,32/earth_radius()))));
  float cost=fabsf(h-18)*.35f+(fabsf(hx-h)+fabsf(hz-h))*12+dist*.001f;if(cost<score){score=cost;chosen=q;}
 }
 if(score==1e20f){ // Wider deterministic fallback for unusual saved seeds.
  for(int i=0;i<96;i++){float a=(float)i*2.399963f,d=1000+sqrtf((float)i)*2000;float3 q=unit(plus(scale(n,earth_radius()),plus(scale(east,cosf(a)*d),scale(back,sinf(a)*d))));float h=terrain_direct(camera,q);if(h>5&&h<2000&&h<score){score=h;chosen=q;}}
 }
 east=unit(minus(east,scale(chosen,dotv(east,chosen))));back=crossv(east,chosen);
 // Normalize the anchor in double precision, then preserve the residual.
 double nx=(double)chosen.x,ny=(double)chosen.y,nz=(double)chosen.z,inv=1.0/sqrt(nx*nx+ny*ny+nz*nz);nx*=inv;ny*=inv;nz*=inv;
 ship[28]=make_float4((float)nx,(float)ny,(float)nz,0);ship[29]=make_float4((float)(nx-(double)(float)nx),(float)(ny-(double)(float)ny),(float)(nz-(double)(float)nz),0);ship[30]=make_float4(east.x,east.y,east.z,0);
 float deck=terrain_direct(camera,chosen);
 for(int i=0;i<9;i++){float x=(float)(i%3-1)*32,z=(float)(i/3)*28-39;float3 q=unit(plus(scale(chosen,earth_radius()),plus(scale(east,x),scale(back,z))));deck=fmaxf(deck,terrain_direct(camera,q));}
 ship[28].w=deck+.3f;ship[21]=make_float4(startParked!=0?2:0,0,0,62);ship[33]=make_float4(1500,500,20,0);ship[27]=make_float4(0,-1,0,0);ship[38]=make_float4(0,0,0,0);
 if(startParked!=0){
  navigationState[0]=make_float2(ship[28].x,ship[29].x);navigationState[1]=make_float2(ship[28].y,ship[29].y);navigationState[2]=make_float2(ship[28].z,ship[29].z);navigationState[3]=make_float2(east.x,0);navigationState[4]=make_float2(east.y,0);navigationState[5]=make_float2(east.z,0);navigationState[6]=make_float2(0,0);navigationState[7]=make_float2(0,0);
  ship[0]=make_float4(0,ship[28].w+2.45f,0,1);ship[1]=make_float4(0,0,0,0);ship[8]=make_float4(0,0,0,0);ship[9]=make_float4(0,0,0,0);ship[10]=make_float4(0,0,22,0);ship[2]=make_float4(.32f,.36f,22,0);ship[22].x=ship[0].y;camera[5].w=-1000;
 }
}
__global__ void outpost_step(float4 *camera,float2 *navigationState,float4 *ship,float deltaTime,float forward,float turn,float rise,float lookX,float lookY,float speed,int boost,int action,int inspect,int flightProfile,float depth,float aspect,int command){
 if(ship[21].w!=62)return;
 float dt=fminf(.1f,fmaxf(0,deltaTime));int mode=(int)ship[21].x,justExited=0;ship[22].y=fmaxf(0,ship[22].y-dt);camera[20].z=mode==3?1:0;
 if(command==7)ship[38].x=0;
 if(command==3&&ship[38].x==2&&ship[21].y==0){ship[21].y=1;ship[21].z=0;ship[38].x=0;}
 if(command==4&&ship[38].x==2&&ship[21].y==6&&ship[27].w==0){ship[21].y=7;ship[27].x+=750;ship[27].w=1;ship[38].x=0;}
 if(command==5&&ship[38].x==3&&ship[21].y!=3&&ship[21].y!=5){ship[14]=make_float4(100,75,20,0);ship[15].z=4;ship[38].x=0;}
 if(command==6&&ship[38].x==2&&ship[21].y==8){ship[21].y=1;ship[21].z=0;ship[33]=make_float4(1500,500,20,0);ship[27].y=-1;ship[38].x=0;}
 if(ship[38].x>0)dt=0;
 if(action!=0){ship[21].x=0;mode=0;ship[38].x=0;}
 if(mode==0){
  outpost_frame(ship,camera,navigationState);
  // Restore the ship tangent before the landing probes, not the chase eye.
  float3 n=vec(navigationState[0].x,navigationState[1].x,navigationState[2].x),e=vec(navigationState[3].x,navigationState[4].x,navigationState[5].x),b=crossv(e,n);
  float h=terrain_height(camera,n),h1=terrain_height(camera,unit(plus(n,scale(e,6/earth_radius())))),h2=terrain_height(camera,unit(plus(n,scale(b,6/earth_radius()))));float slope=fmaxf(fabsf(h1-h),fabsf(h2-h))/6;
  float3 p=outpost_inverse(ship,scale(ship_xyz(ship[32]),-1));int pad=fabsf(p.x)<10&&fabsf(p.z)<9?1:0;if(pad!=0){h=ship[28].w+.4f;slope=0;}
  int reason=h<.5f?1:slope>.32f?2:ship[9].w>22?3:ship[0].y-h>48?4:outpost_landing_blocked(p)!=0?5:0;
  ship[39]=make_float4((float)reason,ship[0].y-h,slope,ship[9].w);
  if(command==1){if(reason==0){ship[21].x=1;ship[22].x=h+2.05f;ship[9]=make_float4(0,0,0,0);ship[1].w=0;}else{ship[22].y=4;ship[22].z=(float)reason;}}
 }else{
  float3 n=unit(vec(navigationState[0].x,navigationState[1].x,navigationState[2].x)),e=unit(vec(navigationState[3].x,navigationState[4].x,navigationState[5].x)),b=crossv(e,n);
  camera[5]=make_float4(e.x,e.y,e.z,camera[5].w);camera[6]=make_float4(n.x,n.y,n.z,0);camera[7]=make_float4(b.x,b.y,b.z,camera[7].w);
  ship[1].y=0;ship[1].z=0;float yaw=ship[1].x;float3 right=vec(cosf(yaw),0,sinf(yaw)),back=vec(-sinf(yaw),0,cosf(yaw));ship[4]=make_float4(right.x,0,right.z,0);ship[5]=make_float4(0,1,0,0);ship[6]=make_float4(back.x,0,back.z,0);ship[9]=make_float4(0,0,0,0);ship[17]=make_float4(0,0,0,0);ship[3].x+=dt;
  outpost_frame(ship,camera,navigationState);
  if(mode==1){ship[0].y=fmaxf(ship[22].x,ship[0].y-dt*10);if(ship[0].y<=ship[22].x+.01f){ship[21].x=2;mode=2;}ship[3].y=.25f;}
  if(mode==4){ship[0].y+=dt*8;ship[3].y=.6f;if(ship[0].y>=ship[22].x+14){ship[21].x=0;ship[8]=make_float4(yaw,0,0,0);}}
  if(mode==2){ship[3].y=0;if(command==1){ship[21].x=4;}if(command==2){float3 exit=ship_axis(ship,vec(-2.7f,0,-8.2f));ship[24]=make_float4(exit.x,0,exit.z,0);ship[25]=make_float4(yaw,0,0,0);ship[21].x=3;mode=3;justExited=1;}}
  float3 offset,f,r,u;
  if(mode==3){
   ship[3].y=0;ship[25].x=ship_angle(ship[25].x+lookX);ship[25].y=fminf(1.45f,fmaxf(-1.45f,ship[25].y+lookY));
   float walk=(boost!=0?6.5f:3.8f)*dt,len=fmaxf(1,sqrtf(forward*forward+turn*turn)),a=ship[25].x;float dx=(sinf(a)*forward+cosf(a)*turn)*walk/len,dz=(-cosf(a)*forward+sinf(a)*turn)*walk/len;
   float3 candidate=vec(ship[24].x+dx,0,ship[24].z+dz),local=outpost_inverse(ship,minus(candidate,ship_xyz(ship[32])));float h=outpost_ground(camera,ship,candidate.x,candidate.z),old=outpost_ground(camera,ship,ship[24].x,ship[24].z);
   float3 body=ship_inverse(ship,candidate);int hull=fabsf(body.x)<7.4f&&body.z>-6.6f&&body.z<6.3f?1:0;
   if(h>.1f&&fabsf(h-old)<.6f+sqrtf(dx*dx+dz*dz)*.6f&&outpost_blocked(local)==0&&hull==0){ship[24].x=candidate.x;ship[24].z=candidate.z;}
   h=outpost_ground(camera,ship,ship[24].x,ship[24].z);ship[24].y=h+1.72f-ship[0].y;
   offset=ship_xyz(ship[24]);f=ship_heading(ship[25].x,ship[25].y);r=vec(cosf(a),0,sinf(a));u=crossv(r,f);
   local=outpost_inverse(ship,minus(vec(offset.x,0,offset.z),ship_xyz(ship[32])));float3 board=ship_axis(ship,vec(-2.7f,0,-8.2f));float bd=sqrtf((offset.x-board.x)*(offset.x-board.x)+(offset.z-board.z)*(offset.z-board.z));float cd=sqrtf((local.x+3)*(local.x+3)+(local.z+18)*(local.z+18)),ed=sqrtf((local.x-10)*(local.x-10)+(local.z+18)*(local.z+18));
   ship[22].w=cd<3?2:ed<3?3:bd<4?1:0;
   if(command==2&&ship[38].x==0&&justExited==0){if(ship[22].w==1){ship[21].x=2;ship[22].w=0;}else if(ship[22].w>1)ship[38].x=ship[22].w;}
  }else{ship[22].w=mode==2?1:0;{ship[2].x+=lookX;ship[2].y=fminf(1.3f,fmaxf(.08f,ship[2].y-lookY));}float a=yaw+ship[2].x,el=ship[2].y,d=ship[2].z*fmaxf(1,.88f/aspect);offset=vec(-sinf(a)*cosf(el)*d,sinf(el)*d,cosf(a)*cosf(el)*d);f=unit(minus(vec(0,.5f,0),offset));r=unit(crossv(f,vec(0,1,0)));u=crossv(r,f);}
  ship_camera(camera,navigationState,ship,offset,f,r,u,depth);camera[8].w=0;outpost_frame(ship,camera,navigationState);
 }
 int stage=(int)ship[21].y;if(flightProfile!=0){
  if(stage==1&&ship[21].x==0){ship[21].y=2;ship[21].z=12;}
  if(stage==2||stage==4){ship[21].z=fmaxf(0,ship[21].z-dt);if(ship[21].z==0){ship[21].y=stage==2?3:5;ship[34].x=stage==2?3:4;}}
 }
 float3 to=minus(ship_xyz(ship[32]),ship_axis(ship,ship_xyz(ship[7])));float z=dotv(to,ship_xyz(camera[2]));ship[36]=make_float4(dotv(to,ship_xyz(camera[3]))/fmaxf(.01f,z)/(.65f*aspect),-dotv(to,ship_xyz(camera[4]))/fmaxf(.01f,z)/.65f,ship[32].w,z>0?1:0);
 for(int i=0;i<2;i++){float3 point=minus(plus(ship_xyz(ship[32]),outpost_axis(ship,vec(i==0?-3:10,2.38f,-18))),ship_axis(ship,ship_xyz(ship[7])));float d=dotv(point,ship_xyz(camera[2]));ship[44+i]=make_float4(dotv(point,ship_xyz(camera[3]))/fmaxf(.01f,d)/(.65f*aspect),-dotv(point,ship_xyz(camera[4]))/fmaxf(.01f,d)/.65f,sqrtf(dotv(point,point)),d>0?1:0);}
 camera[20].z=ship[21].x==3?1:0;
}

// 128 bounded analytic parts, one shared binary BVH. No additional imported
// meshes/textures and no per-pixel terrain or mission work in the browser.
__device__ float4 outpost_part(int id,int size){
 float3 c=vec(0,-3,-11),s=vec(32,3.4f,28);int mat=0,shape=0;
 if(id==1){c=vec(0,.44f,0);s=vec(12,.04f,11);mat=1;}
 if(id>=2&&id<=5){int k=id-2;c=vec(k<2?(k==0?-12:12):0,.50f,k<2?0:(k==2?-11:11));s=vec(k<2?.08f:12,.03f,k<2?11:.08f);mat=8;}
 if(id>=6&&id<=9){int k=id-6;c=vec(k<2?(k==0?-2:2):0,.505f,k<2?0:(k==2?-2:2));s=vec(k<2?.18f:2,.025f,k<2?2:.18f);mat=2;}
 if(id>=10&&id<16){int k=id-10;c=vec(-30+(float)k*12,.65f,16);s=vec(.28f,.25f,.28f);mat=8;}
 if(id>=16&&id<32){int k=id-16;float side=k<8?-1:1;int j=k%8;c=vec(side*31.5f,.7f,-36+(float)j*7);s=vec(.2f,.3f,.25f);mat=8;}
 if(id==32){c=vec(-17,3.25f,-29);s=vec(10,2.85f,7);mat=3;}
 if(id==33){c=vec(-17,6.25f,-29);s=vec(10.5f,.2f,7.5f);mat=1;}
 if(id==34){c=vec(-17,3.85f,-21.91f);s=vec(7.7f,1.05f,.10f);mat=4;}
 if(id==35){c=vec(-17,2.8f,-21.76f);s=vec(8,.06f,.10f);mat=8;}
 if(id==36){c=vec(-9,2,-21.70f);s=vec(.95f,1.6f,.2f);mat=1;}
 if(id==37){c=vec(-9,3.5f,-21.45f);s=vec(.75f,.06f,.08f);mat=8;}
 if(id>=38&&id<42){c=vec(-24+(float)(id-38)*4.7f,3.35f,-21.63f);s=vec(.10f,2.85f,.15f);mat=1;}
 if(id>=42&&id<46){c=vec(-24+(float)(id-42)*4.5f,6.6f,-30);s=vec(1.3f,.2f,2.1f);mat=5;}
 if(id==46){c=vec(16.5f,2.7f,-29.5f);s=vec(7,2.3f,6);mat=3;}
 if(id==47){c=vec(16.5f,5.2f,-29.5f);s=vec(7.5f,.2f,6.5f);mat=1;}
 if(id==48){c=vec(16.5f,2.55f,-23.4f);s=vec(4.6f,1.85f,.12f);mat=5;}
 if(id==49){c=vec(16.5f,4.6f,-23.15f);s=vec(4.6f,.10f,.14f);mat=9;}
 if(id>=50&&id<54){c=vec(11+(float)(id-50)*3.7f,2.7f,-23.1f);s=vec(.12f,2.3f,.2f);mat=1;}
 if(id==54){c=vec(26,7,-21);s=vec(.32f,6.6f,.32f);mat=1;}
 if(id==55){c=vec(26,12,-21);s=vec(3.5f,.16f,.2f);mat=2;}
 if(id==56){c=vec(26,13.6f,-21);s=vec(.28f,.25f,.28f);mat=9;shape=1;}
 if(id==57){c=vec(26,2,-21);s=vec(1.3f,1.6f,1.3f);mat=3;}
 if(id>=58&&id<62){c=vec(-26.5f,1.7f,-12+(float)(id-58)*2.8f);s=vec(3,.12f,1.1f);mat=6;}
 if(id>=62&&id<66){c=vec(-26.5f,.9f,-12+(float)(id-62)*2.8f);s=vec(.12f,.5f,.12f);mat=1;}
 if(id>=66&&id<70){int k=id-66;c=vec(21+(float)(k%2)*2,1.2f,-8+(float)(k/2)*2);s=vec(.85f,.8f,.85f);mat=k==3?2:1;}
 if(id>=70&&id<74){int k=id-70;c=vec(k<2?-6:7,1.1f,-21+(float)(k%2)*.8f);s=vec(.5f,.7f,.25f);mat=3;}
 if(id>=74&&id<78){c=vec(-3+(float)(id-74)*4.4f,.48f,-16);s=vec(1,.04f,.12f);mat=2;}
 if(id>=78&&id<82){int k=id-78;c=vec(-26+(float)k*17,1.4f,-38);s=vec(.10f,1,.12f);mat=1;}
 if(id==82){c=vec(-.5f,2.1f,-38);s=vec(25.5f,.06f,.06f);mat=1;}
 if(id>=83&&id<88){c=vec(-23+(float)(id-83)*3,1.6f,-36.2f);s=vec(.8f,.8f,.3f);mat=5;}
 if(id>=88&&id<128){
  int person=(id-88)/20,k=(id-88)%20;float x=person==0?-3:10;mat=person==0?7:2;shape=1;c=vec(x,1.5f,-18);s=vec(.255f,.36f,.18f);shape=0;
  if(k==1){c.y=1.17f;s=vec(.23f,.17f,.17f);mat=1;}
  if(k==2){c.y=2.0f;s=vec(.22f,.24f,.22f);mat=3;shape=1;}
  if(k==3){c=vec(x,2.02f,-17.81f);s=vec(.18f,.12f,.07f);mat=4;shape=1;}
  if(k>=4&&k<=7){float side=k%2==0?-1:1;c=vec(x+side*.36f,k<6?1.58f:1.27f,-18);s=vec(.105f,k<6?.21f:.18f,.12f);shape=k<6?1:0;}
  if(k==8||k==9){c=vec(x+(k==8?-.37f:.37f),1.04f,-17.97f);s=vec(.095f,.10f,.1f);mat=1;}
  if(k==10||k==11){c=vec(x+(k==10?-.13f:.13f),.90f,-18);s=vec(.11f,.22f,.12f);mat=1;}
  if(k==12||k==13){c=vec(x+(k==12?-.13f:.13f),.62f,-18);s=vec(.10f,.19f,.11f);}
  if(k==14){c=vec(x,1.55f,-18.23f);s=vec(.22f,.30f,.12f);mat=1;}
  if(k==16||k==17){c=vec(x+(k==16?-.13f:.13f),.48f,-17.94f);s=vec(.12f,.08f,.20f);mat=1;}
  if(k==18||k==19){c=vec(x+(k==18?-.13f:.13f),.76f,-17.87f);s=vec(.085f,.105f,.035f);mat=3;}
  if(k==15){c=vec(x,1.62f,-17.805f);s=vec(.12f,.07f,.025f);mat=8;shape=0;}
 }

 return size!=0?make_float4(s.x,s.y,s.z,(float)mat):make_float4(c.x,c.y,c.z,(float)shape);
}
__global__ void outpost_build(float4 *parts,float4 *bounds){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=128)return;float4 c=outpost_part(i,0),s=outpost_part(i,1);parts[i*2]=c;parts[i*2+1]=s;float3 lo=minus(ship_xyz(c),ship_xyz(s)),hi=plus(ship_xyz(c),ship_xyz(s));bounds[(127+i)*2]=make_float4(lo.x,lo.y,lo.z,0);bounds[(127+i)*2+1]=make_float4(hi.x,hi.y,hi.z,0);}
__global__ void outpost_bounds(float4 *bounds,int level){int i=blockIdx.x*blockDim.x+threadIdx.x,count=128>>level;if(i>=count)return;int node=count-1+i;float3 lo=ship_min(ship_xyz(bounds[(node*2+1)*2]),ship_xyz(bounds[(node*2+2)*2])),hi=ship_max(ship_xyz(bounds[(node*2+1)*2+1]),ship_xyz(bounds[(node*2+2)*2+1]));bounds[node*2]=make_float4(lo.x,lo.y,lo.z,0);bounds[node*2+1]=make_float4(hi.x,hi.y,hi.z,0);}
struct OutpostHit{float t;float3 normal;int part;};
__device__ OutpostHit outpost_shape(float3 origin,float3 ray,float4 centre,float4 size){
 OutpostHit h;h.t=1e20f;h.normal=vec(0,1,0);h.part=-1;float3 p=minus(origin,ship_xyz(centre)),s=ship_xyz(size);
 if(centre.w==1){float3 o=vec(p.x/s.x,p.y/s.y,p.z/s.z),d=vec(ray.x/s.x,ray.y/s.y,ray.z/s.z);float a=dotv(d,d),b=dotv(o,d),c=dotv(o,o)-1,disc=b*b-a*c;if(disc<0)return h;float t=(-b-sqrtf(disc))/a;if(t<.002f)t=(-b+sqrtf(disc))/a;if(t<.002f)return h;h.t=t;float3 q=plus(p,scale(ray,t));h.normal=unit(vec(q.x/(s.x*s.x),q.y/(s.y*s.y),q.z/(s.z*s.z)));return h;}
 float enter=-1e20f,leave=1e20f;float3 normal=vec(0,1,0);
 // Clip a box against twelve edge bevel planes for solid machined silhouettes.
 for(int k=0;k<18;k++){float3 n=vec(0,0,0);float d=0;
  if(k<6){int a=k/2;float sign=k%2==0?-1:1;n=a==0?vec(sign,0,0):a==1?vec(0,sign,0):vec(0,0,sign);d=a==0?s.x:a==1?s.y:s.z;}
  else{int pair=(k-6)/4,j=(k-6)%4;float a=j%2==0?-1:1,b=j/2==0?-1:1;float bevel=fminf(.12f,fminf(s.x,fminf(s.y,s.z))*.23f);n=pair==0?vec(a,b,0):pair==1?vec(a,0,b):vec(0,a,b);d=(pair==0?s.x+s.y:pair==1?s.x+s.z:s.y+s.z)-bevel;}
  float den=dotv(n,ray),num=d-dotv(n,p);if(fabsf(den)<.0000001f){if(num<0)return h;continue;}float t=num/den;if(den<0){if(t>enter){enter=t;normal=unit(n);}}else leave=fminf(leave,t);if(enter>leave)return h;
 }
 if(enter>.002f){h.t=enter;h.normal=normal;}return h;
}
__device__ OutpostHit outpost_trace(const float4 *parts,const float4 *bounds,float3 o,float3 r,float limit,int shadow){
 OutpostHit hit;hit.t=limit;hit.part=-1;hit.normal=vec(0,1,0);float3 inv=vec(1/(fabsf(r.x)<.0000001f?.0000001f:r.x),1/(fabsf(r.y)<.0000001f?.0000001f:r.y),1/(fabsf(r.z)<.0000001f?.0000001f:r.z));int node=0;
 while(node>=0){if(ship_box_hit(bounds,node,o,inv,hit.t)==0){node=ship_next(node,2);continue;}if(node<127){node=node*2+1;continue;}int i=node-127;OutpostHit h=outpost_shape(o,r,parts[i*2],parts[i*2+1]);if(h.t<hit.t){hit=h;hit.part=i;if(shadow!=0)return hit;}node=ship_next(node,2);}return hit;
}
__global__ void outpost_render(const float4 *parts,const float4 *bounds,const float4 *camera,const float4 *ship,float *hits,unsigned *image,int width,int height,int samples,float exposure){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height||ship[21].w!=62)return;
 float3 o=ship_xyz(ship[40]),f=outpost_inverse(ship,ship_xyz(camera[2])),r=outpost_inverse(ship,ship_xyz(camera[3])),u=outpost_inverse(ship,ship_xyz(camera[4]));
 float3 centre=unit(plus(f,plus(scale(r,(2*((float)x+.5f)-(float)width)/(float)height*.65f),scale(u,(1-2*((float)y+.5f)/(float)height)*.65f))));
 float3 inv=vec(1/(fabsf(centre.x)<.0000001f?.0000001f:centre.x),1/(fabsf(centre.y)<.0000001f?.0000001f:centre.y),1/(fabsf(centre.z)<.0000001f?.0000001f:centre.z));
 int station=ship[32].w<18000?ship_box_hit(bounds,0,o,inv,18000):0;int gear=0;
 if(ship[21].x!=0){float3 go=ship_xyz(ship[7]),gr=ship_inverse(ship,outpost_axis(ship,centre));OutpostHit box=outpost_shape(go,gr,make_float4(0,-1.5f,-.5f,0),make_float4(5.2f,.65f,4.5f,1));gear=box.t<18000?1:0;}
 if(station==0&&gear==0)return;
 float3 sum=vec(0,0,0),bg=ship_decode(image[y*width+x]),sun=unit(outpost_inverse(ship,ship_xyz(camera[9])));int covered=0;
 for(int sample=0;sample<4;sample++){
  if(sample>=samples)break;float ox=samples==1?.5f:((float)(sample%2)+.5f)*.5f,oy=samples==1?.5f:((float)(sample/2)+.5f)*.5f;
  float3 ray=unit(plus(f,plus(scale(r,(2*((float)x+ox)-(float)width)/(float)height*.65f),scale(u,(1-2*((float)y+oy)/(float)height)*.65f))));float terrain=terrain_pixel_hit(hits,camera,width,height,(float)x+ox,(float)y+oy,0),limit=terrain>0?terrain:1000000;
  OutpostHit h;h.t=limit;h.part=-1;h.normal=vec(0,1,0);if(station!=0)h=outpost_trace(parts,bounds,o,ray,limit,0);int mat=h.part>=0?(int)parts[h.part*2+1].w:1;
  if(gear!=0){float3 go=ship_xyz(ship[7]),gr=ship_inverse(ship,outpost_axis(ship,ray));for(int i=0;i<6;i++){int leg=i/2;float lx=leg==0?0:leg==1?-4.6f:4.6f,lz=leg==0?-4.3f:3.1f;float4 c=make_float4(lx,i%2==0?-1.5f:-1.97f,lz,0),s=i%2==0?make_float4(.12f,.5f,.16f,1):make_float4(.48f,.08f,.58f,1);OutpostHit g=outpost_shape(go,gr,c,s);if(g.t<h.t){h=g;h.part=128;h.normal=outpost_inverse(ship,ship_axis(ship,g.normal));mat=1;}}}
  if(h.part<0){sum=plus(sum,bg);continue;}covered++;float3 p=plus(o,scale(ray,h.t)),n=h.normal;float footprint=h.t*1.3f/(float)height;
  float3 colour=mat==0?vec(.12f,.15f,.17f):mat==1?vec(.035f,.055f,.07f):mat==2?vec(.60f,.24f,.045f):mat==3?vec(.42f,.48f,.49f):mat==4?vec(.022f,.085f,.11f):mat==5?vec(.09f,.12f,.13f):mat==6?vec(.016f,.04f,.12f):mat==7?vec(.045f,.24f,.29f):mat==8?vec(.12f,.9f,1):vec(1,.28f,.035f);
  float3 q=vec(fabsf(n.x),fabsf(n.y),fabsf(n.z));float a=q.y>.7f?p.x:p.x+p.z,b=q.y>.7f?p.z:p.y;
  float seam=fmaxf(ship_line(fract(a*.5f)-.5f,.012f,footprint*.5f),ship_line(fract(b*.5f)-.5f,.012f,footprint*.5f));if((mat==0||mat==3)&&h.part<88)colour=scale(colour,1-seam*.45f);
  if(mat==5)colour=scale(colour,.5f+.5f*(1-ship_line(fract(p.y*8)-.5f,.24f,footprint*8)));
  if(mat==6){float grid=fmaxf(ship_line(fract(a*3)-.5f,.025f,footprint*3),ship_line(fract(b*3)-.5f,.025f,footprint*3));colour=blend(colour,vec(.16f,.22f,.32f),grid);}
  float nd=fmaxf(0,dotv(n,sun)),shadow=1;if(nd>0&&h.part<128){OutpostHit sh=outpost_trace(parts,bounds,plus(p,scale(n,.012f)),sun,200,1);if(sh.part>=0)shadow=.16f;}
  if(n.y>.8f&&ship[21].x!=0&&sun.y>.05f){float3 sp=ship_inverse(ship,plus(ship_xyz(ship[32]),outpost_axis(ship,p))),ss=ship_inverse(ship,outpost_axis(ship,sun));float t=-sp.y/fmaxf(.05f,ss.y);float3 q=plus(sp,scale(ss,t));float hull=(1-eased(1.35f,1.75f,fabsf(q.x)))*(1-eased(4.6f,5.2f,fabsf(q.z+.7f))),engine=(1-eased(1.05f,1.4f,fabsf(fabsf(q.x)-4.65f)))*(1-eased(4.1f,4.5f,fabsf(q.z-.5f))),wing=(1-eased(5,5.5f,fabsf(q.x)))*(1-eased(1.3f,1.6f,fabsf(q.z-1)));if(t>0&&t<100)shadow*=1-fmaxf(hull,fmaxf(engine,wing))*.75f;}
  float daylight=eased(-.12f,.15f,camera[9].y);float3 result=scale(colour,.10f+daylight*(.18f+.13f*(n.y*.5f+.5f)+nd*shadow*.95f));
  if(h.part>=88&&h.part<128){float fres=positive_power(1-fmaxf(0,dotv(n,scale(ray,-1))),3);float3 halfv=unit(minus(sun,ray));float shine=positive_power(fmaxf(0,dotv(n,halfv)),mat==4?64:18)*shadow*daylight;result=plus(result,scale(vec(.3f,.48f,.56f),fres*.24f+shine*.4f));if(mat==2||mat==7){float weave=sinf(p.x*155)*sinf(p.y*190)*.04f*(1-eased(.005f,.015f,footprint));result=scale(result,1+weave);}}

  if(mat==4||mat==6){float fres=positive_power(1-fmaxf(0,dotv(n,scale(ray,-1))),4);result=plus(result,scale(vec(.25f,.42f,.55f),(.18f+fres*.6f)*daylight));}
  // Warm apron lamps and cool suit/display emission retain readable silhouettes at night.
  float lamp=expf(-fabsf(p.y-1)*.24f)/(1+(p.x*p.x+(p.z+16)*(p.z+16))*.006f);result=plus(result,scale(colour,lamp*.3f));
  if(mat>=8)result=scale(colour,1.4f);sum=plus(sum,ship_decode(pack_color(scale(result,exposure))));
  if(camera[21].w!=0)hits[(int)camera[26].x+sample*width*height+y*width+x]=h.t;
 }
 if(covered>0){float3 c=scale(sum,1/(float)samples);image[y*width+x]=(unsigned)(clamp01(c.x)*255)|((unsigned)(clamp01(c.y)*255)<<8)|((unsigned)(clamp01(c.z)*255)<<16)|4278190080u;}
}
