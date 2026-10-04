// Combat simulation, targeting, collisions and effects live on the GPU.
// A bounded pool and screen tiles keep work independent of flight distance.
// combat[0..15] telemetry/frame, [32..191] five 32-float4 enemy records,
// [256..511] 64 swept projectiles, [544..607] 16 transient effects.
__device__ int combat_enemy(int i){return 32+i*32;}
__device__ int combat_round(int i){return 256+i*4;}
__device__ int combat_fx(int i){return 544+i*4;}
__device__ float3 combat_rotate(float3 p,const float4 *state,const float4 *camera){
 float3 world=plus(scale(ship_xyz(state[5]),p.x),plus(scale(ship_xyz(state[6]),p.y),scale(ship_xyz(state[7]),p.z)));
 return vec(dotv(world,ship_xyz(camera[5])),dotv(world,ship_xyz(camera[6])),dotv(world,ship_xyz(camera[7])));
}
__device__ float combat_sweep(float3 a,float3 b,float radius){
 float3 d=minus(b,a);float c=dotv(a,a)-radius*radius;if(c<=0)return 0;float aa=dotv(d,d),bb=dotv(a,d),disc=bb*bb-aa*c;
 if(aa<.000001f||disc<0)return -1;float t=(-bb-sqrtf(disc))/aa;return t>=0&&t<=1?t:-1;
}
__device__ void combat_effect(float4 *state,float3 a,float3 b,int type,float3 colour,float strength){
 int slot=(int)state[11].x%16;state[11].x+=1;int f=combat_fx(slot);float life=type==1?.13f:type==2?1.2f:type==4?.07f:.32f;
 state[f]=make_float4(a.x,a.y,a.z,life);state[f+1]=make_float4(b.x,b.y,b.z,(float)type);state[f+2]=make_float4(colour.x,colour.y,colour.z,strength);state[f+3]=make_float4(life,0,0,0);
}
__device__ void combat_damage(float4 *v,float amount){float shield=fminf(v[14].y,amount);v[14].y-=shield;v[14].x=fmaxf(0,v[14].x-(amount-shield));v[14].z=0;v[14].w=1;}
__device__ void combat_repair(float4 *v,float dt,float maxHull,float maxShield){
 v[20].x=fmaxf(0,v[20].x-dt*18);v[20].y=fmaxf(0,v[20].y-dt*18);
 if(v[14].x<=0)return;v[14].z+=dt;v[14].w=fmaxf(0,v[14].w-dt*3.5f);
 if(v[14].z>4)v[14].y=fminf(maxShield,v[14].y+dt*12);
 if(v[14].z>10)v[14].x=fminf(maxHull,v[14].x+dt*2);
}
__device__ void combat_projectile(float4 *state,float3 p,float3 velocity,int type,int owner,int target,float damage){
 for(int i=0;i<64;i++){int j=combat_round(i);if(state[j].w>0)continue;state[j]=make_float4(p.x,p.y,p.z,type==2?8:2.4f);state[j+1]=make_float4(velocity.x,velocity.y,velocity.z,(float)type);state[j+2]=make_float4(p.x,p.y,p.z,(float)owner);state[j+3]=make_float4((float)target,damage,0,0);return;}
}
__device__ void combat_basis(float4 *enemy,float3 nose,float3 eye){
 nose=unit(nose);float3 right=crossv(nose,vec(0,1,0));if(dotv(right,right)<.001f)right=vec(1,0,0);right=unit(right);float3 back=scale(nose,-1),up=crossv(back,right);
 enemy[4]=make_float4(right.x,right.y,right.z,0);enemy[5]=make_float4(up.x,up.y,up.z,0);enemy[6]=make_float4(back.x,back.y,back.z,0);
 float3 local=ship_inverse(enemy,minus(eye,ship_xyz(enemy[9])));enemy[7]=make_float4(local.x,local.y,local.z,0);
}
__device__ void combat_spawn(float4 *state,const float4 *ship,int i){
 int e=combat_enemy(i);float side=i%2==0?1:-1;float3 rel=ship_axis(ship,vec(i==0?0:side*(80+(float)i*25), i==0?3:35+(float)i*14,-165-(float)i*70));
 rel.y=fmaxf(rel.y,20-ship[0].y);state[e+9]=make_float4(rel.x,rel.y,rel.z,0);state[e]=make_float4(0,0,0,1);state[e+8]=make_float4(0,0,0,7+(float)i*2);
 state[e+11]=make_float4(.022f,.028f,.039f,2);state[e+12]=make_float4(.56f,.016f,.024f,2);state[e+13]=make_float4(1,.025f,.009f,2);state[e+14]=make_float4(90,60,20,0);state[e+15]=make_float4((float)(i%3),0,0,0);state[e+16].w=2;state[e+20]=make_float4(0,0,0,0);
}
__device__ int combat_target(const float4 *state,float3 origin,float3 direction,float range,float cone){
 int target=-1;float best=1000;
 for(int i=0;i<5;i++){int e=combat_enemy(i);if(state[e].w==0||state[e+14].x<=0)continue;float3 to=minus(ship_xyz(state[e+9]),origin);float distance=sqrtf(dotv(to,to)),alignment=dotv(unit(to),direction),score=(1-alignment)*6+distance/8000;if(distance<range&&alignment>cone&&score<best){best=score;target=i;}}
 return target;
}
__device__ float3 combat_outpost(const float4 *ship){return plus(ship_xyz(ship[32]),outpost_axis(ship,vec(0,3,-18)));}
__device__ void combat_base_damage(float4 *ship,float amount){float shield=fminf(ship[33].y,amount);ship[33].y-=shield;ship[33].x=fmaxf(0,ship[33].x-amount+shield);ship[33].z=0;ship[33].w=1;}
__device__ void combat_laser(float4 *state,float4 *ship,float3 start,float3 direction,int owner){
 float reach=1600,nearest=1;int victim=-2;float3 end=plus(start,scale(direction,reach));
 if(owner<0){for(int i=0;i<5;i++){int e=combat_enemy(i);if(state[e].w==0||state[e+14].x<=0)continue;float3 centre=ship_xyz(state[e+9]);float t=combat_sweep(minus(start,centre),minus(end,centre),7);if(t>=0&&t<nearest){nearest=t;victim=i;}}}
 else if(ship[21].w==62&&state[combat_enemy(owner)+15].y==1){float3 base=combat_outpost(ship);float t=combat_sweep(minus(start,base),minus(end,base),24);if(t>=0){nearest=t;victim=-3;}}
 else if(ship[14].x>0){float t=combat_sweep(start,end,7);if(t>=0){nearest=t;victim=-1;}}
 end=plus(start,scale(direction,reach*nearest));float3 hue=owner<0?vec(.08f,.65f,1):vec(1,.04f,.015f);combat_effect(state,start,end,1,hue,1);
 if(victim==-3){combat_base_damage(ship,14);combat_effect(state,end,end,3,hue,1);}
 if(victim>=-1){if(victim<0)combat_damage(ship,9);else combat_damage(state+combat_enemy(victim),16);combat_effect(state,end,end,3,hue,1);state[10].w+=1;}
}
__global__ void combat_step(float4 *state,float4 *ship,const float4 *camera,float4 *brush,float deltaTime,int firing,int weapon,int enemyCount,int resetCombat,int fixture){
 int shoot=firing,count=enemyCount;
 int mission=ship[21].w==62?1:0,stage=(int)ship[21].y;int battle=mission!=0&&(stage==3||stage==5)?1:0;
 if(mission!=0)count=battle!=0?5:0;
 float dt=ship[38].x>0?0:fminf(.1f,fmaxf(0,deltaTime));if(mission!=0&&ship[21].x!=0)shoot=0;float3 player=ship_xyz(ship[0]),velocity=ship_xyz(ship[9]),eye=ship_axis(ship,ship_xyz(ship[7]));
 if(resetCombat!=0||state[0].w==0){
  for(int i=0;i<1536;i++)state[i]=make_float4(0,0,0,0);state[0].w=1;state[1].z=-1;
  state[5]=camera[5];state[6]=camera[6];state[7]=camera[7];
  if(ship[16].w!=61||resetCombat==2){ship[14]=make_float4(100,75,20,0);ship[15]=make_float4(0,0,4,0);ship[16]=make_float4(0,0,0,61);}
  state[0].z=ship[16].x;
  if(mission!=0)ship[27].y=-1;else for(int i=0;i<count;i++)combat_spawn(state,ship,i);
 }
 if(battle!=0&&ship[27].y!=(float)stage){
  ship[27].y=(float)stage;
  for(int i=0;i<5;i++){int e=combat_enemy(i);state[e].w=0;if((float)i<ship[34].x){combat_spawn(state,ship,i);float a=(float)i*1.6f;float3 rel=plus(combat_outpost(ship),outpost_axis(ship,vec(sinf(a)*350,100+25*(float)i,cosf(a)*350)));state[e+9]=make_float4(rel.x,rel.y,rel.z,0);}}
 }
 if(mission!=0){ship[33].z+=dt;ship[33].w=fmaxf(0,ship[33].w-dt*3);if(ship[33].z>8)ship[33].y=fminf(500,ship[33].y+dt*8);}
 state[0].x+=dt;float time=state[0].x;float3 travel=fixture!=0?scale(velocity,dt):ship_xyz(ship[17]),nose=scale(ship_xyz(ship[6]),-1);if(dt==0)travel=vec(0,0,0);
 combat_repair(ship,dt,100,75);ship[15].x=(float)weapon;ship[15].y-=dt;ship[15].w+=dt;if(ship[15].w>=3){ship[15].z=fminf(4,ship[15].z+1);ship[15].w-=3;}
 // Rebase the encounter with the local planet frame; fast pilots can leave it.
 for(int i=0;i<5;i++){
  int e=combat_enemy(i);if(i>=count){state[e].w=0;continue;}
  if(state[e].w==0){state[e+9].w-=dt;if(mission==0&&dt>0&&state[e+9].w<=0&&ship[9].w<(ship[0].y>80000?10000:500)&&fixture==0)combat_spawn(state,ship,i);else continue;}
  float3 rel=minus(combat_rotate(ship_xyz(state[e+9]),state,camera),travel),v=combat_rotate(ship_xyz(state[e+8]),state,camera);state[e+18]=make_float4(rel.x,rel.y,rel.z,0);
  combat_repair(state+e,dt,90,60);state[e+8].w-=dt;
  if(fixture==0&&dt>0){float phase=time*.13f+(float)i*2.1f;float3 desired=ship_axis(ship,vec(sinf(phase)*145,35+cosf(phase*.7f)*20,-190-cosf(phase)*50));desired.y=fmaxf(desired.y,25-player.y);
   if(battle!=0){desired=plus(combat_outpost(ship),outpost_axis(ship,vec(sinf(phase)*240,70+cosf(phase*.7f)*25,cosf(phase)*240)));}
   float3 aim=scale(minus(desired,rel),.55f);float cap=player.y>80000?800:135,len=sqrtf(dotv(aim,aim));if(len>cap)aim=scale(aim,cap/len);v=blend(v,aim,1-expf(-dt*1.7f));rel=plus(rel,scale(v,dt));
   float3 point=plus(player,rel);float floor=0;if(camera[21].w!=0&&player.y<20000)floor=fmaxf(0,terrain_height(camera,to_world(camera,unit(vec(point.x,earth_radius()+point.y,point.z)))));
   if(point.y<floor+18){rel.y=floor+18-player.y;v.y=fmaxf(0,v.y);}
  }
  if(mission==0&&dotv(rel,rel)>25000000){state[e].w=0;state[e+9].w=12;continue;}
  state[e+9]=make_float4(rel.x,rel.y,rel.z,state[e+9].w);state[e+8].x=v.x;state[e+8].y=v.y;state[e+8].z=v.z;float3 world=plus(player,rel);state[e]=make_float4(world.x,world.y,world.z,1);state[e+3]=make_float4(time,.38f,world.y,135);float3 targetPoint=battle!=0&&(i!=0||ship[32].w>1000)?combat_outpost(ship):vec(0,0,0);state[e+15].y=dotv(targetPoint,targetPoint)>0?1:0;combat_basis(state+e,minus(targetPoint,rel),eye);
  if(dt>0&&fixture==0&&ship[14].x>0&&state[e+14].x>0&&state[e+8].w<=0&&dotv(minus(rel,targetPoint),minus(rel,targetPoint))<640000){
   int kind=i%3,shot=(int)state[e+20].w,side=shot%2==0?-1:1;float3 muzzle=plus(rel,ship_axis(state+e,ship_weapon_mount(kind,side,(shot/2)%2))),aim=unit(minus(state[e+15].y==1?targetPoint:scale(velocity,.12f),muzzle));
   if(kind==1)combat_laser(state,ship,muzzle,aim,i);else combat_projectile(state,muzzle,plus(v,scale(aim,kind==2?230:600)),kind==2?2:0,i,state[e+15].y==1?-3:-1,kind==2?32:7);
   combat_effect(state,muzzle,muzzle,4,vec(1,.05f,.015f),1);if(side<0)state[e+20].x=1;else state[e+20].y=1;state[e+20].z=(float)kind;state[e+20].w+=1;
   state[e+8].w=kind==1?2.8f:kind==2?5.0f:.65f;
  }
 }
 int target=combat_target(state,vec(0,0,0),nose,1800,.90f);state[1].z=(float)target;
 if(dt>0&&shoot!=0&&ship[14].x>0&&ship[15].y<=0){
  int shot=(int)ship[20].w,side=shot%2==0?-1:1;float3 muzzle=ship_axis(ship,ship_weapon_mount(weapon,side,(shot/2)%2));int discharged=weapon!=2||(target>=0&&ship[15].z>=1)?1:0;
  if(weapon==1){combat_laser(state,ship,muzzle,nose,-1);ship[15].y+=.24f;state[10].y+=1;}
  else if(weapon==2){if(target>=0&&ship[15].z>=1){combat_projectile(state,muzzle,plus(velocity,scale(nose,170)),2,-1,target,48);ship[15].z-=1;ship[15].y+=.9f;state[10].z+=1;}else ship[15].y=0;}
  else{combat_projectile(state,muzzle,plus(velocity,scale(nose,1000)),0,-1,-2,8);ship[15].y+=.10f;state[10].x+=1;}
  if(discharged!=0){combat_effect(state,muzzle,muzzle,4,weapon==1?vec(.08f,.65f,1):vec(1,.48f,.09f),1);if(side<0)ship[20].x=1;else ship[20].y=1;ship[20].z=(float)weapon;ship[20].w+=1;}
 }else if(shoot==0)ship[15].y=fmaxf(0,ship[15].y);
 // Effects already in flight are rebased before integrating their swept paths.
 for(int i=0;i<64;i++){
  int j=combat_round(i);if(state[j].w<=0)continue;float3 a=ship_xyz(state[j]),v=ship_xyz(state[j+1]);int owner=(int)state[j+2].w,type=(int)state[j+1].w;
  // A new muzzle position is already in the current frame. Start its travel
  // next tick; existing rounds are rebased exactly once, then move by world v.
  if(state[j+3].w==0){state[j+3].w=1;continue;}
  a=minus(combat_rotate(a,state,camera),travel);v=combat_rotate(v,state,camera);
  int lock=(int)state[j+3].x;
  if(type==2){float3 destination=vec(0,0,0),targetVelocity=velocity;int hasTarget=lock==-1?1:0;if(lock==-3){destination=combat_outpost(ship);targetVelocity=vec(0,0,0);hasTarget=1;}if(lock>=0&&state[combat_enemy(lock)].w>0&&state[combat_enemy(lock)+14].x>0){destination=ship_xyz(state[combat_enemy(lock)+9]);targetVelocity=ship_xyz(state[combat_enemy(lock)+8]);hasTarget=1;}
   if(hasTarget!=0){float distance=sqrtf(dotv(minus(destination,a),minus(destination,a))),lead=fminf(.7f,distance/340);float3 desired=unit(minus(plus(destination,scale(minus(targetVelocity,velocity),lead)),a));float3 inherited=owner<0?velocity:ship_xyz(state[combat_enemy(owner)+8]);float3 relative=minus(v,inherited),direction=unit(blend(unit(relative),desired,1-expf(-dt*4.5f)));v=plus(inherited,scale(direction,340));}
  }
  float3 b=plus(a,scale(v,dt));float hit=2;int victim=-2;
  if(dt>0&&owner<0){for(int k=0;k<5;k++){int e=combat_enemy(k);if(state[e].w==0||state[e+14].x<=0)continue;float t=combat_sweep(minus(a,ship_xyz(state[e+18])),minus(b,ship_xyz(state[e+9])),type==2?8:7);if(t>=0&&t<hit){hit=t;victim=k;}}}
  else if(dt>0&&owner>=0&&lock==-3){float3 base=combat_outpost(ship);float t=combat_sweep(minus(a,base),minus(b,base),24);if(t>=0){hit=t;victim=-3;}}
  else if(dt>0&&ship[14].x>0){float t=combat_sweep(plus(a,travel),b,7);if(t>=0){hit=t;victim=-1;}}
  if(player.y+b.y<0&&dt>0){float t=(player.y+a.y)/fmaxf(.0001f,a.y-b.y);if(t>=0&&t<hit){hit=t;victim=-2;}}
  state[j].w-=dt;state[j+2]=make_float4(a.x,a.y,a.z,(float)owner);state[j+3].z+=dt;
  if(hit<=1){float3 point=plus(a,scale(minus(b,a),hit));if(victim>=0)combat_damage(state+combat_enemy(victim),state[j+3].y);else if(victim==-1)combat_damage(ship,state[j+3].y);else if(victim==-3)combat_base_damage(ship,state[j+3].y);
   if(type==2){for(int k=0;k<5;k++){int e=combat_enemy(k);if(owner>=0||k==victim||state[e].w==0)continue;float d=sqrtf(dotv(minus(ship_xyz(state[e+9]),point),minus(ship_xyz(state[e+9]),point)));if(d<22)combat_damage(state+e,24*(1-d/22));}}
   combat_effect(state,point,point,type==2?2:3,type==2?vec(1,.28f,.035f):owner<0?vec(1,.68f,.20f):vec(1,.035f,.01f),1);state[j].w=0;state[10].w+=1;
  }else{state[j].x=b.x;state[j].y=b.y;state[j].z=b.z;state[j+1].x=v.x;state[j+1].y=v.y;state[j+1].z=v.z;}
 }
 int alive=0;for(int i=0;i<5;i++){int e=combat_enemy(i);if(state[e].w==0)continue;if(state[e+14].x<=0){combat_effect(state,ship_xyz(state[e+9]),ship_xyz(state[e+9]),2,vec(1,.23f,.025f),2);state[e].w=0;state[e+9].w=12;state[0].z+=1;if(battle!=0)ship[34].x=fmaxf(0,ship[34].x-1);}else alive++;}
 if(battle!=0&&dt>0){if(ship[33].x<=0){ship[21].y=8;ship[21].z=0;}else if(ship[34].x==0){ship[21].y=stage==3?4:6;ship[21].z=stage==3?8:0;}}
 brush[12]=make_float4(0,0,0,0);int lightCount=0;
 for(int i=0;i<16;i++){int f=combat_fx(i);if(state[f].w<=0)continue;float3 a=ship_xyz(state[f]),b=ship_xyz(state[f+1]);if(state[f+3].y>0){a=minus(combat_rotate(a,state,camera),travel);b=minus(combat_rotate(b,state,camera),travel);}state[f].x=a.x;state[f].y=a.y;state[f].z=a.z;state[f+1].x=b.x;state[f+1].y=b.y;state[f+1].z=b.z;state[f].w=fmaxf(0,state[f].w-dt);state[f+3].y+=dt;
  if(lightCount<2&&state[f+1].w==2&&state[f].w>0){float3 point=plus(player,a);brush[13+lightCount*2]=make_float4(point.x,point.y,point.z,180*state[f].w*state[f+2].w);brush[14+lightCount*2]=state[f+2];lightCount++;}
 }
 if(target>=0&&state[combat_enemy(target)].w==0)target=-1;
 ship[16].x=state[0].z;brush[12].x=(float)lightCount;state[5]=camera[5];state[6]=camera[6];state[7]=camera[7];state[1]=make_float4(ship[14].x,ship[14].y,(float)target,(float)alive);
 state[2]=make_float4(target>=0?state[combat_enemy(target)+14].x:0,target>=0?state[combat_enemy(target)+14].y:0,ship[15].z,ship[14].x<=0?1:0);state[3]=make_float4(ship[14].w,ship[15].y,(float)weapon,state[0].z);state[4]=make_float4(ship[14].z,0,0,0);
}

// Projection is once per object; the full-resolution pass only visits objects
// in its 32-pixel tile. All enemies share one mesh and one acceleration tree.
__device__ float3 combat_view(float3 p,const float4 *ship,const float4 *camera){
 float3 q=minus(p,ship_axis(ship,ship_xyz(ship[7])));return vec(dotv(q,ship_xyz(camera[3])),dotv(q,ship_xyz(camera[4])),dotv(q,ship_xyz(camera[2])));
}
__device__ float2 combat_screen(float3 p,int width,int height){float k=(float)height/(1.3f*fmaxf(.5f,p.z));return make_float2((float)width*.5f+p.x*k,(float)height*.5f-p.y*k);}
__global__ void combat_prepare(float4 *state,const float4 *ship,const float4 *camera,int width,int height){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=85)return;int out=1024+i*3;state[out]=make_float4(-1,-1,-1,-1);
 float3 a=vec(0,0,0),b=a;float radius=0;int active=0;float3 player=ship_xyz(ship[0]);
 if(i==0){float3 aim=scale(ship_xyz(ship[6]),-1000);float3 q=combat_view(aim,ship,camera);float2 p=combat_screen(q,width,height);state[16]=make_float4(p.x/(float)width,p.y/(float)height,0,q.z>1?1:0);}
 if(i<5){int e=combat_enemy(i);active=state[e].w>0?1:0;a=ship_xyz(state[e+9]);b=a;radius=18;state[17+i*3]=make_float4(0,0,0,0);}
 else if(i<69){int j=combat_round(i-5);active=state[j].w>0?1:0;b=ship_xyz(state[j]);float3 v=ship_xyz(state[j+1]);a=minus(b,scale(unit(v),state[j+1].w==2?9:14));radius=state[j+1].w==2?.9f:.28f;}
 else{int j=combat_fx(i-69);active=state[j].w>0?1:0;a=ship_xyz(state[j]);b=ship_xyz(state[j+1]);int type=(int)state[j+1].w;radius=type==1?.6f:type==2?(4+state[j+3].y*12)*state[j+2].w:type==4?.32f:1.4f+state[j+3].y*7;}
 if(active==0)return;a=combat_view(a,ship,camera);b=combat_view(b,ship,camera);if(fmaxf(a.z,b.z)+radius<.5f)return;
 if(a.z<.5f&&b.z>.5f)a=plus(a,scale(minus(b,a),(.5f-a.z)/(b.z-a.z)));
 if(b.z<.5f&&a.z>.5f)b=plus(b,scale(minus(a,b),(.5f-b.z)/(a.z-b.z)));
 float2 sa=combat_screen(a,width,height),sb=combat_screen(b,width,height);float pixelRadius=fmaxf(i<5?2:2.5f,radius*(float)height/(1.3f*fmaxf(1,fminf(a.z,b.z)-radius)));
 state[out]=make_float4(fminf(sa.x,sb.x)-pixelRadius-2,fminf(sa.y,sb.y)-pixelRadius-2,fmaxf(sa.x,sb.x)+pixelRadius+2,fmaxf(sa.y,sb.y)+pixelRadius+2);
 state[out+1]=make_float4(sa.x,sa.y,sb.x,sb.y);state[out+2]=make_float4(a.z,b.z,pixelRadius,radius);
 if(i<5){int e=combat_enemy(i);state[17+i*3]=make_float4(sa.x/(float)width,sa.y/(float)height,sqrtf(dotv(a,a)),a.z>1?state[e].w:0);state[18+i*3]=state[e+14];}
}
__global__ void combat_tiles(const float4 *state,unsigned *tiles,int width,int height){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y,tw=(width+31)/32,th=(height+31)/32;if(x>=tw||y>=th)return;
 unsigned m0=0u,m1=0u,m2=0u;float left=(float)(x*32),top=(float)(y*32);
 for(int i=0;i<85;i++){float4 r=state[1024+i*3];if(r.z<0||r.z<left||r.w<top||r.x>left+32||r.y>top+32)continue;unsigned bit=1u<<(unsigned)(i%32);if(i<32)m0|=bit;else if(i<64)m1|=bit;else m2|=bit;}
 int p=(y*tw+x)*3;tiles[p]=m0;tiles[p+1]=m1;tiles[p+2]=m2;
}
__device__ int combat_visible(unsigned a,unsigned b,unsigned c,int i){unsigned mask=i<32?a:i<64?b:c;return (mask&(1u<<(unsigned)(i%32)))!=0u?1:0;}
__global__ void combat_render(const float4 *mesh,const float4 *bounds,const float4 *camera,const float4 *state,const unsigned *tiles,const float *hits,unsigned *image,int width,int height,int samples,float exposure){
 int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=width||y>=height)return;
 int tile=((y/32)*((width+31)/32)+x/32)*3;unsigned m0=tiles[tile],m1=tiles[tile+1],m2=tiles[tile+2];if((m0|m1|m2)==0u)return;
 float px=(float)x+.5f,py=(float)y+.5f,sx=(2*px-(float)width)/(float)height,sy=1-2*py/(float)height;
 float3 forward=ship_xyz(camera[2]),right=ship_xyz(camera[3]),up=ship_xyz(camera[4]),ray=unit(plus(forward,plus(scale(right,sx*.65f),scale(up,sy*.65f))));
 float terrain=terrain_pixel_hit(hits,camera,width,height,px,py,0),sea=globe_hit(camera[0].y,ray),limit=terrain>0?terrain:10000000;if(sea>0)limit=fminf(limit,sea);
 float3 bg=ship_decode(image[y*width+x]),result=bg,exhaust=vec(0,0,0);float closest=limit;int enemy=-1;
 for(int i=0;i<5;i++){if(combat_visible(m0,m1,m2,i)==0)continue;int e=combat_enemy(i);float3 localRay=ship_inverse(state+e,ray);ShipHit hit=ship_trace(mesh,bounds,ship_xyz(state[e+7]),localRay,0);exhaust=plus(exhaust,ship_plume(state+e,ship_xyz(state[e+7]),localRay,fminf(hit.t,limit)));if(hit.triangle>=0&&hit.t<closest){closest=hit.t;enemy=i;}}
 if(enemy>=0){int e=combat_enemy(enemy),previousPart=-1;float shadow=-1;float3 sum=vec(0,0,0),origin=ship_xyz(state[e+7]);
  for(int s=0;s<4;s++){if(s>=samples)break;float ox=samples==1?.5f:((float)(s%2)+.5f)*.5f,oy=samples==1?.5f:((float)(s/2)+.5f)*.5f;
   float3 wr=unit(plus(forward,plus(scale(right,(2*((float)x+ox)-(float)width)/(float)height*.65f),scale(up,(1-2*((float)y+oy)/(float)height)*.65f)))),lr=ship_inverse(state+e,wr);ShipHit h=ship_trace(mesh,bounds,origin,lr,0);
   if(h.triangle<0||h.t>limit){sum=plus(sum,bg);continue;}float4 light=ship_radiance(mesh,bounds,camera,state+e,h,lr,h.t*1.3f/(float)height,h.part==previousPart?shadow:-1);previousPart=h.part;shadow=light.w;sum=plus(sum,ship_decode(pack_color(scale(ship_xyz(light),exposure))));
  }result=scale(sum,1/(float)samples);
 }
 float3 glow=exhaust;float forwardCos=fmaxf(.01f,dotv(ray,forward));
 for(int group=0;group<3;group++){
  unsigned pending=group==0?(m0&4294967264u):group==1?m1:m2;
  while(pending!=0u){int i=group*32+__ffs(pending)-1;pending=pending&(pending-1u);
  int out=1024+i*3;float4 rect=state[out];if(px<rect.x||py<rect.y||px>rect.z||py>rect.w)continue;
  float4 q=state[out+1],meta=state[out+2];float dx=q.z-q.x,dy=q.w-q.y,t=clamp01(((px-q.x)*dx+(py-q.y)*dy)/fmaxf(.0001f,dx*dx+dy*dy));
  float viewDepth=1/fmaxf(.000001f,(1-t)/fmaxf(.5f,meta.x)+t/fmaxf(.5f,meta.y)),distance=viewDepth/forwardCos;if(distance>closest+1)continue;
  // The conservative tile bound can be huge when a beam crosses the near
  // plane. Shade with its local perspective width, not that bound's radius.
  float radius=fmaxf(2.5f,meta.w*(float)height/(1.3f*fmaxf(1,viewDepth)));
  float ddx=px-q.x-dx*t,ddy=py-q.y-dy*t,d=sqrtf(ddx*ddx+ddy*ddy);float3 colour=vec(0,0,0);float intensity=0;
  if(i<69){int j=combat_round(i-5),rocket=state[j+1].w==2?1:0;colour=state[j+2].w<0?(rocket!=0?vec(1,.32f,.07f):vec(1,.7f,.22f)):vec(1,.035f,.014f);float core=fmaxf(.65f,radius*.18f);intensity=expf(-d*d/(core*core))*3+expf(-d*d/(radius*radius)*3)*.5f;if(rocket!=0)intensity*=.4f+.6f*t;}
  else{int j=combat_fx(i-69),type=(int)state[j+1].w;float fade=clamp01(state[j].w/state[j+3].x),age=state[j+3].y;colour=ship_xyz(state[j+2]);
   if(type==1){float core=fmaxf(.65f,radius*.12f);intensity=(expf(-d*d/(core*core))*4+expf(-d*d/(radius*radius)*4)*.7f)*sqrtf(fade);}
   else if(type==2){float r=d/fmaxf(1,radius),angle=atan2f(ddy,ddx),lobes=1+.10f*sinf(angle*7+age*5)+.08f*cosf(angle*11-age*8);float fire=expf(-r*r*lobes*5)*fmaxf(0,1-age)*5,ring=expf(-(r-.65f)*(r-.65f)*180)*fade*.6f,sparks=positive_power(fmaxf(0,cosf(angle*19+age*2)),24)*expf(-(r-.83f)*(r-.83f)*110)*fade;
    intensity=(fire+ring+sparks*2)*fade;float smoke=clamp01(expf(-r*r*5)*age*fade*.32f);result=scale(result,1-smoke);
   }else{float r=d/fmaxf(1,radius),angle=atan2f(ddy,ddx);intensity=(expf(-r*r*10)+positive_power(fmaxf(0,cosf(angle*9)),20)*expf(-r*r*2))*fade*2;}
  }glow=plus(glow,scale(colour,intensity));
  }
 }
 if(enemy<0&&dotv(glow,glow)<.0000001f&&dotv(minus(result,bg),minus(result,bg))<.0000001f)return;
 float3 linear=vec(ship_unfilm(result.x),ship_unfilm(result.y),ship_unfilm(result.z));image[y*width+x]=pack_color(plus(linear,scale(glow,exposure)));
}
