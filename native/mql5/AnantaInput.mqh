// Native input adapter for frozen A-nanTa V8. No order execution.
// Quote-only mode is experimental; activity is NOT exchange notional.
#ifndef ANANTA_INPUT_MQH
#define ANANTA_INPUT_MQH
#define AN_F 40
#define AN_LB 128
#define AN_W 32
#define AN_K 4
#define AN_NA EMPTY_VALUE
struct AnBar {
 long end_ms;
 double o,h,l,c,bid,ask,volume,notional,signed_notional,buy_notional,count,dt;
 bool real_volume;
};
bool AnValid(double v) { return v!=AN_NA && MathIsValidNumber(v); }
double AnClip(double v,double lo,double hi) {return MathMax(lo,MathMin(hi,v));}
double AnLog1p(double v){return MathLog(1.0+v);}
double AnAsinh(double v){return (v<0?-1:1)*MathLog(MathAbs(v)+MathSqrt(v*v+1));}
void AnAppend(AnBar &a[],const AnBar &b){int n=ArraySize(a);ArrayResize(a,n+1,1024);a[n]=b;}
class AnTickBars {
 private:
  AnBar b;
  bool active;
  double previous;
  int sign;
  long start_ms,last_ms;
 public:
  bool proxy;
  double quote_weight,contract_size;
  AnTickBars(){active=false;previous=0;sign=0;start_ms=0;last_ms=0;proxy=false;quote_weight=0;contract_size=0;}
  void Reset(){active=false;previous=0;sign=0;start_ms=0;last_ms=0;}
  // quote_weight must be calibrated on separate paired MT5/exchange history.
  // contract_size must convert tick real volume into BTC units, not assumed.
  bool Push(const MqlTick &t,AnBar &closed){
   if(t.bid<=0 || t.ask<t.bid || t.time_msc<last_ms)return false;
   if(last_ms>0 && t.time_msc-last_ms>3600000)Reset();
   last_ms=t.time_msc;
   bool trade=((t.flags&TICK_FLAG_BUY)!=0 || (t.flags&TICK_FLAG_SELL)!=0);
   if(!proxy && (!trade || t.last<=0 || t.volume_real<=0 || contract_size<=0))return false;
   if(proxy && (quote_weight<=0 || (t.flags&(TICK_FLAG_BID|TICK_FLAG_ASK))==0))return false;
   double p=proxy?(t.bid+t.ask)*0.5:t.last;
   if(p>previous && previous>0)sign=1;
   else if(p<previous)sign=-1;
   previous=p;
   int s=proxy?sign:(((t.flags&TICK_FLAG_BUY)!=0 && (t.flags&TICK_FLAG_SELL)==0)?1:(((t.flags&TICK_FLAG_SELL)!=0 && (t.flags&TICK_FLAG_BUY)==0)?-1:0));
   double v=proxy?quote_weight/p:t.volume_real*contract_size;
   double usd=v*p;
   if(!active){
    ZeroMemory(b);b.o=p;b.h=p;b.l=p;start_ms=t.time_msc;active=true;b.real_volume=!proxy;
   }
   b.h=MathMax(b.h,p);b.l=MathMin(b.l,p);b.c=p;b.bid=t.bid;b.ask=t.ask;
   b.volume+=v;b.notional+=usd;b.signed_notional+=s*usd;b.buy_notional+=(s+1)*0.5*usd;b.count++;
   b.end_ms=t.time_msc;b.dt=MathMax(0.001,(t.time_msc-start_ms)*0.001);
   if(b.notional<500000.0)return false;
   closed=b;active=false;return true;
  }
};
// Aggregate completed event bars into right-closed UTC time buckets.
// Pass UTC-converted timestamps; never emit the current incomplete bucket.
void AnTimeframe(const AnBar &src[],const long period_ms,const long now_ms,AnBar &dst[]){
 ArrayResize(dst,0);AnBar b;bool active=false;long end=0;
 for(int i=0;i<ArraySize(src);i++){
  long e=((src[i].end_ms+period_ms-1)/period_ms)*period_ms;
  if(e>now_ms)break;
  if(!active || e!=end){
   if(active)AnAppend(dst,b);
   b=src[i];b.end_ms=e;b.bid=AN_NA;b.ask=AN_NA;end=e;active=true;
  }else{
   b.h=MathMax(b.h,src[i].h);b.l=MathMin(b.l,src[i].l);b.c=src[i].c;
   b.volume+=src[i].volume;b.notional+=src[i].notional;b.signed_notional+=src[i].signed_notional;
   b.buy_notional+=src[i].buy_notional;b.count+=src[i].count;b.dt+=src[i].dt;
   b.real_volume=b.real_volume && src[i].real_volume;
  }
 }
 if(active)AnAppend(dst,b);
}
// Rolling primitives, sample standard deviation (pandas ddof=1).
double AnStat(const double &a[],int i,int w,int min_n,int mode){
 double vals[];ArrayResize(vals,w);int n=0;double sum=0;
 for(int j=MathMax(0,i-w+1);j<=i;j++)if(AnValid(a[j])){vals[n++]=a[j];sum+=a[j];}
 if(n<min_n || n==0)return AN_NA;
 double m=sum/n;
 if(mode==0)return m;
 if(mode==1){if(n<2)return AN_NA;double q=0;for(int j=0;j<n;j++)q+=MathPow(vals[j]-m,2);return MathSqrt(q/(n-1));}
 if(mode==2){ArrayResize(vals,n);ArraySort(vals);return (vals[(n-1)/2]+vals[n/2])*0.5;}
 if(mode==3){double z=vals[0];for(int j=1;j<n;j++)z=MathMin(z,vals[j]);return z;}
 if(mode==4){double z=vals[0];for(int j=1;j<n;j++)z=MathMax(z,vals[j]);return z;}
 return sum;
}
double AnMom(const double &r[],int i,bool kurt){
 if(i<23)return AN_NA;
 double m=AnStat(r,i,24,24,0),sd=AnStat(r,i,24,24,1);
 if(!AnValid(m)||!AnValid(sd))return AN_NA;
 if(sd<1e-15)return kurt?-3.0:0.0;
 double z=0;for(int j=i-23;j<=i;j++)z+=MathPow((r[j]-m)/sd,kurt?4:3);
 return kurt?(24.0*25/(23*22*21)*z-3.0*23*23/(22*21)):24.0/(23*22)*z;
}
double AnEntropy(const double &r[],int i){
 if(i<23)return AN_NA;int counts[27];ArrayInitialize(counts,0);
 for(int j=i-21;j<=i;j++){
  double x=r[j-2],y=r[j-1],z=r[j];if(!AnValid(x)||!AnValid(y)||!AnValid(z))return AN_NA;
  int rx=(y<x?1:0)+(z<x?1:0),ry=(x<=y?1:0)+(z<y?1:0);
  counts[rx*9+ry*3+3-rx-ry]++;
 }
 double h=0;for(int j=0;j<27;j++)if(counts[j]>0){double p=counts[j]/22.0;h-=p*MathLog(p);}
 return h/MathLog(6.0);
}
double AnCorr(const double &r[],int i){
 if(i<24)return AN_NA;double x=0,y=0,xx=0,yy=0,xy=0;
 for(int j=i-23;j<=i;j++){if(!AnValid(r[j])||!AnValid(r[j-1]))return AN_NA;double a=r[j],b=r[j-1];x+=a;y+=b;xx+=a*a;yy+=b*b;xy+=a*b;}
 double d=(xx-x*x/24)*(yy-y*y/24);return d>0?(xy-x*y/24)/MathSqrt(d):AN_NA;
}
// One contiguous segment; caller must reset at gaps, as in training.
bool AnFeatures(const AnBar &b[],double &f[]){
 int n=ArraySize(b);if(n<1)return false;
 ArrayResize(f,n*AN_F);ArrayInitialize(f,AN_NA);
 double p[],lp[],r[],absr[],r2[],down[],dp[],vol[],imb[],notl[],cnt[],signedabs[],speed[],kyle[],gk[],ami[],avg[],vpin[];
 ArrayResize(p,n);ArrayResize(lp,n);ArrayResize(r,n);ArrayResize(absr,n);ArrayResize(r2,n);ArrayResize(down,n);ArrayResize(dp,n);ArrayResize(vol,n);ArrayResize(imb,n);ArrayResize(notl,n);ArrayResize(cnt,n);ArrayResize(signedabs,n);ArrayResize(speed,n);ArrayResize(kyle,n);ArrayResize(gk,n);ArrayResize(ami,n);ArrayResize(avg,n);ArrayResize(vpin,n);
 double ew=0;int en=0;
 for(int i=0;i<n;i++){
  if(b[i].c<=0||b[i].o<=0||b[i].l<=0||b[i].h<=0||b[i].notional<=0||b[i].dt<=0)return false;
  p[i]=b[i].c;lp[i]=MathLog(p[i]);
  r[i]=i>0?lp[i]-lp[i-1]:AN_NA;
  absr[i]=i>0?MathAbs(r[i]):AN_NA;r2[i]=i>0?r[i]*r[i]:AN_NA;
  down[i]=i>0?MathPow(MathMin(r[i],0),2):AN_NA;dp[i]=i>0?MathAbs(p[i]-p[i-1]):AN_NA;
  vol[i]=AnLog1p(b[i].volume);imb[i]=b[i].signed_notional/(b[i].notional+1e-9);
  notl[i]=AnLog1p(b[i].notional);cnt[i]=AnLog1p(b[i].count);signedabs[i]=MathAbs(b[i].signed_notional);
  speed[i]=AnLog1p(b[i].notional/b[i].dt);vpin[i]=signedabs[i]/(b[i].notional+1e-9);
  kyle[i]=i>0?absr[i]/MathMax(signedabs[i]/1e6,1e-6):AN_NA;
  ami[i]=i>0?absr[i]/MathMax(b[i].notional/1e6,1e-6):AN_NA;
  double hl=MathLog(b[i].h/b[i].l),co=MathLog(b[i].c/b[i].o);
  gk[i]=MathMax(0,0.5*hl*hl-(2*MathLog(2.0)-1)*co*co);
  avg[i]=AnLog1p(b[i].notional/MathMax(b[i].count,1));
  int k=i*AN_F;double v=AnStat(r,i,48,24,1),v12=AnStat(r,i,12,12,1);
  f[k]=r[i];if(i>=3)f[k+1]=lp[i]-lp[i-3];if(i>=12)f[k+2]=lp[i]-lp[i-12];
  f[k+3]=v12;f[k+4]=v;if(AnValid(v)&&AnValid(v12))f[k+5]=v12/(v+1e-8);
  double d=AnStat(dp,i,24,24,5);if(i>=24&&AnValid(d))f[k+6]=MathAbs(p[i]-p[i-24])/(d+1e-8);
  if(AnValid(v)&&v>0)f[k+7]=(b[i].h-b[i].l)/(p[i]*v+1e-8);
  double mn=AnStat(p,i,24,24,3),mx=AnStat(p,i,24,24,4);
  if(AnValid(mn)&&AnValid(mx))f[k+8]=(p[i]-mn)/(mx-mn+1e-8);
  d=AnStat(down,i,24,24,0);if(AnValid(d))f[k+9]=MathSqrt(d);
  f[k+10]=AnMom(r,i,false);f[k+11]=AnMom(r,i,true);f[k+12]=AnEntropy(r,i);
  bool spread=AnValid(b[i].bid)&&AnValid(b[i].ask)&&b[i].bid>0&&b[i].ask>=b[i].bid;
  if(spread)f[k+13]=(b[i].ask-b[i].bid)/p[i];f[k+14]=spread?1:0;
  // Proxy volume is explicitly marked missing, not passed off as real volume.
  d=AnStat(vol,i,48,24,0);if(b[i].real_volume&&AnValid(d))f[k+15]=vol[i]-d;
  f[k+16]=b[i].real_volume?1:0;
  d=AnStat(r,i,96,48,1);if(i>=96&&AnValid(d))f[k+17]=(lp[i]-lp[i-96])/(d*MathSqrt(96.0)+1e-9);
  f[k+18]=AnStat(r2,i,12,12,0);f[k+19]=AnStat(r2,i,48,48,0);f[k+20]=AnStat(r2,i,288,144,0);
  double s=AnStat(r,i,48,24,5),ab=AnStat(absr,i,48,24,5);if(AnValid(s)&&AnValid(ab))f[k+21]=MathLog((MathAbs(s)+1e-9)/(ab+1e-9));
  s=AnStat(r,i,8,8,0);d=AnStat(r,i,48,24,0);if(AnValid(s)&&AnValid(d))f[k+22]=s-d;
  d=AnStat(absr,i,48,24,2);if(AnValid(d)&&AnValid(absr[i]))f[k+23]=MathMin(50,absr[i]/(d+1e-9));
  d=AnStat(lp,i,48,24,0);if(AnValid(d)&&AnValid(v))f[k+24]=(lp[i]-d)/(v*MathSqrt(48.0)+1e-8);
  f[k+25]=AnCorr(r,i);f[k+26]=imb[i];f[k+27]=b[i].buy_notional/(b[i].notional+1e-9);
  d=AnStat(notl,i,48,24,0);if(AnValid(d))f[k+28]=notl[i]-d;
  d=AnStat(cnt,i,48,24,0);if(AnValid(d))f[k+29]=cnt[i]-d;
  ew=en==0?imb[i]:ew+(2.0/25)*(imb[i]-ew);en++;if(en>=12)f[k+30]=ew;
  d=AnStat(signedabs,i,48,24,2);if(AnValid(d))f[k+31]=AnClip(signedabs[i]/(d+1e-9),0,50);
  f[k+32]=AnLog1p(b[i].dt);d=AnStat(speed,i,48,24,0);if(AnValid(d))f[k+33]=speed[i]-d;
  f[k+34]=AnAsinh(b[i].signed_notional/(b[i].dt*1000+1e-3));f[k+35]=AnStat(vpin,i,24,12,0);
  d=AnStat(kyle,i,24,12,0);if(AnValid(d))f[k+36]=AnLog1p(d);
  d=AnStat(gk,i,24,12,0);if(AnValid(d))f[k+37]=MathSqrt(d);
  d=AnStat(ami,i,24,12,0);if(AnValid(d))f[k+38]=AnLog1p(d);
  d=AnStat(avg,i,48,24,0);if(AnValid(d))f[k+39]=avg[i]-d;
 }
 return true;
}
bool AnNormalize(const double &f[],const double &med[],const double &iqr[],float &z[]){
 int n=ArraySize(f)/AN_F;if(ArraySize(med)!=AN_F||ArraySize(iqr)!=AN_F)return false;
 ArrayResize(z,n*AN_F);double raw[];ArrayResize(raw,n);
 for(int j=0;j<AN_F;j++){
  if(!AnValid(med[j])||!AnValid(iqr[j])||iqr[j]<=0)return false;
  for(int i=0;i<n;i++)raw[i]=AnValid(f[i*AN_F+j])?(f[i*AN_F+j]-med[j])/iqr[j]:AN_NA;
  for(int i=0;i<n;i++){
   double v=raw[i];
   if(AnValid(v)&&j!=14&&j!=16){
    double m=AnStat(raw,i,200,48,0),s=AnStat(raw,i,200,48,1);
    if(AnValid(m)&&AnValid(s))v=(v-m)/MathMax(s,0.5);
   }
   z[i*AN_F+j]=(float)(AnValid(v)?AnClip(v,-12,12):0);
  }
 }
 return true;
}
// Tensor layout matches ONNX: x[1,128,40], mtf[1,4,32,40], mask[1,4,32].
bool AnPackCore(const double &f[],const float &z[],float &x[]){
 int n=ArraySize(z)/AN_F;if(n<AN_LB||ArraySize(f)!=ArraySize(z))return false;
 ArrayResize(x,AN_LB*AN_F);
 for(int i=0;i<AN_LB;i++){int k=(n-AN_LB+i)*AN_F;
  if(!AnValid(f[k])||!AnValid(f[k+3])||!AnValid(f[k+4]))return false;
  for(int j=0;j<AN_F;j++)x[i*AN_F+j]=z[k+j];
 }return true;
}
bool AnPackTF(const double &f[],const float &z[],int tf,float &mtf[],bool &mask[]){
 if(tf<0||tf>=AN_K||ArraySize(f)!=ArraySize(z))return false;
 if(ArraySize(mtf)!=AN_K*AN_W*AN_F||ArraySize(mask)!=AN_K*AN_W)return false;
 int n=ArraySize(z)/AN_F;
 for(int i=0;i<AN_W;i++){
  int q=n-AN_W+i,k=q*AN_F;bool valid=q>=0;
  if(valid)valid=AnValid(f[k])&&AnValid(f[k+3])&&AnValid(f[k+4]);
  mask[tf*AN_W+i]=valid;
  for(int j=0;j<AN_F;j++)mtf[(tf*AN_W+i)*AN_F+j]=q>=0?z[k+j]:0;
 }return true;
}
#endif
