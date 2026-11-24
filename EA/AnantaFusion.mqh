// Native forecast fusion; no order execution or entry/exit rules.
// All return-like horizon summaries are expressed at a 24-bar reference scale.
// Scaling is heuristic (linear drift, sqrt-time dispersion), not a calibrated law.
#ifndef ANANTA_FUSION_MQH
#define ANANTA_FUSION_MQH
#include "AnantaOutput.mqh"
#define AN_FUSION_COUNT 12
struct AnFusionConfig {
 double horizon[4];     // prior horizon weights
 double reliability[4]; // externally estimated on past, resolved labels only
 double head[5];       // direction, barrier, return, path, decision
 double risk_scale;    // reference log-return risk scale
};
struct AnComposite {
 double direction_score,direction_confidence,expected_return,expected_utility;
 double risk_score,expected_volatility,expected_mfe,expected_mae;
 double trend_strength,reversal_risk,market_stability,signal_agreement;
};
void AnFusionDefaults(AnFusionConfig &c){
 c.horizon[0]=0.10;c.horizon[1]=0.20;c.horizon[2]=0.40;c.horizon[3]=0.30;
 for(int i=0;i<4;i++)c.reliability[i]=1.0;
 c.head[0]=0.30;c.head[1]=0.15;c.head[2]=0.20;c.head[3]=0.15;c.head[4]=0.20;
 c.risk_scale=0.005;
}
double AnFClip(double x,double lo,double hi){return MathMax(lo,MathMin(hi,x));}
double AnFTanh(double x){double z=MathExp(-2*MathAbs(x));return (x<0?-1:1)*(1-z)/(1+z);}
bool AnFusionWeight(double x){return MathIsValidNumber(x)&&x>=0;}
bool AnFuse(const AnForecast &f,const AnFusionConfig &c,AnComposite &o){
 ZeroMemory(o);
 if(!f.valid||!MathIsValidNumber(c.risk_scale)||c.risk_scale<=0)return false;
 int knot[4]={1,3,7,11};
 double w[4],hw[5],sum=0,hs=0;
 for(int h=0;h<4;h++){
  if(!AnFusionWeight(c.horizon[h])||!AnFusionWeight(c.reliability[h]))return false;
  int k=knot[h];
  double qsd=(f.returns[h][2]-f.returns[h][0])/2.5631;
  double psd=MathSqrt(MathMax(0,f.path_cumulative_cov[k][k]));
  double scale=MathSqrt(24.0/AnHorizons[h]);
  double uncertainty=0.5*(qsd+psd)*scale;
  double stability=1-0.5*AnFClip(f.jump[h],0,1);
  // Bounded uncertainty discount, avoids unbounded precision for zero variance.
  w[h]=c.horizon[h]*c.reliability[h]*stability/(1+uncertainty/c.risk_scale);
  sum+=w[h];
 }
 for(int j=0;j<5;j++){if(!AnFusionWeight(c.head[j]))return false;hs+=c.head[j];}
 if(!MathIsValidNumber(sum)||!MathIsValidNumber(hs)||sum<=0||hs<=0)return false;
 for(int j=0;j<5;j++)hw[j]=c.head[j]/hs;
 double score[5];ArrayInitialize(score,0);
 double risk=0,ret=0,vol=0,mfe=0,mae=0,rev=0,stable=0,trend=0;
 for(int h=0;h<4;h++){
  double a=w[h]/sum;int k=knot[h];
  double drift=24.0/AnHorizons[h],scale=MathSqrt(drift);
  double median=f.returns[h][1],pm=f.path_cumulative_mu[k];
  double qsd=(f.returns[h][2]-f.returns[h][0])/2.5631;
  double psd=MathSqrt(MathMax(0,f.path_cumulative_cov[k][k]));
  score[0]+=a*(f.direction[h][AN_UP]-f.direction[h][AN_DOWN]);
  score[1]+=a*(f.barrier[h][AN_LONG]-f.barrier[h][AN_SHORT]);
  score[2]+=a*AnFTanh(median/MathMax(qsd,1e-8));
  score[3]+=a*AnFTanh(pm/MathMax(psd,1e-8));
  // Median proxy, not a proven conditional expectation.
  ret+=a*0.5*(median+pm)*drift;
  vol+=a*f.volatility[h]*scale;
  mfe+=a*MathMax(0,f.mfe[h])*scale;
  mae+=a*MathAbs(f.mae[h])*scale;
  double tail=MathMax(0,MathMax(-f.returns[h][0],f.returns[h][2]));
  risk+=a*scale*(tail+psd+MathAbs(f.mae[h])+f.volatility[h])/4;
  rev+=a*AnFClip(f.pattern_detail[h][1],0,1);
  stable+=a*(1-(AnFClip(f.jump[h],0,1)+AnFClip(f.pattern_detail[h][0],0,1)+AnFClip(f.regime[2],0,1))/3);
  trend+=a*(AnFClip(f.regime[0],0,1)+AnFClip(f.pattern_detail[h][2],0,1)+MathAbs(f.direction[h][AN_UP]-f.direction[h][AN_DOWN]))/3;
 }
 score[4]=f.decision[AN_LONG]-f.decision[AN_SHORT];
 double d=0,absmean=0;
 for(int j=0;j<5;j++){d+=hw[j]*score[j];absmean+=hw[j]*MathAbs(score[j]);}
 double dispersion=0;for(int j=0;j<5;j++)dispersion+=hw[j]*MathPow(score[j]-d,2);
 // Agreement is weighted directional coherence, not independent evidence.
 double agreement=absmean>1e-12?MathAbs(d)/absmean:0;
 o.direction_score=AnFClip(d,-1,1);
 o.signal_agreement=AnFClip(agreement,0,1);
 o.direction_confidence=AnFClip(MathAbs(d)*(1-MathMin(1,MathSqrt(dispersion))),0,1);
 o.expected_return=ret;
 // Probability-weighted median utility proxy, in original utility units.
 for(int i=0;i<3;i++)o.expected_utility+=f.decision[i]*f.utility[i][AN_Q50];
 o.risk_score=AnFClip(risk/(risk+c.risk_scale),0,1);
 o.expected_volatility=vol;o.expected_mfe=mfe;o.expected_mae=mae;
 o.trend_strength=AnFClip(trend,0,1);o.reversal_risk=AnFClip(rev,0,1);
 o.market_stability=AnFClip(stable,0,1);
 return MathIsValidNumber(o.expected_return)&&MathIsValidNumber(o.expected_utility)&&MathIsValidNumber(o.risk_score)&&MathIsValidNumber(o.expected_volatility)&&MathIsValidNumber(o.expected_mfe)&&MathIsValidNumber(o.expected_mae)&&MathIsValidNumber(o.direction_score)&&MathIsValidNumber(o.direction_confidence)&&MathIsValidNumber(o.signal_agreement)&&MathIsValidNumber(o.trend_strength)&&MathIsValidNumber(o.reversal_risk)&&MathIsValidNumber(o.market_stability);
}
bool AnDecodeAndFuse(const AnRawOutput &raw,const AnFusionConfig &cfg,AnForecast &decoded,AnComposite &out){
 ZeroMemory(out);if(!AnDecode(raw,decoded))return false;
 return AnFuse(decoded,cfg,out);
}
#endif
