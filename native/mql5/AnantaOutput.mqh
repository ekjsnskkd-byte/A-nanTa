// Frozen V8 output decoder. No trading rules, thresholds or order execution.
#ifndef ANANTA_OUTPUT_MQH
#define ANANTA_OUTPUT_MQH
#define AN_OUTPUT_COUNT 15
#define AN_OUTPUT_VALUES 163
#define AN_DECISION_TEMPERATURE 1.3820429178368876
enum AnAction { AN_SHORT=0, AN_WAIT=1, AN_LONG=2 };
enum AnDirection { AN_DOWN=0, AN_NEUTRAL=1, AN_UP=2 };
enum AnQuantile { AN_Q10=0, AN_Q50=1, AN_Q90=2 };
const int AnHorizons[4]={2,6,24,96};
const int AnPathKnots[12]={1,2,4,6,8,12,16,24,32,48,64,96};
struct AnRawOutput {
 float direction[12],quantiles[12],rv[4],excursions[8],barrier[12];
 float pattern[4],pattern_detail[12],jump[4];
 float path_mu[12],path_diag[12],path_factors[48];
 float action_quantiles[9],decision[3],regime[7],mtf_attention[4];
};
struct AnForecast {
 double direction[4][3],returns[4][3],variance[4],volatility[4];
 double mfe[4],mae[4],barrier[4][3],pattern[4],pattern_detail[4][3],jump[4];
 double decision[3],utility[3][3],regime[7],mtf_attention[4];
 double path_increment_mu[12],path_increment_cov[12][12];
 double path_cumulative_mu[12],path_cumulative_cov[12][12];
 double path_cumulative_quantiles[12][3];
 bool valid;
};
double AnOutputSigmoid(double x){
 if(x>=0)return 1.0/(1.0+MathExp(-x));
 double e=MathExp(x);return e/(1.0+e);
}
bool AnOutputFinite(const float &a[]){
 for(int i=0;i<ArraySize(a);i++)if(!MathIsValidNumber((double)a[i])||a[i]==EMPTY_VALUE)return false;
 return true;
}
void AnOutputSoftmax3(double a,double b,double c,double t,double &p0,double &p1,double &p2){
 double m=MathMax(a,MathMax(b,c));
 double x=MathExp((a-m)/t),y=MathExp((b-m)/t),z=MathExp((c-m)/t);
 double s=x+y+z;p0=x/s;p1=y/s;p2=z/s;
}
bool AnDecode(const AnRawOutput &r,AnForecast &o,double temperature=AN_DECISION_TEMPERATURE){
 ZeroMemory(o);
 if(!MathIsValidNumber(temperature)||temperature<=0)return false;
 if(!AnOutputFinite(r.direction)||!AnOutputFinite(r.quantiles)||!AnOutputFinite(r.rv)||
 !AnOutputFinite(r.excursions)||!AnOutputFinite(r.barrier)||!AnOutputFinite(r.pattern)||
 !AnOutputFinite(r.pattern_detail)||!AnOutputFinite(r.jump)||!AnOutputFinite(r.path_mu)||
 !AnOutputFinite(r.path_diag)||!AnOutputFinite(r.path_factors)||!AnOutputFinite(r.action_quantiles)||
 !AnOutputFinite(r.decision)||!AnOutputFinite(r.regime)||!AnOutputFinite(r.mtf_attention))return false;
 AnOutputSoftmax3(r.decision[0],r.decision[1],r.decision[2],temperature,o.decision[0],o.decision[1],o.decision[2]);
 for(int a=0;a<3;a++){
  for(int q=0;q<3;q++)o.utility[a][q]=r.action_quantiles[3*a+q];
  if(o.utility[a][0]>o.utility[a][1]||o.utility[a][1]>o.utility[a][2])return false;
 }
 for(int h=0;h<4;h++){
  int k=3*h;
  AnOutputSoftmax3(r.direction[k],r.direction[k+1],r.direction[k+2],1,o.direction[h][0],o.direction[h][1],o.direction[h][2]);
  AnOutputSoftmax3(r.barrier[k],r.barrier[k+1],r.barrier[k+2],1,o.barrier[h][0],o.barrier[h][1],o.barrier[h][2]);
  for(int q=0;q<3;q++){
   o.returns[h][q]=r.quantiles[k+q];
   o.pattern_detail[h][q]=AnOutputSigmoid(r.pattern_detail[k+q]);
  }
  if(o.returns[h][0]>o.returns[h][1]||o.returns[h][1]>o.returns[h][2]||r.rv[h]<0)return false;
  o.variance[h]=r.rv[h];o.volatility[h]=MathSqrt(o.variance[h]);
  // Preserve original signed excursion predictions, without clipping.
  o.mfe[h]=r.excursions[2*h];o.mae[h]=r.excursions[2*h+1];
  o.pattern[h]=AnOutputSigmoid(r.pattern[h]);o.jump[h]=AnOutputSigmoid(r.jump[h]);
  o.mtf_attention[h]=r.mtf_attention[h];
 }
 for(int i=0;i<7;i++)o.regime[i]=AnOutputSigmoid(r.regime[i]);
 // Increment covariance = diag(path_diag^2) + factors * factors^T.
 // Cumulative covariance = L * increment_cov * L^T, L lower triangular ones.
 double mu=0;
 for(int i=0;i<12;i++){
  if(r.path_diag[i]<=0)return false;
  o.path_increment_mu[i]=r.path_mu[i];mu+=r.path_mu[i];o.path_cumulative_mu[i]=mu;
  for(int j=0;j<12;j++){
   double c=i==j?(double)r.path_diag[i]*r.path_diag[i]:0;
   for(int k=0;k<4;k++)c+=(double)r.path_factors[4*i+k]*r.path_factors[4*j+k];
   o.path_increment_cov[i][j]=c;
  }
 }
 for(int i=0;i<12;i++)for(int j=0;j<12;j++){
  double c=0;
  for(int a=0;a<=i;a++)for(int b=0;b<=j;b++)c+=o.path_increment_cov[a][b];
  o.path_cumulative_cov[i][j]=c;
 }
 for(int i=0;i<12;i++){
  double s=MathSqrt(MathMax(1e-15,o.path_cumulative_cov[i][i]));
  double m=o.path_cumulative_mu[i];
  o.path_cumulative_quantiles[i][0]=m-1.28155*s;
  o.path_cumulative_quantiles[i][1]=m;
  o.path_cumulative_quantiles[i][2]=m+1.28155*s;
 }
 o.valid=true;return true;
}
// Convert a predicted log return to price; no trading signal is generated.
bool AnForecastPrice(double reference,double log_return,double &price){
 if(reference<=0||!MathIsValidNumber(reference)||!MathIsValidNumber(log_return))return false;
 price=reference*MathExp(log_return);return MathIsValidNumber(price)&&price>0;
}
#endif
