// EA component version 1.0 (V8 model)
// Integration entry point: no policy, no trading. Feed chronological UTC ticks.
#ifndef ANANTA_PIPELINE_MQH
#define ANANTA_PIPELINE_MQH
#include "AnantaInput_v1.0.mqh"
#include "AnantaScalers_v1.0.mqh"
class AnInputPipeline {
 private:
  AnTickBars engine;
  AnBar bars[];
  long last_tick;
  bool BuildTF(int tf,long period,const double &med[],const double &iqr[],float &mtf[],bool &mask[]){
   AnBar all[],segment[];AnTimeframe(bars,period,bars[ArraySize(bars)-1].end_ms,all);
   int n=ArraySize(all),start=0;long gap=(long)MathMax(3600000,period*2);
   for(int i=1;i<n;i++)if(all[i].end_ms-all[i-1].end_ms>gap)start=i;
   for(int i=start;i<n;i++)AnAppend(segment,all[i]);
   double f[];float z[];
   if(n>0 && (!AnFeatures(segment,f)||!AnNormalize(f,med,iqr,z)))return false;
   return AnPackTF(f,z,tf,mtf,mask);
  }
 public:
  AnInputPipeline(){last_tick=0;}
  bool Configure(bool proxy,double quote_weight,double contract_size){
   if(proxy && quote_weight<=0)return false;
   if(!proxy && contract_size<=0)return false;
   engine.Reset();engine.proxy=proxy;engine.quote_weight=quote_weight;engine.contract_size=contract_size;
   ArrayResize(bars,0);last_tick=0;return true;
  }
  bool Push(const MqlTick &utc_tick){
   if(utc_tick.time_msc<last_tick)return false;
   last_tick=utc_tick.time_msc;AnBar b;
   if(!engine.Push(utc_tick,b))return false;
   AnAppend(bars,b);return true;
  }
  int BarCount(){return ArraySize(bars);}
  bool Build(float &x[],float &mtf[],bool &mask[]){
   int n=ArraySize(bars);if(n<AN_LB)return false;
   AnBar core[];int start=0;
   for(int i=1;i<n;i++)if(bars[i].end_ms-bars[i-1].end_ms>3600000)start=i;
   for(int i=start;i<n;i++)AnAppend(core,bars[i]);
   double f[];float z[];
   if(!AnFeatures(core,f)||!AnNormalize(f,AnMed,AnIqr,z)||!AnPackCore(f,z,x))return false;
   ArrayResize(mtf,AN_K*AN_W*AN_F);ArrayResize(mask,AN_K*AN_W);
   return BuildTF(0,900000,AnMedM15,AnIqrM15,mtf,mask)
       && BuildTF(1,3600000,AnMedH1,AnIqrH1,mtf,mask)
       && BuildTF(2,14400000,AnMedH4,AnIqrH4,mtf,mask)
       && BuildTF(3,86400000,AnMedD1,AnIqrD1,mtf,mask);
  }
};
#endif
