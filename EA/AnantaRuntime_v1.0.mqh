// EA component version 1.0 (V8 model)
// Native MT5 ONNX inference; no policy, no external process.
#ifndef ANANTA_RUNTIME_MQH
#define ANANTA_RUNTIME_MQH
#include "AnantaOutput_v1.0.mqh"
class AnantaRuntime {
 private:
  long handle;
  int error;
  string stage;
  bool Fail(string where,int code){
   stage=where;error=code;return false;
  }
  bool OutShape(int index,string name,int d1,int d2=0){
   if(OnnxGetOutputName(handle,index)!=name)return Fail("output name: "+name,-1);
   ulong dims[];
   ArrayResize(dims,d2>0?3:2);
   dims[0]=1;dims[1]=(ulong)d1;if(d2>0)dims[2]=(ulong)d2;
   ResetLastError();
   if(!OnnxSetOutputShape(handle,index,dims))return Fail("output shape: "+name,GetLastError());
   return true;
  }
  bool Setup(){
   if(OnnxGetInputCount(handle)!=3||OnnxGetOutputCount(handle)!=15)return Fail("tensor count",-1);
   if(OnnxGetInputName(handle,0)!="x"||OnnxGetInputName(handle,1)!="mtf"||
      OnnxGetInputName(handle,2)!="mtf_mask")return Fail("input names",-1);
   const ulong xshape[]={1,128,40};
   const ulong mshape[]={1,4,32,40};
   const ulong bshape[]={1,4,32};
   ResetLastError();
   if(!OnnxSetInputShape(handle,0,xshape)||!OnnxSetInputShape(handle,1,mshape)||
      !OnnxSetInputShape(handle,2,bshape))return Fail("input shape",GetLastError());
   return OutShape(0,"direction",4,3)&&OutShape(1,"quantiles",4,3)&&
    OutShape(2,"rv",4)&&OutShape(3,"excursions",4,2)&&OutShape(4,"barrier",4,3)&&
    OutShape(5,"pattern",4)&&OutShape(6,"pattern_detail",4,3)&&OutShape(7,"jump",4)&&
    OutShape(8,"path_mu",12)&&OutShape(9,"path_diag",12)&&OutShape(10,"path_factors",12,4)&&
    OutShape(11,"action_quantiles",3,3)&&OutShape(12,"decision",3)&&OutShape(13,"regime",7)&&
    OutShape(14,"mtf_attention",4);
  }
 public:
  AnantaRuntime(){handle=INVALID_HANDLE;error=0;stage="not initialized";}
  ~AnantaRuntime(){Release();}
  void Release(){
   if(handle!=INVALID_HANDLE){OnnxRelease(handle);handle=INVALID_HANDLE;}
  }
  bool Ready(){return handle!=INVALID_HANDLE;}
  int LastErrorCode(){return error;}
  string LastStage(){return stage;}
  bool Init(const uchar &model[]){
   Release();error=0;stage="";
   if(ArraySize(model)==0)return Fail("empty model resource",-1);
   ResetLastError();handle=OnnxCreateFromBuffer(model,ONNX_DEFAULT);
   if(handle==INVALID_HANDLE)return Fail("create",GetLastError());
   if(!Setup()){Release();return false;}
   stage="ready";return true;
  }
  bool Run(const float &x[],const float &mtf[],const bool &mask[],AnRawOutput &out){
   ZeroMemory(out);error=0;stage="";
   if(!Ready())return Fail("not initialized",-1);
   if(ArraySize(x)!=5120||ArraySize(mtf)!=5120||ArraySize(mask)!=128)
      return Fail("input element count",-1);
   if(!AnOutputFinite(x)||!AnOutputFinite(mtf))return Fail("nonfinite input",-1);
   // Flat contiguous arrays; shape metadata configured once in Setup.
   // Use native one-byte mask values, corresponding to ONNX BOOL.
   uchar mask_bytes[128];
   for(int i=0;i<128;i++)mask_bytes[i]=(uchar)(mask[i]?1:0);
   ResetLastError();
   if(!OnnxRun(handle,ONNX_NO_CONVERSION,x,mtf,mask_bytes,
      out.direction,out.quantiles,out.rv,out.excursions,out.barrier,
      out.pattern,out.pattern_detail,out.jump,out.path_mu,out.path_diag,
      out.path_factors,out.action_quantiles,out.decision,out.regime,out.mtf_attention)){
    int code=GetLastError();ZeroMemory(out);return Fail("inference",code);
   }
   if(!AnOutputFinite(out.direction)||!AnOutputFinite(out.quantiles)||
      !AnOutputFinite(out.rv)||!AnOutputFinite(out.excursions)||!AnOutputFinite(out.barrier)||
      !AnOutputFinite(out.pattern)||!AnOutputFinite(out.pattern_detail)||!AnOutputFinite(out.jump)||
      !AnOutputFinite(out.path_mu)||!AnOutputFinite(out.path_diag)||!AnOutputFinite(out.path_factors)||
      !AnOutputFinite(out.action_quantiles)||!AnOutputFinite(out.decision)||
      !AnOutputFinite(out.regime)||!AnOutputFinite(out.mtf_attention)){
    ZeroMemory(out);return Fail("nonfinite output",-1);
   }
   stage="inference ok";return true;
  }
};
#endif
