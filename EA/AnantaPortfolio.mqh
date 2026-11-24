// Live broker book snapshot for Ananta. One EA instance per symbol/magic.
// Hedging accounts only: netting merges legs and cannot enforce per-leg rules.
#ifndef ANANTA_PORTFOLIO_MQH
#define ANANTA_PORTFOLIO_MQH
#include "AnantaPolicy.mqh"
#include <Trade/Trade.mqh>
// Initial risk must be recorded on fill and restored by position identifier.
// Missing risk history blocks additions rather than assuming a favorable value.
struct AnLegRisk { ulong identifier; double initial_risk_cash; };
double AnInitialRisk(const AnLegRisk &ledger[],ulong id){
 for(int i=0;i<ArraySize(ledger);i++)if(ledger[i].identifier==id)return ledger[i].initial_risk_cash;
 return 0;
}
bool AnSnapshot(const string symbol,const ulong magic,const AnPolicyConfig &c,
 const AnLegRisk &ledger[],AnBookSnapshot &b){
 ZeroMemory(b);b.all_positions_profitable=true;b.min_position_profit_r=1e100;
 if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)return false;
 MqlTick t;if(!SymbolInfoTick(symbol,t)||t.bid<=0||t.ask<t.bid)return false;
 long newest=-1;
 for(int i=PositionsTotal()-1;i>=0;i--){
  ulong ticket=PositionGetTicket(i);if(ticket==0)return false;
  if(PositionGetString(POSITION_SYMBOL)!=symbol||PositionGetInteger(POSITION_MAGIC)!=(long)magic)continue;
  b.layers++;
  int side=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
  if(b.side!=0&&b.side!=side)b.has_opposite=true;
  if(b.side==0)b.side=side;
  double vol=PositionGetDouble(POSITION_VOLUME),entry=PositionGetDouble(POSITION_PRICE_OPEN);
  double sl=PositionGetDouble(POSITION_SL),mark=side>0?t.bid:t.ask;
  double costs=vol*(c.roundtrip_cost_per_lot+c.gap_reserve_per_lot);
  double pnl=PositionGetDouble(POSITION_PROFIT)+PositionGetDouble(POSITION_SWAP)-costs;
  ulong id=(ulong)PositionGetInteger(POSITION_IDENTIFIER);
  double initial=AnInitialRisk(ledger,id);
  if(pnl<=0||!AnPFinite(pnl))b.all_positions_profitable=false;
  if(initial<=0||!AnPFinite(initial))b.min_position_profit_r=0;
  else b.min_position_profit_r=MathMin(b.min_position_profit_r,pnl/initial);
  if(sl<=0||vol<=0||entry<=0||side*(mark-sl)<=0)return false;
  ENUM_ORDER_TYPE type=side>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL;
  double downside=0,locked=0;
  if(!OrderCalcProfit(type,symbol,vol,mark,sl,downside)||
     !OrderCalcProfit(type,symbol,vol,entry,sl,locked))return false;
  b.exposure_risk+=MathMax(0,-downside)+costs;
  b.protected_profit+=MathMax(0,locked+PositionGetDouble(POSITION_SWAP)-costs);
  long ms=PositionGetInteger(POSITION_TIME_MSC);
  if(ms>newest){newest=ms;b.last_fill=entry;}
 }
 for(int i=OrdersTotal()-1;i>=0;i--){
  ulong ticket=OrderGetTicket(i);if(ticket==0)return false;
  if(OrderGetString(ORDER_SYMBOL)!=symbol||OrderGetInteger(ORDER_MAGIC)!=(long)magic)continue;
  b.has_pending=true;
  ENUM_ORDER_TYPE ot=(ENUM_ORDER_TYPE)OrderGetInteger(ORDER_TYPE);
  if(ot!=ORDER_TYPE_BUY_LIMIT&&ot!=ORDER_TYPE_SELL_LIMIT)return false;
  int side=ot==ORDER_TYPE_BUY_LIMIT?1:-1;
  if(b.side!=0&&b.side!=side)b.has_opposite=true;
  double vol=OrderGetDouble(ORDER_VOLUME_CURRENT),entry=OrderGetDouble(ORDER_PRICE_OPEN),sl=OrderGetDouble(ORDER_SL);
  if(vol<=0||sl<=0||entry<=0||side*(entry-sl)<=0)return false;
  double loss=0;ENUM_ORDER_TYPE type=side>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL;
  if(!OrderCalcProfit(type,symbol,vol,entry,sl,loss)||loss>=0)return false;
  b.exposure_risk+=-loss+vol*(c.roundtrip_cost_per_lot+c.gap_reserve_per_lot);
 }
 if(b.layers==0)b.min_position_profit_r=0;
 b.valid=AnPFinite(b.exposure_risk)&&AnPFinite(b.protected_profit)&&!b.has_opposite;
 return b.valid;
}
// Call on every tick and immediately before submitting a new order.
// Cancellation cannot guarantee prevention of a fill already racing at broker.
bool AnGuardPending(const string symbol,const ulong magic,const AnPolicyConfig &c,
 const AnLegRisk &ledger[]){
 AnBookSnapshot b;bool ok=AnSnapshot(symbol,magic,c,ledger,b);
 int pending=0;
 for(int i=OrdersTotal()-1;i>=0;i--){
  if(OrderGetTicket(i)==0)return false;
  if(OrderGetString(ORDER_SYMBOL)==symbol&&OrderGetInteger(ORDER_MAGIC)==(long)magic)pending++;
 }
 if(ok&&b.layers+pending<=3&&(!b.layers||b.all_positions_profitable)&&
    b.exposure_risk<=AccountInfoDouble(ACCOUNT_EQUITY)*c.basket_risk)return true;
 CTrade trade;trade.SetAsyncMode(false);bool success=true;
 for(int i=OrdersTotal()-1;i>=0;i--){
  ulong ticket=OrderGetTicket(i);if(ticket==0){success=false;continue;}
  if(OrderGetString(ORDER_SYMBOL)!=symbol||OrderGetInteger(ORDER_MAGIC)!=(long)magic)continue;
  if(!trade.OrderDelete(ticket)||trade.ResultRetcode()!=TRADE_RETCODE_DONE)success=false;
 }
 return success;
}
#endif
