// EA component version 1.2 (V8 model)
// Model-driven policy planner. Includes checked limit submission. Native MQL5.
#ifndef ANANTA_POLICY_MQH
#define ANANTA_POLICY_MQH
#include "AnantaFusion_v1.0.mqh"
enum AnPolicySignal { AN_SIGNAL_SHORT=-1, AN_SIGNAL_WAIT=0, AN_SIGNAL_LONG=1 };
enum AnPolicyExit { AN_KEEP=0, AN_CUT_LOSS=1 };
struct AnPolicyConfig {
 double base_risk, basket_risk, max_margin_fraction, min_direction, min_agreement;
 double min_utility, min_reward_risk, max_reversal, min_stability;
 double pyramid_multiplier, profit_credit_fraction, exit_pressure;
 double min_add_profit_r, roundtrip_cost_per_lot, gap_reserve_per_lot;
 double max_drawdown, marginal_ev_r, decay_threshold, max_spread_fraction;
 int max_layers, expiry_bars, decay_bars;
};
void AnPolicyDefaults(AnPolicyConfig &c){
 c.base_risk=0.01;c.basket_risk=0.02;c.max_margin_fraction=0.20;
 c.min_direction=0.12;c.min_agreement=0.55;c.min_utility=0;
 c.min_reward_risk=1.2;c.max_reversal=0.70;c.min_stability=0.35;
 c.pyramid_multiplier=1.25;c.profit_credit_fraction=0.25;c.exit_pressure=0.55;
 c.min_add_profit_r=0.5;c.roundtrip_cost_per_lot=0;c.gap_reserve_per_lot=0;
 c.max_layers=3;c.expiry_bars=6;
 c.max_spread_fraction=0.10;c.max_drawdown=0.10;c.marginal_ev_r=0.15;c.decay_threshold=0.60;c.decay_bars=2;
}
// Caller MUST reconcile all positions and pending orders before each plan.
// exposure_risk: account-currency downside from current executable mark to SL,
// including pending orders and costs. Unprotected exposure => snapshot invalid.
// protected_profit: net positive profit locked by actual broker-accepted stops,
// never merely floating PnL. Equity already includes floating profit.
struct AnBookSnapshot {
 bool valid,has_pending,all_positions_profitable,has_opposite;
 int side,layers;
 double exposure_risk,protected_profit,min_position_profit_r,last_fill;
};
struct AnOrderPlan {
 bool valid;
 AnPolicySignal signal;
 double entry,tp,sl,lots,risk_cash,margin_cash;
 int expiry_bars;
};
double AnPClamp(double x,double a,double b){return MathMax(a,MathMin(b,x));}
bool AnPFinite(double x){return MathIsValidNumber(x)&&x!=EMPTY_VALUE;}
bool AnPolicyConfigOK(const AnPolicyConfig &c){
 return c.base_risk>0&&c.base_risk<=0.01&&c.basket_risk>=c.base_risk&&c.basket_risk<=0.05&&
 c.max_margin_fraction>0&&c.max_margin_fraction<=1&&c.max_layers>=1&&c.max_layers<=3&&
 c.pyramid_multiplier>=1&&c.pyramid_multiplier<=2&&c.profit_credit_fraction>=0&&c.profit_credit_fraction<=1&&
 c.min_direction>0&&c.min_direction<=1&&c.min_agreement>=0&&c.min_agreement<=1&&
 c.min_reward_risk>0&&c.max_reversal>=0&&c.max_reversal<=1&&c.min_stability>=0&&c.min_stability<=1&&
 c.exit_pressure>0&&c.exit_pressure<=1&&c.min_add_profit_r>=0&&c.expiry_bars>0&&
 c.roundtrip_cost_per_lot>=0&&c.gap_reserve_per_lot>=0&&
 c.max_spread_fraction>0&&c.max_spread_fraction<1&&c.max_drawdown>0&&c.max_drawdown<1&&c.marginal_ev_r>=0&&c.decay_threshold>0&&c.decay_threshold<=1&&c.decay_bars>=1;
}
AnPolicySignal AnSignal(const AnForecast &f,const AnComposite &s,const AnPolicyConfig &c){
 if(!f.valid||!AnPolicyConfigOK(c)||s.signal_agreement<c.min_agreement||
 s.market_stability<c.min_stability||s.reversal_risk>c.max_reversal)return AN_SIGNAL_WAIT;
 int side=s.direction_score>=c.min_direction?1:(s.direction_score<=-c.min_direction?-1:0);
 if(side==0)return AN_SIGNAL_WAIT;
 int a=side>0?AN_LONG:AN_SHORT,other=side>0?AN_SHORT:AN_LONG;
 if(f.decision[a]<=f.decision[other]||f.decision[a]<=f.decision[AN_WAIT])return AN_SIGNAL_WAIT;
 if(f.utility[a][AN_Q50]<=MathMax(c.min_utility,f.utility[AN_WAIT][AN_Q50]))return AN_SIGNAL_WAIT;
 if(side*s.expected_return<=0)return AN_SIGNAL_WAIT;
 return side>0?AN_SIGNAL_LONG:AN_SIGNAL_SHORT;
}
// Only losing positions are eligible for discretionary early exit.
// Never override broker TP/SL; no trailing or breakeven rule is introduced.
AnPolicyExit AnEarlyExit(const AnForecast &f,const AnComposite &s,const AnPolicyConfig &c,
 int side,double net_profit,double initial_risk_cash){
 if(!f.valid||!AnPolicyConfigOK(c)||(side!=1&&side!=-1)||!AnPFinite(net_profit)||
 !AnPFinite(initial_risk_cash)||initial_risk_cash<=0||net_profit>=0)return AN_KEEP;
 int a=side>0?AN_LONG:AN_SHORT,opp=side>0?AN_SHORT:AN_LONG;
 double contrary=AnPClamp(-side*s.direction_score,0,1);
 double utility_bad=f.utility[a][AN_Q50]<f.utility[AN_WAIT][AN_Q50]?1:0;
 double adverse=AnPClamp(f.decision[opp]-f.decision[a],0,1);
 double draw=AnPClamp(-net_profit/initial_risk_cash,0,1);
 double pressure=0.35*contrary+0.20*adverse+0.20*utility_bad+
                 0.15*s.reversal_risk+0.10*draw;
 return pressure>=c.exit_pressure?AN_CUT_LOSS:AN_KEEP;
}
double AnTickDown(double p,double tick){return MathFloor(p/tick+1e-9)*tick;}
double AnTickUp(double p,double tick){return MathCeil(p/tick-1e-9)*tick;}
// For the 96-bar forecast, anchor levels to the SAME close used for inference.
// MFE/MAE heads are not side-specific; use magnitudes as candidate excursions.
// Reject incompatible broker levels rather than inventing or widening stops.
bool AnPlanLimit(const string symbol,const AnForecast &f,const AnComposite &s,
 const AnPolicyConfig &c,const AnBookSnapshot &b,double reference,AnOrderPlan &p){
 ZeroMemory(p);p.signal=AN_SIGNAL_WAIT;
 if(!b.valid||!AnPolicyConfigOK(c)||!AnPFinite(reference)||reference<=0||
 !AnPFinite(b.exposure_risk)||b.exposure_risk<0||!AnPFinite(b.protected_profit)||
 b.protected_profit<0||b.layers<0||b.layers>=c.max_layers||b.has_pending||b.has_opposite)return false;
 AnPolicySignal sig=AnSignal(f,s,c);if(sig==AN_SIGNAL_WAIT)return false;
 int side=(int)sig;
 if(b.layers>0&& (b.side!=side||!b.all_positions_profitable||
 b.min_position_profit_r<c.min_add_profit_r||b.last_fill<=0))return false;
 MqlTick t;if(!SymbolInfoTick(symbol,t)||t.bid<=0||t.ask<t.bid)return false;
 double tick=SymbolInfoDouble(symbol,SYMBOL_TRADE_TICK_SIZE);
 double point=SymbolInfoDouble(symbol,SYMBOL_POINT);
 double step=SymbolInfoDouble(symbol,SYMBOL_VOLUME_STEP);
 double vmin=SymbolInfoDouble(symbol,SYMBOL_VOLUME_MIN);
 double vmax=SymbolInfoDouble(symbol,SYMBOL_VOLUME_MAX);
 if(tick<=0||point<=0||step<=0||vmin<=0||vmax<vmin)return false;
 double min_dist=MathMax(tick,point*SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL));
 // Near-term adverse excursion for passive entry, using 2/6-bar path quantiles.
 // Use the nearest conservative nonzero pullback predicted by the model.
 double q2=side>0?f.path_cumulative_quantiles[1][0]:f.path_cumulative_quantiles[1][2];
 double q6=side>0?f.path_cumulative_quantiles[3][0]:f.path_cumulative_quantiles[3][2];
 double pull=MathMin(MathAbs(q2),MathAbs(q6));
 if(side*q2>=0||side*q6>=0||pull<=0)return false;
 double entry=reference*MathExp(-side*pull);
 entry=side>0?AnTickDown(entry,tick):AnTickUp(entry,tick);
 if(entry<=0||(side>0&&entry>t.ask-min_dist)||(side<0&&entry<t.bid+min_dist))return false;
 // Pyramiding only at a better trend-progress price than previous entry.
 if(b.layers>0&&side*(entry-b.last_fill)<=0)return false;
 // Excursions represent upper/lower path movement; swap for shorts.
 double upper=MathMax(0,f.mfe[3]),lower=MathMax(0,-f.mae[3]);
 double favorable=side>0?upper:lower,adverse=side>0?lower:upper;
 if(favorable<=0||adverse<=0)return false;
 double tp=reference*MathExp(side*favorable),sl=reference*MathExp(-side*adverse);
 // Round toward entry (do not increase model stop distance).
 tp=side>0?AnTickDown(tp,tick):AnTickUp(tp,tick);
 sl=side>0?AnTickUp(sl,tick):AnTickDown(sl,tick);
 if(t.ask-t.bid>c.max_spread_fraction*MathAbs(tp-entry))return false;
 if(!AnPFinite(tp)||!AnPFinite(sl)||tp<=0||sl<=0||
 side*(tp-entry)<min_dist||side*(entry-sl)<min_dist)return false;
 ENUM_ORDER_TYPE type=side>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL;
 double loss=0,gain=0;
 if(!OrderCalcProfit(type,symbol,vmin,entry,sl,loss)||
 !OrderCalcProfit(type,symbol,vmin,entry,tp,gain)||loss>=0||gain<=0)return false;
 double cost=c.roundtrip_cost_per_lot+c.gap_reserve_per_lot;
 double perlot=-loss/vmin+cost;
 double netgain=gain/vmin-c.roundtrip_cost_per_lot;
 if(perlot<=0||netgain/perlot<c.min_reward_risk)return false;
 // Conservative model barrier proxy: unresolved mass treated as loss.
 // Not a calibrated probability conditional on the actual limit fill.
 int action=side>0?AN_LONG:AN_SHORT;
 double win_proxy=AnPClamp(MathMin(f.barrier[3][action],f.decision[action]),0,1);
 double marginal_ev=win_proxy*netgain-(1-win_proxy)*perlot;
 if(b.layers>0&&marginal_ev/perlot<c.marginal_ev_r)return false;
 double eq=AccountInfoDouble(ACCOUNT_EQUITY),free=AccountInfoDouble(ACCOUNT_MARGIN_FREE);
 if(eq<=0||free<=0)return false;
 // Diminish risk in unstable or uncertain forecasts, never increase base risk
 // for confidence alone. Pyramiding can raise target lots, but cap basket loss.
 double quality=AnPClamp(s.direction_confidence*s.market_stability*(1-s.risk_score),0,1);
 double target=b.layers==0?eq*c.base_risk*quality:eq*c.base_risk*quality*MathPow(c.pyramid_multiplier,b.layers);
 // Basket budget is binding; never credit unsecured floating profit.
 if(b.layers>0)target=MathMin(target,eq*c.base_risk*MathPow(c.pyramid_multiplier,b.layers));
 double budget=MathMin(target,eq*c.basket_risk-b.exposure_risk);
 if(budget<=0)return false;
 double margin_min=0;
 if(!OrderCalcMargin(type,symbol,vmin,entry,margin_min)||margin_min<=0)return false;
 double margin_cap=MathMin(free*c.max_margin_fraction,eq*c.max_margin_fraction);
 double raw=MathMin(vmax,MathMin(budget/perlot,margin_cap*vmin/margin_min));
 // Floor to broker volume step; never round UP to minimum lot.
 double lots=NormalizeDouble(MathFloor(raw/step+1e-10)*step,8);
 if(lots<vmin||lots>vmax)return false;
 double margin=0,actual_loss=0;
 if(!OrderCalcMargin(type,symbol,lots,entry,margin)||
 !OrderCalcProfit(type,symbol,lots,entry,sl,actual_loss)||actual_loss>=0)return false;
 double risk=-actual_loss+lots*cost;
 if(risk>budget+1e-8||margin>margin_cap)return false;
 p.signal=sig;p.entry=entry;p.tp=tp;p.sl=sl;p.lots=lots;
 p.risk_cash=risk;p.margin_cash=margin;p.expiry_bars=c.expiry_bars;p.valid=true;
 return true;
}
// Caller must cancel expired or invalidated unfilled orders; never chase by market order.
bool AnCancelPending(const AnForecast &f,const AnComposite &s,const AnPolicyConfig &c,
 int pending_side,int elapsed_closed_bars){
 return elapsed_closed_bars>=c.expiry_bars||(int)AnSignal(f,s,c)!=pending_side;
}


// Peak equity is account-wide. Persist with terminal globals; do not auto-reset
// on restart or day rollover. Deposits/withdrawals need an explicit rebaseline.
string AnPolicyKey(const string symbol,const ulong magic,const string suffix){
 return StringFormat("AN11.%I64d.%I64u.%s.%s",AccountInfoInteger(ACCOUNT_LOGIN),magic,symbol,suffix);
}
string AnPeakKey(){
 return StringFormat("AN11.%I64d.%s.peak",AccountInfoInteger(ACCOUNT_LOGIN),AccountInfoString(ACCOUNT_SERVER));
}
bool AnDrawdownOK(const AnPolicyConfig &c,double &dd){
 dd=1;
 double eq=AccountInfoDouble(ACCOUNT_EQUITY);
 if(!AnPolicyConfigOK(c)||!AnPFinite(eq)||eq<=0)return false;
 string key=AnPeakKey();
 if(!GlobalVariableCheck(key)&&GlobalVariableSet(key,eq)==0)return false;
 double peak=0;
 for(int i=0;i<5;i++){
  if(!GlobalVariableGet(key,peak)||!AnPFinite(peak)||peak<=0)return false;
  if(eq<=peak)break;
  if(GlobalVariableSetOnCondition(key,eq,peak)){peak=eq;break;}
 }
 if(peak<eq)return false;
 dd=MathMax(0,1-eq/peak);
 return dd<c.max_drawdown;
}
// Persistent per-bar claim prevents a repeated send after timeout or restart.
// A single controller per symbol/magic is required; never auto-retry ambiguous sends.
bool AnClaimBar(const string symbol,const ulong magic,const long bar_ms){
 if(bar_ms<=0||bar_ms>9007199254740991)return false;
 string key=AnPolicyKey(symbol,magic,"bar");
 if(!GlobalVariableCheck(key)&&GlobalVariableSet(key,0)==0)return false;
 double prev=0;
 if(!GlobalVariableGet(key,prev)||prev>=bar_ms)return false;
 if(!GlobalVariableSetOnCondition(key,(double)bar_ms,prev))return false;
 GlobalVariablesFlush();return true;
}
// Recheck actual live positions and orders before sending, independent of
// caller snapshot. Only hedging accounts support per-leg semantics.
bool AnLiveEntryOK(const string symbol,const ulong magic,const int side,
 const AnPolicyConfig &c,const AnBookSnapshot &book,const AnOrderPlan &p){
 if(!AnPFinite(p.risk_cash)||p.risk_cash<=0||!book.valid||!p.valid||book.has_pending||book.has_opposite||
 AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)return false;
 int count=0;double risk=0;
 MqlTick t;if(!SymbolInfoTick(symbol,t)||t.bid<=0||t.ask<t.bid)return false;
 for(int i=PositionsTotal()-1;i>=0;i--){
  if(PositionGetTicket(i)==0)return false;
  if(PositionGetString(POSITION_SYMBOL)!=symbol||PositionGetInteger(POSITION_MAGIC)!=(long)magic)continue;
  count++;
  int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
  double v=PositionGetDouble(POSITION_VOLUME),sl=PositionGetDouble(POSITION_SL);
  double mark=dir>0?t.bid:t.ask;
  double cost=v*(c.roundtrip_cost_per_lot+c.gap_reserve_per_lot);
  if(dir!=side||PositionGetDouble(POSITION_PROFIT)+PositionGetDouble(POSITION_SWAP)-cost<=0||
     v<=0||sl<=0||dir*(mark-sl)<=0)return false;
  double loss=0;
  if(!OrderCalcProfit(dir>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL,symbol,v,mark,sl,loss))return false;
  risk+=MathMax(0,-loss)+cost;
 }
 for(int i=OrdersTotal()-1;i>=0;i--){
  if(OrderGetTicket(i)==0)return false;
  if(OrderGetString(ORDER_SYMBOL)==symbol&&OrderGetInteger(ORDER_MAGIC)==(long)magic)return false;
 }
 double eq=AccountInfoDouble(ACCOUNT_EQUITY);
 if(count!=book.layers||count>=MathMin(3,c.max_layers)||eq<=0)return false;
 return risk+p.risk_cash<=eq*c.basket_risk&&(count>0||p.risk_cash<=eq*c.base_risk);
}
// Forecast decay relative to the entry forecast, per position. Update only on
// newly closed event bars; a winning position is never cut by this function.
struct AnDecayState {
 long last_bar;
 int bad_bars;
 double entry_edge;
 bool initialized;
};
double AnSideEdge(const AnForecast &f,const AnComposite &s,const int side){
 int a=side>0?AN_LONG:AN_SHORT,opp=side>0?AN_SHORT:AN_LONG;
 return 0.40*side*s.direction_score+0.30*(f.decision[a]-f.decision[opp])+
        0.30*AnPClamp((f.utility[a][AN_Q50]-f.utility[AN_WAIT][AN_Q50])/0.005,-1,1);
}
bool AnDecayInit(AnDecayState &d,const AnForecast &f,const AnComposite &s,int side,long bar_ms){
 ZeroMemory(d);
 if(!f.valid||(side!=1&&side!=-1)||bar_ms<=0)return false;
 d.entry_edge=AnSideEdge(f,s,side);
 if(!AnPFinite(d.entry_edge)||d.entry_edge<=0)return false;
 d.last_bar=bar_ms;d.initialized=true;return true;
}
AnPolicyExit AnDecayExit(AnDecayState &d,const AnForecast &f,const AnComposite &s,
 const AnPolicyConfig &c,int side,double pnl,double initial_risk,long bar_ms){
 if(!f.valid||!AnPolicyConfigOK(c)||(side!=1&&side!=-1)||!AnPFinite(pnl))return AN_KEEP;
 if(d.initialized&&bar_ms>d.last_bar){
  double edge=AnSideEdge(f,s,side);
  double decay=1-edge/MathMax(d.entry_edge,1e-8);
  if(AnPFinite(decay)&&decay>=c.decay_threshold)d.bad_bars++;
  else d.bad_bars=0;
  d.last_bar=bar_ms;
 }
 if(pnl>=0)return AN_KEEP;
 if(AnEarlyExit(f,s,c,side,pnl,initial_risk)==AN_CUT_LOSS)return AN_CUT_LOSS;
 if(d.initialized&&d.bad_bars>=c.decay_bars)return AN_CUT_LOSS;
 return AN_KEEP;
}
// Checked submission. Main controller must use this entry point, not direct
// OrderSend, and reconcile accepted/ambiguous results through broker state.
bool AnSubmitLimit(const string symbol,const ulong magic,const long bar_ms,
 const AnPolicyConfig &c,const AnBookSnapshot &book,const AnOrderPlan &p,
 MqlTradeResult &result){
 ZeroMemory(result);
 double dd=0;
 if(!AnPolicyConfigOK(c)||!AnDrawdownOK(c,dd)||!p.valid||
 (p.signal!=AN_SIGNAL_LONG&&p.signal!=AN_SIGNAL_SHORT))return false;
 int side=(int)p.signal;
 if(!AnPFinite(p.entry)||!AnPFinite(p.sl)||!AnPFinite(p.tp)||!AnPFinite(p.lots)||
 p.entry<=0||p.sl<=0||p.tp<=0||p.lots<=0||side*(p.entry-p.sl)<=0||side*(p.tp-p.entry)<=0)return false;
 if(!AnLiveEntryOK(symbol,magic,side,c,book,p))return false;
 if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)||!MQLInfoInteger(MQL_TRADE_ALLOWED)||
 !AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)||!AccountInfoInteger(ACCOUNT_TRADE_EXPERT))return false;
 ENUM_ORDER_TYPE market=side>0?ORDER_TYPE_BUY:ORDER_TYPE_SELL;
 double loss=0,margin=0;
 if(!OrderCalcProfit(market,symbol,p.lots,p.entry,p.sl,loss)||loss>=0||
 !OrderCalcMargin(market,symbol,p.lots,p.entry,margin))return false;
 double risk=-loss+p.lots*(c.roundtrip_cost_per_lot+c.gap_reserve_per_lot);
 if(risk>p.risk_cash+1e-8||margin>MathMin(AccountInfoDouble(ACCOUNT_MARGIN_FREE),AccountInfoDouble(ACCOUNT_EQUITY))*c.max_margin_fraction)return false;
 MqlTradeRequest req;ZeroMemory(req);
 req.action=TRADE_ACTION_PENDING;req.magic=magic;req.symbol=symbol;
 req.type=side>0?ORDER_TYPE_BUY_LIMIT:ORDER_TYPE_SELL_LIMIT;
 req.price=p.entry;req.sl=p.sl;req.tp=p.tp;req.volume=p.lots;
 req.type_filling=ORDER_FILLING_RETURN;
 // Fail closed if broker cannot enforce a server-side expiry.
 if((SymbolInfoInteger(symbol,SYMBOL_EXPIRATION_MODE)&SYMBOL_EXPIRATION_SPECIFIED)==0)return false;
 req.type_time=ORDER_TIME_SPECIFIED;req.expiration=TimeTradeServer()+3600;
 req.comment=StringFormat("AN11-%I64d",bar_ms);
 MqlTradeCheckResult check;ZeroMemory(check);
 ResetLastError();
 if(!OrderCheck(req,check)||(check.retcode!=0&&check.retcode!=TRADE_RETCODE_DONE))return false;
 if(!AnClaimBar(symbol,magic,bar_ms))return false;
 if(!AnDrawdownOK(c,dd)||!AnLiveEntryOK(symbol,magic,side,c,book,p))return false;
 ResetLastError();
 bool sent=OrderSend(req,result);
 return sent&&(result.retcode==TRADE_RETCODE_PLACED||result.retcode==TRADE_RETCODE_DONE)&&
        (result.order>0||result.deal>0);
}
// Call each tick even without a fresh forecast: cancel pending on drawdown,
// layer limit or any losing existing leg. Existing broker TP/SL are untouched.
bool AnPolicyGuardPending(const string symbol,const ulong magic,const AnPolicyConfig &c){
 double dd=0;bool safe=AnDrawdownOK(c,dd);int count=0,pending=0;
 for(int i=PositionsTotal()-1;i>=0;i--){
  if(PositionGetTicket(i)==0){safe=false;continue;}
  if(PositionGetString(POSITION_SYMBOL)!=symbol||PositionGetInteger(POSITION_MAGIC)!=(long)magic)continue;
  count++;double v=PositionGetDouble(POSITION_VOLUME);
  if(PositionGetDouble(POSITION_PROFIT)+PositionGetDouble(POSITION_SWAP)-v*(c.roundtrip_cost_per_lot+c.gap_reserve_per_lot)<=0)safe=false;
 }
 for(int i=OrdersTotal()-1;i>=0;i--){
  if(OrderGetTicket(i)==0){safe=false;continue;}
  if(OrderGetString(ORDER_SYMBOL)==symbol&&OrderGetInteger(ORDER_MAGIC)==(long)magic)pending++;
 }
 if(safe&&count+pending<=MathMin(3,c.max_layers)&&pending<=1)return true;
 bool ok=true;
 for(int i=OrdersTotal()-1;i>=0;i--){
  ulong ticket=OrderGetTicket(i);if(ticket==0){ok=false;continue;}
  if(OrderGetString(ORDER_SYMBOL)!=symbol||OrderGetInteger(ORDER_MAGIC)!=(long)magic)continue;
  MqlTradeRequest req;MqlTradeResult result;ZeroMemory(req);ZeroMemory(result);
  req.action=TRADE_ACTION_REMOVE;req.order=ticket;
  if(!OrderSend(req,result)||result.retcode!=TRADE_RETCODE_DONE)ok=false;
 }
 return ok;
}
#endif
