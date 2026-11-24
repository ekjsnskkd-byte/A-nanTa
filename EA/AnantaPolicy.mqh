// Model-driven policy planner. Does not send orders. Native MQL5.
#ifndef ANANTA_POLICY_MQH
#define ANANTA_POLICY_MQH
#include "AnantaFusion.mqh"
enum AnPolicySignal { AN_SIGNAL_SHORT=-1, AN_SIGNAL_WAIT=0, AN_SIGNAL_LONG=1 };
enum AnPolicyExit { AN_KEEP=0, AN_CUT_LOSS=1 };
struct AnPolicyConfig {
 double base_risk, basket_risk, max_margin_fraction, min_direction, min_agreement;
 double min_utility, min_reward_risk, max_reversal, min_stability;
 double pyramid_multiplier, profit_credit_fraction, exit_pressure;
 double min_add_profit_r, roundtrip_cost_per_lot, gap_reserve_per_lot;
 int max_layers, expiry_bars;
};
void AnPolicyDefaults(AnPolicyConfig &c){
 c.base_risk=0.005;c.basket_risk=0.02;c.max_margin_fraction=0.20;
 c.min_direction=0.12;c.min_agreement=0.55;c.min_utility=0;
 c.min_reward_risk=1.2;c.max_reversal=0.70;c.min_stability=0.35;
 c.pyramid_multiplier=1.25;c.profit_credit_fraction=0.25;c.exit_pressure=0.55;
 c.min_add_profit_r=0.5;c.roundtrip_cost_per_lot=0;c.gap_reserve_per_lot=0;
 c.max_layers=3;c.expiry_bars=6;
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
 return c.base_risk>0&&c.basket_risk>=c.base_risk&&c.basket_risk<=0.05&&
 c.max_margin_fraction>0&&c.max_margin_fraction<=1&&c.max_layers>=1&&c.max_layers<=5&&
 c.pyramid_multiplier>=1&&c.pyramid_multiplier<=2&&c.profit_credit_fraction>=0&&c.profit_credit_fraction<=1&&
 c.min_direction>0&&c.min_direction<=1&&c.min_agreement>=0&&c.min_agreement<=1&&
 c.min_reward_risk>0&&c.max_reversal>=0&&c.max_reversal<=1&&c.min_stability>=0&&c.min_stability<=1&&
 c.exit_pressure>0&&c.exit_pressure<=1&&c.min_add_profit_r>=0&&c.expiry_bars>0&&
 c.roundtrip_cost_per_lot>=0&&c.gap_reserve_per_lot>=0;
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
 double favorable=MathMax(0,f.mfe[3]),adverse=MathAbs(f.mae[3]);
 if(favorable<=0||adverse<=0)return false;
 double tp=reference*MathExp(side*favorable),sl=reference*MathExp(-side*adverse);
 // Round toward entry (do not increase model stop distance).
 tp=side>0?AnTickDown(tp,tick):AnTickUp(tp,tick);
 sl=side>0?AnTickUp(sl,tick):AnTickDown(sl,tick);
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
 double eq=AccountInfoDouble(ACCOUNT_EQUITY),free=AccountInfoDouble(ACCOUNT_MARGIN_FREE);
 if(eq<=0||free<=0)return false;
 // Diminish risk in unstable or uncertain forecasts, never increase base risk
 // for confidence alone. Pyramiding can raise target lots, but cap basket loss.
 double quality=AnPClamp(s.direction_confidence*s.market_stability*(1-s.risk_score),0,1);
 double target=eq*c.base_risk*quality*MathPow(c.pyramid_multiplier,b.layers);
 if(b.layers>0)target=MathMin(target,eq*c.base_risk*quality+c.profit_credit_fraction*b.protected_profit);
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
#endif
