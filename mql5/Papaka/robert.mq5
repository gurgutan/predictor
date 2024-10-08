//+------------------------------------------------------------------+
//|                                                       robert.mq5 |
//|                                  Copyright 2022, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Include                                                          |
//+------------------------------------------------------------------+
#include <Expert\Expert.mqh>
//--- available signals
#include <Expert\MySignals\SampleSignal.mqh>
//--- available trailing
#include <Expert\Trailing\TrailingFixedPips.mqh>
//--- available money management
#include <Expert\Money\MoneyFixedLot.mqh>
//+------------------------------------------------------------------+
//| Inputs                                                           |
//+------------------------------------------------------------------+
//--- inputs for expert
input string Expert_Title                  ="robert"; // Document name
ulong        Expert_MagicNumber            =11446;    //
bool         Expert_EveryTick              =true;    //
//--- inputs for main signal
input int    Signal_ThresholdOpen          =10;       // Signal threshold value to open [0...100]
input int    Signal_ThresholdClose         =10;       // Signal threshold value to close [0...100]
input double Signal_PriceLevel             =0.0;      // Price level to execute a deal
input double Signal_StopLevel              =50.0;     // Stop Loss level (in points)
input double Signal_TakeLevel              =50.0;     // Take Profit level (in points)
input int    Signal_Expiration             =4;        // Expiration of pending orders (in bars)
input int    Signal_DB_OpenThreshold       =128;      // OpenThreshold
input int    Signal_DB_CloseThreshold      =0;        // CloseThreshold
 double Signal_DB_StopLoss            =256.0;    // PredictionDB(128,0,256.0,...)
 double Signal_DB_TakeProfit          =256.0;    // PredictionDB(128,0,256.0,...)
input int    Signal_DB_ExpirationSeconds   =600;      // PredictionDB(128,0,256.0,...)
input double Signal_DB_Weight              =1.0;      // PredictionDB(128,0,256.0,...) Weight [0...1.0]
//--- inputs for trailing
input int    Trailing_FixedPips_StopLevel  =30;       // Stop Loss trailing level (in points)
input int    Trailing_FixedPips_ProfitLevel=50;       // Take Profit trailing level (in points)
//--- inputs for money
input double Money_FixLot_Percent          =10.0;     // Percent
input double Money_FixLot_Lots             =0.1;      // Fixed volume
//+------------------------------------------------------------------+
//| Global expert object                                             |
//+------------------------------------------------------------------+
CExpert ExtExpert;
//+------------------------------------------------------------------+
//| Initialization function of the expert                            |
//+------------------------------------------------------------------+
int OnInit()
{   
   Print("ММТЕ запущен");
   AlignTime();
//--- Initializing expert
   if(!ExtExpert.Init(Symbol(),Period(),Expert_EveryTick,Expert_MagicNumber)) {
      //--- failed
      printf(__FUNCTION__+": error initializing expert");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//--- Creating signal
   CExpertSignal *signal=new CExpertSignal;
   
   if(signal==NULL) {
      //--- failed
      printf(__FUNCTION__+": error creating signal");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//---
   ExtExpert.InitSignal(signal);
   signal.ThresholdOpen(Signal_ThresholdOpen);
   signal.ThresholdClose(Signal_ThresholdClose);
   signal.PriceLevel(Signal_PriceLevel);
   signal.StopLevel(Signal_StopLevel);
   signal.TakeLevel(Signal_TakeLevel);
   signal.Expiration(Signal_Expiration);
//--- Creating filter CSampleSignal
   CSampleSignal *filter0=new CSampleSignal;
   if(filter0==NULL) {
      //--- failed
      printf(__FUNCTION__+": error creating filter0");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
   signal.AddFilter(filter0);
//--- Set filter parameters
   filter0.SetTestMode();
   filter0.OpenThreshold(Signal_DB_OpenThreshold);
   filter0.CloseThreshold(Signal_DB_CloseThreshold);
   filter0.StopLoss(Signal_DB_StopLoss);
   filter0.TakeProfit(Signal_DB_TakeProfit);
   filter0.ExpirationSeconds(Signal_DB_ExpirationSeconds);
   filter0.Weight(Signal_DB_Weight);
//--- Creation of trailing object
   CTrailingFixedPips *trailing=new CTrailingFixedPips;
   if(trailing==NULL) {
      //--- failed
      printf(__FUNCTION__+": error creating trailing");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//--- Add trailing to expert (will be deleted automatically))
   if(!ExtExpert.InitTrailing(trailing)) {
      //--- failed
      printf(__FUNCTION__+": error initializing trailing");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//--- Set trailing parameters
   trailing.StopLevel(Trailing_FixedPips_StopLevel);
   trailing.ProfitLevel(Trailing_FixedPips_ProfitLevel);

//--- Creation of money object
   CMoneyFixedLot *money=new CMoneyFixedLot;
   if(money==NULL) {
      //--- failed
      printf(__FUNCTION__+": error creating money");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//--- Add money to expert (will be deleted automatically))
   if(!ExtExpert.InitMoney(money)) {
      //--- failed
      printf(__FUNCTION__+": error initializing money");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//--- Set money parameters
   money.Percent(Money_FixLot_Percent);
   money.Lots(Money_FixLot_Lots);
//--- Check all trading objects parameters
   if(!ExtExpert.ValidationSettings()) {
      //--- failed
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//--- Tuning of all necessary indicators
   if(!ExtExpert.InitIndicators()) {
      //--- failed
      printf(__FUNCTION__+": error initializing indicators");
      ExtExpert.Deinit();
      return(INIT_FAILED);
   }
//--- ok
   return(INIT_SUCCEEDED);
}

void AlignTime()
{
   int DELAY = 2;
   ulong align_sec = (DELAY - (ulong)TimeLocal() % DELAY) % 3600;
   Print("Выравнивание времени: " + TimeLocal() + "  " + IntegerToString(align_sec));
   Sleep((int)align_sec * 1000 + 2000);
   EventSetTimer(DELAY);
}

//+------------------------------------------------------------------+
//| Deinitialization function of the expert                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ExtExpert.Deinit();
}
//+------------------------------------------------------------------+
//| "Tick" event handler function                                    |
//+------------------------------------------------------------------+
void OnTick()
{
   ExtExpert.OnTick();
}
//+------------------------------------------------------------------+
//| "Trade" event handler function                                   |
//+------------------------------------------------------------------+
void OnTrade()
{
   ExtExpert.OnTrade();
}
//+------------------------------------------------------------------+
//| "Timer" event handler function                                   |
//+------------------------------------------------------------------+
void OnTimer()
{
   ExtExpert.OnTimer();
}
//+------------------------------------------------------------------+
