//+------------------------------------------------------------------+
//|                                                         MMTE.mq5 |
//|                                                 Slepovichev Ivan |
//|                                               http://www.mama.ru |
//+------------------------------------------------------------------+
#property copyright "Slepovichev Ivan"
#property link      "http://www.mama.ru"
#property version   "1.00"

#include <Expert.mqh>

input int     StopLoss      = 1000;
input int     TakeProfit    = 4000;
input double  MAX_VOL       = 0.4;
input double  MIN_LOT       = 0.1;
input int     DELAY         = 300;
input double CONFIDENCE = 0.7;
input double STD = 0.00300;   // Стандартное отклонение
input double THRESHOLD = 0.00100;

// ==================================================================
// Переменные
int         MAGIC=2000;
CExpert     e;
int db=0;
string filename="predictions.sqlite";
// Параметры эксперта
int         expert_delay = DELAY;   // задержка перед итерацией действия эксперта
// Параметры трала
int         tral_dist    = 400; // уровень трейлинга позиции в пунктах
int         tral_step    = 400;    // величина шага трейлинга в пунктах
int         tral_delay   = 10;   // задержка перед трэйлингом в секундах


//+------------------------------------------------------------------+
//| Типы данных                                   |
//+------------------------------------------------------------------+
struct Prediction {
   int               id;
   datetime          rdate;
   float             rprice;
   string            pmodel;
   datetime          pdate;
   float             plow;
   float             phigh;
   float             prob;
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   if(!InitExpert()) {
      Print("Не смог инициализировать эксперта!");
      ExpertRemove();
   };
   if(!InitDB()) {
      Print("Не смог подключиться к БД");
      ExpertRemove();
   }
   EventSetTimer(DELAY);
   return(0);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnTesterInit()
{
   if(!InitDB()) {
      Print("Не смог подключиться к БД");
      ExpertRemove();
      return(1);
   }
   return 0;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTesterDeinit()
{
   DatabaseClose(db);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool InitDB()
{
   db=DatabaseOpen(filename, DATABASE_OPEN_READONLY | DATABASE_OPEN_COMMON);
   if(db==INVALID_HANDLE) {
      Print("DB: ", filename, " ошибка открытия ", GetLastError());
      return false;
   }
   return true;
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   DatabaseClose(db);
   EventKillTimer();
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   /*
       e.SetNoLoss(15);
       if(!e.Main()) return;
       if(!e.TimerCheck()) return;
       if(fabs(e.GetPosVolume())>=MAX_VOL) return; // Проверка на совокупный объем
       CheckDealOpen();
   */
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
{
//e.SetNoLoss(15);
   if(!e.Main())
      return;
//if(!e.TimerCheck()) return;
//if(fabs(e.GetPosVolume())>=MAX_VOL) return; // Проверка на совокупный объем
   CheckDealOpen();
}
//+------------------------------------------------------------------+
//| Инициализация экспертов                                          |
//+------------------------------------------------------------------+
bool InitExpert()
{
   bool res=e.Init(MAGIC,Symbol(),Period());
   expert_delay = PeriodSeconds(PERIOD_M5);
   e.InitTimer(expert_delay);
//e.InitTral(tral_dist, tral_step, tral_delay);
   return(res);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CheckDealOpen()
{
   double t = GetTrend();
   double trend=fmax(-1, fmin(1,t));
   double target_vol = MAX_VOL*trend;
   int pos_type = e.GetPosType(); 
   double pos_volume = e.GetPosVolume();
   //double pos_dest = pos_type==POSITION_TYPE_BUY ? 1 : (pos_type==POSITION_TYPE_SELL ? -1 : 0);
   double d = target_vol-pos_volume;// * pos_dest;
   if(d==0)
      return;
   if(d>0 && fabs(pos_volume)<MAX_VOL) {
      e.DealOpen(ORDER_TYPE_BUY,MIN_LOT,StopLoss,TakeProfit);
   }
   if(d<0 && fabs(pos_volume)<MAX_VOL) {
      e.DealOpen(ORDER_TYPE_SELL,MIN_LOT,StopLoss,TakeProfit);
   }
   /*
   if(deal_volume>0 && e.GetPosVolume()<MAX_VOL) {
       e.DealOpen(ORDER_TYPE_BUY,MIN_LOT,StopLoss,TakeProfit);
   }
   if(deal_volume<0 && e.GetPosVolume()<MAX_VOL) {
       e.DealOpen(ORDER_TYPE_SELL,MIN_LOT,StopLoss,TakeProfit);
   }
   */
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetTrend()
{
   Prediction p;
   double cur_price = (e.BasePrice(ORDER_TYPE_BUY)+e.BasePrice(ORDER_TYPE_SELL))/2;
   if(cur_price==0)
      return 0;
   if(!Request(p))
      return 0;
   if(p.prob < CONFIDENCE)
      return 0;
   //ShowInfo(cur_price, p);
   //DrawTrend(cur_price, p);
   double d = (p.phigh+p.plow)/2-cur_price;
   return d/STD;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Request(Prediction& p)
{
// интервал - последняя минута
   long startdate = (long)TimeCurrent()-60;
   long enddate = (long)TimeCurrent()+1;
   string req_str ="SELECT * FROM pdata WHERE rdate>"+ IntegerToString(startdate)+ " and rdate<"+IntegerToString(enddate)+" ORDER BY pdate DESC";
   if(db==0) {
      db=DatabaseOpen(filename, DATABASE_OPEN_READONLY);
      if(db==INVALID_HANDLE) {
         Print("DB: ", filename, " ошибка открытия ", GetLastError());
         return false;
      }
   }
   int request = DatabasePrepare(db,req_str);
   if(request==INVALID_HANDLE) {
      Print("DB: ", filename, " ошибка запроса с кодом ", GetLastError());
      return false;
   }
   bool success = DatabaseReadBind(request, p);
   DatabaseFinalize(request);
   return success;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ShowInfo(double price, Prediction& p)
{
   double high = NormalizeDouble(p.phigh, _Digits);
   double low = NormalizeDouble(p.plow, _Digits);
   double curprice = NormalizeDouble(price, _Digits);
   double prob = NormalizeDouble(p.prob, 5);
   string text = "Prediction: dest="+DoubleToString(curprice, 5)+" low="+DoubleToString(low, 5)+" high="+DoubleToString(high, 5)+" prob="+DoubleToString(prob, 4);
   string label_name = "prediction";
   ObjectCreate(0,label_name,OBJ_LABEL,0,0,0);
   ObjectSetString(0,label_name,OBJPROP_TEXT,text);
   ObjectSetInteger(0,label_name,OBJPROP_XDISTANCE,10);
   ObjectSetInteger(0,label_name,OBJPROP_YDISTANCE,30);
   ObjectSetInteger(0,label_name,OBJPROP_COLOR,clrWhite);
   ObjectSetString(0,label_name,OBJPROP_FONT,"Consolas");
   ObjectSetInteger(0,label_name,OBJPROP_FONTSIZE,8);
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DrawTrend(double price, Prediction& p)
{
   datetime t1 = TimeCurrent();
   double price1 = NormalizeDouble(price, _Digits);
   datetime t2 = p.pdate;
   double price2 = NormalizeDouble((p.plow+p.phigh)/2, _Digits);
   double price3 = NormalizeDouble(p.plow, _Digits);
   double price4 = NormalizeDouble(p.phigh, _Digits);
   ObjectCreate(0, "line_trend", OBJ_ARROWED_LINE, 0, t1, price1, t2, price2, _Digits);
   //ObjectCreate(0, "rect_trend", OBJ_RECTANGLE, 0, t2, price3, t2+4, price4, _Digits);
   //ObjectSetInteger(0,"rect_trend",OBJPROP_COLOR,clrDeepSkyBlue);
}
//+------------------------------------------------------------------+
