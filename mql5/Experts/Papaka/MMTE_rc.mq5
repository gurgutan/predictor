//+------------------------------------------------------------------+
//|                                                         MMTE.mq5 |
//|                                                 Slepovichev Ivan |
//|                                               http://www.mama.ru |
//+------------------------------------------------------------------+
#property copyright "Slepovichev Ivan"
#property link      "http://www.хертебеанеграаль.хер"
#property version   "1.01"


#include <Expert.mqh>

// ==================================================================
// настройки для модели trendencoder
input int     ORD_SL          = 1024;
input int     ORD_TP          = 256;
input int     SIG_MIN_POINTS  = 128;
input int     SIG_DIST        = 128;
input double  SIG_K_LOW       = -0.5;
input double  SIG_K_HIGH      = 4.0;
input double  VOL_HIGH        = 0.1;
input double  VOL_MIN         = 0.1;
input int     EQUITY_UNIT     = 80000;// маржа для единицы реинвеста
input ulong   DELAY           = 8;   // задержка таймера в секундах
input string  SYMBOL_NAME     = "EURUSD";
input ENUM_TIMEFRAMES FRAME   = PERIOD_H1;
input bool    TRAIL_ON        = false;
input int     TRAIL_POINTS    = 64;  // уровень трейлинга позиции в пунктах

string        model           = "trendencoder";
int           tral_delay      = 10;   // задержка перед трэйлингом в секундах
int           tral_step       = 32;
// ==================================================================
// Переменные
int         MAGIC = 1979;
string      filename = "predictions.sqlite";
int         db = 0;
uint        DB_FLAGS = DATABASE_OPEN_READONLY | DATABASE_OPEN_COMMON;
bool        DEBUG = false;
long        SIG_OBSOLETE = 60*30; // Количество секунд, за которое устаревает прогноз

CExpert     e;  // эксперт

//+------------------------------------------------------------------+
//| Типы данных                                                      |
//+------------------------------------------------------------------+
struct Prediction {
   ulong             rdate;
   float             rprice;
   string            symbol;
   string            model;
   ulong             pdate;
   float             price;
   float             low;
   float             high;
   float             confidence;
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
   if(!InitDB(DB_FLAGS)) {
      Print("Не смог подключиться к БД");
      ExpertRemove();
   }
   ulong align_sec = (DELAY - (ulong)TimeLocal() % DELAY) % 3600;
   Print("Выравнивание времени: " + TimeLocal() + "  " + IntegerToString(align_sec));
   Sleep((int)align_sec * 1000 + 2000);
   EventSetTimer(DELAY);
   Print("ММТЕ запущен");
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnTesterInit()
{
   if(!InitDB(DB_FLAGS)) {
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
bool InitDB(uint flags)
{
   db = DatabaseOpen(filename, flags);
   if(db == INVALID_HANDLE) {
      Print("DB: ", filename, " ошибка открытия ", GetLastError());
      return false;
   }
   Print("Подключено к ", filename);
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
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(!e.Main()) return;
   e.SetNoLoss(SIG_DIST);
//if(!e.TimerCheck()) return;
   Prediction prediction;
   if(!Request(prediction)) return;
   CheckDealClose(prediction);
   CheckDealOpen(prediction);
   double price = GetCurrentPrice();
   DrawTrend(price, prediction);
   ShowInfo(price, prediction);
}
//+------------------------------------------------------------------+
//| Инициализация экспертов                                          |
//+------------------------------------------------------------------+
bool InitExpert()
{
   bool res = e.Init(MAGIC, SYMBOL_NAME, FRAME);
//    e.SetNoLoss(15);
   if(TRAIL_ON)
      e.InitTral(TRAIL_POINTS, tral_step, tral_delay);
   return(res);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CheckDealClose(Prediction & p)
{
   double pos_volume = e.GetPosVolume();
   if(fabs(pos_volume)==0) return;
   double predicted_price = p.price;
   double open_price = p.rprice;
   double cur_price = GetCurrentPrice();
   double difference = NormalizeDouble(predicted_price - cur_price, 5);
// Закрываем позицию, если прогноз цены и позиция разнонаправлены
   if(sign(pos_volume) != sign(difference)) {
      e.DealClose();
   }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CheckDealOpen(Prediction & p)
{
   double cur_price = GetCurrentPrice();
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double reinvest = fmin(32, fmax(1, NormalizeDouble(equity / EQUITY_UNIT, 0)));
   double predicted_price = p.price;
   double open_price = p.rprice;
// отклонение текущей цены от предсказанной
   double difference = NormalizeDouble(predicted_price - cur_price, 5);
// отклонение текущей цены от цены открытия бара
   double difference_open = NormalizeDouble(predicted_price - open_price, 5);
   double pos_price = e.GetPosPrice();
   double pos_volume = e.GetPosVolume();
   double order_volume = NormalizeDouble(VOL_MIN*reinvest, 2);

   int tp = ORD_TP; // fabs(trend / _Point)-5;

// Проверка прогноза и текущей цены
   if(!IsDifferenceNormal(open_price, cur_price, predicted_price)) return;

// Проверка времени
   if(!IsTimeToOpen(p.rdate)) return;

// Проверка объемов
   if(!IsVolumeNormal(order_volume, VOL_HIGH*reinvest)) return;

   if(difference > 0) {
      if(pos_volume > 0 && pos_price - cur_price < SIG_DIST * _Point) return;
      e.DealOpen(ORDER_TYPE_BUY, order_volume, ORD_SL, tp, p.model);
   }
   if(difference < 0) {
      if(pos_volume < 0 && cur_price - pos_price < SIG_DIST * _Point) return;
      e.DealOpen(ORDER_TYPE_SELL, order_volume, ORD_SL, tp, p.model);
   }
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool IsVolumeNormal(double order_volume, double max_volume)
{
   double pos_volume = e.GetPosVolume();
   if(fabs(pos_volume) + order_volume > max_volume)
      return false;
   return true;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool IsDifferenceNormal(double open_price, double current_price, double predicted_price)
{
   double k;
   double difference_current = NormalizeDouble(predicted_price - current_price, 6);
   double difference_open = NormalizeDouble(predicted_price - open_price, 6);

// Условие на диапазон отклонений цены
   double difference_points = fabs(difference_current / _Point);
   if(difference_points < SIG_MIN_POINTS) return false;

// Если прогноз == 0, условие не выполнено
   if(difference_open==0) return false;

// Условие на отношение отклонений от цены открытия:
// текущее_отклонение/предсказанное_отклонение
   k = fabs(difference_current/difference_open);
   if(k<SIG_K_LOW || k>SIG_K_HIGH) {
      if(DEBUG) {
         Print("Не выполнено условие отношения отклонений: "+
               DoubleToString(k, 4)+
               " не в ["+
               DoubleToString(SIG_K_LOW) +","+DoubleToString(SIG_K_HIGH)+"]");
         return false;
      }
   }
   return true;
}

//+------------------------------------------------------------------+
//| Функция для проверки условия открытия ордера по времени          |
//+------------------------------------------------------------------+
bool IsTimeToOpen(datetime start_time)
{
   datetime    current_time=TimeCurrent();
   MqlDateTime current_time_struct;
   TimeToStruct(current_time, current_time_struct);
// Прогноз устаревает за SIG_OBSOLETE секунд
   if((long)(TimeCurrent()-start_time)>SIG_OBSOLETE) return false;
// В пятницу после 21:00 сделки не открываем
   if(current_time_struct.day_of_week==5 && current_time_struct.hour>21) return false;
   return true;
   /*
      Alert("Год: "        +(string)stm.year);
      Alert("Месяц: "      +(string)stm.mon);
      Alert("Число: "      +(string)stm.day);
      Alert("Час: "        +(string)stm.hour);
      Alert("Минута: "     +(string)stm.min);
      Alert("Секунда: "    +(string)stm.sec);
      Alert("День недели: "+(string)stm.day_of_week);
      Alert("День года: "  +(string)stm.day_of_year);
   */
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Request(Prediction & p)
{
   ulong startdate = (ulong)TimeCurrent() - 3600;
   ulong enddate = (ulong)TimeCurrent();
   string req_str = 
      "SELECT * FROM pdata WHERE rdate>=" +
      IntegerToString(startdate) + 
      " and rdate<=" + 
      IntegerToString(enddate) + 
      " ORDER BY rdate DESC, pdate ASC";
   if(db == 0) {
      if(!InitDB(DB_FLAGS)) {
         return false;
      }
   }
   int request = DatabasePrepare(db, req_str);
   if(request == INVALID_HANDLE) {
      Print("DB: ", filename, " ошибка запроса с кодом ", GetLastError());
      return false;
   }
   /*for(i=0; DatabaseReadBind(request, p); i++)
   {
   }*/
   bool success = DatabaseReadBind(request, p);
   DatabaseFinalize(request);
   return success;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ShowInfo(double price, Prediction & p)
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double reinvest = NormalizeDouble(fmin(32, fmax(1, equity / EQUITY_UNIT)),2);
   double cur_price = GetCurrentPrice();
   double pred_price = p.price;
   double trend = NormalizeDouble(pred_price - cur_price, 5);
   double difference_open = NormalizeDouble(pred_price - p.rprice, 5);

   string text = TimeToString((datetime)p.rdate, TIME_MINUTES) +
                 " (" + DoubleToString(difference_open/_Point, 0)+"),  " +
                 TimeToString(TimeCurrent(), TIME_MINUTES) +
                 " (" + DoubleToString(trend/_Point, 0) + ")" +
                 "   reinvest=" + DoubleToString(reinvest,2);
   string label_name = "prediction";
   if(ObjectFind(0, label_name) < 0)
      ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);
   ObjectSetString(0, label_name, OBJPROP_TEXT, text);
   ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, 30);
   ObjectSetInteger(0, label_name, OBJPROP_COLOR, clrYellow);
   ObjectSetString(0, label_name, OBJPROP_FONT, "Arial Black");
   ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 11);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DrawTrend(double price, Prediction & p)
{
   int rect_length = 5 * 16 * 60;
   datetime t1 = (datetime)p.rdate; // TimeCurrent();
   double price1 = NormalizeDouble(price, _Digits);
   double open_price = NormalizeDouble(p.rprice, _Digits);
   datetime t2 = (datetime)p.pdate;
   double price2 = NormalizeDouble(p.price, _Digits);
   double price3 = NormalizeDouble(p.low, _Digits);
   double price4 = NormalizeDouble(p.high, _Digits);
   int rect_fill = (int)(p.confidence * rect_length);
   string line_min = "line_min";
   if(ObjectFind(0, line_min) >= 0)
      ObjectDelete(0, line_min);
   ObjectCreate(0, line_min, OBJ_ARROWED_LINE, 0, TimeCurrent(), price1, t2, price3, _Digits);
   string line_max = "line_max";
   if(ObjectFind(0, line_max) >= 0)
      ObjectDelete(0, line_max);
   ObjectCreate(0, line_max, OBJ_ARROWED_LINE, 0, TimeCurrent(), price1, t2, price4, _Digits);
   string line_open = "line_open"+IntegerToString(p.rdate);
   if(ObjectFind(0, line_open) >= 0)
      ObjectDelete(0, line_open);
   ObjectCreate(0, line_open, OBJ_ARROWED_LINE, 0, t1, open_price, t2, price2, _Digits);
   ObjectSetInteger(0, line_open, OBJPROP_COLOR, clrCyan);
   DrawRect(t2, price3, t2 + rect_length, price4, p.confidence);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DrawRect(double t1, double p1, double t2, double p2, double confidence)
{
   string rect_name = "range_rect";
   if(ObjectFind(0, rect_name) >= 0)
      ObjectDelete(0, rect_name);
   ObjectCreate(0, rect_name, OBJ_RECTANGLE, 0, t1, p1, t2, p2, _Digits);
   ObjectSetInteger(0, rect_name, OBJPROP_FILL, false);
   ObjectSetInteger(0, rect_name, OBJPROP_COLOR, clrAzure);
   string rect_name2 = "range_rect_fill";
   if(ObjectFind(0, rect_name2) >= 0)
      ObjectDelete(0, rect_name2);
   int rect_length = t2 - t1;
   ObjectCreate(0, rect_name2, OBJ_RECTANGLE, 0, t1 + 1, p1 + _Point, t1 + rect_length * confidence, p2 - _Point, _Digits);
   ObjectSetInteger(0, rect_name2, OBJPROP_FILL, true);
   ObjectSetInteger(0, rect_name2, OBJPROP_COLOR, clrDeepSkyBlue);
   /*
   string label_name = "text_confidence";
   if(ObjectFind(0, label_name) >= 0)
       ObjectDelete(0, label_name);
   ObjectCreate(0, label_name, OBJ_TEXT, 0, t1 + 5 * 5 * 60, NormalizeDouble((p1 + p2) / 2 + 20 * _Point, 6));
   ObjectSetString(0, label_name, OBJPROP_TEXT, DoubleToString(confidence, 2));
   ObjectSetInteger(0, label_name, OBJPROP_COLOR, clrWhite);
   ObjectSetString(0, label_name, OBJPROP_FONT, "Arial Black");
   ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 12);
   */
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double sign(double a)
{
   if(a > 0) return 1;
   if(a == 0) return 0;
   return -1;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetCurrentPrice()
{
   return NormalizeDouble((e.BasePrice(ORDER_TYPE_BUY) + e.BasePrice(ORDER_TYPE_SELL)) / 2, 5);
}
//+------------------------------------------------------------------+
