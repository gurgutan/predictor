//+------------------------------------------------------------------+
//|                                                         MMTE.mq5 |
//|                                                 Slepovichev Ivan |
//|                                               http://www.mama.ru |
//+------------------------------------------------------------------+
#property copyright "Slepovichev Ivan"
#property link      "http://www.дфштчппвгтуцп.ru"
#property version   "1.00"

#include <Expert.mqh>

input int     StopLoss      = 1000;
input int     TakeProfit    = 1000;
input double  MAX_VOL       = 0.4;
input double  MIN_LOT       = 0.1;
input int     DELAY         = 10; // задержка в секундах
input double  CONFIDENCE    = 0.5;
input double  STD           = 0.00300;   // Стандартное отклонение
//input double  REINVEST = 0.01; // реинвестировать в %

// ==================================================================
// Переменные
int         MAGIC = 2000;
CExpert     e;
int         db = 0;
string      filename = "predictions.sqlite";
// Параметры эксперта
int         expert_delay = DELAY;   // задержка перед итерацией действия эксперта
// Параметры трала
int         tral_dist    = 400;  // уровень трейлинга позиции в пунктах
int         tral_step    = 400;    // величина шага трейлинга в пунктах
int         tral_delay   = 10;   // задержка перед трэйлингом в секундах


//+------------------------------------------------------------------+
//| Типы данных                                   |
//+------------------------------------------------------------------+
struct Prediction {
    ulong             rdate;
    float            rprice;
    string           pmodel;
    ulong             pdate;
    float            plow;
    float            phigh;
    float            prob;
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
    return(INIT_SUCCEEDED);
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
    db = DatabaseOpen(filename, DATABASE_OPEN_READONLY /*| DATABASE_OPEN_COMMON*/);
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
    bool res = e.Init(MAGIC, Symbol(), PERIOD_M5);
//    expert_delay = PeriodSeconds(PERIOD_M5);
//    e.InitTimer(expert_delay);
//    e.InitTral(tral_dist, tral_step, tral_delay);
    return(res);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CheckDealOpen()
{
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double reinvest_k = 1; //REINVEST * equity / 300 ;
    double t = GetTrend();
    double trend = fmax(-1, fmin(1, t));
    double target_vol = MAX_VOL * trend * reinvest_k;
    double pos_volume = e.GetPosVolume();
    double d = target_vol - pos_volume;
    double vol = NormalizeDouble(MIN_LOT * reinvest_k, 2);
    Print("trend=", DoubleToString(t, 4));
    if(d == 0)
        return;
    if(d >= MIN_LOT && pos_volume < MAX_VOL * reinvest_k) {
        e.DealOpen(ORDER_TYPE_BUY, vol, StopLoss, TakeProfit);
    }
    if(d <= -MIN_LOT && pos_volume > -MAX_VOL * reinvest_k) {
        e.DealOpen(ORDER_TYPE_SELL, vol, StopLoss, TakeProfit);
    }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double GetTrend()
{
    Prediction p;
    double cur_price = (e.BasePrice(ORDER_TYPE_BUY) + e.BasePrice(ORDER_TYPE_SELL)) / 2;
    if(cur_price == 0)
        return 0;
    if(!Request(p)) {
        return 0;
    }
    ShowInfo(cur_price, p);
    DrawTrend(cur_price, p);
    if(p.prob < CONFIDENCE)
        return 0;
    double d = ((p.phigh + p.plow) / 2 - cur_price) / STD;
    return d;
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Request(Prediction& p)
{
// интервал - последняя минута
    ulong startdate = (ulong)TimeCurrent() - 360;
    ulong enddate = (ulong)TimeCurrent() + 60;
    string req_str = "SELECT * FROM pdata WHERE rdate>" + IntegerToString(startdate) + " and rdate<" + IntegerToString(enddate) + " ORDER BY pdate DESC";
    Print(startdate, "  ", enddate); // DEBUG
    if(db == 0) {
        if(!InitDB()) {
            return false;
        }
    }
    int request = DatabasePrepare(db, req_str);
    if(request == INVALID_HANDLE) {
        Print("DB: ", filename, " ошибка запроса с кодом ", GetLastError());
        return false;
    }
    bool success = DatabaseReadBind(request, p);
//if(success)
//Print(enddate, " ", p.rprice, " ", p.phigh, " ", p.plow, " ", p.pdate);
//else
//Print("Нет данных");
    DatabaseFinalize(request);
    return success;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void ShowInfo(double price, Prediction & p)
{
    double high = NormalizeDouble(p.phigh, _Digits);
    double low = NormalizeDouble(p.plow, _Digits);
    double curprice = NormalizeDouble(price, _Digits);
    double prob = NormalizeDouble(p.prob, 5);
    double trend = ((p.phigh + p.plow) / 2 - price) / STD;
    string text = "(" + DoubleToString(low, 5) + "," + DoubleToString(high, 5) + ") p=" + DoubleToString(prob, 2) + " trend=" + DoubleToString(trend, 2);
    string label_name = "prediction";
    if(ObjectFind(0, label_name) < 0)
        ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);
    ObjectSetString(0, label_name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, 10);
    ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, 30);
    ObjectSetInteger(0, label_name, OBJPROP_COLOR, clrWhite);
    ObjectSetString(0, label_name, OBJPROP_FONT, "Consolas");
    ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 8);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DrawTrend(double price, Prediction & p)
{
    datetime t1 = (datetime)p.rdate; // TimeCurrent();
    double price1 = NormalizeDouble(price, _Digits);
    datetime t2 = (datetime)p.pdate;
    double price2 = NormalizeDouble((p.plow + p.phigh) / 2, _Digits);
    double price3 = NormalizeDouble(p.plow, _Digits);
    double price4 = NormalizeDouble(p.phigh, _Digits);
    string label_name = "trend";
    if(ObjectFind(0, label_name) >= 0) {
        ObjectDelete(0, label_name);
    }
    ObjectCreate(0, label_name, OBJ_ARROWED_LINE, 0, t1, price1, t2, price2, _Digits);
    string rect_name = "range_rect";
    if(ObjectFind(0, rect_name) >= 0) {
        ObjectDelete(0, rect_name);
    }
   ObjectCreate(0, rect_name, OBJ_RECTANGLE, 0, t2, price3, t2+2, price4, _Digits);
   ObjectSetInteger(0,rect_name,OBJPROP_COLOR,clrDeepSkyBlue);
}
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
