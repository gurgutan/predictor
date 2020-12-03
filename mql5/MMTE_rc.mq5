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
input int     StopLoss      = 1024;
input int     TakeProfit    = 256;
input int MaxTP = 1024;
input int MinTP = 128;
input double  MAX_VOL       = 0.1;
input double  MIN_LOT       = 0.1;
input int     MIN_DIST      = 64;
input ulong   DELAY         = 60;   // задержка в секундах таймера
//input double  CONFIDENCE    = 0;  // порог вероятности прогноза
//input double  THRESHOLD     = 1;  // порог чувствительности
//input double  RANGE         = 0.0003; // отклонение цены
input string  SYMBOL_NAME   = "EURUSD";
input ENUM_TIMEFRAMES FRAME = PERIOD_H1;
input bool    USE_TRAL      = false;
input int     tral_dist     = 64;  // уровень трейлинга позиции в пунктах

string        model         = "trendencoder";
int           tral_delay    = 10;   // задержка перед трэйлингом в секундах
int           tral_step     = 32;
// ==================================================================
// Переменные

int         MAGIC = 1979;
string      filename = "predictions.sqlite";
int         expert_delay = DELAY;   // задержка перед итерацией действия эксперта
int         db = 0;
uint        DB_FLAGS = DATABASE_OPEN_READONLY | DATABASE_OPEN_COMMON;

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

struct TargetVolume {
    double            v;
    double            confidence;
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
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
    Sleep((int)align_sec * 1000 + 10000);
    EventSetTimer(DELAY);
    Print("ММТЕ запущен");
    return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
int OnTesterInit() {
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
void OnTesterDeinit() {
    DatabaseClose(db);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool InitDB(uint flags) {
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
void OnDeinit(const int reason) {
    DatabaseClose(db);
    EventKillTimer();
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer() {
    if(!e.Main())
        return;
    e.SetNoLoss(MIN_DIST);
//if(!e.TimerCheck()) return;
//if(fabs(e.GetPosVolume())>=MAX_VOL) return; // Проверка на совокупный объем
    CheckDealClose();
    CheckDealOpen();
}
//+------------------------------------------------------------------+
//| Инициализация экспертов                                          |
//+------------------------------------------------------------------+
bool InitExpert() {
    bool res = e.Init(MAGIC, SYMBOL_NAME, FRAME);
//    expert_delay = PeriodSeconds(PERIOD_M5);
//    e.SetNoLoss(15);
//    e.InitTimer(expert_delay);
    if(USE_TRAL)
        e.InitTral(tral_dist, tral_step, tral_delay);
    return(res);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CheckDealClose() {
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double sign(double a) {
    if(a > 0) return 1;
    if(a == 0) return 0;
    return -1;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CheckDealOpen() {
    TargetVolume target;
    Prediction p;
    if(!GetTargetVolume(target, p))
        return;
//double equity = AccountInfoDouble(ACCOUNT_EQUITY);
//double reinvest_k = 1;// + REINVEST * equity / 10000;
    double cur_price = (e.BasePrice(ORDER_TYPE_BUY) + e.BasePrice(ORDER_TYPE_SELL)) / 2;
    double pred_price = p.price;
    double pos_volume = e.GetPosVolume();
    double d = NormalizeDouble(pred_price - cur_price, 5);
    double pos_price = e.GetPosPrice();
    double vol = NormalizeDouble(MIN_LOT, 2);
    int tp = fabs(d / _Point);
//Print("d=" + DoubleToString(d / _Point, 0));
    if(fabs(pos_volume) > 0 && sign(pos_volume) != sign(d)) {
        //e.DealClose();
        if(pos_volume > 0)
            e.DealOpen(ORDER_TYPE_SELL, vol, StopLoss, TakeProfit, p.model);
        if(pos_volume < 0)
            e.DealOpen(ORDER_TYPE_BUY, vol, StopLoss, TakeProfit, p.model);
    }
    if(fabs(d / _Point) > MaxTP || fabs(d / _Point) < MinTP) return;
    // если прошло 30 мин
    //if((long)(TimeCurrent()-p.rdate)>60*30) return;
    if(d > 0) {
        if(fabs(pos_volume) + vol > MAX_VOL)
            return;
        if(pos_volume > 0 && cur_price - pos_price > -MIN_DIST * _Point) {
            Print(cur_price - pos_price);
            return;
        }
        e.DealOpen(ORDER_TYPE_BUY, vol, StopLoss, TakeProfit, p.model);
    }
    if(d < 0) {
        if(fabs(pos_volume) + vol > MAX_VOL)
            return;
        if(pos_volume < 0 && pos_price - cur_price > -MIN_DIST * _Point) {
            Print(pos_price - cur_price);
            return;
        }
        e.DealOpen(ORDER_TYPE_SELL, vol, StopLoss, TakeProfit, p.model);
    }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool GetTargetVolume(TargetVolume & target, Prediction & p) {
    if(!Request(p)) {
        Print("Нет котировок");
        return false;
    }
    double cur_price = (e.BasePrice(ORDER_TYPE_BUY) + e.BasePrice(ORDER_TYPE_SELL)) / 2;
    if(cur_price == 0)
        return false;
    ShowInfo(cur_price, p);
    DrawTrend(cur_price, p);
    target.confidence = p.confidence;
    target.v = p.price - cur_price;
    return true;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool Request(Prediction & p) {
    ulong startdate = (ulong)TimeCurrent() - 3600;
    ulong enddate = (ulong)TimeCurrent();
    string req_str = "SELECT * FROM pdata WHERE rdate>=" + IntegerToString(startdate) + " and rdate<=" + IntegerToString(enddate) + " ORDER BY rdate DESC, pdate ASC";
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
void ShowInfo(double price, Prediction & p) {
    double high = NormalizeDouble(p.high, _Digits);
    double low = NormalizeDouble(p.low, _Digits);
    double curprice = NormalizeDouble(p.rprice, _Digits);
    double confidence = NormalizeDouble(p.confidence, 5);
    double trend = p.price - p.rprice;
    string text = (datetime)p.rdate + " " + DoubleToString((low + high) / 2, 5) + " p=(" + DoubleToString(trend, 2) + "," + DoubleToString(confidence, 2) + ")";
    string label_name = "prediction";
    if(ObjectFind(0, label_name) < 0)
        ObjectCreate(0, label_name, OBJ_LABEL, 0, 0, 0);
    ObjectSetString(0, label_name, OBJPROP_TEXT, text);
    ObjectSetInteger(0, label_name, OBJPROP_XDISTANCE, 10);
    ObjectSetInteger(0, label_name, OBJPROP_YDISTANCE, 30);
    ObjectSetInteger(0, label_name, OBJPROP_COLOR, clrYellow);
    ObjectSetString(0, label_name, OBJPROP_FONT, "Arial Black");
    ObjectSetInteger(0, label_name, OBJPROP_FONTSIZE, 12);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DrawTrend(double price, Prediction & p) {
    int rect_length = 5 * 16 * 60;
    datetime t1 = (datetime)p.rdate; // TimeCurrent();
    double price1 = NormalizeDouble(price, _Digits);
    double open_price = NormalizeDouble(p.rprice, _Digits);
    datetime t2 = (datetime)p.pdate;
    double price2 = NormalizeDouble((p.low + p.high) / 2, _Digits);
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
    string line_open = "line_open";
    if(ObjectFind(0, line_open) >= 0)
        ObjectDelete(0, line_open);
    ObjectCreate(0, line_open, OBJ_ARROWED_LINE, 0, t1, open_price, t2, price2, _Digits);
    ObjectSetInteger(0, line_open, OBJPROP_COLOR, clrCyan);
    DrawRect(t2, price3, t2 + rect_length, price4, p.confidence);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void DrawRect(double t1, double p1, double t2, double p2, double confidence) {
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
