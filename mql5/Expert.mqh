//+------------------------------------------------------------------+
//|                                                       Expert.mqh |
//|                                                 Slepovichev Ivan |
//|                                               http://www.mama.ru |
//+------------------------------------------------------------------+
#property copyright "Slepovichev Ivan"
#property link      "http://www.mama.ru"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\DealInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Timer.mqh>
#include <TrailingStop.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CExpert {
  private:
  protected:
    string            info;
    bool              ready;
    string            symbol;
    int               magic;            // магик эксперта
    ulong             delay;
    ENUM_TIMEFRAMES   timeframe;        // рабочий таймфрейм
    CTimer            timer;
    CSymbolInfo       syminfo;
    CTrailingStop     tral;
    CTrade            trade;
    CPositionInfo     pos;
  public:
                     CExpert();
                    ~CExpert();
    //--- Функции инициализации. Позаимствовал у Алексеева Сергея
    virtual bool      Init(int _magic, string _symbol, ENUM_TIMEFRAMES tf);
    virtual bool      Init(int _magic, string _symbol, ENUM_TIMEFRAMES tf, int _level, int _step, int _period);
    virtual bool      Main();                          // Обновить инфо о символе,
    void              InitTral(int _level, int _step, int _period);
    void              InitTimer(int _period);
    //--- Функции управления позицией
    ulong             DealOpen(long dir, double lot, int SL, int TP, string comment);  // совершение сделки с указанными параметрами
    bool              DealClose();
    ulong             GetDealByOrder(ulong order);                     // получить тикет сделки по тикету ордера
    double            CountProfitByDeal(ulong ticket);                 // посчитать полученную прибыль по тикету сделки
    //--- Функции получения сигналов/событий.
    bool              CheckNewBar();
    bool              CheckTime(datetime start, datetime end);
    bool              TimerCheck() {
        return(timer.Check());
    };  // истина, если прошло больше _period времени с предыдущей проверки
    //--- Макросы нормализации.
    double            NormalPrice(double d);          // нормализация цены
    double            NormalDbl(double d, int n = -1); // нормализация цены на тик
    double            BasePrice(long dir);            // возвращает цену Bid/Ask для указанного направления
    double            ReversPrice(long dir);          // возвращает цену Bid/Ask для обратного направления
    double            NormalOpen(long dir, double op, double stop); // нормализация цены открытия отложенного одера
    double            NormalTP(long dir, double op, double pr, int TP, double stop); // нормализация тейкпрофита с учетом стопуровня и спреда
    double            NormalSL(long dir, double op, double pr, int SL, double stop); // нормализация стоплоса с учетом стопуровня и спреда
    double            NormalLot(double lot);          // нормализация лота с учетом свойств символа
    //--- Информационные функции
    void              AddInfo(string st, bool ini = false);
    void              ErrorHandle(int err, ulong ticket, string str);
    //---- Информации об открытой позиции
    long              GetPosType();
    double            GetPosVolume();
    double            GetPosPrice();
    double            GetPosTP();
    double            GetPosSL();
    //---- Функции изменения позиции
    bool              SetNoLoss(int _dist);  // переводит позицию в безубыток на расстоянии _dist
    bool              SetNoLoss();           // переводит позицию в безубыток на минимальном расстоянии
    //---- Информация о символе
    double            GetSymPoint();
    string            GetSymName();

    double            GetLastDealProfit();
    double            GetLastDealPrice();
    double            GetLastDealClosePrice();
    double            GetLastDealVolume();
};
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpert::CExpert() {
    ready = false;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
CExpert::~CExpert(void) {  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpert::Init(int _magic, string _symbol, ENUM_TIMEFRAMES tf) {
    delay = 0;
    magic = _magic;
    symbol = _symbol;
    timeframe = tf;
    trade.SetExpertMagicNumber(magic);
    syminfo.Name(_symbol);
    syminfo.RefreshRates();
    syminfo.Select(true);
    pos.Select(symbol);
    tral.Off();
    ready = true;
    return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpert::Init(int _magic, string _symbol, ENUM_TIMEFRAMES tf, int _level, int _step, int _period) {
    delay = _period;
    magic = _magic;
    symbol = _symbol;
    timeframe = tf;
    syminfo.Name(_symbol);
    syminfo.RefreshRates();
    syminfo.Select(true);
    pos.Select(symbol);
    trade.SetExpertMagicNumber(magic);
    InitTral(_level, _step, _period);
    tral.On();
    InitTimer(_period);
    if(delay > 0) timer.On(delay);
    ready = true;
    return(true);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CExpert::InitTral(int _level, int _step, int _period) {
    tral.Init(symbol, magic, _level, _step, _period);
    tral.On();
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CExpert::InitTimer(int _period) {
    timer.On(_period);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpert::Main() { // Главный модуль
    if(!ready) return(false);
    if(!MQL5InfoInteger(MQL5_TRADE_ALLOWED) || !TerminalInfoInteger(TERMINAL_CONNECTED))
        return(false);                            // если торговля невозможна, то выходим
    info = "";                                  // сбросили информационную строку
    syminfo.Refresh();
    syminfo.RefreshRates(); // обновили параметры символа
    pos.Select(symbol);     // выберем позу
    tral.Tral();
    return(true);
}
//===================================================================
//+------------------------------------------------------------------+
//|  Работа с открытой позицией                                      |
//|  GetPosVolume возвращает -volume если позиция SELL и             |
//|  volume, если позиция BUY                                        |
//+------------------------------------------------------------------+
double CExpert::GetPosVolume() {
    if(pos.Select(symbol))
        if(pos.Magic() == magic) {
            double vol = NormalLot(pos.Volume());
            if(pos.PositionType() == POSITION_TYPE_SELL)
                vol = -vol;
            return(vol);
        }
    return(0);
};
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
long CExpert::GetPosType() {
    if(pos.Select(symbol))
        if(pos.Magic() == magic)
            return(pos.PositionType());
    return(-1);
};
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CExpert::GetPosPrice() {
    if(pos.Select(symbol))
        if(pos.Magic() == magic)
            return(NormalDbl(pos.PriceOpen()));
    return(0);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CExpert::GetPosTP() {
    if(pos.Select(symbol))
        if(pos.Magic() == magic)
            return(NormalDbl(pos.Volume()));
    return(0);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CExpert::GetPosSL() {
    if(pos.Select(symbol))
        if(pos.Magic() == magic)
            return(NormalDbl(pos.StopLoss()));
    return(0);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpert::SetNoLoss(int _dist) {
    tral.SetMinStop(_dist);
    return(tral.NoLossMove());
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpert::SetNoLoss() {
    tral.SetMinStop(0);
    return(tral.NoLossMove());
}
//===================================================================
//+------------------------------------------------------------------+
//| Работа со сделками                                               |
//+------------------------------------------------------------------+
double CExpert::GetLastDealVolume() {
    PositionSelect(symbol);
    HistorySelectByPosition(PositionGetInteger(POSITION_IDENTIFIER));
    uint total = HistoryDealsTotal();
    for(uint i = total - 1; i >= 0; i--) {
        ulong deal = HistoryDealGetTicket(i);
        if(magic == HistoryDealGetInteger(deal, DEAL_MAGIC))
            return(HistoryDealGetDouble(deal, DEAL_VOLUME));
    }
    return(0);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CExpert::GetLastDealProfit() {
    PositionSelect(symbol);
    HistorySelectByPosition(PositionGetInteger(POSITION_IDENTIFIER));
    uint total = HistoryDealsTotal();
    for(uint i = total - 1; i >= 0; i--) {
        ulong deal = HistoryDealGetTicket(i);
        if(magic == HistoryDealGetInteger(deal, DEAL_MAGIC))
            return(HistoryDealGetDouble(deal, DEAL_PROFIT));
    }
    return(0);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CExpert::GetLastDealPrice() {
    PositionSelect(symbol);
    HistorySelectByPosition(PositionGetInteger(POSITION_IDENTIFIER));
    uint total = HistoryDealsTotal();
    for(uint i = total - 1; i >= 0; i--) {
        ulong deal = HistoryDealGetTicket(i);
        if(magic == HistoryDealGetInteger(deal, DEAL_MAGIC))
            return(HistoryDealGetDouble(deal, DEAL_PRICE));
    }
    return(0);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
double CExpert::GetSymPoint() {
    syminfo.Refresh();
    return(syminfo.Point());
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string CExpert::GetSymName() {
    return(syminfo.Name());
}
//===================================================================
//+------------------------------------------------------------------+
//|   функция проверки появления нового бара                         |
//+------------------------------------------------------------------+
bool CExpert::CheckNewBar() {
    static datetime prevTime[2];
    datetime currentTime[1];
    CopyTime(symbol, timeframe, 0, 1, currentTime);
    int _ = timeframe == PERIOD_M30;
    if(currentTime[0] == prevTime[_])return(false);
    else {
        prevTime[_] = currentTime[0];
        return(true);
    }
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpert::CheckTime(datetime start, datetime end) {
    datetime dt = TimeCurrent();                        // текущее время
    if(start < end) if(dt >= start && dt < end) return(true); // проверяем нахождение в промежутке
    if(start >= end) if(dt >= start || dt < end) return(true);
    return(false);
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
ulong CExpert::DealOpen(long dir, double lot, int SL, int TP, string comment = "") { // совершение сделки с указанными параметрами
    double op, sl, tp, apr, StopLvl;
// определили параметры цены
    syminfo.RefreshRates();
    syminfo.Refresh();
    StopLvl = syminfo.StopsLevel() * syminfo.Point(); // запомнили стопуровень
    apr = ReversPrice(dir);
    op = BasePrice(dir);      // цена открытия
    sl = NormalSL(dir, op, apr, SL, StopLvl);       // стоплос
    tp = NormalTP(dir, op, apr, TP, StopLvl);       // тейкпрофит
// открываем позицию
    trade.PositionOpen(symbol, (ENUM_ORDER_TYPE)dir, lot, op, sl, tp, comment);
    ulong order = trade.ResultOrder();
    if(order <= 0) return(0); // тикет одера
    return(GetDealByOrder(order));                  // вернули тикет сделки
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CExpert::DealClose() {
    double op, sl, tp, apr, StopLvl;
// определили параметры цены
    syminfo.RefreshRates();
    syminfo.Refresh();
    trade.PositionClose(symbol);
    return(0);
}
//---------------------------------------------------------------   GetDealByOrder
ulong CExpert::GetDealByOrder(ulong order) { // получение тикета сделки по тикету ордера
    PositionSelect(symbol);
    HistorySelectByPosition(PositionGetInteger(POSITION_IDENTIFIER));
    uint total = HistoryDealsTotal();
    for(uint i = 0; i < total; i++) {
        ulong deal = HistoryDealGetTicket(i);
        if(order == HistoryDealGetInteger(deal, DEAL_ORDER))
            return(deal);                            // запомнили тикет сделки
    }
    return(0);
}
//---------------------------------------------------------------   CountProfit
double CExpert::CountProfitByDeal(ulong ticket) { // профит позиции по тикету сделки
    CDealInfo deal;
    deal.Ticket(ticket);               // тикет сделки
    HistorySelect(deal.Time(), TimeCurrent());         // выбрать все сделки после данной
    uint total = HistoryDealsTotal();
    long pos_id = deal.PositionId();                   // получаем идентификатор позиции
    double prof = 0;
    for(uint i = 0; i < total; i++) { // ищем все сделки с этим идентификатором
        ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_POSITION_ID) != pos_id) continue;
        prof += HistoryDealGetDouble(ticket, DEAL_PROFIT); // суммируем профит
    }
    return(prof);                                      // возвращаем профит
}

//+------------------------------------------------------------------+
//| Макросы нормализации                                             |
//+------------------------------------------------------------------+
double CExpert::NormalDbl(double d, int n = -1) {
    if(n < 0) return(::NormalizeDouble(d, syminfo.Digits()));
    return(NormalizeDouble(d, n));
}
//---------------------------------------------------------------   NP
double CExpert::NormalPrice(double d) {
    return(NormalDbl(MathRound(d / syminfo.TickSize()) * syminfo.TickSize()));
}
//---------------------------------------------------------------   NPR
double CExpert::BasePrice(long dir) {
    if(dir == (long)ORDER_TYPE_BUY) return(syminfo.Ask());
    if(dir == (long)ORDER_TYPE_SELL) return(syminfo.Bid());
    return(WRONG_VALUE);
}
//---------------------------------------------------------------   APR
double CExpert::ReversPrice(long dir) {
    if(dir == (long)ORDER_TYPE_BUY) return(syminfo.Bid());
    if(dir == (long)ORDER_TYPE_SELL) return(syminfo.Ask());
    return(WRONG_VALUE);
}
//---------------------------------------------------------------   NOP
double CExpert::NormalOpen(long dir, double op, double stop) {
    if(dir == ORDER_TYPE_BUY_LIMIT) return(NormalPrice(MathMin(op, syminfo.Ask() - stop)));
    if(dir == ORDER_TYPE_BUY_STOP) return(NormalPrice(MathMax(op, syminfo.Ask() + stop)));
    if(dir == ORDER_TYPE_SELL_LIMIT) return(NormalPrice(MathMax(op, syminfo.Bid() + stop)));
    if(dir == ORDER_TYPE_SELL_STOP) return(NormalPrice(MathMin(op, syminfo.Bid() - stop)));
    return(WRONG_VALUE);
}
//---------------------------------------------------------------   NTP
double CExpert::NormalTP(long dir, double op, double pr, int TP, double stop) {
    if(TP == 0) return(NormalPrice(0));
    if(dir == ORDER_TYPE_BUY || dir == ORDER_TYPE_BUY_STOP || dir == ORDER_TYPE_BUY_LIMIT) return(NormalPrice(MathMax(op + TP * syminfo.Point(), pr + stop)));
    if(dir == ORDER_TYPE_SELL || dir == ORDER_TYPE_SELL_STOP || dir == ORDER_TYPE_SELL_LIMIT) return(NormalPrice(MathMin(op - TP * syminfo.Point(), pr - stop)));
    return(WRONG_VALUE);
}
//---------------------------------------------------------------   NSL
double CExpert::NormalSL(long dir, double op, double pr, int SL, double stop) {
    if(SL == 0) return(NormalPrice(0));
    if(dir == ORDER_TYPE_BUY || dir == ORDER_TYPE_BUY_STOP || dir == ORDER_TYPE_BUY_LIMIT) return(NormalPrice(MathMin(op - SL * syminfo.Point(), pr - stop)));
    if(dir == ORDER_TYPE_SELL || dir == ORDER_TYPE_SELL_STOP || dir == ORDER_TYPE_SELL_LIMIT) return(NormalPrice(MathMax(op + SL * syminfo.Point(), pr + stop)));
    return(WRONG_VALUE);
}
//---------------------------------------------------------------   NL
double CExpert::NormalLot(double lot) {
    int k = 0;
    double ll = lot, ls = syminfo.LotsStep();
    if(ls <= 0.001) k = 3;
    else if(ls <= 0.01) k = 2;
    else if(ls <= 0.1) k = 1;
    ll = NormalDbl(MathMin(syminfo.LotsMax(), MathMax(syminfo.LotsMin(), ll)), k);
    return(ll);
}
//--- Информационные функции
//---------------------------------------------------------------   INF
void CExpert::AddInfo(string st, bool ini = false) {
    string zn = "\n      ", zzn = "\n               ";
    if(ini) info = info + zn + st;
    else info = info + zzn + st;
}
//---------------------------------------------------------------   ErrorHandle
void CExpert::ErrorHandle(int err, ulong ticket, string str) {
    Print("-Err(", err, ") ", magic, " #", ticket, " | " + str);
    switch(err) {
    case TRADE_RETCODE_REJECT:
    case TRADE_RETCODE_TOO_MANY_REQUESTS:
        Sleep(2000);        // wait 2 seconds
        break;
    case TRADE_RETCODE_PRICE_OFF:
    case TRADE_RETCODE_PRICE_CHANGED:
    case TRADE_RETCODE_REQUOTE:
        syminfo.Refresh(); // refresh symbol info
        syminfo.RefreshRates();
        break;
    }
}
//+------------------------------------------------------------------+
