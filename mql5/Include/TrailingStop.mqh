//+------------------------------------------------------------------+
//|                                                 TrailingStop.mqh |
//|                                                 Slepovichev Ivan |
//|                                               http://www.mama.ru |
//+------------------------------------------------------------------+
#property copyright "Slepovichev Ivan"
#property link      "http://www.mama.ru"

#include <Timer.mqh>
#include <Trade\Trade.mqh>
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CTrailingStop
  {
private:
   CTimer            timer;      // таймер для отслеживания задержки
   long              min_stop;   // минимальное расстояние от цены позиции на котором будет сработает трал (перевод в безубыток)
protected:
   bool              active;  // Текущее состояние активности
   string            symbol;         // Символ
   long              magic;     // для опознавания "своих" ордеров
   long              level;     // уровень траления (расстояние в пунктах от текущей цены на котором будет двигаться СЛ)
   long              step;      // шаг в пунктах, с которым будет тралить
   long              period;    // задержка в секундах с которой будет проверятся трал
public:
   void              CTrailingStop() { active=false; };
   void             ~CTrailingStop() {};
   void              Init(string _symbol,int _magic,int _level,int _step,int _period);  // Инициализация класса
   void              SetLevel(int _level);
   void              SetMinStop(long _ms);
   void              On()  { active=true; };                                   // Включение трейлинг стопа
   void              Off() { active=false; };                                  // Выключение трейлинг стопа
   bool              Tral();  // Тралит позицию по символу symbol, с периодичностью раз в period секунд, через шаг step, на расстоянии level от текущей цены
                              // возвращает true, если операция удалась

   bool              NoLossMove();  //перевести в безубыток
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTrailingStop::Init(string _symbol,int _magic,int _level,int _step,int _period)
  {
   symbol= _symbol;
   magic = _magic;
   level = _level;
   step  = _step;
   period=_period;
   active=false;
   min_stop=SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL)+1;
   if(level<min_stop) level=min_stop;
   if(step<1) step=4;
   if(period<0) period=0;
   timer.On(period);
  }
//+------------------------------------------------------------------+
void CTrailingStop::SetLevel(int _level)
  {
   min_stop=(int)SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL);
   level=_level;
   if(level<min_stop) level=min_stop;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void CTrailingStop::SetMinStop(long _ms)
  {
   long m=SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL)+1;
   if(_ms>=m) min_stop=_ms;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CTrailingStop::Tral()
  {
   if(!active || timer.GetRound()<1 || !PositionSelect(symbol)) return(false);
   CTrade   trade;
   double pos_price=PositionGetDouble(POSITION_PRICE_OPEN);
   double pos_sl = PositionGetDouble(POSITION_SL);
   double pos_tp = PositionGetDouble(POSITION_TP);
   if(PositionGetInteger(POSITION_MAGIC)==magic)
     {
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
        {
         double curr_price=SymbolInfoDouble(symbol,SYMBOL_BID);
         if(curr_price-pos_sl>(level+step)*_Point)
           {
            double sl = NormalizeDouble(curr_price-level*_Point, _Digits);
            double tp = NormalizeDouble(pos_tp, _Digits);
            bool res=trade.PositionModify(symbol,sl,tp);
            printf(trade.ResultRetcodeDescription());
            return(res);
           }
        } // if(pos.Type()==POSITION_TYPE_BUY)
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
        {
         double curr_price=SymbolInfoDouble(symbol,SYMBOL_ASK);
         if(pos_sl-curr_price>(level+step)*_Point)
           {
            double sl = NormalizeDouble(curr_price+level*_Point, _Digits);
            double tp = NormalizeDouble(pos_tp, _Digits);
            bool res=trade.PositionModify(symbol,sl,tp);
            printf(trade.ResultRetcodeDescription());
            return(res);
           }
        }  // if(pos.Type()==POSITION_TYPE_SELL)
     }   // if(pos.Magic()==magic)
   return(false);
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CTrailingStop::NoLossMove()
  {
   if(!active || !timer.Check() || !PositionSelect(symbol)) return(false);
   CTrade   trade;
   double pos_price=PositionGetDouble(POSITION_PRICE_OPEN);
   double pos_sl = PositionGetDouble(POSITION_SL);
   double pos_tp = PositionGetDouble(POSITION_TP);
//   min_stop=SymbolInfoInteger(symbol,SYMBOL_TRADE_STOPS_LEVEL)+1;

   if(PositionGetInteger(POSITION_MAGIC)==magic)
     {
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY)
        {
         double curr_price=SymbolInfoDouble(symbol,SYMBOL_BID);
         if(curr_price-pos_price>min_stop*_Point && curr_price-pos_sl>0 && pos_sl!=pos_price)
           {
            double sl = NormalizeDouble(pos_price, _Digits);
            double tp = NormalizeDouble(pos_tp, _Digits);
            bool res=trade.PositionModify(symbol,sl,tp);
            return(res);
           }
        } // if(pos.Type()==POSITION_TYPE_BUY)
      if(PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_SELL)
        {
         double curr_price=SymbolInfoDouble(symbol,SYMBOL_ASK);
         if(pos_price-curr_price>min_stop*_Point && pos_sl-curr_price>0 && pos_sl!=pos_price)
           {
            double sl = NormalizeDouble(pos_price, _Digits);
            double tp = NormalizeDouble(pos_tp, _Digits);
            bool res=trade.PositionModify(symbol,sl,tp);
            return(res);
           }
        }  // if(pos.Type()==POSITION_TYPE_SELL)
     }   // if(pos.Magic()==magic)
   return(false);
  }

//+------------------------------------------------------------------+
