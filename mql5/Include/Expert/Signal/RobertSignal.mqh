//+------------------------------------------------------------------+
//|                                                 SampleSignal.mqh |
//|                        Copyright 2010, MetaQuotes Software Corp. |
//|                                              https://www.mql5.com |
//+------------------------------------------------------------------+
//|                                               Signal2EMA-ITF.mqh |
//|                      Copyright © 2010, MetaQuotes Software Corp. |
//|                                        http://www.metaquotes.net |
//|                                              Revision 2010.11.15 |
//+------------------------------------------------------------------+
#include <Expert\ExpertSignal.mqh>
// wizard description start
//+------------------------------------------------------------------+
//| Description of the class                                         |
//| Title=Сигналы по БД                                              |
//| Type=SignalAdvanced                                              |
//| Name=PredictionDB                                                |
//| ShortName=DB                                                |
//| Class=CSampleSignal                                              |
//| Page=                                                            |
//| Parameter=OpenThreshold,int,128                                  |
//| Parameter=CloseThreshold,int,0                                   |
//| Parameter=StopLoss,double,50.0                                   |
//| Parameter=TakeProfit,double,50.0                                 |
//| Parameter=ExpirationSeconds,int,600                                     |
//+------------------------------------------------------------------+
// wizard description end
//+------------------------------------------------------------------+
//| Класс CSampleSignal.                                             |
//| Назначение: Класс генератора торговых сигналов по значениям      |
//|             прогноза из подключаемой БД.                         |
//+------------------------------------------------------------------+
class CSampleSignal : public CExpertSignal {
struct Prediction {
   ulong             time; // rdate;
   float             open; // rprice;
   string            symbol;
   string            model;
   ulong             p_time; //pdate;
   float             p_price;
   float             p_low;
   float             p_high;
   float             p_confidence;
};
protected:
   CiMA           m_MA;             // объект для доступа к значениям скользящей средней
   CiOpen         m_open;           // объект для доступа к ценам открытия баров
   CiClose        m_close;          // объект для доступа к ценам закрытия баров
   string         db_name;          // имя БД
   int            db_handle;        // хэндл открытой БД
   int            db_flags;
   int            r_period;         // период для использования в запросе БД
   double         m_stop_loss;      // уровень установки ордера "stop loss" относительно цены открытия
   double         m_take_profit;    // уровень установки ордера "take profit" относительно цены открытия
   int            m_expiration_sec; // время в секундах устаревания прогноза
   int            m_open_threshold;
   int            m_close_threshold;
   Prediction     prediction;       // структура для хранения прогноза
public:
                  CSampleSignal();
   virtual bool   ValidationSettings();
   virtual bool   InitIndicators(CIndicators* indicators);
   virtual int    LongCondition();
   virtual int    ShortCondition();
//   virtual bool   CheckCloseLong(double &price);
//   virtual bool   CheckCloseShort(double &price);
   bool           SetTestMode();
   bool           SetDBName(string dbname);
   bool           Request();
   void           StopLoss(double value)              { m_stop_loss=value;   }
   void           TakeProfit(double value)            { m_take_profit=value; }
   void           OpenThreshold(int value)            { m_open_threshold=value;  }
   void           CloseThreshold(int value)           { m_close_threshold=value;  }
   void           ExpirationSeconds(int value)        { m_expiration_sec=value;  }
           

protected:
   bool           InitDB();
   bool           IsTimeToOpen();
   //--- метод инициализации объектов
   bool           InitMA(CIndicators* indicators);
   bool           InitOpen(CIndicators* indicators);
   bool           InitClose(CIndicators* indicators);
   //--- методы доступа к данным объектов
   double         MA(int index)                       { return(m_MA.Main(index)); }
   double         Open(int index)                     { return(m_open.GetData(index)); }
   double         Close(int index)                    { return(m_close.GetData(index));}
};



//+------------------------------------------------------------------+
//| Конструктор CSampleSignal.                                       |
//| INPUT:  нет.                                                     |
//| OUTPUT: нет.                                                     |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
void CSampleSignal::CSampleSignal()
{
   db_name = "predictions.sqlite";
   db_flags = DATABASE_OPEN_READONLY;
   r_period = 3600;  // в секундах
   m_expiration_sec = 600;   // прогноз устаревает за 600 секунд
   m_stop_loss = 256 * _Point;
   m_take_profit = 256 * _Point;
   InitDB();
}

//+------------------------------------------------------------------+
//| Переключение сигнала в режим тестирования                        |
//+------------------------------------------------------------------+
bool CSampleSignal::SetTestMode()
{
   db_flags = DATABASE_OPEN_READONLY | DATABASE_OPEN_COMMON;
   return(InitDB());
}

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool CSampleSignal::SetDBName(string dbname)
{
   db_name = dbname;
   return(InitDB());
}

//+------------------------------------------------------------------+
//| Проверка параметров настройки.                                   |
//| INPUT:  нет.                                                     |
//| OUTPUT: true-если настройки правильные, иначе false.             |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
bool CSampleSignal::ValidationSettings()
{
//--- проверка параметров
   if(db_handle == 0 || db_handle == INVALID_HANDLE) {
      printf(__FUNCTION__+": ошибка подключения к БД");
      return(false);
   }

   if(r_period<=0) {
      printf(__FUNCTION__+": период запроса должен быть больше нуля");
      return(false);
   }
//--- ok
   return(true);
}

//+------------------------------------------------------------------+
//| Инициализация индикаторов и таймсерий.                           |
//| INPUT:  indicators - указатель на объект-коллекцию               |
//|                      индикаторов и таймсерий.                    |
//| OUTPUT: true-в случае успешного завершения, иначе false.         |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
bool CSampleSignal::InitIndicators(CIndicators* indicators)
{
//--- проверка указателя
   if(indicators==NULL)       return(false);
//--- инициализация скользящей средней
   if(!InitMA(indicators))    return(false);
//--- инициализация таймсерии цен открытия
   if(!InitOpen(indicators))  return(false);
//--- инициализация таймсерии цен закрытия
   if(!InitClose(indicators)) return(false);
//--- успешное завершение
   return(true);
}

//+------------------------------------------------------------------+
//| Инициализация скользящей средней.                                |
//| INPUT:  indicators - указатель на объект-коллекцию               |
//|                      индикаторов и таймсерий.                    |
//| OUTPUT: true-в случае успешного завершения, иначе false.         |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
bool CSampleSignal::InitMA(CIndicators* indicators)
{
//--- инициализация объекта скользящей средней
   if(!m_MA.Create(m_symbol.Name(), PERIOD_H1, 2, 0, MODE_EMA, PRICE_OPEN)) {
      printf(__FUNCTION__+": ошибка инициализации объекта");
      return(false);
   }
   m_MA.BufferResize(3);
//--- добавление объекта в коллекцию
   if(!indicators.Add(GetPointer(m_MA))) {
      printf(__FUNCTION__+": ошибка добавления объекта");
      return(false);
   }
//--- успешное завершение
   return(true);
}

//+------------------------------------------------------------------+
//| Инициализация таймсерии цен открытия.                            |
//| INPUT:  indicators - указатель на объект-коллекцию               |
//|                      индикаторов и таймсерий.                    |
//| OUTPUT: true-в случае успешного завершения, иначе false.         |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
bool CSampleSignal::InitOpen(CIndicators* indicators)
{
//--- инициализация объекта таймсерии
   if(!m_open.Create(m_symbol.Name(),m_period)) {
      printf(__FUNCTION__+": ошибка инициализации объекта");
      return(false);
   }
//--- добавление объекта в коллекцию
   if(!indicators.Add(GetPointer(m_open))) {
      printf(__FUNCTION__+": ошибка добавления объекта");
      return(false);
   }
//--- успешное завершение
   return(true);
}

//+------------------------------------------------------------------+
//| Инициализация таймсерии цен закрытия.                            |
//| INPUT:  indicators - указатель на объект-коллекцию таймсерий.    |
//| OUTPUT: true-в случае успешного завершения, иначе false.         |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
bool CSampleSignal::InitClose(CIndicators* indicators)
{
//--- инициализация объекта таймсерии
   if(!m_close.Create(m_symbol.Name(),m_period)) {
      printf(__FUNCTION__+": ошибка инициализации объекта");
      return(false);
   }
//--- добавление объекта в коллекцию
   if(!indicators.Add(GetPointer(m_close))) {
      printf(__FUNCTION__+": ошибка добавления объекта");
      return(false);
   }
//--- успешное завершение
   return(true);
}

//+------------------------------------------------------------------+
//| Проверка выполнения условия для покупки.                         |
//+------------------------------------------------------------------+
int CSampleSignal::LongCondition()
{
   if(!Request()) 
      {
      printf("Ошибка запроса БД");
      return(false);
      }
   double unit = _Point;
   double current_price = m_symbol.Ask();
   datetime current_time = TimeCurrent();
   double price = current_price;
   double sl = m_symbol.NormalizePrice(price-m_stop_loss*unit);
   double tp = m_symbol.NormalizePrice(price+m_take_profit*unit);   
   double predicted_price = prediction.p_price;
   double open_price = prediction.open;   
   // отклонение текущей цены от предсказанной
   double difference_current = NormalizeDouble(predicted_price - current_price, 6);
   // отклонение текущей цены от цены открытия бара   
   double difference_open = NormalizeDouble(predicted_price - open_price, 6);
   // Условие на диапазон отклонений цены
   double difference_points = difference_current / unit;   
   // проверка устаревания прогноза
   if((long)(current_time-prediction.time) > m_expiration_sec) return(0);
   // проверка порога открытия в пунктах
   if(difference_points > m_open_threshold) return(100);
   // Проверка времени
   //if(!IsTimeToOpen()) return(0);

   return(0);
}

//+------------------------------------------------------------------+
//| Проверка выполнения условия для продажи.                         |
//+------------------------------------------------------------------+
int CSampleSignal::ShortCondition()
{
   if(!Request()) 
      {
      printf("Ошибка запроса БД");
      return(0);
      }
   double unit = _Point;
   double current_price = m_symbol.Bid();
   datetime current_time = TimeCurrent();
   double price = current_price;
   double sl = m_symbol.NormalizePrice(price+m_stop_loss*unit);
   double tp = m_symbol.NormalizePrice(price-m_take_profit*unit);   
   double predicted_price = prediction.p_price;
   double open_price = prediction.open;   
   // отклонение текущей цены от предсказанной
   double difference_current = NormalizeDouble(current_price - predicted_price, 6);
   // отклонение текущей цены от цены открытия бара   
   double difference_open = NormalizeDouble(open_price - predicted_price, 6);
   // Условие на диапазон отклонений цены
   double difference_points = difference_current / unit;
   // проверка устаревания прогноза
   if((long)(current_time-prediction.time) > m_expiration_sec) return(0);
   // проверка порога открытия в пунктах
   if(difference_points > m_open_threshold) return(100);
   // Проверка времени
   //if(!IsTimeToOpen()) return(false);

   return(0);
}
/*
bool CSampleSignal::CheckCloseLong(double &price)
{
   double predicted_price = prediction.p_price;
   double current_price = m_symbol.Bid();
   if(current_price - predicted_price > m_threshold_close) return(true);
   return(false);
}

bool CSampleSignal::CheckCloseShort(double &price)
{
   double predicted_price = prediction.p_price;
   double current_price = m_symbol.Ask();
   if(predicted_price - current_price > m_threshold_close) return(true);
   return(false);
}
*/
//+------------------------------------------------------------------+
//| Функция для проверки условия открытия ордера по времени          |
//+------------------------------------------------------------------+
bool CSampleSignal::IsTimeToOpen()
{
   datetime    current_time=TimeCurrent();
   MqlDateTime current_time_struct;
   TimeToStruct(current_time, current_time_struct);   
// В пятницу после 21:00 сделки не открываем
   if(current_time_struct.day_of_week==5 && current_time_struct.hour>21) return false;
   return true;
}

//+------------------------------------------------------------------+
//| Инициализация БД                                                 |
//| INPUT:  flags      - флаги открытия БД                           |
//| OUTPUT: true-если БД открыта на чтение                           |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
bool CSampleSignal::InitDB()
{
   db_handle = DatabaseOpen(db_name, db_flags);
   if(db_handle == INVALID_HANDLE) {
      Print("DB: ", db_name, " ошибка подключения ", GetLastError());
      return false;
   }
   Print("Подключено к ", db_name);
   return true;
}
//+------------------------------------------------------------------+
//| Запрос прогноза из БД.                                           |
//| INPUT:  price      - ссылка для размещения цены открытия,        |
//| OUTPUT: true-если запрос успешен и данные получены               |
//| REMARK: нет.                                                     |
//+------------------------------------------------------------------+
bool CSampleSignal::Request()
{
   ulong startdate = (ulong)TimeCurrent() - r_period;
   ulong enddate = (ulong)TimeCurrent();
   string req_str =
      "SELECT * FROM pdata WHERE rdate>=" +
      IntegerToString(startdate) +
      " and rdate<=" +
      IntegerToString(enddate) +
      " ORDER BY rdate DESC, pdate ASC";
   if(db_handle == 0) {
      Print("Ошибка подключения к БД");
      return false;
   }
   int request = DatabasePrepare(db_handle, req_str);
   if(request == INVALID_HANDLE) {
      Print("DB: ", db_name, " ошибка запроса с кодом ", GetLastError());
      return false;
   }
   bool success = DatabaseReadBind(request, prediction);
   DatabaseFinalize(request);
   return success;
}
//+------------------------------------------------------------------+
