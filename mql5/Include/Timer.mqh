//+------------------------------------------------------------------+
//|                                                        Timer.mqh |
//|                                                 Slepovichev Ivan |
//|                                               http://www.mama.ru |
//+------------------------------------------------------------------+
#property copyright "Slepovichev Ivan"
#property link      "http://www.mama.ru"
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
class CTimer
  {
protected:
   bool              active;
   long              cycle;   // корличество секунд в круге
   datetime          last;    // "последнее" запомненое время
public:
                     CTimer() { active=false; };
                    ~CTimer() {};
   void              On(long c);     // Включение таймера
   void              Off() { active=false; };        // Выключение таймера
   long              GetRound(); // Вернуть количество полных кругов таймера с прошедших с последнего обращения к этой функции
   bool              Check();    // возвращает истину, если прошло больше или в точности один круг
  };
//+------------------------------------------------------------------+
void CTimer::On(long c)
  {
   if(c<0) c=1;
   active=true;
   last=TimeCurrent();
   cycle=c;
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
long CTimer::GetRound(void)
  {
   if(!active) return(0);
   //if(last>TimeCurrent()) return(0);
   long d=TimeCurrent()-last;
   long res=(long)(d/cycle);
   if(res>0) last=TimeCurrent()-d%cycle;
   return(res);
  }
//+------------------------------------------------------------------+
bool CTimer::Check()
  {
   long res=GetRound();
   return(res>0);
  }
//+------------------------------------------------------------------+
