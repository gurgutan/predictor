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
   long              cycle;   // ����������� ������ � �����
   datetime          last;    // "���������" ���������� �����
public:
                     CTimer() { active=false; };
                    ~CTimer() {};
   void              On(long c);     // ��������� �������
   void              Off() { active=false; };        // ���������� �������
   long              GetRound(); // ������� ���������� ������ ������ ������� � ��������� � ���������� ��������� � ���� �������
   bool              Check();    // ���������� ������, ���� ������ ������ ��� � �������� ���� ����
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
