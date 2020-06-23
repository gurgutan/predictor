import mtcommon
import dbcommon
from predictor import Predictor
import json
import sqlite3
import datetime as dt
import pandas as pd


class Server(object):
    def __init__(self):
        self.p = None
        self.ready = False
        self.mt = None
        self.lowdate = None
        with open("config.json") as config_file:
            data = json.load(config_file)
            self.dbname = data["dbname"]
            self.initialdate = dt.datetime.fromisoformat(data["initialdate"])
            self.modelname = data["modelname"]
            self.symbol = data["symbol"]
            self.timeframe = data["timeframe"]
        if self.__init_db__() and self.__init_mt5__() and self.__init_predictor__():
            print("Сервер инициализирован")
            self.ready = True
        else:
            print("Ошибка инициализации сервера")
            self.ready = False

    def __init_db__(self):
        self.db = dbcommon.db_open(self.dbname)
        if(self.db is None):
            print("Ошибка открытия БД '%s':" % self.dbname)
            return False
        return True

    def __init_mt5__(self):
        # подключимся к MetaTrader 5
        self.mt = mtcommon.initmt5()
        if self.mt is None:
            print("Ошибка инициализации mt5")
            return False
        print("Терминал MT5 инициализирован")
        print("  версия:", self.mt.version())
        return True

    def __init_predictor__(self):
        self.p = Predictor(modelname=self.modelname)
        if(not self.p.trained):
            print("Ошибка инициализации модели '%s'" % self.modelname)
            return False
        print("Модель '%s' загружена" % self.modelname)
        return True

    def request_data(self):
        rates = pd.DataFrame(
            mtcommon.get_rates(
                self.mt, self.lowdate, dt.datetime.utcnow()))
        print("Получено "+str(len(rates))+" котировок")
        return rates

    def compute(self, rates):
        in_size = self.p.datainfo._in_size()
        closes = rates[in_size+1:]['close'].array
        times = rates[in_size+1:]['time'].array
        last_idx = len(closes)-in_size-1
        results = []
        for i in range(last_idx):
            input_data = closes[i:i+in_size+1]
            output_data = self.p.infer(input_data)
            rdate = times[i+in_size]
            rprice = closes[i+in_size]
            pmodel = self.p.name
            pdate = rdate + 60*5*self.p.datainfo.future  # секунды*M5*future
            plow, phigh, prob = self.p.infer(input_data)
            db_row = (rdate, rprice, pmodel, pdate, plow, phigh, prob)
            results += db_row
            if(DEBUG):
                print(db_row)
        dbcommon.db_replace(self.db, results)

    def start(self):
        # подготовка данных
        self.lowdate = dbcommon.db_get_lowdate(self.db)[0]
        if(self.lowdate is None):
            self.lowdate = self.initialdate
        while True:
            self.compute(self.request_data())
            if DEBUG:
                return 0


DEBUG = True
