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
        if self.db is None:
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
        if not self.p.trained:
            print("Ошибка инициализации модели '%s'" % self.modelname)
            return False
        print("Модель '%s' загружена" % self.modelname)
        return True

    def request_data(self):
        rates = pd.DataFrame(
            mtcommon.get_rates(self.mt, self.lowdate, dt.datetime.now())
        )
        print("Получено " + str(len(rates)) + " котировок")
        return rates

    def compute(self, rates):
        in_size = self.p.datainfo._in_size()
        closes = rates["close"]
        times = rates["time"]
        last_idx = len(closes)
        shift = in_size + 1
        results = []
        input_data = []
        for i in range(shift, last_idx):
            x = closes[i - shift : i].to_numpy()
            input_data.append(x)
        output_data = self.p.predict(input_data)
        if output_data is None:
            return None

        for i in range(shift, last_idx):
            plow, phigh, prob = output_data[i - shift]
            rdate = int(times[i - 1])
            rprice = closes[i - 1]
            pmodel = self.p.name
            pdate = int(rdate + 60 * 5 * self.p.datainfo.future)  # секунды*M5*future
            db_row = (
                rdate,
                round(rprice, 6),
                pmodel,
                pdate,
                round(rprice + plow, 6),
                round(rprice + phigh, 6),
                round(prob, 8),
            )
            results.append(db_row)
            if DEBUG:
                print(db_row)
        print("Вычисления завершены: " + str(len(results)) + " строк")
        return results

    def start(self):
        # подготовка данных
        dbvalue = dbcommon.db_get_lowdate(self.db)[0]
        if not dbvalue is None:
            self.lowdate = dt.datetime.fromtimestamp(int(dbvalue))
        else:
            self.lowdate = self.initialdate
        while True:
            rates = self.request_data()
            results = self.compute(rates)
            if not results is None:
                dbcommon.db_replace(self.db, results)
            if DEBUG:
                return 0


DEBUG = True
