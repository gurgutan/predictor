import mtcommon
import dbcommon
from predictor import Predictor
import json
import sqlite3
import datetime as dt
import pandas as pd
from time import sleep
from timer import DelayTimer


class Server(object):
    def __init__(self):
        configname = "config.json"
        self.p = None
        self.ready = False
        self.mt = None
        self.lowdate = None
        with open(configname) as config_file:
            print("Загрузка конфигурации " + configname)
            data = json.load(config_file)
            self.dbname = data["dbname"]  # полное имя БД
            self.initialdate = dt.datetime.fromisoformat(data["initialdate"])
            self.modelname = data["modelname"]  # полное имя модели (с путем)
            self.symbol = data["symbol"]  # символ инструмента
            self.timeframe = data["timeframe"]  # тайм-фрэйм сервера
            self.delay = data["delay"]  # задержка в секундах цикла сервера
        if self.__init_db__() and self.__init_mt5__() and self.__init_predictor__():
            print("Сервер инициализирован")
            self.ready = True
        else:
            print("Ошибка инициализации сервера")
            self.ready = False

    def __init_db__(self):
        print("Открытие БД " + self.dbname)
        self.db = dbcommon.db_open(self.dbname)
        if self.db is None:
            print("Ошибка открытия БД '%s':" % self.dbname)
            return False
        return True

    def __init_mt5__(self):
        # подключимся к MetaTrader 5
        print("Подключение к терминалу MT5 ", end="")
        self.mt = mtcommon.initmt5()
        if self.mt is None:
            print("ошибка инициализации терминала MT5")
            return False
        print("версия MT5:", self.mt.version())
        return True

    def __init_predictor__(self):
        print("Загрузка и инициализация модели " + self.modelname)
        self.p = Predictor(modelname=self.modelname)
        if not self.p.trained:
            print("Ошибка инициализации модели '%s'" % self.modelname)
            return False
        return True

    def request_data(self):
        mt5rates = mtcommon.get_rates(self.mt, self.lowdate, dt.datetime.now())
        if mt5rates is None:
            return None
        rates = pd.DataFrame(mt5rates)
        print("Получено " + str(len(rates)) + " котировок")
        return rates

    def compute(self, rates):
        if len(rates) == 0:
            return None
        closes, times = rates["close"], rates["time"]
        last_idx = len(closes)
        shift = self.p.datainfo._in_size() + 1
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
                round(prob, 6),
            )
            results.append(db_row)
        print("Вычисления завершены. Получено " + str(len(results)) + " строк")
        return results

    def start(self):
        # подготовка данных
        date = dbcommon.db_get_lowdate(self.db)
        dtimer = DelayTimer(30)
        if date is None:
            self.lowdate = self.initialdate
        else:
            # нужно 257 пятиминутных баров
            delta = dt.timedelta(minutes=(self.p.datainfo._in_size() + 1) * 5)
            self.lowdate = date - delta
        while True:
            if not dtimer.elapsed():
                continue
            rates = self.request_data()
            if rates is None:
                print("Отсутствуют новые котировки")
                continue
            results = self.compute(rates)
            if not results is None:
                dbcommon.db_replace(self.db, results)


DEBUG = False
